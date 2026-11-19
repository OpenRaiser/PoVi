"""Shared preparation stage for independent machine and human evaluation."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Iterable

from ..artifacts import has_submitted_content_text, resolve_artifact_path
from ..human_protocol import review_units_for_metric
from ..metrics.registry import MetricRegistry
from ..metrics.runner import build_context, run_rule_metrics
from ..progress import elapsed_seconds, emit_progress
from ..resources.loader import resource_path
from ..storage.atomic import sha256_file
from ..storage.json_files import write_json
from ..transcription.service import run_speech_transcription
from ..validation import (
    load_json,
    validate_gold_annotation,
    validate_pair,
    validate_submission,
    validate_task_input,
)
from .manifest import build_manifest, manifest_matches


SCHEMAS = resource_path("schemas")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_active_ids(plan: Any, only: set[str] | None) -> set[str]:
    active = set(plan.active_metric_ids)
    gates = {
        item.metric_id
        for item in plan.activations
        if item.active and item.stage == "gate"
    }
    return active if only is None else gates | (active & only)


def gate_passed(gates: dict[str, Any], metric_id: str) -> bool:
    item = (gates.get("metrics") or {}).get(metric_id)
    return bool(
        item and not item.get("not_applicable") and item.get("score") == 1
    )


def _walk_artifact_values(value: Any, prefix: str = "artifacts") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_artifact_values(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_artifact_values(child, f"{prefix}[{index}]")


def _artifact_fingerprints(
    submission: dict[str, Any], submission_dir: Path, context: Any
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for key, value in _walk_artifact_values(submission.get("artifacts") or {}):
        try:
            path = resolve_artifact_path(submission_dir, value)
        except ValueError:
            continue
        if path.is_file():
            result[key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    audio = context.resolve_audio_artifact().path
    if audio is not None and audio.is_file():
        result["__authoritative_audio__"] = {
            "path": str(audio.resolve()),
            "sha256": sha256_file(audio),
        }
    video = context.artifact_path("final_video")
    if video is not None and video.is_file():
        result["__authoritative_video__"] = {
            "path": str(video.resolve()),
            "sha256": sha256_file(video),
        }
    return dict(sorted(result.items()))


def verify_preparation_contract(
    output_dir: Path,
    task_input_path: Path,
    gold_path: Path,
    submission_path: Path,
    registry: MetricRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or MetricRegistry()
    output_dir = output_dir.resolve()
    manifest = load_json(output_dir / "run_manifest.json")
    plan = load_json(output_dir / "evaluation_plan.json")
    input_hashes = {
        "task_input": _sha256(task_input_path.resolve()),
        "gold": _sha256(gold_path.resolve()),
        "submission": _sha256(submission_path.resolve()),
    }
    if not manifest_matches(
        manifest,
        input_hashes=input_hashes,
        registry_version=registry.version,
        evaluation_plan=plan,
    ):
        raise ValueError(
            "Preparation contract no longer matches inputs, registry, or evaluation plan"
        )
    for name, entry in (manifest.get("prepared_assets") or {}).items():
        path = Path(str((entry or {}).get("path") or ""))
        expected = str((entry or {}).get("sha256") or "")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Prepared artifact changed or is missing: {name}")
    contract = manifest.get("preparation_contract") or {}
    for name, entry in (contract.get("outputs") or {}).items():
        path = Path(str((entry or {}).get("path") or ""))
        expected = str((entry or {}).get("sha256") or "")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Preparation output changed or is missing: {name}")
    return manifest


def prepare_pipeline(
    task_input_path: Path,
    gold_path: Path,
    submission_path: Path,
    output_dir: Path,
    *,
    only: set[str] | None = None,
    resume: bool = False,
    force_asr: bool = False,
    registry: MetricRegistry | None = None,
) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress("prepare", "start", "开始准备评测", output_dir=output_dir)
    input_paths = {
        "task_input": task_input_path.resolve(),
        "gold": gold_path.resolve(),
        "submission": submission_path.resolve(),
    }
    input_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    manifest_path = output_dir / "run_manifest.json"
    previous = load_json(manifest_path) if manifest_path.exists() else {}
    task = load_json(task_input_path)
    gold = load_json(gold_path)
    submission = load_json(submission_path)
    stages: dict[str, Any] = {}
    errors: list[str] = []

    validation_errors = validate_task_input(task, SCHEMAS / "task_input.schema.json")
    validation_errors.extend(
        validate_gold_annotation(gold, SCHEMAS / "gold_annotation.schema.json")
    )
    validation_errors.extend(
        validate_submission(
            submission,
            submission_path.resolve().parent,
            SCHEMAS / "submission.schema.json",
        )
    )
    validation_errors.extend(validate_pair(task, submission, gold))
    validation_report = {
        "ok": not validation_errors,
        "errors": sorted(set(validation_errors)),
    }
    write_json(output_dir / "00_validation.json", validation_report)
    stages["validate"] = validation_report
    if validation_errors:
        errors.extend(validation_errors)
        manifest = build_manifest(
            input_hashes=input_hashes,
            stages=stages,
            errors=errors,
            registry_version="unavailable",
            evaluation_plan={},
        )
        manifest["preparation"] = {
            "status": "FAILED",
            "human_ready": False,
            "machine_ready": False,
        }
        write_json(manifest_path, manifest)
        return manifest, 1

    registry = registry or MetricRegistry()
    plan = registry.build_evaluation_plan(submission)
    selected_active = _selected_active_ids(plan, only)
    plan_payload = plan.as_dict()
    plan_payload["selected_active_metric_ids"] = sorted(selected_active)
    plan_payload["expected_review_units"] = {
        metric_id: review_units_for_metric(submission, metric_id)
        for metric_id in selected_active
        if registry.get(metric_id).human_required
        or registry.get(metric_id).human_policy == "review"
        or registry.get(metric_id).formal_channel in {"human", "learner"}
    }
    write_json(output_dir / "evaluation_plan.json", plan_payload)
    stages["plan"] = {
        "ok": True,
        "active_metric_count": len(selected_active),
        "output": str(output_dir / "evaluation_plan.json"),
    }
    fingerprints_match = manifest_matches(
        previous,
        input_hashes=input_hashes,
        registry_version=registry.version,
        evaluation_plan=plan_payload,
    )

    context = build_context(task, submission, submission_path.resolve().parent)
    gate_ids = {
        metric_id
        for metric_id in selected_active
        if metric_id in registry.metrics and registry.get(metric_id).gate
    }
    gates = {
        "schema_version": "1.0",
        "metrics": run_rule_metrics(
            context, metric_ids=gate_ids, registry=registry
        ),
    }
    gate_path = output_dir / "01_deterministic" / "gates.json"
    write_json(gate_path, gates)
    text_ok = gate_passed(gates, "G2.1")
    video_ok = gate_passed(gates, "G2.2")
    audio_ok = gate_passed(gates, "G2.3")
    stages["gates"] = {
        "ok": True,
        "text_ok": text_ok,
        "video_ok": video_ok,
        "audio_ok": audio_ok,
        "output": str(gate_path),
    }

    audio_resolution = context.resolve_audio_artifact()
    stages["audio_source"] = audio_resolution.as_dict()
    submitted_text_available = has_submitted_content_text(
        submission, submission_path.resolve().parent
    )
    needs_asr = (
        any(
            registry.get(metric_id).domain in {"av", "content"}
            for metric_id in selected_active
            if metric_id in registry.metrics
        )
        and plan.route.track_id != "text_planning"
        and audio_ok
        and audio_resolution.path is not None
        and (not submitted_text_available or force_asr)
    )
    asr_error: str | None = None
    asr_dir = output_dir / "02_asr"
    asr_result = asr_dir / "asr_result.json"
    asr_transcript = asr_dir / "asr_transcript.txt"
    if needs_asr:
        try:
            hashes = (previous.get("stage_output_hashes") or {}).get("asr") or {}
            resumable = (
                resume
                and not force_asr
                and fingerprints_match
                and bool(((previous.get("stages") or {}).get("asr") or {}).get("ok"))
                and asr_result.is_file()
                and asr_transcript.is_file()
                and hashes.get("result_path") == sha256_file(asr_result)
                and hashes.get("transcript_path") == sha256_file(asr_transcript)
            )
            if resumable:
                if not asr_transcript.read_text(encoding="utf-8").strip():
                    raise ValueError("empty evaluator transcript")
                stages["asr"] = {
                    "ok": True,
                    "resumed": True,
                    "result_path": str(asr_result),
                    "transcript_path": str(asr_transcript),
                }
            else:
                stages["asr"] = run_speech_transcription(
                    audio_resolution.path, asr_dir
                )
                if (
                    not stages["asr"].get("ok", False)
                    or not asr_transcript.read_text(encoding="utf-8").strip()
                ):
                    raise ValueError("empty or unsuccessful evaluator transcript")
        except Exception as exc:
            asr_error = f"asr: {type(exc).__name__}: {exc}"
            stages["asr"] = {"ok": False, "error": asr_error}
            errors.append(asr_error)
    else:
        stages["asr"] = {"ok": True, "skipped": True, "required": False}

    human_ready = True
    machine_ready = asr_error is None
    preparation_status = "COMPLETED" if machine_ready else "PARTIAL"
    manifest = build_manifest(
        input_hashes=input_hashes,
        stages=stages,
        errors=errors,
        registry_version=registry.version,
        evaluation_plan=plan_payload,
    )
    manifest["prepared_assets"] = _artifact_fingerprints(
        submission, submission_path.resolve().parent, context
    )
    preparation_outputs = {
        "gates": {"path": str(gate_path), "sha256": sha256_file(gate_path)}
    }
    if asr_result.is_file():
        preparation_outputs["asr_result"] = {
            "path": str(asr_result),
            "sha256": sha256_file(asr_result),
        }
    if asr_transcript.is_file():
        preparation_outputs["asr_transcript"] = {
            "path": str(asr_transcript),
            "sha256": sha256_file(asr_transcript),
        }
    manifest["preparation_contract"] = {"outputs": preparation_outputs}
    manifest["preparation"] = {
        "status": preparation_status,
        "human_ready": human_ready,
        "machine_ready": machine_ready,
    }
    write_json(manifest_path, manifest)
    emit_progress(
        "prepare",
        "ok" if machine_ready else "partial",
        "准备阶段结束",
        errors=len(errors),
        elapsed_seconds=elapsed_seconds(started),
        manifest=manifest_path,
    )
    return manifest, 0 if machine_ready else 1
