from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from ..scoring.aggregate import aggregate_judgments
from ..reporting import render_text_report
from ..artifacts import has_submitted_content_text, resolve_artifact_path
from ..transcription.service import run_speech_transcription
from ..media.video import extract_timestamped_frames
from ..storage.json_files import write_json
from ..storage.atomic import sha256_file
from ..progress import elapsed_seconds, emit_progress
from ..metrics.registry import EvaluationPlan, MetricRegistry
from ..human_protocol import review_units_for_metric
from ..validation import (
    load_json,
    validate_gold_annotation,
    validate_judgments,
    validate_pair,
    validate_submission,
    validate_task_input,
)
from ..metrics.runner import (
    AV_METRICS,
    TEXT_METRICS,
    VISUAL_METRICS,
    build_context,
    deterministic_report,
    judge_av,
    judge_text,
    judge_visual,
)
from ..resources.loader import resource_path
from .manifest import build_manifest, manifest_matches


SCHEMAS = resource_path("schemas")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _judgment_id(*parts: str) -> str:
    return "j_" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _stage_can_resume(
    expected: Path,
    *,
    resume: bool,
    fingerprints_match: bool,
    previous_manifest: dict[str, Any],
    stage: str,
    force_stages: set[str],
    manifest_key: str = "output",
    required_outputs: dict[str, Path] | None = None,
) -> bool:
    if not (resume and fingerprints_match and stage not in force_stages):
        return False
    previous_stage = (previous_manifest.get("stages") or {}).get(stage) or {}
    if not previous_stage.get("ok", False):
        return False
    paths = required_outputs or {manifest_key: expected}
    hashes = (previous_manifest.get("stage_output_hashes") or {}).get(stage) or {}
    return all(
        path.exists()
        and path.is_file()
        and hashes.get(key) == sha256_file(path)
        for key, path in paths.items()
    )


def deterministic_to_judgments(
    report: dict[str, Any],
    submission: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for metric_id, item in report.get("metrics", {}).items():
        if item.get("not_applicable", False):
            rows.append(
                {
                    "judgment_id": _judgment_id(
                        str(submission["sample_id"]),
                        str(submission["system_id"]),
                        str(metric_id),
                        "sample",
                        "rule",
                    ),
                    "sample_id": str(submission["sample_id"]),
                    "system_id": str(submission["system_id"]),
                    "track": str(submission["track"]),
                    "setting": str(submission["setting"]),
                    "metric_id": str(metric_id),
                    "unit_id": "sample",
                    "score": None,
                    "label": "not_applicable",
                    "weight": 1.0,
                    "not_applicable": True,
                    "critical": False,
                    "evidence": {
                        "gold_ids": [],
                        "text_spans": [],
                        "time_ranges": [],
                    },
                    "judge": {
                        "type": "rule",
                        "name": "bench.deterministic",
                        "version": "0.4",
                        "role": "primary",
                    },
                    "confidence": 1.0,
                    "needs_review": False,
                    "notes": "",
                    "raw": item.get("raw"),
                }
            )
            continue
        rows.append(
            {
                "judgment_id": _judgment_id(
                    str(submission["sample_id"]),
                    str(submission["system_id"]),
                    str(metric_id),
                    str(item.get("unit_id") or "sample"),
                    "rule",
                ),
                "sample_id": str(submission["sample_id"]),
                "system_id": str(submission["system_id"]),
                "track": str(submission["track"]),
                "setting": str(submission["setting"]),
                "metric_id": str(metric_id),
                "unit_id": str(item.get("unit_id") or "sample"),
                "score": item.get("score"),
                "label": str(item.get("label") or "rule"),
                "weight": 1.0,
                "not_applicable": False,
                "critical": bool(item.get("critical", False)),
                "evidence": item.get("evidence")
                or {
                    "gold_ids": [],
                    "text_spans": [],
                    "time_ranges": [],
                },
                "judge": {
                    "type": "rule",
                    "name": "bench.deterministic",
                    "version": "0.4",
                    "role": "primary",
                },
                "confidence": 1.0,
                "needs_review": bool(item.get("critical", False)),
                "notes": "",
                "raw": item.get("raw"),
            }
        )
    return {"schema_version": "0.4", "judgments": rows}


def merge_judgment_payloads(
    payloads: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in payloads:
        for item in payload.get("judgments") or []:
            judgment_id = str(item.get("judgment_id") or "")
            fallback = "|".join(
                (
                    str(item.get("system_id")),
                    str(item.get("sample_id")),
                    str(item.get("metric_id")),
                    str(item.get("unit_id")),
                    str((item.get("judge") or {}).get("type")),
                    str((item.get("judge") or {}).get("annotator_id")),
                )
            )
            key = judgment_id or fallback
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
    return {"schema_version": "0.4", "judgments": result}


def merge_judgment_files(
    paths: Iterable[Path], output_path: Path
) -> dict[str, Any]:
    payloads = [load_json(path) for path in paths]
    merged = merge_judgment_payloads(payloads)
    write_json(output_path, merged)
    return merged


def _active_ids(
    plan: EvaluationPlan,
    only: set[str] | None,
) -> set[str]:
    active = set(plan.active_metric_ids)
    gates = {
        item.metric_id
        for item in plan.activations
        if item.active and item.stage == "gate"
    }
    if only is None:
        return active
    return gates | (active & only)


def _gate_passed(
    deterministic_payload: dict[str, Any], metric_id: str
) -> bool:
    item = deterministic_payload.get("metrics", {}).get(metric_id)
    return bool(item and not item.get("not_applicable") and item.get("score") == 1)


def _frame_files_exist(manifest_path: Path) -> bool:
    try:
        payload = load_json(manifest_path)
        return all(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and (manifest_path.parent / item["path"]).is_file()
            for item in payload.get("frames") or []
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _eligible_human_metrics(
    plan: EvaluationPlan,
    registry: MetricRegistry,
    *,
    text_ok: bool,
    video_ok: bool,
    audio_ok: bool,
    selected: set[str],
) -> set[str]:
    result: set[str] = set()
    for metric_id in selected:
        spec = registry.get(metric_id)
        if spec.gate or spec.formal_channel == "learner":
            continue
        if spec.domain == "content":
            if plan.route.track_id == "text_planning" and not text_ok:
                continue
            if plan.route.track_id == "full_task_teaching" and not audio_ok:
                continue
        if spec.domain == "visual" and not video_ok:
            continue
        if spec.domain == "av" and not (video_ok and audio_ok):
            continue
        result.add(metric_id)
    return result


def machine_evaluate_pipeline(
    task_input_path: Path,
    gold_path: Path,
    submission_path: Path,
    output_dir: Path,
    *,
    only: set[str] | None = None,
    resume: bool = False,
    force_stages: set[str] | None = None,
    registry: MetricRegistry | None = None,
) -> tuple[dict[str, Any], int]:
    pipeline_started = time.monotonic()
    force_stages = force_stages or set()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(
        "pipeline",
        "start",
        "开始评测",
        output_dir=output_dir,
        resume=resume,
        only=",".join(sorted(only)) if only else "all",
    )
    input_paths = {
        "task_input": task_input_path.resolve(),
        "gold": gold_path.resolve(),
        "submission": submission_path.resolve(),
    }
    fingerprints = {
        name: _sha256(path) for name, path in input_paths.items()
    }
    manifest_path = output_dir / "run_manifest.json"
    previous = load_json(manifest_path) if manifest_path.exists() else {}
    fingerprints_match = False
    task = load_json(task_input_path)
    gold = load_json(gold_path)
    submission = load_json(submission_path)
    stages: dict[str, Any] = {}
    errors: list[str] = []

    validation_errors = validate_task_input(
        task, SCHEMAS / "task_input.schema.json"
    )
    validation_errors.extend(
        validate_gold_annotation(
            gold, SCHEMAS / "gold_annotation.schema.json"
        )
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
            input_hashes=fingerprints,
            stages=stages,
            errors=errors,
            registry_version="unavailable",
            evaluation_plan={},
        )
        write_json(manifest_path, manifest)
        return manifest, 1

    registry = registry or MetricRegistry()
    plan = registry.build_evaluation_plan(submission)
    selected_active = _active_ids(plan, only)
    plan_payload = plan.as_dict()
    plan_payload["selected_active_metric_ids"] = sorted(selected_active)
    plan_payload["expected_review_units"] = {
        metric_id: review_units_for_metric(submission, metric_id)
        for metric_id in selected_active
        if registry.get(metric_id).human_required
        or registry.get(metric_id).formal_channel in {"human", "learner"}
    }
    fingerprints_match = manifest_matches(
        previous,
        input_hashes=fingerprints,
        registry_version=registry.version,
        evaluation_plan=plan_payload,
    )
    write_json(output_dir / "evaluation_plan.json", plan_payload)
    stages["plan"] = {
        "ok": True,
        "active_metric_count": len(selected_active),
        "output": str(output_dir / "evaluation_plan.json"),
    }

    deterministic_dir = output_dir / "01_deterministic"
    deterministic_dir.mkdir(parents=True, exist_ok=True)
    metric_context = build_context(
        task, submission, submission_path.resolve().parent
    )
    deterministic_payload = deterministic_report(metric_context, registry=registry)
    write_json(deterministic_dir / "report.json", deterministic_payload)
    deterministic_judgments = deterministic_to_judgments(
        deterministic_payload, submission
    )
    deterministic_path = (
        deterministic_dir / "judgments.deterministic.json"
    )
    write_json(deterministic_path, deterministic_judgments)
    stages["deterministic"] = {
        "ok": True,
        "judgment_count": len(deterministic_judgments["judgments"]),
        "output": str(deterministic_path),
    }

    text_ok = _gate_passed(deterministic_payload, "G2.1")
    video_ok = _gate_passed(deterministic_payload, "G2.2")
    audio_ok = _gate_passed(deterministic_payload, "G2.3")
    presence_ok = _gate_passed(deterministic_payload, "G1.1")
    stages["gates"] = {
        "ok": presence_ok
        and all(
            _gate_passed(deterministic_payload, metric_id)
            for metric_id in plan.route.gate_ids
            if metric_id != "G1.1"
        ),
        "presence_ok": presence_ok,
        "text_ok": text_ok,
        "video_ok": video_ok,
        "audio_ok": audio_ok,
    }

    artifacts = submission.get("artifacts") or {}
    submitted_content_text = has_submitted_content_text(
        submission, submission_path.resolve().parent
    )
    video_candidate = metric_context.artifact_path("final_video")
    video_path = (
        video_candidate
        if video_candidate is not None and video_candidate.is_file()
        else None
    )
    audio_resolution = metric_context.resolve_audio_artifact()
    audio_source_path = audio_resolution.path
    stages["audio_source"] = audio_resolution.as_dict()

    transcript = ""
    evaluator_generated_transcript = False
    needs_audio_processing = any(
        registry.get(metric_id).domain in {"av", "content"}
        for metric_id in selected_active
        if metric_id in registry.metrics
    ) and plan.route.track_id != "text_planning" and (
        not submitted_content_text or "asr" in force_stages
    )
    if needs_audio_processing and audio_ok and audio_source_path is not None:
        asr_dir = output_dir / "02_asr"
        asr_expected = asr_dir / "asr_result.json"
        try:
            if _stage_can_resume(
                asr_expected,
                resume=resume,
                fingerprints_match=fingerprints_match,
                previous_manifest=previous,
                stage="asr",
                force_stages=force_stages,
                manifest_key="result_path",
                required_outputs={
                    "result_path": asr_expected,
                    "transcript_path": asr_dir / "asr_transcript.txt",
                },
            ):
                transcript = (asr_dir / "asr_transcript.txt").read_text(
                    encoding="utf-8"
                )
                if not transcript.strip():
                    raise ValueError("empty evaluator transcript")
                evaluator_generated_transcript = True
                stages["asr"] = {
                    "ok": True,
                    "resumed": True,
                    "result_path": str(asr_expected),
                    "transcript_path": str(asr_dir / "asr_transcript.txt"),
                }
            else:
                stages["asr"] = run_speech_transcription(
                    audio_source_path,
                    asr_dir,
                )
                transcript = (asr_dir / "asr_transcript.txt").read_text(
                    encoding="utf-8"
                )
                if not stages["asr"].get("ok", False) or not transcript.strip():
                    raise ValueError("empty or unsuccessful evaluator transcript")
                evaluator_generated_transcript = True
        except Exception as exc:
            message = f"asr: {type(exc).__name__}: {exc}"
            stages["asr"] = {"ok": False, "error": message}
            errors.append(message)
    elif isinstance(artifacts.get("asr_transcript"), str):
        # Submission-provided ASR is auxiliary only and never outranks evaluator ASR.
        try:
            transcript = resolve_artifact_path(
                submission_path.resolve().parent,
                artifacts["asr_transcript"],
            ).read_text(encoding="utf-8")
            stages["asr"] = {
                "ok": bool(transcript.strip()),
                "source": "submission_auxiliary",
            }
        except (OSError, UnicodeError, ValueError):
            stages["asr"] = {"ok": False, "source": "submission_auxiliary"}
    else:
        stages["asr"] = {"ok": True, "skipped": True}

    judgment_paths = [deterministic_path]
    text_ids = {
        metric_id
        for metric_id in selected_active
        if metric_id in TEXT_METRICS
        and registry.get(metric_id).prelabel_channel not in {"none", "rule"}
        and registry.get(metric_id).domain == "content"
        and (
            (plan.route.track_id == "text_planning" and text_ok)
            or (
                plan.route.track_id == "full_task_teaching"
                and audio_ok
                and (
                    evaluator_generated_transcript
                    or submitted_content_text
                )
            )
        )
    }
    text_dir = output_dir / "04_text"
    text_expected = text_dir / "judgments.text.json"
    if text_ids:
        try:
            if _stage_can_resume(
                text_expected,
                resume=resume,
                fingerprints_match=fingerprints_match,
                previous_manifest=previous,
                stage="text",
                force_stages=force_stages,
            ):
                stages["text"] = {"ok": True, "resumed": True, "output": str(text_expected)}
            else:
                stages["text"] = judge_text(
                    task_input_path,
                    gold_path,
                    submission_path,
                    text_dir,
                    only=text_ids,
                    system_text_override=(
                        transcript
                        if evaluator_generated_transcript
                        else None
                    ),
                    registry=registry,
                )
            if text_expected.exists():
                judgment_paths.append(text_expected)
            if not stages["text"].get("ok", False):
                errors.extend(
                    f"text: {message}"
                    for message in stages["text"].get("errors", [])
                )
        except Exception as exc:
            message = f"text: {type(exc).__name__}: {exc}"
            stages["text"] = {"ok": False, "error": message}
            errors.append(message)
    else:
        stages["text"] = {"ok": True, "skipped": True}

    visual_ids = {
        metric_id
        for metric_id in selected_active
        if metric_id in VISUAL_METRICS
    }
    av_ids = {
        metric_id
        for metric_id in selected_active
        if metric_id in AV_METRICS
    }
    frame_manifest_path: Path | None = None
    if video_ok and video_path is not None and (visual_ids or av_ids):
        frame_dir = output_dir / "03_frames"
        frame_manifest_path = frame_dir / "frame_manifest.json"
        try:
            if _stage_can_resume(
                frame_manifest_path,
                resume=resume,
                fingerprints_match=fingerprints_match,
                previous_manifest=previous,
                stage="frames",
                force_stages=force_stages,
                manifest_key="manifest",
            ) and _frame_files_exist(frame_manifest_path):
                frame_payload = load_json(frame_manifest_path)
                stages["frames"] = {
                    "ok": True,
                    "resumed": True,
                    "frame_count": len(frame_payload.get("frames") or []),
                    "manifest": str(frame_manifest_path),
                }
            else:
                frame_payload = extract_timestamped_frames(
                    video_path, frame_dir
                )
                stages["frames"] = {
                    "ok": True,
                    "frame_count": len(frame_payload["frames"]),
                    "manifest": str(frame_manifest_path),
                }
        except Exception as exc:
            message = f"frames: {type(exc).__name__}: {exc}"
            stages["frames"] = {"ok": False, "error": message}
            errors.append(message)
            frame_manifest_path = None
    else:
        stages["frames"] = {"ok": True, "skipped": True}

    if frame_manifest_path is not None and visual_ids:
        visual_dir = output_dir / "05_visual"
        expected_path = visual_dir / "judgments.visual.json"
        try:
            if _stage_can_resume(
                expected_path,
                resume=resume,
                fingerprints_match=fingerprints_match,
                previous_manifest=previous,
                stage="visual",
                force_stages=force_stages,
            ):
                stages["visual"] = {"ok": True, "resumed": True, "output": str(expected_path)}
            else:
                stages["visual"] = judge_visual(
                    task_input_path,
                    gold_path,
                    submission_path,
                    frame_manifest_path,
                    visual_dir,
                    only=visual_ids,
                    resume=resume and "visual" not in force_stages,
                    registry=registry,
                )
            if expected_path.exists():
                judgment_paths.append(expected_path)
            if not stages["visual"].get("ok", False):
                errors.extend(
                    f"visual: {message}"
                    for message in stages["visual"].get("errors", [])
                )
        except Exception as exc:
            message = f"visual: {type(exc).__name__}: {exc}"
            stages["visual"] = {"ok": False, "error": message}
            errors.append(message)
    else:
        stages["visual"] = {"ok": True, "skipped": True}

    if frame_manifest_path is not None and audio_ok and av_ids:
        av_dir = output_dir / "06_av"
        expected_path = av_dir / "judgments.av.json"
        try:
            if _stage_can_resume(
                expected_path,
                resume=resume,
                fingerprints_match=fingerprints_match,
                previous_manifest=previous,
                stage="av",
                force_stages=force_stages,
            ):
                stages["av"] = {"ok": True, "resumed": True, "output": str(expected_path)}
            else:
                stages["av"] = judge_av(
                    task_input_path,
                    gold_path,
                    submission_path,
                    frame_manifest_path,
                    av_dir,
                    only=av_ids,
                    transcript_override=(
                        transcript if evaluator_generated_transcript else None
                    ),
                    resume=resume and "av" not in force_stages,
                    registry=registry,
                )
            if expected_path.exists():
                judgment_paths.append(expected_path)
            if not stages["av"].get("ok", False):
                errors.extend(
                    f"av: {message}"
                    for message in stages["av"].get("errors", [])
                )
        except Exception as exc:
            message = f"av: {type(exc).__name__}: {exc}"
            stages["av"] = {"ok": False, "error": message}
            errors.append(message)
    else:
        stages["av"] = {"ok": True, "skipped": True}

    merged_dir = output_dir / "08_merged"
    merged_path = merged_dir / "judgments.json"
    merged = merge_judgment_files(judgment_paths, merged_path)
    judgment_errors = validate_judgments(
        merged, SCHEMAS / "judgment.schema.json", registry
    )
    if judgment_errors:
        errors.extend(judgment_errors)
    eligible_human = _eligible_human_metrics(
        plan,
        registry,
        text_ok=text_ok,
        video_ok=video_ok,
        audio_ok=audio_ok,
        selected=selected_active,
    )
    required_human = {
        metric_id
        for metric_id in eligible_human
        if registry.get(metric_id).human_required
        or registry.get(metric_id).formal_channel == "human"
    }
    report = aggregate_judgments(
        merged["judgments"],
        registry,
        track=str(submission["track"]),
        setting=str(submission["setting"]),
        include_diagnostics=False,
        active_metric_ids=selected_active,
        required_review_metric_ids=required_human,
        expected_units_by_metric={
            metric_id: review_units_for_metric(submission, metric_id)
            for metric_id in required_human
        },
        auxiliary_reports=deterministic_payload.get("auxiliary") or {},
    )
    write_json(output_dir / "auto_report.json", report)
    (output_dir / "auto_report.txt").write_text(
        render_text_report(report, merged["judgments"], registry), encoding="utf-8"
    )
    stages["merge"] = {
        "ok": not judgment_errors,
        "judgment_count": len(merged["judgments"]),
        "output": str(merged_path),
    }
    stages["aggregate"] = {
        "ok": True,
        "official_complete": bool(
            report.get("review_status", {}).get("complete")
        ),
        "pending_human": not bool(
            report.get("review_status", {}).get("complete")
        ),
        "output": str(output_dir / "auto_report.txt"),
    }
    manifest = build_manifest(
        input_hashes=fingerprints,
        stages=stages,
        errors=errors,
        registry_version=registry.version,
        evaluation_plan=plan_payload,
    )
    if previous.get("prepared_assets"):
        manifest["prepared_assets"] = previous["prepared_assets"]
    if previous.get("preparation_contract"):
        manifest["preparation_contract"] = previous["preparation_contract"]
    if previous.get("preparation"):
        manifest["preparation"] = previous["preparation"]
    write_json(manifest_path, manifest)
    emit_progress(
        "pipeline",
        "ok" if not errors else "partial",
        "评测结束",
        errors=len(errors),
        elapsed_seconds=elapsed_seconds(pipeline_started),
        manifest=manifest_path,
    )
    return manifest, 1 if errors else 0


def evaluate_pipeline(
    task_input_path: Path,
    gold_path: Path,
    submission_path: Path,
    output_dir: Path,
    *,
    only: set[str] | None = None,
    resume: bool = False,
    force_stages: set[str] | None = None,
    registry: MetricRegistry | None = None,
) -> tuple[dict[str, Any], int]:
    """Compatibility wrapper: prepare first, then run machine evaluation."""
    from .preparation import prepare_pipeline

    force_stages = force_stages or set()
    preparation, preparation_code = prepare_pipeline(
        task_input_path,
        gold_path,
        submission_path,
        output_dir,
        only=only,
        resume=resume,
        force_asr="asr" in force_stages,
        registry=registry,
    )
    if preparation_code != 0:
        return preparation, preparation_code
    return machine_evaluate_pipeline(
        task_input_path,
        gold_path,
        submission_path,
        output_dir,
        only=only,
        resume=True,
        force_stages=force_stages - {"asr"},
        registry=registry,
    )
