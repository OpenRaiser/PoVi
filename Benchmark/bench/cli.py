from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

from .scoring.aggregate import aggregate_judgments
from .transcription.service import run_speech_transcription
from .media.video import extract_timestamped_frames
from .storage.json_files import write_json
from .providers import doctor_report
from .metrics.registry import DEFAULT_REGISTRY, MetricRegistry
from .runs import (
    HumanStatus,
    MachineStatus,
    PreparationStatus,
    RunState,
    RunStore,
    utc_now,
)
from .human_workflow import (
    export_human_eval,
    finalize_run,
    import_human_annotations,
    run_status,
)
from .pipeline.orchestrator import (
    evaluate_pipeline,
    machine_evaluate_pipeline,
    merge_judgment_files,
)
from .pipeline.preparation import prepare_pipeline, verify_preparation_contract
from .validation import (
    load_json,
    validate_gold_annotation,
    validate_pair,
    validate_judgments,
    validate_submission,
    validate_task_input,
)
from .metrics.runner import (
    build_context,
    deterministic_report,
    judge_av,
    judge_text,
    judge_visual,
    parse_metric_ids,
)
from .resources.loader import resource_path


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = resource_path("schemas")


def command_validate(args: argparse.Namespace) -> int:
    task_input_path = Path(args.task_input).resolve()
    submission_path = Path(args.submission).resolve()
    task_input = load_json(task_input_path)
    submission = load_json(submission_path)
    gold = load_json(Path(args.gold).resolve()) if args.gold else None
    errors = validate_task_input(task_input, SCHEMAS / "task_input.schema.json")
    if gold is not None:
        errors.extend(
            validate_gold_annotation(gold, SCHEMAS / "gold_annotation.schema.json")
        )
    errors.extend(
        validate_submission(
            submission,
            submission_path.parent,
            SCHEMAS / "submission.schema.json",
        )
    )
    errors.extend(validate_pair(task_input, submission, gold))
    report = {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "sample_id": task_input.get("sample_id"),
        "system_id": submission.get("system_id"),
    }
    if not errors and args.deterministic:
        report["deterministic"] = deterministic_report(
            build_context(task_input, submission, submission_path.parent),
            registry=MetricRegistry(args.registry),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        write_json(Path(args.output), report)
    return 0 if not errors else 1


def command_deterministic(args: argparse.Namespace) -> int:
    task_input_path = Path(args.task_input).resolve()
    submission_path = Path(args.submission).resolve()
    report = deterministic_report(
        build_context(
            load_json(task_input_path),
            load_json(submission_path),
            submission_path.parent,
        ),
        registry=MetricRegistry(args.registry),
    )
    if args.output:
        write_json(Path(args.output), report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_aggregate(args: argparse.Namespace) -> int:
    payload = load_json(Path(args.judgments))
    registry = MetricRegistry(args.registry)
    errors = validate_judgments(
        payload,
        SCHEMAS / "judgment.schema.json",
        registry,
    )
    if errors:
        raise ValueError("Invalid judgments: " + "; ".join(errors))
    judgments = payload.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError("judgments JSON must contain a judgments array")
    plan_payload = load_json(Path(args.plan)) if args.plan else {}
    active_metric_ids = (
        plan_payload.get("selected_active_metric_ids")
        or plan_payload.get("active_metric_ids")
    )
    required_review_metric_ids = (
        {
            metric_id
            for metric_id in active_metric_ids
            if metric_id in registry.metrics
            and (
                registry.get(metric_id).human_required
                or registry.get(metric_id).formal_channel == "learner"
            )
        }
        if isinstance(active_metric_ids, list)
        else None
    )
    report = aggregate_judgments(
        judgments,
        registry,
        bootstrap=args.bootstrap,
        seed=args.seed,
        track=args.track,
        setting=args.setting,
        include_diagnostics=args.include_diagnostics,
        active_metric_ids=active_metric_ids,
        required_review_metric_ids=required_review_metric_ids,
        expected_units_by_metric=(
            plan_payload.get("expected_review_units")
            if isinstance(
                plan_payload.get("expected_review_units"),
                dict,
            )
            else None
        ),
    )
    if args.output:
        write_json(Path(args.output), report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    report = doctor_report(probe_remote=args.probe_remote)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        write_json(Path(args.output), report)
    return 0 if report["ok"] else 1


def command_asr(args: argparse.Namespace) -> int:
    report = run_speech_transcription(
        Path(args.video),
        Path(args.output_dir),
        submission_path=Path(args.submission) if args.submission else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def command_extract_frames(args: argparse.Namespace) -> int:
    report = extract_timestamped_frames(
        Path(args.video),
        Path(args.output_dir),
        interval_seconds=args.interval_seconds,
        segment_seconds=args.segment_seconds,
        max_frames_per_segment=args.max_frames_per_segment,
    )
    summary = {
        "manifest": str(Path(args.output_dir).resolve() / "frame_manifest.json"),
        "frame_count": len(report["frames"]),
        "segment_count": len(report["segments"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_judge_text(args: argparse.Namespace) -> int:
    registry = MetricRegistry(args.registry)
    report = judge_text(
        Path(args.task_input),
        Path(args.gold),
        Path(args.submission),
        Path(args.output_dir),
        only=parse_metric_ids(args.metric, registry),
        system_text_override=(
            Path(args.system_text).read_text(encoding="utf-8")
            if args.system_text
            else None
        ),
        registry=registry,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def command_judge_visual(args: argparse.Namespace) -> int:
    registry = MetricRegistry(args.registry)
    report = judge_visual(
        Path(args.task_input),
        Path(args.gold),
        Path(args.submission),
        Path(args.frames),
        Path(args.output_dir),
        only=parse_metric_ids(args.metric, registry),
        resume=args.resume,
        registry=registry,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def command_judge_av(args: argparse.Namespace) -> int:
    registry = MetricRegistry(args.registry)
    report = judge_av(
        Path(args.task_input),
        Path(args.gold),
        Path(args.submission),
        Path(args.frames),
        Path(args.output_dir),
        only=parse_metric_ids(args.metric, registry),
        transcript_override=(
            Path(args.transcript).read_text(encoding="utf-8")
            if args.transcript
            else None
        ),
        resume=args.resume,
        registry=registry,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def command_merge_judgments(args: argparse.Namespace) -> int:
    merged = merge_judgment_files(
        [Path(value) for value in args.inputs], Path(args.output)
    )
    print(json.dumps({"output": str(Path(args.output).resolve()), "judgment_count": len(merged["judgments"])}, ensure_ascii=False, indent=2))
    return 0


def _registry_path(value: str | Path | None) -> Path:
    return Path(value or DEFAULT_REGISTRY).resolve()


def _registry_for_record(
    record: dict[str, Any], requested: str | Path | None = None
) -> MetricRegistry:
    options = record.get("options") or {}
    frozen = _registry_path(options.get("registry"))
    if requested is not None and _registry_path(requested) != frozen:
        raise ValueError("--registry is frozen when the run is created")
    return MetricRegistry(
        str(frozen),
        process_evaluation_enabled=not bool(options.get("no_process")),
    )


def _resolve_preparation_run(
    args: argparse.Namespace, store: RunStore
) -> tuple[dict[str, Any], Path, Path, Path, set[str] | None, bool]:
    if args.run_id and args.new_run_id:
        raise ValueError("--run-id and --new-run-id cannot be used together")
    if args.run_id:
        if any((args.task_input, args.gold, args.submission)):
            raise ValueError("--run-id cannot be combined with new input paths")
        record = store.load(args.run_id)
        if RunState(record["state"]) == RunState.FINALIZED:
            raise ValueError("A finalized run cannot be prepared again")
        stored_metrics = list((record.get("options") or {}).get("metric") or [])
        if args.metric is not None and list(args.metric) != stored_metrics:
            raise ValueError("--metric is frozen when the run is created")
        stored_no_process = bool((record.get("options") or {}).get("no_process"))
        if getattr(args, "no_process", False) != stored_no_process:
            raise ValueError("--no-process is frozen when the run is created")
        _registry_for_record(record, getattr(args, "registry", None))
        metric_values = stored_metrics or None
        return (
            record,
            store.input_path(record, "task_input"),
            store.input_path(record, "gold"),
            store.input_path(record, "submission"),
            parse_metric_ids(metric_values),
            True,
        )
    missing = [
        name
        for name in ("task_input", "gold", "submission")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(
            "New runs require --task-input, --gold, and --submission"
        )
    task_input = Path(args.task_input)
    gold = Path(args.gold)
    submission = Path(args.submission)
    record = store.create(
        task_input=task_input,
        gold=gold,
        submission=submission,
        run_id=args.new_run_id,
        options={
            "metric": list(args.metric or []),
            "registry": str(_registry_path(getattr(args, "registry", None))),
            "no_process": bool(getattr(args, "no_process", False)),
        },
    )
    return (
        record,
        task_input,
        gold,
        submission,
        parse_metric_ids(args.metric),
        False,
    )


def _verify_prepared_run_before_retry(
    store: RunStore,
    record: dict[str, Any],
    task_input: Path,
    gold: Path,
    submission: Path,
    registry: MetricRegistry,
) -> None:
    preparation_status = PreparationStatus(
        str(
            (record.get("preparation") or {}).get(
                "status", PreparationStatus.PENDING.value
            )
        )
    )
    if preparation_status in {
        PreparationStatus.COMPLETED,
        PreparationStatus.PARTIAL,
    }:
        verify_preparation_contract(
            store.run_dir(str(record["run_id"])),
            task_input,
            gold,
            submission,
            registry,
        )


def _save_preparation_result(
    store: RunStore,
    record: dict[str, Any],
    manifest: dict[str, Any],
    exit_code: int,
    *,
    force_asr: bool,
) -> None:
    preparation = manifest.get("preparation") or {}
    status = str(
        preparation.get("status")
        or ("COMPLETED" if exit_code == 0 else "FAILED")
    )
    PreparationStatus(status)
    store.set_phase_status(
        record,
        "preparation",
        status,
        operation="prepare",
        human_ready=bool(preparation.get("human_ready")),
        machine_ready=bool(preparation.get("machine_ready")),
        manifest=str(store.run_dir(str(record["run_id"])) / "run_manifest.json"),
        prepared_at=utc_now(),
        last_error=None,
    )
    asr_stage = (manifest.get("stages") or {}).get("asr") or {}
    asr_regenerated = force_asr and not bool(asr_stage.get("skipped"))
    if asr_regenerated:
        store.mark_machine_stale(
            record, reason="ASR output was regenerated after machine evaluation"
        )
    if exit_code != 0:
        store.record_phase_error(
            record,
            phase="preparation",
            operation="prepare",
            error="; ".join(manifest.get("errors") or ["preparation failed"]),
        )


def command_prepare(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    if args.run_id:
        run_id = str(args.run_id)
    else:
        record, task_input, gold, submission, only, resume = (
            _resolve_preparation_run(args, store)
        )
        run_id = str(record["run_id"])
    force_asr = "asr" in set(args.force_stage or [])
    with store.operation_lock(run_id, "prepare"):
        if args.run_id:
            record, task_input, gold, submission, only, resume = (
                _resolve_preparation_run(args, store)
            )
        registry = _registry_for_record(
            record, getattr(args, "registry", None)
        )
        _verify_prepared_run_before_retry(
            store, record, task_input, gold, submission, registry
        )
        try:
            manifest, exit_code = prepare_pipeline(
                task_input,
                gold,
                submission,
                store.run_dir(run_id),
                only=only,
                resume=resume,
                force_asr=force_asr,
                registry=registry,
            )
            _save_preparation_result(
                store, record, manifest, exit_code, force_asr=force_asr
            )
        except Exception as exc:
            store.record_phase_error(
                record,
                phase="preparation",
                operation="prepare",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
    current = store.load(run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "state": current["state"],
                "preparation": current["preparation"],
                "run_dir": str(store.run_dir(run_id)),
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exit_code


def command_machine_evaluate(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    run_id = str(args.run_id)
    force_stages = set(args.force_stage or [])
    with store.operation_lock(run_id, "machine-evaluate"):
        record = store.load(run_id)
        if RunState(record["state"]) == RunState.FINALIZED:
            raise ValueError("A finalized run cannot be evaluated again")
        task_input = store.input_path(record, "task_input")
        gold = store.input_path(record, "gold")
        submission = store.input_path(record, "submission")
        registry = _registry_for_record(
            record, getattr(args, "registry", None)
        )
        verify_preparation_contract(
            store.run_dir(run_id), task_input, gold, submission, registry
        )
        if (
            (record.get("machine") or {}).get("status")
            == MachineStatus.COMPLETED.value
            and not force_stages
        ):
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "state": record["state"],
                        "machine": record["machine"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if not bool((record.get("preparation") or {}).get("machine_ready")):
            raise ValueError(
                "machine-evaluate is blocked until required preparation, including ASR, succeeds"
            )
        plan = load_json(store.run_dir(run_id) / "evaluation_plan.json")
        only = set(plan.get("selected_active_metric_ids") or [])
        try:
            manifest, exit_code = machine_evaluate_pipeline(
                task_input,
                gold,
                submission,
                store.run_dir(run_id),
                only=only,
                resume=True,
                force_stages=force_stages,
                registry=registry,
            )
            status = (
                MachineStatus.COMPLETED.value
                if exit_code == 0
                else MachineStatus.FAILED.value
            )
            store.set_phase_status(
                record,
                "machine",
                status,
                operation="machine-evaluate",
                manifest=str(store.run_dir(run_id) / "run_manifest.json"),
                completed_at=utc_now() if exit_code == 0 else None,
                last_error=None,
            )
            if exit_code != 0:
                store.record_phase_error(
                    record,
                    phase="machine",
                    operation="machine-evaluate",
                    error="; ".join(
                        manifest.get("errors") or ["machine evaluation failed"]
                    ),
                )
        except Exception as exc:
            store.set_phase_status(
                record,
                "machine",
                MachineStatus.FAILED.value,
                operation="machine-evaluate",
                completed_at=None,
            )
            store.record_phase_error(
                record,
                phase="machine",
                operation="machine-evaluate",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
    current = store.load(run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "state": current["state"],
                "machine": current["machine"],
                "run_dir": str(store.run_dir(run_id)),
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exit_code


def command_evaluate(args: argparse.Namespace) -> int:
    """Compatibility command that runs prepare followed by machine evaluation."""
    store = RunStore(args.runs_dir)
    if args.run_id:
        run_id = str(args.run_id)
    else:
        record, task_input, gold, submission, only, resume = (
            _resolve_preparation_run(args, store)
        )
        run_id = str(record["run_id"])
    force_stages = set(args.force_stage or [])
    with store.operation_lock(run_id, "evaluate"):
        if args.run_id:
            record, task_input, gold, submission, only, resume = (
                _resolve_preparation_run(args, store)
            )
        registry = _registry_for_record(
            record, getattr(args, "registry", None)
        )
        _verify_prepared_run_before_retry(
            store, record, task_input, gold, submission, registry
        )
        if (
            (record.get("machine") or {}).get("status")
            == MachineStatus.COMPLETED.value
        ):
            store.mark_machine_stale(
                record, reason="Compatibility evaluate rerun started"
            )
        try:
            manifest, exit_code = evaluate_pipeline(
                task_input,
                gold,
                submission,
                store.run_dir(run_id),
                only=only,
                resume=resume,
                force_stages=force_stages,
                registry=registry,
            )
            _save_preparation_result(
                store,
                record,
                manifest,
                exit_code if not (manifest.get("preparation") or {}).get("machine_ready") else 0,
                force_asr="asr" in force_stages,
            )
            if bool((manifest.get("preparation") or {}).get("machine_ready")):
                machine_status = (
                    MachineStatus.COMPLETED.value
                    if exit_code == 0
                    else MachineStatus.FAILED.value
                )
                store.set_phase_status(
                    record,
                    "machine",
                    machine_status,
                    operation="evaluate",
                    manifest=str(store.run_dir(run_id) / "run_manifest.json"),
                    completed_at=utc_now() if exit_code == 0 else None,
                )
                if exit_code != 0:
                    store.record_phase_error(
                        record,
                        phase="machine",
                        operation="evaluate",
                        error="; ".join(
                            manifest.get("errors") or ["automatic evaluation failed"]
                        ),
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if bool((record.get("preparation") or {}).get("machine_ready")):
                store.set_phase_status(
                    record,
                    "machine",
                    MachineStatus.FAILED.value,
                    operation="evaluate",
                    completed_at=None,
                )
                store.record_phase_error(
                    record,
                    phase="machine",
                    operation="evaluate",
                    error=error,
                )
            else:
                store.record_error(
                    record,
                    operation="evaluate",
                    error=error,
                )
            raise
    current = store.load(run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "state": current["state"],
                "run_dir": str(store.run_dir(run_id)),
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exit_code

def command_human_export(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    with store.operation_lock(args.run_id, "human-export"):
        record = store.load(args.run_id)
        registry = _registry_for_record(record, args.registry)
        report = export_human_eval(store, args.run_id, registry)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_human_import(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    with store.operation_lock(args.run_id, "human-import"):
        record = store.load(args.run_id)
        registry = _registry_for_record(record, args.registry)
        report = import_human_annotations(
            store,
            args.run_id,
            Path(args.input),
            registry,
            annotator_id=args.annotator_id,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_status(args: argparse.Namespace) -> int:
    report = run_status(RunStore(args.runs_dir), args.run_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    with store.operation_lock(args.run_id, "finalize"):
        record = store.load(args.run_id)
        registry = _registry_for_record(record, args.registry)
        report = finalize_run(store, args.run_id, registry)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Poetry metaphor benchmark utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--probe-remote", action="store_true")
    doctor_parser.add_argument("--output")
    doctor_parser.set_defaults(func=command_doctor)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--task-input", required=True)
    validate_parser.add_argument("--submission", required=True)
    validate_parser.add_argument("--gold")
    validate_parser.add_argument("--output")
    validate_parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    validate_parser.add_argument(
        "--registry", default=str(DEFAULT_REGISTRY)
    )
    validate_parser.set_defaults(func=command_validate)

    deterministic_parser = subparsers.add_parser("deterministic")
    deterministic_parser.add_argument("--task-input", required=True)
    deterministic_parser.add_argument("--submission", required=True)
    deterministic_parser.add_argument("--output")
    deterministic_parser.add_argument(
        "--registry", default=str(DEFAULT_REGISTRY)
    )
    deterministic_parser.set_defaults(func=command_deterministic)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--judgments", required=True)
    aggregate_parser.add_argument("--output")
    aggregate_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    aggregate_parser.add_argument(
        "--plan",
        help="evaluation_plan.json；提供时按其中的激活原子执行严格完整性检查。",
    )
    aggregate_parser.add_argument("--bootstrap", type=int, default=1000)
    aggregate_parser.add_argument("--seed", type=int, default=20260718)
    aggregate_parser.add_argument("--track")
    aggregate_parser.add_argument("--setting")
    aggregate_parser.add_argument(
        "--include-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    aggregate_parser.set_defaults(func=command_aggregate)

    asr_parser = subparsers.add_parser("asr")
    asr_parser.add_argument("--video", required=True)
    asr_parser.add_argument("--output-dir", required=True)
    asr_parser.add_argument("--submission")
    asr_parser.set_defaults(func=command_asr)

    frames_parser = subparsers.add_parser("extract-frames")
    frames_parser.add_argument("--video", required=True)
    frames_parser.add_argument("--output-dir", required=True)
    frames_parser.add_argument("--interval-seconds", type=float, default=2.0)
    frames_parser.add_argument("--segment-seconds", type=float, default=30.0)
    frames_parser.add_argument("--max-frames-per-segment", type=int, default=16)
    frames_parser.set_defaults(func=command_extract_frames)

    text_parser = subparsers.add_parser("judge-text")
    text_parser.add_argument("--task-input", required=True)
    text_parser.add_argument("--gold", required=True)
    text_parser.add_argument("--submission", required=True)
    text_parser.add_argument("--output-dir", required=True)
    text_parser.add_argument("--system-text")
    text_parser.add_argument(
        "--metric",
        action="append",
        help="只运行指定原子指标 ID；可重复使用，例如 --metric S4。",
    )
    text_parser.add_argument(
        "--registry", default=str(DEFAULT_REGISTRY)
    )
    text_parser.set_defaults(func=command_judge_text)

    visual_parser = subparsers.add_parser("judge-visual")
    visual_parser.add_argument("--task-input", required=True)
    visual_parser.add_argument("--gold", required=True)
    visual_parser.add_argument("--submission", required=True)
    visual_parser.add_argument("--frames", required=True)
    visual_parser.add_argument("--output-dir", required=True)
    visual_parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    visual_parser.add_argument(
        "--metric",
        action="append",
        help="只运行指定原子指标 ID；可重复使用，例如 --metric R1。",
    )
    visual_parser.add_argument(
        "--registry", default=str(DEFAULT_REGISTRY)
    )
    visual_parser.set_defaults(func=command_judge_visual)

    av_parser = subparsers.add_parser("judge-av")
    av_parser.add_argument("--task-input", required=True)
    av_parser.add_argument("--gold", required=True)
    av_parser.add_argument("--submission", required=True)
    av_parser.add_argument("--frames", required=True)
    av_parser.add_argument("--output-dir", required=True)
    av_parser.add_argument("--transcript")
    av_parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    av_parser.add_argument(
        "--metric",
        action="append",
        help="只运行指定原子指标 ID；可重复使用，例如 --metric R6。",
    )
    av_parser.add_argument(
        "--registry", default=str(DEFAULT_REGISTRY)
    )
    av_parser.set_defaults(func=command_judge_av)

    merge_parser = subparsers.add_parser("merge-judgments")
    merge_parser.add_argument("--inputs", nargs="+", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.set_defaults(func=command_merge_judgments)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--task-input")
    prepare_parser.add_argument("--gold")
    prepare_parser.add_argument("--submission")
    prepare_parser.add_argument("--run-id", help="Resume preparation for an existing run.")
    prepare_parser.add_argument("--new-run-id")
    prepare_parser.add_argument("--runs-dir", default="runs")
    prepare_parser.add_argument("--metric", action="append")
    prepare_parser.add_argument("--registry")
    prepare_parser.add_argument("--no-process", action="store_true", default=False)
    prepare_parser.add_argument("--force-stage", action="append", choices=("asr",))
    prepare_parser.set_defaults(func=command_prepare)

    machine_parser = subparsers.add_parser("machine-evaluate")
    machine_parser.add_argument("--run-id", required=True)
    machine_parser.add_argument("--runs-dir", default="runs")
    machine_parser.add_argument("--registry")
    machine_parser.add_argument(
        "--force-stage",
        action="append",
        choices=("frames", "text", "visual", "av"),
    )
    machine_parser.set_defaults(func=command_machine_evaluate)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--task-input")
    evaluate_parser.add_argument("--gold")
    evaluate_parser.add_argument("--submission")
    evaluate_parser.add_argument(
        "--run-id",
        help="Resume preparation or machine evaluation for an existing run.",
    )
    evaluate_parser.add_argument(
        "--new-run-id",
        help="Specify the ID for a new run; an existing ID is rejected to avoid overwriting results.",
    )
    evaluate_parser.add_argument("--runs-dir", default="runs")
    evaluate_parser.add_argument("--registry")
    evaluate_parser.add_argument("--no-process", action="store_true", default=False)
    evaluate_parser.add_argument(
        "--metric",
        action="append",
        help="只运行指定底层指标 ID；可重复使用。",
    )
    evaluate_parser.add_argument("--force-stage", action="append", choices=("asr", "frames", "text", "visual", "av"))
    evaluate_parser.set_defaults(func=command_evaluate)

    human_export_parser = subparsers.add_parser("human-export")
    human_export_parser.add_argument("--run-id", required=True)
    human_export_parser.add_argument("--runs-dir", default="runs")
    human_export_parser.add_argument(
        "--registry"
    )
    human_export_parser.set_defaults(func=command_human_export)

    human_import_parser = subparsers.add_parser("human-import")
    human_import_parser.add_argument("--run-id", required=True)
    human_import_parser.add_argument(
        "--input",
        default="annotations.csv",
        help=(
            "标注文件；相对路径按 runs/<run_id>/human_eval/ 解析，"
            "默认 annotations.csv。"
        ),
    )
    human_import_parser.add_argument("--runs-dir", default="runs")
    human_import_parser.add_argument(
        "--annotator-id",
        help="导入简洁评分表时必填；评审者只需填写评分表中的分数。",
    )
    human_import_parser.add_argument(
        "--registry"
    )
    human_import_parser.set_defaults(func=command_human_import)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-id", required=True)
    status_parser.add_argument("--runs-dir", default="runs")
    status_parser.set_defaults(func=command_status)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-id", required=True)
    finalize_parser.add_argument("--runs-dir", default="runs")
    finalize_parser.add_argument(
        "--registry"
    )
    finalize_parser.set_defaults(func=command_finalize)
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
