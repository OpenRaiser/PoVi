"""The only automated metric execution entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..media.probe import probe_audio
from .context import MetricContext
from .engines import llm_judge
from .implementations.auxiliary import duration_fit_report
from .implementations.gates import RULE_EXECUTORS as GATE_RULES
from .registry import MetricRegistry


RULE_EXECUTORS = dict(GATE_RULES)
TEXT_METRICS = llm_judge.TEXT_METRICS
VISUAL_METRICS = llm_judge.VISUAL_METRICS
AV_METRICS = llm_judge.AV_METRICS


def validate_execution_contract(registry: MetricRegistry | None = None) -> list[str]:
    """Return configuration entries whose declared prelabel has no executor.

    The registry remains the scoring contract; this check deliberately verifies
    only machine prelabels.  Human formal channels need no runner.
    """
    registry = registry or MetricRegistry()
    executable_by_channel = {
        "rule": set(RULE_EXECUTORS),
        "llm": TEXT_METRICS,
        "mllm": VISUAL_METRICS | AV_METRICS,
    }
    errors: list[str] = []
    for spec in registry.metrics.values():
        channel = spec.prelabel_channel
        if channel in {"", "none"}:
            continue
        available = executable_by_channel.get(channel)
        if available is None:
            errors.append(f"{spec.metric_id}: unsupported prelabel channel {channel!r}")
        elif spec.metric_id not in available:
            errors.append(
                f"{spec.metric_id}: prelabel_channel={channel!r} has no runner executor"
            )
    return errors


def run_rule_metrics(
    context: MetricContext,
    *,
    metric_ids: Iterable[str] | None = None,
    registry: MetricRegistry | None = None,
) -> dict[str, dict[str, Any]]:
    registry = registry or MetricRegistry()
    active = set(registry.build_evaluation_plan(context.submission).active_metric_ids)
    selected = (
        active if metric_ids is None else active & {str(value) for value in metric_ids}
    )
    results: dict[str, dict[str, Any]] = {}
    for metric_id, evaluator in RULE_EXECUTORS.items():
        if metric_id not in selected:
            continue
        result = evaluator(context)
        results[str(result["metric_id"])] = result
    return results


def judge_text(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return llm_judge.judge_text(*args, **kwargs)


def judge_visual(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return llm_judge.judge_visual(*args, **kwargs)


def judge_av(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return llm_judge.judge_av(*args, **kwargs)


def parse_metric_ids(values: Iterable[str] | None) -> set[str] | None:
    return llm_judge.parse_metric_ids(values)


def deterministic_report(
    context: MetricContext,
    registry: MetricRegistry | None = None,
) -> dict[str, Any]:
    metrics = run_rule_metrics(context, registry=registry)
    video_result = metrics.get("G2.2")
    video_probe = video_result.get("raw") if video_result is not None else None
    audio_resolution = context.resolve_audio_artifact()
    audio_probe = (
        probe_audio(audio_resolution.path)
        if audio_resolution.path is not None
        else {}
    )
    audio_probe["artifact_resolution"] = audio_resolution.as_dict()
    auxiliary: dict[str, Any] = {}
    if context.route.track_id in {"audio_visual_generation", "full_task_teaching"}:
        duration = duration_fit_report(context)
        if not duration.get("not_applicable"):
            auxiliary["duration_fit"] = {
                "label": duration.get("label"),
                "raw": duration.get("raw"),
            }
    return {
        "schema_version": "0.4",
        "sample_id": context.task_input.get("sample_id"),
        "system_id": context.submission.get("system_id"),
        "route": {
            "track": context.route.track_id,
            "setting": context.route.setting,
            "public_name": context.route.public_name,
            "main_domains": list(context.route.main_domains),
        },
        "metrics": metrics,
        "auxiliary": auxiliary,
        "media": {"video": video_probe, "audio": audio_probe},
    }


def build_context(
    task_input: dict[str, Any],
    submission: dict[str, Any],
    submission_dir: Path,
) -> MetricContext:
    from ..tracks import resolve_route

    route = resolve_route(
        str(submission.get("track") or ""),
        str(submission.get("setting") or ""),
    )
    return MetricContext(
        task_input=task_input,
        submission=submission,
        submission_dir=submission_dir,
        route=route,
    )
