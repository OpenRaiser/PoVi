"""Submission evaluability Gates.

Gate checks only required-artifact presence and readability. Protocol/schema
validation, path safety, and content quality are intentionally out of scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...artifacts import resolve_artifact_path
from ...media.probe import probe_audio, probe_video
from ..context import MetricContext


def _artifact_files(context: MetricContext, name: str) -> tuple[Path, ...]:
    value = context.artifacts.get(name)
    values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    result: list[Path] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            continue
        try:
            result.append(resolve_artifact_path(context.submission_dir, item))
        except ValueError:
            continue
    return tuple(result)


def required_artifact_presence(context: MetricContext) -> dict[str, Any]:
    option_status = []
    for option in context.route.required_artifacts:
        status = {
            name: bool(_artifact_files(context, name))
            and all(path.exists() and path.is_file() for path in _artifact_files(context, name))
            for name in sorted(option)
        }
        option_status.append(
            {"artifacts": sorted(option), "status": status, "passed": all(status.values())}
        )
    passed = any(item["passed"] for item in option_status)
    return _gate_result("G1.1", passed, {"required_options": option_status})


def text_readability(context: MetricContext) -> dict[str, Any]:
    options: list[dict[str, Any]] = []
    for option in context.route.required_artifacts:
        rows = []
        for name in sorted(option):
            files = _artifact_files(context, name)
            try:
                readable = bool(files) and all(path.read_text(encoding="utf-8").strip() for path in files)
            except (OSError, UnicodeError):
                readable = False
            rows.append({"artifact": name, "readable": readable})
        options.append(
            {"artifacts": sorted(option), "items": rows, "passed": bool(rows) and all(row["readable"] for row in rows)}
        )
    declared = any(all(name in context.artifacts for name in option) for option in context.route.required_artifacts)
    if not declared:
        return {"metric_id": "G2.1", "not_applicable": True}
    passed = any(item["passed"] for item in options)
    return _gate_result("G2.1", passed, {"options": options})


def video_readability(context: MetricContext) -> dict[str, Any]:
    path = context.artifact_path("final_video")
    if path is None:
        return {"metric_id": "G2.2", "not_applicable": True}
    report = probe_video(path)
    passed = bool(report.get("readable") and report.get("has_video_stream"))
    return _gate_result("G2.2", passed, report)


def audio_readability(context: MetricContext) -> dict[str, Any]:
    resolution = context.resolve_audio_artifact()
    if resolution.path is None:
        return {
            "metric_id": "G2.3",
            "not_applicable": True,
            "raw": {"artifact_resolution": resolution.as_dict()},
        }
    report = probe_audio(resolution.path)
    report["artifact_resolution"] = resolution.as_dict()
    return _gate_result("G2.3", bool(report.get("readable")), report)


def _gate_result(metric_id: str, passed: bool, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "unit_id": "sample",
        "score": 1.0 if passed else 0.0,
        "label": "pass" if passed else "fail",
        "critical": not passed,
        "evidence": {"gold_ids": [], "text_spans": [], "time_ranges": []},
        "raw": raw,
    }


RULE_EXECUTORS = {
    "G1.1": required_artifact_presence,
    "G2.1": text_readability,
    "G2.2": video_readability,
    "G2.3": audio_readability,
}
