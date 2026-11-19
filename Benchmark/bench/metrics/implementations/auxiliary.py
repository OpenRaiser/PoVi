"""Non-scoring auxiliary calculations."""

from __future__ import annotations

from typing import Any

from ...media.duration import duration_fit
from ...media.probe import probe_video
from ..context import MetricContext


def duration_fit_report(context: MetricContext) -> dict[str, Any]:
    path = context.artifact_path("final_video")
    task = context.task_input.get("task")
    target = task.get("target_duration_seconds") if isinstance(task, dict) else None
    if path is None or target is None:
        return {"report_id": "duration_fit", "not_applicable": True}
    report = probe_video(path)
    actual = float(report.get("container_duration_seconds") or report.get("duration_seconds") or 0.0)
    result = duration_fit(actual, target)
    return {
        "report_id": "duration_fit",
        "unit_id": "sample",
        "score": result["score"],
        "label": "duration_fit",
        "evidence": {"gold_ids": [], "text_spans": [], "time_ranges": []},
        "raw": result,
    }
