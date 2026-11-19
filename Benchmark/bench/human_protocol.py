"""Shared helpers for the single formal human-review protocol."""

from __future__ import annotations

from typing import Any


def scale_anchors(scale: str) -> list[dict[str, Any]]:
    if scale == "1_5":
        return [
            {"score": 1, "meaning": "严重失败，核心要求基本未满足"},
            {"score": 2, "meaning": "存在重大问题，仅少量要求满足"},
            {"score": 3, "meaning": "部分满足，仍有明显问题"},
            {"score": 4, "meaning": "基本满足，仅有轻微问题"},
            {"score": 5, "meaning": "充分满足，没有实质问题"},
        ]
    if scale in {"0_1", "rate"}:
        return [
            {"score": 0.0, "meaning": "完全不满足或比例为 0"},
            {"score": 0.25, "meaning": "少量满足"},
            {"score": 0.5, "meaning": "约一半满足"},
            {"score": 0.75, "meaning": "大部分满足"},
            {"score": 1.0, "meaning": "完全满足或比例为 1"},
        ]
    if scale == "pass_fail":
        return [
            {"score": 0, "meaning": "不通过"},
            {"score": 1, "meaning": "通过"},
        ]
    return [{"score": None, "meaning": f"按 {scale} 原生量表填写"}]


def review_units_for_metric(
    submission: dict[str, Any], metric_id: str
) -> list[str]:
    """Return run-scoped review units for the formal workflow."""
    modality = {
        "P6.I": "image",
        "P6.V": "video",
        "P6.A": "audio",
    }.get(metric_id)
    if modality is None:
        return ["sample"]
    artifacts = submission.get("artifacts") or {}
    pairs = (
        artifacts.get("prompt_media_pairs")
        if isinstance(artifacts, dict)
        else None
    )
    if not isinstance(pairs, list):
        return []
    return [
        str(item["pair_id"])
        for item in pairs
        if isinstance(item, dict)
        and item.get("modality") == modality
        and item.get("pair_id")
    ]
