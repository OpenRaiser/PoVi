"""Duration-fit auxiliary calculations."""

from __future__ import annotations

from typing import Any


def duration_fit(
    actual_seconds: float,
    target: float | list[float] | tuple[float, ...],
) -> dict[str, Any]:
    if actual_seconds <= 0:
        return {"score": 0.0, "actual_seconds": actual_seconds, "target": target}
    if isinstance(target, (list, tuple)):
        values = [float(value) for value in target]
        if len(values) != 2:
            raise ValueError("Duration range must contain exactly two values")
        lower, upper = sorted(values)
        if lower <= actual_seconds <= upper:
            return {
                "score": 1.0,
                "actual_seconds": actual_seconds,
                "target_range": [lower, upper],
                "nearest_target_seconds": actual_seconds,
                "absolute_error_seconds": 0.0,
            }
        nearest = lower if actual_seconds < lower else upper
        score = min(actual_seconds, nearest) / max(actual_seconds, nearest)
        return {
            "score": score,
            "actual_seconds": actual_seconds,
            "target_range": [lower, upper],
            "nearest_target_seconds": nearest,
            "absolute_error_seconds": abs(actual_seconds - nearest),
        }
    target_seconds = float(target)
    if target_seconds <= 0:
        raise ValueError("Target duration must be positive")
    score = min(actual_seconds, target_seconds) / max(actual_seconds, target_seconds)
    return {
        "score": score,
        "actual_seconds": actual_seconds,
        "target_seconds": target_seconds,
        "ratio": actual_seconds / target_seconds,
        "absolute_error_seconds": abs(actual_seconds - target_seconds),
    }
