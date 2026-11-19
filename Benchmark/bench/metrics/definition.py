"""Executable metric definition types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteMetricDefinition:
    metric_id: str
    name: str
    judge_kind: str
    prompt_name: str
