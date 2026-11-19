"""Metric implementation catalog.

Configuration under ``bench.resources.metrics`` is the scoring contract.
This module only binds executable machine/rule implementations.
"""

from collections.abc import Iterable

from ..definition import RemoteMetricDefinition
from .audiovisual import DEFINITIONS as AUDIOVISUAL_DEFINITIONS
from .content import DEFINITIONS as CONTENT_DEFINITIONS
from .visual import DEFINITIONS as VISUAL_DEFINITIONS


ALL_DEFINITIONS = (
    *CONTENT_DEFINITIONS,
    *VISUAL_DEFINITIONS,
    *AUDIOVISUAL_DEFINITIONS,
)

REMOTE_METRICS: dict[str, RemoteMetricDefinition] = {}
for definition in ALL_DEFINITIONS:
    if definition.metric_id in REMOTE_METRICS:
        raise ValueError(f"Duplicate executable metric id: {definition.metric_id}")
    REMOTE_METRICS[definition.metric_id] = definition


def metric_ids_for(judge_kind: str) -> set[str]:
    return {
        metric_id
        for metric_id, definition in REMOTE_METRICS.items()
        if definition.judge_kind == judge_kind
    }


def select_metric_ids(
    judge_kind: str,
    selected: Iterable[str] | None = None,
) -> list[str]:
    available = metric_ids_for(judge_kind)
    if selected is None:
        return sorted(available)
    from ..registry import MetricRegistry

    registry = MetricRegistry()
    requested: set[str] = set()
    for value in selected:
        token = str(value).strip()
        if not token:
            continue
        requested.add(registry.resolve_metric_id(token))
    unknown = requested - set(REMOTE_METRICS)
    if unknown:
        raise ValueError(f"Unknown metric ids: {sorted(unknown)}")
    return sorted(available & requested)
