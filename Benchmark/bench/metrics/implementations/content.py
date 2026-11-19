"""Semantic / narrative text metrics (paper S1–S6, D1–D2)."""

from ..definition import RemoteMetricDefinition


def _metric(metric_id: str, name: str, kind: str = "text") -> RemoteMetricDefinition:
    return RemoteMetricDefinition(metric_id, name, kind, "content_judge.md")


DEFINITIONS = (
    _metric("S1", "Information Coverage"),
    _metric("S2", "Metaphor Source"),
    _metric("S3", "Metaphor Target"),
    _metric("S4", "Mapping Bridge"),
    _metric("S5", "Metaphor Contextual Support"),
    _metric("S6", "Interpretive Calibration"),
    _metric("D1", "Narrative Structure"),
    _metric("D2", "Prelude-Explication Linkage"),
)
