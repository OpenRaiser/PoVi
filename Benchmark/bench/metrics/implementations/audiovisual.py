"""Cross-modal metrics (paper D3–D4, R6–R7)."""

from ..definition import RemoteMetricDefinition


def _metric(metric_id: str, name: str, kind: str = "av") -> RemoteMetricDefinition:
    return RemoteMetricDefinition(metric_id, name, kind, "av_alignment_judge.md")


DEFINITIONS = (
    _metric("D3", "Modality Complementarity"),
    _metric("D4", "Audiovisual Pacing"),
    _metric("R6", "Cross-modal Correspondence"),
    _metric("R7", "Narration-Event Synchrony"),
)
