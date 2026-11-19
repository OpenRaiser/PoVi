"""Visual realization metrics (paper R1–R5)."""

from ..definition import RemoteMetricDefinition


def _metric(metric_id: str, name: str, kind: str = "human_only") -> RemoteMetricDefinition:
    return RemoteMetricDefinition(metric_id, name, kind, "visual_judge.md")


DEFINITIONS = (
    _metric("R1", "Visual Requirement Coverage", "visual"),
    _metric("R2", "Visual Semantic Correctness", "visual"),
    _metric("R3", "Static Visual Quality", "visual"),
    _metric("R4", "Character Identity Continuity", "visual"),
    _metric("R5", "Visual Metaphor Realization", "visual"),
)
