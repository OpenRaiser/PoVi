"""Canonical runtime result models used at module boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Judgment:
    judgment_id: str
    system_id: str
    sample_id: str
    metric_id: str
    unit_id: str
    score: float | None
    judge: dict[str, Any]
    evidence: dict[str, Any]
    label: str | None = None
    confidence: float | None = None
    not_applicable: bool = False
    critical: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Judgment":
        return cls(
            judgment_id=str(payload["judgment_id"]),
            system_id=str(payload["system_id"]),
            sample_id=str(payload["sample_id"]),
            metric_id=str(payload["metric_id"]),
            unit_id=str(payload.get("unit_id") or "sample"),
            score=float(payload["score"]) if payload.get("score") is not None else None,
            judge=dict(payload.get("judge") or {}),
            evidence=dict(payload.get("evidence") or {}),
            label=str(payload["label"]) if payload.get("label") is not None else None,
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
            not_applicable=bool(payload.get("not_applicable", False)),
            critical=bool(payload.get("critical", False)),
            raw=dict(payload.get("raw") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricResult:
    system_id: str
    sample_id: str
    metric_id: str
    official_score: float | None
    provisional_score: float | None
    not_applicable: bool
    status: str
    weight_total: float = 0.0
    critical_count: int = 0
    judgment_count: int = 0
    human_count: int = 0
    annotator_count: int = 0
    disagreement: bool = False
    missing_units: tuple[str, ...] = ()
    judgment_ids: tuple[str, ...] = ()


@dataclass
class EvaluationResult:
    systems: list[str]
    atoms: dict[str, Any]
    families: dict[str, Any]
    domains: dict[str, Any]
    gates: dict[str, Any]
    diagnostics: dict[str, Any]
    auxiliary: dict[str, Any]
    review_status: dict[str, Any]
    overall: dict[str, Any]
    review_queue: list[dict[str, Any]]
    metadata: dict[str, Any]
    schema_version: str = "0.4"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
