from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tracks import Route, resolve_route
from ..resources.loader import resource_path


DEFAULT_REGISTRY = resource_path("metrics")

INTERMEDIATE_ARTIFACTS = frozenset(
    {
        "planning_output",
        "storyboard",
        "prompts",
        "timeline",
        "character_assets",
        "intermediate_outputs",
        "system_input_snapshot",
        "timed_narration",
        "prompt_media_pairs",
    }
)

HUMAN_REVIEWED_APPLICABILITY = frozenset(
    {
        "delayed_test_conducted",
        "recurring_character",
    }
)


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    name: str
    domain: str
    description: str = ""
    default_weight: float = 1.0


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    family_id: str
    name: str
    domain: str
    method: str
    scale: str
    direction: str
    score_role: str
    stage: str
    formal_channel: str
    prelabel_channel: str
    human_policy: str
    routes: tuple[str, ...]
    rubric: str
    gate: bool = False
    auxiliary: bool = False
    optional: bool = False
    default_weight: float = 1.0
    requires_artifacts_any: tuple[str, ...] = ()
    requires_intermediate_any: tuple[str, ...] = ()
    requires_intermediate_count: int = 0
    requires_prompt_media_modality: str | None = None
    applicability: str | None = None
    experimental: bool = False
    enabled: bool = True
    accepted_source: str | None = None

    @property
    def diagnostic(self) -> bool:
        return self.score_role == "diagnostic"

    @property
    def human_required(self) -> bool:
        return self.human_policy == "required"

    def validate_score(self, score: Any) -> float:
        """Validate and return a metric score on its declared native scale."""
        if isinstance(score, bool):
            raise ValueError(
                f"Metric {self.metric_id} score must be numeric, not boolean"
            )
        try:
            numeric = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Metric {self.metric_id} score={score!r} is not numeric"
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(
                f"Metric {self.metric_id} score={score!r} must be finite"
            )

        ranges = {
            "0_1": (0.0, 1.0),
            "1_5": (1.0, 5.0),
            "0_100": (0.0, 100.0),
            "pass_fail": (0.0, 1.0),
        }
        bounds = ranges.get(self.scale)
        if bounds is None:
            raise ValueError(
                f"Metric {self.metric_id} has unsupported score scale "
                f"{self.scale!r}"
            )
        minimum, maximum = bounds
        if not minimum <= numeric <= maximum:
            raise ValueError(
                f"Metric {self.metric_id} score={numeric:g} is outside "
                f"native scale {self.scale} [{minimum:g}, {maximum:g}]"
            )
        if self.scale == "1_5" and not numeric.is_integer():
            raise ValueError(
                f"Metric {self.metric_id} score={numeric:g} must be an "
                "integer on native scale 1_5"
            )
        if self.scale == "pass_fail" and numeric not in {0.0, 1.0}:
            raise ValueError(
                f"Metric {self.metric_id} score={numeric:g} must be 0 or 1 "
                "on native scale pass_fail"
            )
        return numeric

    def normalize(self, score: float) -> float:
        """Map supported official-score scales to [0, 1]."""
        numeric = self.validate_score(score)
        if self.scale == "0_1":
            value = numeric
        elif self.scale == "1_5":
            # Paper convention: normalize 1–5 scores by dividing by 5.
            value = numeric / 5.0
        elif self.scale == "0_100":
            value = numeric / 100.0
        elif self.scale == "pass_fail":
            value = numeric
        else:
            raise ValueError(
                f"Metric {self.metric_id} scale={self.scale!r} "
                "has no generic normalization"
            )
        return value


@dataclass(frozen=True)
class MetricActivation:
    metric_id: str
    family_id: str
    active: bool
    reason: str
    stage: str
    formal_channel: str
    prelabel_channel: str
    human_policy: str
    experimental: bool
    enabled: bool
    accepted_source: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "family_id": self.family_id,
            "active": self.active,
            "reason": self.reason,
            "stage": self.stage,
            "formal_channel": self.formal_channel,
            "prelabel_channel": self.prelabel_channel,
            "human_policy": self.human_policy,
            "experimental": self.experimental,
            "enabled": self.enabled,
            "accepted_source": self.accepted_source,
        }


@dataclass(frozen=True)
class EvaluationPlan:
    route: Route
    activations: tuple[MetricActivation, ...]

    @property
    def active_metric_ids(self) -> tuple[str, ...]:
        return tuple(
            item.metric_id for item in self.activations if item.active
        )

    def metric_ids_for_stage(
        self, stage: str, *, prelabel_only: bool = False
    ) -> tuple[str, ...]:
        result = []
        for item in self.activations:
            if not item.active or item.stage != stage:
                continue
            if prelabel_only and item.prelabel_channel in {"none", ""}:
                continue
            result.append(item.metric_id)
        return tuple(result)

    def as_dict(self) -> dict[str, Any]:
        stages: dict[str, list[str]] = {}
        for item in self.activations:
            if item.active:
                stages.setdefault(item.stage, []).append(item.metric_id)
        return {
            "schema_version": "0.1",
            "route": {
                "track": self.route.track_id,
                "setting": self.route.setting,
                "public_name": self.route.public_name,
                "main_domains": list(self.route.main_domains),
            },
            "active_metric_ids": list(self.active_metric_ids),
            "stages": stages,
            "metrics": [item.as_dict() for item in self.activations],
        }


def _route_matches(pattern: str, route: Route) -> bool:
    if pattern == "*":
        return True
    track_pattern, _, setting_pattern = pattern.partition("/")
    return (
        track_pattern in {"*", route.track_id}
        and setting_pattern in {"*", route.setting}
    )


def _declared_artifacts(submission: dict[str, Any]) -> set[str]:
    artifacts = submission.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        return set()
    result = set()
    for name, value in artifacts.items():
        if isinstance(value, str) and value.strip():
            result.add(name)
        elif isinstance(value, list) and value:
            result.add(name)
    return result


def _prompt_media_modalities(submission: dict[str, Any]) -> set[str]:
    artifacts = submission.get("artifacts") or {}
    pairs = artifacts.get("prompt_media_pairs") if isinstance(artifacts, dict) else None
    if not isinstance(pairs, list):
        return set()
    return {
        str(item.get("modality"))
        for item in pairs
        if isinstance(item, dict) and item.get("modality")
    }


class MetricRegistry:
    def __init__(
        self,
        path: Path | str = DEFAULT_REGISTRY,
        *,
        process_evaluation_enabled: bool = True,
    ):
        self.process_evaluation_enabled = process_evaluation_enabled
        self.path = Path(path)
        files = (
            sorted(self.path.glob("*.json"))
            if self.path.is_dir()
            else [self.path]
        )
        if not files:
            raise ValueError(f"No metric definition files found in {self.path}")
        payloads = [json.loads(file.read_text(encoding="utf-8")) for file in files]
        versions = {str(payload["version"]) for payload in payloads}
        if len(versions) != 1:
            raise ValueError(
                f"Metric definition versions do not match: {sorted(versions)}"
            )
        self.version = versions.pop()
        self.domains: dict[str, dict[str, Any]] = {}
        self.families: dict[str, FamilySpec] = {}
        self.metrics: dict[str, MetricSpec] = {}
        for payload in payloads:
            domain = dict(payload["domain"])
            domain_id = str(domain.pop("id"))
            if domain_id in self.domains:
                raise ValueError(f"Duplicate domain id: {domain_id}")
            self.domains[domain_id] = domain
            for item in payload.get("families") or []:
                family = FamilySpec(
                    family_id=str(item["id"]),
                    name=str(item["name"]),
                    domain=domain_id,
                    description=str(item.get("description", item["name"])),
                    default_weight=float(item.get("default_weight", 1.0)),
                )
                if family.family_id in self.families:
                    raise ValueError(
                        f"Duplicate family id: {family.family_id}"
                    )
                self.families[family.family_id] = family
            for item in payload["metrics"]:
                score_role = str(
                    item.get(
                        "score_role",
                        "gate"
                        if item.get("gate")
                        else "auxiliary"
                        if item.get("auxiliary")
                        else "main",
                    )
                )
                spec = MetricSpec(
                    metric_id=str(item["id"]),
                    family_id=str(item["family_id"]),
                    name=str(item["name"]),
                    domain=domain_id,
                    method=str(item["method"]),
                    scale=str(item["scale"]),
                    direction=str(item["direction"]),
                    score_role=score_role,
                    stage=str(item.get("stage", domain_id)),
                    formal_channel=str(item.get("formal_channel", "rule")),
                    prelabel_channel=str(item.get("prelabel_channel", "none")),
                    human_policy=str(item.get("human_policy", "none")),
                    routes=tuple(str(value) for value in item.get("routes", ["*"])),
                    rubric=str(item.get("rubric", "")),
                    gate=bool(item.get("gate", score_role == "gate")),
                    auxiliary=bool(
                        item.get("auxiliary", score_role == "auxiliary")
                    ),
                    optional=bool(item.get("optional", False)),
                    default_weight=float(item.get("default_weight", 1.0)),
                    requires_artifacts_any=tuple(
                        str(value)
                        for value in item.get("requires_artifacts_any", [])
                    ),
                    requires_intermediate_any=tuple(
                        str(value)
                        for value in item.get("requires_intermediate_any", [])
                    ),
                    requires_intermediate_count=int(
                        item.get("requires_intermediate_count", 0)
                    ),
                    requires_prompt_media_modality=(
                        str(item["requires_prompt_media_modality"])
                        if item.get("requires_prompt_media_modality")
                        else None
                    ),
                    applicability=(
                        str(item["applicability"])
                        if item.get("applicability")
                        else None
                    ),
                    experimental=bool(item.get("experimental", False)),
                    enabled=bool(item.get("enabled", True)),
                    accepted_source=(
                        str(item["accepted_source"])
                        if item.get("accepted_source")
                        else None
                    ),
                )
                if (
                    spec.applicability
                    and spec.applicability not in HUMAN_REVIEWED_APPLICABILITY
                ):
                    raise ValueError(
                        f"Metric {spec.metric_id} has unsupported applicability "
                        f"{spec.applicability!r}"
                    )
                if spec.family_id not in self.families:
                    raise ValueError(
                        f"Metric {spec.metric_id} has unknown family "
                        f"{spec.family_id}"
                    )
                if self.families[spec.family_id].domain != domain_id:
                    raise ValueError(
                        f"Metric {spec.metric_id} family/domain mismatch"
                    )
                if spec.metric_id in self.metrics:
                    raise ValueError(f"Duplicate metric id: {spec.metric_id}")
                self.metrics[spec.metric_id] = spec

    def resolve_metric_id(self, metric_id: str) -> str:
        """Return the canonical registry id."""
        key = str(metric_id).strip()
        if key in self.metrics:
            return key
        upper = key.upper()
        if upper in self.metrics:
            return upper
        raise KeyError(f"Unknown metric id: {metric_id}")

    def get(self, metric_id: str) -> MetricSpec:
        try:
            return self.metrics[self.resolve_metric_id(metric_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown metric id: {metric_id}") from exc

    def main_metric_ids(self, domain: str | None = None) -> list[str]:
        result = []
        for spec in self.metrics.values():
            if not spec.enabled or spec.score_role != "main":
                continue
            if not self.domains.get(spec.domain, {}).get("main", False):
                continue
            if domain is not None and spec.domain != domain:
                continue
            result.append(spec.metric_id)
        return result

    def applicable_metric_ids(
        self,
        track: str,
        setting: str,
        *,
        include_diagnostics: bool = False,
        include_gates: bool = True,
        include_disabled: bool = False,
    ) -> list[str]:
        route = resolve_route(track, setting)
        result = []
        for spec in self.metrics.values():
            if not any(_route_matches(pattern, route) for pattern in spec.routes):
                continue
            if not spec.enabled:
                if include_disabled:
                    result.append(spec.metric_id)
                continue
            if spec.gate:
                if include_gates and spec.metric_id in route.gate_ids:
                    result.append(spec.metric_id)
                continue
            if spec.diagnostic:
                if include_diagnostics:
                    result.append(spec.metric_id)
                continue
            if spec.domain in route.main_domains:
                result.append(spec.metric_id)
        return result

    def build_evaluation_plan(
        self,
        submission: dict[str, Any],
        *,
        include_diagnostics: bool = True,
    ) -> EvaluationPlan:
        route = resolve_route(
            str(submission.get("track") or ""),
            str(submission.get("setting") or ""),
        )
        declared = _declared_artifacts(submission)
        intermediate = declared & INTERMEDIATE_ARTIFACTS
        prompt_modalities = _prompt_media_modalities(submission)
        diagnostics_enabled = bool(intermediate)
        activations: list[MetricActivation] = []
        for spec in self.metrics.values():
            active = True
            reason = "active"
            if not spec.enabled:
                active, reason = False, "experimental_disabled"
            elif not any(_route_matches(pattern, route) for pattern in spec.routes):
                active, reason = False, "route_not_applicable"
            elif spec.gate and spec.metric_id not in route.gate_ids:
                active, reason = False, "gate_not_required"
            elif spec.diagnostic and not self.process_evaluation_enabled:
                active, reason = False, "process_evaluation_paused"
            elif spec.diagnostic and (
                not include_diagnostics or not diagnostics_enabled
            ):
                active, reason = False, "no_traceable_intermediate_artifacts"
            elif (
                not spec.gate
                and not spec.diagnostic
                and spec.domain not in route.main_domains
            ):
                active, reason = False, "domain_not_scored_for_route"
            elif spec.requires_artifacts_any and not (
                set(spec.requires_artifacts_any) & declared
            ):
                active, reason = False, "required_artifact_not_declared"
            elif spec.requires_intermediate_any and not (
                set(spec.requires_intermediate_any) & intermediate
            ):
                active, reason = False, "required_intermediate_not_declared"
            elif (
                spec.requires_intermediate_count
                and len(intermediate) < spec.requires_intermediate_count
            ):
                active, reason = False, "insufficient_intermediate_artifacts"
            elif (
                spec.requires_prompt_media_modality
                and spec.requires_prompt_media_modality not in prompt_modalities
            ):
                active, reason = False, "prompt_media_pair_not_declared"
            elif spec.applicability:
                reason = "human_applicability_review"
            activations.append(
                MetricActivation(
                    metric_id=spec.metric_id,
                    family_id=spec.family_id,
                    active=active,
                    reason=reason,
                    stage=spec.stage,
                    formal_channel=spec.formal_channel,
                    prelabel_channel=spec.prelabel_channel,
                    human_policy=spec.human_policy,
                    experimental=spec.experimental,
                    enabled=spec.enabled,
                    accepted_source=spec.accepted_source,
                )
            )
        return EvaluationPlan(route=route, activations=tuple(activations))
