from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

from ..metrics.registry import MetricRegistry, MetricSpec
from ..results.models import EvaluationResult, MetricResult
from ..tracks import Route, resolve_route


def _normalized_item_score(
    item: dict[str, Any], spec: MetricSpec
) -> float | None:
    if item.get("not_applicable", False):
        return None
    score = item.get("score")
    if score is None:
        label = item.get("label")
        if spec.scale == "pass_fail":
            if label in {"pass", "passed", "true"}:
                score = 1.0
            elif label in {"fail", "failed", "false"}:
                score = 0.0
    if score is None:
        return None
    numeric = float(score)
    return spec.normalize(numeric)


def _weighted_score(
    items: list[dict[str, Any]], spec: MetricSpec
) -> tuple[float | None, float]:
    values: list[tuple[float, float]] = []
    for item in items:
        score = _normalized_item_score(item, spec)
        if score is None:
            continue
        weight = max(0.0001, float(item.get("weight") or 1.0))
        values.append((score, weight))
    if not values:
        return None, 0.0
    weight_total = sum(weight for _, weight in values)
    return (
        sum(score * weight for score, weight in values) / weight_total,
        weight_total,
    )


def _annotator_id(item: dict[str, Any]) -> str:
    judge = item.get("judge") or {}
    return str(judge.get("annotator_id") or judge.get("name") or "")


def _is_human(item: dict[str, Any]) -> bool:
    return str((item.get("judge") or {}).get("type")) == "human"


def _is_learner(item: dict[str, Any]) -> bool:
    return str((item.get("judge") or {}).get("type")) == "learner"


def _is_adjudicator(item: dict[str, Any]) -> bool:
    judge = item.get("judge") or {}
    return (
        str(judge.get("type") or "") in {"human", "learner"}
        and str(judge.get("role") or "") == "adjudicator"
    )


def _human_disagreement(
    items: list[dict[str, Any]], spec: MetricSpec
) -> bool:
    scored = [item for item in items if item.get("score") is not None]
    if len({_annotator_id(item) for item in scored}) < 2:
        return False
    critical = {bool(item.get("critical", False)) for item in scored}
    if len(critical) > 1:
        return True
    values = [float(item["score"]) for item in scored]
    if spec.scale == "1_5":
        return max(values) - min(values) >= 2.0
    if spec.scale in {"0_1", "pass_fail"}:
        return max(values) - min(values) > 0.25
    return False


def _reduce_group(
    system_id: str,
    sample_id: str,
    metric_id: str,
    items: list[dict[str, Any]],
    spec: MetricSpec,
) -> MetricResult:
    scored = [
        item
        for item in items
        if item.get("score") is not None
        and not bool(item.get("not_applicable", False))
    ]
    adjudicator_items = [item for item in items if _is_adjudicator(item)]
    adjudicated_na = bool(adjudicator_items) and all(
        bool(item.get("not_applicable", False)) for item in adjudicator_items
    )
    all_na = adjudicated_na or (
        bool(items)
        and not scored
        and all(bool(item.get("not_applicable", False)) for item in items)
    )
    adjudications = [item for item in scored if _is_adjudicator(item)]
    independent_human = [
        item
        for item in scored
        if _is_human(item) and not _is_adjudicator(item)
    ]
    learners = [item for item in scored if _is_learner(item)]
    machine = [
        item
        for item in scored
        if not _is_human(item) and not _is_learner(item)
    ]
    annotators = {
        _annotator_id(item)
        for item in independent_human
        if _annotator_id(item)
    }
    disagreement = _human_disagreement(independent_human, spec)

    adjudicated_score, adjudicated_weight = _weighted_score(
        adjudications, spec
    )
    human_score, human_weight = _weighted_score(independent_human, spec)
    learner_score, learner_weight = _weighted_score(learners, spec)
    formal_machine = [
        item
        for item in machine
        if str((item.get("judge") or {}).get("type"))
        == spec.formal_channel
    ]
    machine_score, machine_weight = _weighted_score(formal_machine, spec)
    any_machine_score, any_machine_weight = _weighted_score(machine, spec)

    official: float | None = None
    status = "missing"
    if all_na:
        status = "not_applicable"
    elif adjudicated_score is not None:
        official = adjudicated_score
        status = "adjudicated"
    elif spec.formal_channel == "learner":
        official = learner_score
        status = "complete" if official is not None else "awaiting_learner"
    elif spec.human_required or spec.formal_channel == "human":
        if len(annotators) >= 2 and not disagreement:
            official = human_score
            status = "human_complete"
        elif len(annotators) >= 2:
            status = "awaiting_adjudication"
        else:
            status = "awaiting_human"
    elif len(annotators) >= 2 and not disagreement:
        official = human_score
        status = "human_override"
    elif machine_score is not None:
        official = machine_score
        status = "complete"
    elif machine:
        status = "machine_scored_only"

    if all_na:
        provisional = None
    elif adjudicated_score is not None:
        provisional = adjudicated_score
    elif human_score is not None:
        provisional = human_score
    elif learner_score is not None:
        provisional = learner_score
    else:
        provisional = (
            machine_score
            if machine_score is not None
            else any_machine_score
        )

    weight_total = max(
        adjudicated_weight,
        human_weight,
        learner_weight,
        machine_weight,
        any_machine_weight,
    )
    return MetricResult(
        system_id=system_id,
        sample_id=sample_id,
        metric_id=metric_id,
        official_score=official,
        provisional_score=provisional,
        not_applicable=all_na,
        status=status,
        weight_total=weight_total,
        critical_count=sum(
            1 for item in items if bool(item.get("critical", False))
        ),
        judgment_count=len(items),
        human_count=len(independent_human) + len(adjudications),
        annotator_count=len(annotators),
        disagreement=disagreement,
    )


def reduce_to_sample_metrics(
    judgments: Iterable[dict[str, Any]],
    registry: MetricRegistry,
    expected_units_by_metric: dict[str, Iterable[str]] | None = None,
) -> list[MetricResult]:
    grouped: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for item in judgments:
        metric_id = str(item["metric_id"])
        registry.get(metric_id)
        key = (
            str(item["system_id"]),
            str(item["sample_id"]),
            metric_id,
            str(item.get("unit_id") or "sample"),
        )
        grouped[key].append(item)
    unit_rows: dict[
        tuple[str, str, str], list[tuple[str, MetricResult]]
    ] = defaultdict(list)
    for (
        system_id,
        sample_id,
        metric_id,
        unit_id,
    ), values in sorted(grouped.items()):
        unit_rows[(system_id, sample_id, metric_id)].append(
            (
                unit_id,
                _reduce_group(
                    system_id,
                    sample_id,
                    metric_id,
                    values,
                    registry.get(metric_id),
                ),
            )
        )

    result: list[MetricResult] = []
    for (system_id, sample_id, metric_id), unit_entries in sorted(
        unit_rows.items()
    ):
        expected_units = {
            str(value)
            for value in (
                (expected_units_by_metric or {}).get(metric_id) or ()
            )
        }
        if not expected_units:
            spec = registry.get(metric_id)
            sample_unit = next(
                (
                    row
                    for unit_id, row in unit_entries
                    if unit_id == "sample"
                ),
                None,
            )
            if (
                sample_unit is not None
                and sample_unit.official_score is not None
                and (
                    spec.human_required
                    or spec.formal_channel in {"human", "learner"}
                )
            ):
                expected_units = {"sample"}
        observed_units = {unit_id for unit_id, _ in unit_entries}
        missing_expected_units = expected_units - observed_units
        scoring_entries = (
            [
                (unit_id, row)
                for unit_id, row in unit_entries
                if unit_id in expected_units
            ]
            if expected_units
            else unit_entries
        )
        rows = [row for _, row in scoring_entries]
        applicable = [row for row in rows if not row.not_applicable]
        all_na = (
            bool(rows)
            and not applicable
            and not missing_expected_units
        )
        complete = (
            bool(applicable)
            and not missing_expected_units
            and all(
            row.official_score is not None for row in applicable
            )
        )
        official_values = [
            float(row.official_score)
            for row in applicable
            if row.official_score is not None
        ]
        provisional_values = [
            float(row.provisional_score)
            for row in applicable
            if row.provisional_score is not None
        ]
        if not provisional_values and expected_units:
            provisional_values = [
                float(row.provisional_score)
                for _, row in unit_entries
                if not row.not_applicable
                and row.provisional_score is not None
            ]
        statuses = {row.status for row in applicable}
        if missing_expected_units:
            spec = registry.get(metric_id)
            statuses.add(
                "awaiting_learner"
                if spec.formal_channel == "learner"
                else (
                    "awaiting_human"
                    if spec.human_required
                    or spec.formal_channel == "human"
                    else "missing"
                )
            )
        if all_na:
            status = "not_applicable"
        elif not complete:
            status = next(
                (
                    candidate
                    for candidate in (
                        "awaiting_adjudication",
                        "awaiting_learner",
                        "awaiting_human",
        "machine_scored_only",
                        "missing",
                    )
                    if candidate in statuses
                ),
                "missing",
            )
        elif "adjudicated" in statuses:
            status = "adjudicated"
        elif statuses <= {"human_complete", "human_override"}:
            status = "human_complete"
        else:
            status = "complete"
        result.append(
            MetricResult(
                system_id=system_id,
                sample_id=sample_id,
                metric_id=metric_id,
                official_score=mean(official_values)
                if complete and official_values
                else None,
                provisional_score=mean(provisional_values)
                if provisional_values
                else None,
                not_applicable=all_na,
                status=status,
                weight_total=sum(
                    row.weight_total for _, row in unit_entries
                ),
                critical_count=sum(
                    row.critical_count for _, row in unit_entries
                ),
                judgment_count=sum(
                    row.judgment_count for _, row in unit_entries
                ),
                human_count=sum(
                    row.human_count for _, row in unit_entries
                ),
                annotator_count=max(
                    (
                        row.annotator_count
                        for _, row in unit_entries
                    ),
                    default=0,
                ),
                disagreement=any(
                    row.disagreement for _, row in unit_entries
                ),
            )
        )
    return result


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    values: list[float],
    repetitions: int,
    rng: random.Random,
) -> tuple[float, float] | None:
    if repetitions <= 0 or len(values) < 2:
        return None
    draws = []
    for _ in range(repetitions):
        sample = [rng.choice(values) for _ in range(len(values))]
        draws.append(mean(sample))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def _report_atom(
    rows_by_sample: dict[str, MetricResult],
    sample_ids: list[str],
    *,
    bootstrap: int,
    rng: random.Random,
    scale: str,
    direction: str,
) -> dict[str, Any]:
    applicable = [
        row
        for row in rows_by_sample.values()
        if not row.not_applicable
    ]
    not_applicable_samples = sorted(
        sample_id
        for sample_id, row in rows_by_sample.items()
        if row.not_applicable
    )
    missing_samples = sorted(
        sample_id
        for sample_id in sample_ids
        if sample_id not in rows_by_sample
        or (
            not rows_by_sample[sample_id].not_applicable
            and rows_by_sample[sample_id].official_score is None
        )
    )
    official_values = [
        float(row.official_score)
        for row in applicable
        if row.official_score is not None
    ]
    provisional_values = [
        float(row.provisional_score)
        for row in applicable
        if row.provisional_score is not None
    ]
    complete = not missing_samples
    ci = bootstrap_ci(official_values, bootstrap, rng) if complete else None
    return {
        "score": mean(official_values)
        if complete and official_values
        else None,
        "provisional_score": mean(provisional_values)
        if provisional_values
        else None,
        "complete": complete,
        "sample_count": len(official_values),
        "expected_sample_count": len(sample_ids),
        "not_applicable_samples": not_applicable_samples,
        "missing_samples": missing_samples,
        "judgment_count": sum(row.judgment_count for row in rows_by_sample.values()),
        "human_judgment_count": sum(
            row.human_count for row in rows_by_sample.values()
        ),
        "critical_count": sum(
            row.critical_count for row in rows_by_sample.values()
        ),
        "disagreement_count": sum(
            1 for row in rows_by_sample.values() if row.disagreement
        ),
        "ci95": list(ci) if ci else None,
        "scale": scale,
        "direction": direction,
        "sample_status": {
            sample_id: row.status for sample_id, row in rows_by_sample.items()
        },
    }


def _weighted_average(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    weight_total = sum(weight for _, weight in values)
    return sum(value * weight for value, weight in values) / weight_total


def aggregate_judgments(
    judgments: list[dict[str, Any]],
    registry: MetricRegistry,
    *,
    bootstrap: int = 1000,
    seed: int = 20260718,
    track: str | None = None,
    setting: str | None = None,
    include_diagnostics: bool = False,
    active_metric_ids: Iterable[str] | None = None,
    required_review_metric_ids: Iterable[str] | None = None,
    expected_units_by_metric: dict[str, Iterable[str]] | None = None,
    auxiliary_reports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route: Route | None = None
    if (track is None) != (setting is None):
        raise ValueError("track and setting must be provided together")
    declared_routes = {
        (str(item["track"]), str(item["setting"]))
        for item in judgments
        if item.get("track") is not None and item.get("setting") is not None
    }
    if track is None and setting is None:
        if len(declared_routes) == 1:
            track, setting = next(iter(declared_routes))
        elif len(declared_routes) > 1:
            raise ValueError(
                "Cannot aggregate judgments from multiple Track/Setting routes"
            )
    elif declared_routes and declared_routes != {(track, setting)}:
        raise ValueError(
            "Judgment Track/Setting does not match the requested aggregation route"
        )

    if track is not None and setting is not None:
        route = resolve_route(track, setting)
        route_allowed = set(
            registry.applicable_metric_ids(
                route.track_id,
                route.setting,
                include_diagnostics=include_diagnostics,
            )
        )
    else:
        route_allowed = set(registry.metrics)
    if active_metric_ids is not None:
        expected = {str(value) for value in active_metric_ids}
    elif include_diagnostics:
        # Without an EvaluationPlan, diagnostics are evidence-driven only.
        # Merely enabling diagnostics must not activate every P metric.
        observed_diagnostics = {
            str(item.get("metric_id"))
            for item in judgments
            if str(item.get("metric_id")) in registry.metrics
            and registry.get(str(item.get("metric_id"))).diagnostic
        }
        expected = {
            metric_id
            for metric_id in route_allowed
            if not registry.get(metric_id).diagnostic
        } | observed_diagnostics
    else:
        expected = route_allowed
    allowed = route_allowed & expected
    filtered_judgments = [
        item for item in judgments if str(item.get("metric_id")) in allowed
    ]
    excluded_metric_ids = sorted(
        {
            str(item.get("metric_id"))
            for item in judgments
            if str(item.get("metric_id")) not in allowed
        }
    )
    sample_metrics = reduce_to_sample_metrics(
        filtered_judgments,
        registry,
        expected_units_by_metric=expected_units_by_metric,
    )
    rng = random.Random(seed)
    systems = sorted(
        {
            str(item.get("system_id"))
            for item in judgments
            if item.get("system_id") is not None
        }
    )
    samples_by_system: dict[str, list[str]] = {}
    for system_id in systems:
        samples_by_system[system_id] = sorted(
            {
                str(item.get("sample_id"))
                for item in judgments
                if str(item.get("system_id")) == system_id
            }
        )

    sample_lookup = {
        (item.system_id, item.sample_id, item.metric_id): item
        for item in sample_metrics
    }
    atoms: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    auxiliary: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    families: dict[str, Any] = {}
    domains: dict[str, Any] = {}

    for system_id in systems:
        sample_ids = samples_by_system[system_id]
        atoms[system_id] = {}
        gates[system_id] = {}
        auxiliary[system_id] = dict(auxiliary_reports or {})
        diagnostics[system_id] = {"atoms": {}, "families": {}}
        for metric_id in sorted(allowed):
            spec = registry.get(metric_id)
            rows = {
                sample_id: sample_lookup[(system_id, sample_id, metric_id)]
                for sample_id in sample_ids
                if (system_id, sample_id, metric_id) in sample_lookup
            }
            entry = _report_atom(
                rows,
                sample_ids,
                bootstrap=bootstrap,
                rng=rng,
                scale=spec.scale,
                direction=spec.direction,
            )
            if spec.gate:
                gate_values = [
                    row.official_score
                    for row in rows.values()
                    if row.official_score is not None
                ]
                gates[system_id][metric_id] = {
                    **entry,
                    "all_passed": bool(gate_values)
                    and entry["complete"]
                    and all(value >= 1.0 for value in gate_values),
                }
            elif spec.auxiliary:
                auxiliary[system_id][metric_id] = entry
            elif spec.diagnostic:
                diagnostics[system_id]["atoms"][metric_id] = entry
            else:
                atoms[system_id][metric_id] = entry

        family_sample_values: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        family_report: dict[str, Any] = {}
        diagnostic_family_report: dict[str, Any] = {}
        for family_id, family in registry.families.items():
            family_metrics = [
                registry.get(metric_id)
                for metric_id in sorted(allowed)
                if registry.get(metric_id).family_id == family_id
                and registry.get(metric_id).score_role in {"main", "diagnostic"}
            ]
            if not family_metrics:
                continue
            for sample_id in sample_ids:
                official_values: list[tuple[float, float]] = []
                provisional_values: list[tuple[float, float]] = []
                missing: list[str] = []
                applicable_count = 0
                for spec in family_metrics:
                    row = sample_lookup.get(
                        (system_id, sample_id, spec.metric_id)
                    )
                    if row is not None and row.not_applicable:
                        continue
                    applicable_count += 1
                    if row is None or row.official_score is None:
                        missing.append(spec.metric_id)
                    else:
                        official_values.append(
                            (row.official_score, spec.default_weight)
                        )
                    if row is not None and row.provisional_score is not None:
                        provisional_values.append(
                            (row.provisional_score, spec.default_weight)
                        )
                family_sample_values[(family_id, sample_id)] = {
                    "score": _weighted_average(official_values)
                    if not missing and applicable_count
                    else None,
                    "provisional_score": _weighted_average(
                        provisional_values
                    ),
                    "complete": not missing,
                    "not_applicable": applicable_count == 0,
                    "missing_atomic_ids": missing,
                }
            applicable_samples = [
                value
                for (fid, _), value in family_sample_values.items()
                if fid == family_id and not value["not_applicable"]
            ]
            complete = all(value["complete"] for value in applicable_samples)
            official_values = [
                value["score"]
                for value in applicable_samples
                if value["score"] is not None
            ]
            provisional_values = [
                value["provisional_score"]
                for value in applicable_samples
                if value["provisional_score"] is not None
            ]
            entry = {
                "score": mean(official_values)
                if complete and official_values
                else None,
                "provisional_score": mean(provisional_values)
                if provisional_values
                else None,
                "complete": complete,
                "not_applicable": not applicable_samples,
                "sample_count": len(official_values),
                "expected_atomic_ids": [
                    spec.metric_id for spec in family_metrics
                ],
                "missing_atomic_ids": sorted(
                    {
                        metric_id
                        for value in applicable_samples
                        for metric_id in value["missing_atomic_ids"]
                    }
                ),
            }
            if family.domain == "process":
                diagnostic_family_report[family_id] = entry
            elif family.domain != "gate":
                family_report[family_id] = entry
        families[system_id] = family_report
        diagnostics[system_id]["families"] = diagnostic_family_report

        domain_report: dict[str, Any] = {}
        expected_domains = (
            route.main_domains
            if route is not None
            else tuple(
                domain
                for domain, meta in registry.domains.items()
                if meta.get("main", False)
            )
        )
        for domain in expected_domains:
            expected_domain_families = [
                family
                for family in registry.families.values()
                if family.domain == domain
                and family.family_id in family_report
            ]
            domain_families = [
                family
                for family in expected_domain_families
                if not family_report[family.family_id]["not_applicable"]
            ]
            not_applicable_families = [
                family.family_id
                for family in expected_domain_families
                if family_report[family.family_id]["not_applicable"]
            ]
            missing_families = [
                family.family_id
                for family in domain_families
                if family_report[family.family_id]["score"] is None
            ]
            official_values = [
                (
                    float(family_report[family.family_id]["score"]),
                    family.default_weight,
                )
                for family in domain_families
                if family_report[family.family_id]["score"] is not None
            ]
            provisional_values = [
                (
                    float(
                        family_report[family.family_id][
                            "provisional_score"
                        ]
                    ),
                    family.default_weight,
                )
                for family in domain_families
                if family_report[family.family_id][
                    "provisional_score"
                ]
                is not None
            ]
            complete = bool(domain_families) and not missing_families
            domain_report[domain] = {
                "score": _weighted_average(official_values)
                if complete
                else None,
                "provisional_score": _weighted_average(
                    provisional_values
                ),
                "complete": complete,
                "expected_family_ids": [
                    family.family_id for family in expected_domain_families
                ],
                "not_applicable_family_ids": not_applicable_families,
                "missing_family_ids": missing_families,
            }
        domains[system_id] = domain_report

    overall: dict[str, Any] = {}
    if route is not None and route.track_id == "full_task_teaching":
        for system_id in systems:
            values = [
                domains.get(system_id, {}).get(domain, {}).get("score")
                for domain in route.main_domains
            ]
            provisional_values = [
                domains.get(system_id, {})
                .get(domain, {})
                .get("provisional_score")
                for domain in route.main_domains
            ]
            available = [float(value) for value in values if value is not None]
            provisional_available = [
                float(value)
                for value in provisional_values
                if value is not None
            ]
            complete = len(available) == len(route.main_domains)
            overall[system_id] = {
                "arithmetic_mean": mean(available) if complete else None,
                "geometric_mean": (
                    math.prod(available)
                    ** (1.0 / len(available))
                    if complete
                    else None
                ),
                "minimum_domain": min(available) if complete else None,
                "provisional_arithmetic_mean": mean(provisional_available)
                if provisional_available
                else None,
                "domain_count": len(available),
                "complete_expected_domains": complete,
            }

    review_queue = [
        item
        for item in filtered_judgments
        if bool(item.get("needs_review", False))
        or bool(item.get("critical", False))
        or (
            item.get("confidence") is not None
            and float(item["confidence"]) < 0.6
        )
    ]
    awaiting_set = {
        (item.system_id, item.sample_id, item.metric_id, item.status)
        for item in sample_metrics
        if item.status
        in {"awaiting_human", "awaiting_adjudication", "awaiting_learner"}
    }
    required_review = {
        str(metric_id)
        for metric_id in (required_review_metric_ids or ())
        if str(metric_id) in allowed
    }
    for system_id in systems:
        for sample_id in samples_by_system[system_id]:
            for metric_id in required_review:
                row = sample_lookup.get((system_id, sample_id, metric_id))
                if row is not None and (
                    row.not_applicable or row.official_score is not None
                ):
                    continue
                spec = registry.get(metric_id)
                status = (
                    row.status
                    if row is not None
                    else (
                        "awaiting_learner"
                        if spec.formal_channel == "learner"
                        else "awaiting_human"
                    )
                )
                awaiting_set.add((system_id, sample_id, metric_id, status))
    awaiting = sorted(awaiting_set)
    return EvaluationResult(
        systems=systems,
        atoms=atoms,
        families=families,
        domains=domains,
        gates=gates,
        diagnostics=diagnostics,
        auxiliary=auxiliary,
        overall=overall,
        review_status={
            "complete": not awaiting,
            "awaiting": [
                {
                    "system_id": system_id,
                    "sample_id": sample_id,
                    "metric_id": metric_id,
                    "status": status,
                }
                for system_id, sample_id, metric_id, status in awaiting
            ],
            "review_queue_count": len(review_queue),
        },
        review_queue=review_queue,
        metadata={
            "registry_version": registry.version,
            "bootstrap_repetitions": bootstrap,
            "bootstrap_seed": seed,
            "aggregation_unit": "judgment_unit_to_atom_to_family_to_domain",
            "family_policy": "registry_weighted_applicable_main_atoms",
            "domain_policy": "registry_weighted_applicable_families",
            "track": route.track_id if route is not None else None,
            "setting": route.setting if route is not None else None,
            "track_name": route.public_name if route is not None else None,
            "expected_main_domains": list(route.main_domains)
            if route is not None
            else [],
            "active_metric_ids": sorted(allowed),
            "include_diagnostics": include_diagnostics,
            "excluded_metric_ids": excluded_metric_ids,
            "overall_policy": "five_domain_full_task_only",
            "missing_policy": "strict_null_with_provisional",
        },
    ).to_dict()
