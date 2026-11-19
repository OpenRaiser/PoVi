"""Inter-rater agreement statistics for the benchmark's native scales.

The official score is the mean of two independent ratings unless adjudicated.
Accordingly, continuous 0-1 ratings report absolute-agreement ICC for both a
single rater and the mean of the raters.  Ordered 1-5 ratings additionally use
quadratic-weighted Cohen's kappa.  Statistics that require variation or more
than one paired item remain ``None`` instead of being presented as reliable.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable


RatingPair = tuple[float, float]


def _quadratic_weighted_kappa(
    pairs: list[RatingPair], categories: tuple[float, ...]
) -> float | None:
    if len(pairs) < 2 or len(categories) < 2:
        return None
    index = {value: position for position, value in enumerate(categories)}
    if any(left not in index or right not in index for left, right in pairs):
        return None

    size = len(categories)
    observed = [[0.0 for _ in categories] for _ in categories]
    left_counts = [0.0 for _ in categories]
    right_counts = [0.0 for _ in categories]
    for left, right in pairs:
        i = index[left]
        j = index[right]
        observed[i][j] += 1.0
        left_counts[i] += 1.0
        right_counts[j] += 1.0

    pair_count = float(len(pairs))
    max_distance = float((size - 1) ** 2)
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(size):
        for j in range(size):
            weight = ((i - j) ** 2) / max_distance
            observed_disagreement += weight * observed[i][j] / pair_count
            expected_disagreement += (
                weight * left_counts[i] * right_counts[j] / (pair_count**2)
            )
    if expected_disagreement <= 0.0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def _absolute_agreement_icc(
    pairs: list[RatingPair],
) -> tuple[float | None, float | None]:
    """Return ICC(A,1) and ICC(A,k) for a complete two-rater design."""
    target_count = len(pairs)
    rater_count = 2
    if target_count < 2:
        return None, None

    row_means = [mean(pair) for pair in pairs]
    column_means = [
        mean(pair[column] for pair in pairs) for column in range(rater_count)
    ]
    grand_mean = mean(row_means)
    ms_rows = (
        rater_count
        * sum((value - grand_mean) ** 2 for value in row_means)
        / (target_count - 1)
    )
    ms_columns = (
        target_count
        * sum((value - grand_mean) ** 2 for value in column_means)
        / (rater_count - 1)
    )
    residual = sum(
        (
            pairs[row][column]
            - row_means[row]
            - column_means[column]
            + grand_mean
        )
        ** 2
        for row in range(target_count)
        for column in range(rater_count)
    )
    ms_error = residual / ((target_count - 1) * (rater_count - 1))

    single_denominator = (
        ms_rows
        + (rater_count - 1) * ms_error
        + rater_count * (ms_columns - ms_error) / target_count
    )
    average_denominator = ms_rows + (ms_columns - ms_error) / target_count
    single = (
        (ms_rows - ms_error) / single_denominator
        if single_denominator > 0
        else None
    )
    average = (
        (ms_rows - ms_error) / average_denominator
        if average_denominator > 0
        else None
    )
    return single, average


def agreement_statistics(
    pairs: Iterable[RatingPair], scale: str
) -> dict[str, Any]:
    values = [(float(left), float(right)) for left, right in pairs]
    differences = [abs(left - right) for left, right in values]
    result: dict[str, Any] = {
        "pair_count": len(values),
        "mean_absolute_difference": mean(differences) if differences else None,
        "exact_agreement_rate": (
            sum(left == right for left, right in values) / len(values)
            if values
            else None
        ),
        "quadratic_weighted_kappa": None,
        "icc_absolute_agreement_single": None,
        "icc_absolute_agreement_average": None,
        "reliability_status": "insufficient_pairs",
    }
    if len(values) < 2:
        return result

    if scale == "1_5":
        result["quadratic_weighted_kappa"] = _quadratic_weighted_kappa(
            values, (1.0, 2.0, 3.0, 4.0, 5.0)
        )
    elif scale == "pass_fail":
        result["quadratic_weighted_kappa"] = _quadratic_weighted_kappa(
            values, (0.0, 1.0)
        )

    if scale in {"0_1", "rate", "1_5"}:
        single, average = _absolute_agreement_icc(values)
        result["icc_absolute_agreement_single"] = single
        result["icc_absolute_agreement_average"] = average

    computed = any(
        result[key] is not None
        for key in (
            "quadratic_weighted_kappa",
            "icc_absolute_agreement_single",
            "icc_absolute_agreement_average",
        )
    )
    result["reliability_status"] = (
        "estimated" if computed else "undefined_no_variation"
    )
    return result
