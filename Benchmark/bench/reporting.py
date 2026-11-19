"""Compact, human-readable benchmark reports."""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable

from .metrics.registry import MetricRegistry


def _channel_score(
    items: Iterable[dict[str, Any]], registry: MetricRegistry, metric_id: str
) -> tuple[float, float] | None:
    """Return native-scale and normalized macro means for one atomic metric."""
    native_values: list[float] = []
    normalized_values: list[float] = []
    for item in items:
        score = item.get("score")
        if score is None or bool(item.get("not_applicable")):
            continue
        if str(item.get("metric_id") or "") != metric_id:
            continue
        spec = registry.get(metric_id)
        if spec.scale not in {"0_1", "1_5", "pass_fail"}:
            continue
        value = float(score)
        native_values.append(value)
        normalized_values.append(spec.normalize(value))
    if not native_values:
        return None
    return mean(native_values), mean(normalized_values)


def _format_score(value: tuple[float, float] | None, scale: str) -> str:
    if value is None:
        return "未评分"
    native, normalized = value
    if scale == "1_5":
        return f"{native:.3f}/5（标准化 {normalized:.3f}）"
    if scale == "pass_fail":
        return f"{'通过' if native >= 1 else '未通过'}（{native:.0f}）"
    return f"{native:.3f}"


def render_text_report(
    report: dict[str, Any],
    judgments: Iterable[dict[str, Any]],
    registry: MetricRegistry,
) -> str:
    """Render every evaluated atomic metric and every available score channel."""
    rows = list(judgments)
    systems = list(report.get("systems") or [])
    lines = [
        "古诗词隐喻理解音视频生成 Benchmark 评分报告",
        "评分说明：LLM/MLLM 与人类专家对同一待评材料独立评分；各评分渠道分别呈现。",
        "",
    ]
    for system_id in systems:
        lines.append(f"系统：{system_id}")
        lines.append("")
        metric_ids = set((report.get("atoms") or {}).get(system_id, {}))
        metric_ids.update((report.get("gates") or {}).get(system_id, {}))
        metric_ids.update(
            ((report.get("diagnostics") or {}).get(system_id, {}).get("atoms") or {})
        )
        metric_ids.update(
            str(item.get("metric_id"))
            for item in rows
            if str(item.get("system_id")) == system_id
            and str(item.get("metric_id")) in registry.metrics
        )
        for metric_id in sorted(metric_ids):
            if metric_id not in registry.metrics:
                continue
            spec = registry.get(metric_id)
            metric_rows = [
                item for item in rows
                if str(item.get("system_id")) == system_id
                and str(item.get("metric_id")) == metric_id
            ]
            rule = _channel_score(
                [item for item in metric_rows if str((item.get("judge") or {}).get("type")) == "rule"],
                registry,
                metric_id,
            )
            machine = _channel_score(
                [
                    item for item in metric_rows
                    if str((item.get("judge") or {}).get("type")) in {"llm", "mllm", "model"}
                ],
                registry,
                metric_id,
            )
            adjudicated = [
                item for item in metric_rows
                if str((item.get("judge") or {}).get("role")) == "adjudicator"
            ]
            human_rows = adjudicated or [
                item for item in metric_rows
                if str((item.get("judge") or {}).get("type")) in {"human", "learner"}
            ]
            human = _channel_score(human_rows, registry, metric_id)
            lines.extend(
                [
                    f"{metric_id}｜{spec.name}",
                    f"简介：{spec.rubric}",
                    f"量表：{spec.scale}",
                    f"规则评分：{_format_score(rule, spec.scale)}",
                    f"LLM/MLLM 评分：{_format_score(machine, spec.scale)}",
                    f"人类评分（L 域为学习者，仲裁优先）：{_format_score(human, spec.scale)}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"
