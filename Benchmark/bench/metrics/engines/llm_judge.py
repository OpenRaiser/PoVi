from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from bench.artifacts import resolve_artifact_path
from bench.progress import elapsed_seconds, emit_progress
from bench.providers import OpenAIProvider, TextProviderRouter, provider_config
from bench.storage.atomic import canonical_json_hash, sha256_file
from bench.storage.json_files import write_json
from ..registry import MetricRegistry
from ..implementations import REMOTE_METRICS, metric_ids_for
from ...resources.loader import resource_path


TEXT_METRICS = metric_ids_for("text")
VISUAL_METRICS = metric_ids_for("visual")
AV_METRICS = metric_ids_for("av")
PROMPTS_DIR = resource_path("prompts")
JUDGE_RUBRICS_DIR = resource_path("rubrics")

SCORE_RULES = {
    "0_1": "score 必须是区间 [0,1]（0 到 1，含端点）内的数字，例如 0、0.35、1；禁止使用 1 到 5 分制。",
    "rate": "score 必须是区间 [0,1]（0 到 1，含端点）内的数字比例，例如 0.1 表示 10%；禁止输出百分数字符串。",
    "pass_fail": "score 必须是数字 1（通过）或 0（不通过）；pass/fail 只能写入 label，禁止写入 score。",
    "1_5": "score 必须是 1、2、3、4、5 中的一个整数；禁止使用小数或 0 到 1 比例。",
    "seconds": "score 必须是大于等于 0 的秒数。",
}

SCORE_CHECK_INSTRUCTION = """评分量表是硬约束。必须逐项读取 metrics 中的 native_scale 和 score_rule，不能根据 label 或主观习惯改用另一种量表。
输出前逐项自检：0_1/rate 只能在 [0,1]；1_5 只能是整数 1、2、3、4、5；pass_fail 的 score 只能是数字 0 或 1，不能是字符串 pass/fail。若 not_applicable=true，可令 score=null。"""

FRAME_CHECKPOINT_VERSION = "1.0"


def parse_metric_ids(
    value: str | Iterable[str] | None,
    registry: MetricRegistry | None = None,
) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = [part for item in value for part in str(item).split(",")]
    raw = [item.strip() for item in values if item.strip()]
    registry = registry or MetricRegistry()
    result: set[str] = set()
    errors: list[str] = []
    for metric_id in raw:
        # Prefer exact / alias resolution; fall back to upper-case for G1.1-style ids.
        candidates = (metric_id, metric_id.upper())
        resolved = None
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                resolved = registry.resolve_metric_id(candidate)
                break
            except KeyError as exc:
                last_error = exc
        if resolved is None:
            errors.append(
                last_error.args[0] if last_error is not None else f"Unknown metric id: {metric_id}"
            )
        else:
            result.add(resolved)
    if errors:
        raise ValueError("; ".join(errors))
    return result


def _selected_metrics(available: set[str], selected: set[str] | None) -> list[str]:
    return sorted(available if selected is None else available & selected)


PROMPT_CONTRACT_START = "<!-- BENCH_PROMPT_CONTRACT\n"
PROMPT_CONTRACT_END = "\n-->"


@dataclass(frozen=True)
class PromptContract:
    contract_id: str
    prompt_version: str
    output_schema: dict[str, Any]
    runtime_rules: tuple[str, ...]
    template: str
    template_sha256: str


def _load_prompt_contract(name: str) -> PromptContract:
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    source = path.read_text(encoding="utf-8").strip()
    if not source.startswith(PROMPT_CONTRACT_START):
        raise ValueError(f"Prompt {name} is missing BENCH_PROMPT_CONTRACT metadata")
    header_end = source.find(PROMPT_CONTRACT_END, len(PROMPT_CONTRACT_START))
    if header_end < 0:
        raise ValueError(f"Prompt {name} has an unterminated BENCH_PROMPT_CONTRACT")
    metadata = json.loads(source[len(PROMPT_CONTRACT_START):header_end])
    template = source[header_end + len(PROMPT_CONTRACT_END):].strip()
    if not isinstance(metadata, dict) or not template:
        raise ValueError(f"Prompt {name} must define metadata and a static template")
    output_schema = metadata.get("output_schema")
    if not isinstance(output_schema, dict):
        raise ValueError(f"Prompt {name} output_schema must be an object")
    required = {"contract_id", "prompt_version", "runtime_rules"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Prompt {name} is missing metadata: {', '.join(missing)}")
    rules = metadata["runtime_rules"]
    if not isinstance(rules, list) or not all(isinstance(rule, str) for rule in rules):
        raise ValueError(f"Prompt {name} runtime_rules must be a string array")
    return PromptContract(
        contract_id=str(metadata["contract_id"]),
        prompt_version=str(metadata["prompt_version"]),
        output_schema=output_schema,
        runtime_rules=tuple(rules),
        template=template,
        template_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def _render_prompt_contract(
    contract: PromptContract, *, runtime_additions: Iterable[str] = ()
) -> str:
    additions = [rule for rule in (*contract.runtime_rules, *runtime_additions) if rule]
    return "\n\n".join(
        [
            contract.template,
            "## Runtime Output Contract (generated)\n"
            f"contract_id: {contract.contract_id}\n"
            f"prompt_version: {contract.prompt_version}\n"
            "output_schema:\n"
            + json.dumps(contract.output_schema, ensure_ascii=False, sort_keys=True),
            "## Runtime Additions (generated)\n" + "\n".join(
                f"- {item}" for item in additions
            ),
        ]
    )


def _prompt_metadata(contract: PromptContract, rendered_prompt: str) -> dict[str, str]:
    return {
        "contract_id": contract.contract_id,
        "prompt_version": contract.prompt_version,
        "template_sha256": contract.template_sha256,
        "rendered_prompt_sha256": hashlib.sha256(
            rendered_prompt.encode("utf-8")
        ).hexdigest(),
    }


def _write_prompt_snapshot(
    output_dir: Path,
    *,
    name: str,
    contract: PromptContract,
    rendered_prompt: str,
) -> dict[str, str]:
    metadata = _prompt_metadata(contract, rendered_prompt)
    write_json(
        output_dir / "prompt_snapshots" / f"{name}.json",
        {**metadata, "rendered_prompt": rendered_prompt},
    )
    return metadata


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


@lru_cache(maxsize=1)
def _load_judge_rubric() -> dict[str, Any]:
    """Load the versioned scoring rules used by remote machine judges."""
    path = JUDGE_RUBRICS_DIR / "judge_rubrics.json"
    payload = _load_json(path)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not all(
        isinstance(metric_id, str) and isinstance(rule, dict)
        for metric_id, rule in metrics.items()
    ):
        raise ValueError(f"Judge rubric has an invalid metrics registry: {path}")
    policies = payload.get("shared_policies", {})
    if not isinstance(policies, dict):
        raise ValueError(f"Judge rubric has invalid shared_policies: {path}")
    return payload


def _judge_rubric_metadata() -> dict[str, str]:
    path = JUDGE_RUBRICS_DIR / "judge_rubrics.json"
    rubric = _load_judge_rubric()
    return {
        "judge_rubric_path": "rubrics/judge_rubrics.json",
        "judge_rubric_sha256": sha256_file(path),
        "judge_rubric_schema_version": str(rubric.get("schema_version", "")),
        "judge_rubric_revision": str(rubric.get("rubric_revision", "")),
        "judge_rubric_registry_version": str(rubric.get("registry_version", "")),
    }


def _judge_rubric_context(metric_ids: Iterable[str]) -> dict[str, Any]:
    """Project only rules relevant to this request into the remote judge input."""
    rubric = _load_judge_rubric()
    requested = [str(metric_id) for metric_id in metric_ids]
    applicable_metrics = {
        metric_id: rubric["metrics"][metric_id]
        for metric_id in requested
        if metric_id in rubric["metrics"]
    }
    has_visual_or_av = any(
        # Poem-card exemption applies to visual/cross-modal metrics (R*, D3/D4, disabled V*).
        (metric_id.startswith("R") or metric_id in {"D3", "D4"})
        for metric_id in applicable_metrics
    )
    return {
        "source": "bench/resources/rubrics/judge_rubrics.json",
        "metadata": _judge_rubric_metadata(),
        "shared_policies": rubric["shared_policies"] if has_visual_or_av else {},
        "metrics": applicable_metrics,
    }


def _artifact_texts(submission: dict[str, Any], submission_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    artifacts = submission.get("artifacts") or {}
    for name in (
        "narration", "asr_transcript", "planning_output", "storyboard", "prompts",
        "timeline", "subtitles", "timed_narration", "evidence_citations",
        "system_input_snapshot", "character_assets", "intermediate_outputs",
    ):
        value = artifacts.get(name)
        values = value if isinstance(value, list) else [value]
        texts: list[str] = []
        for item in values:
            if not isinstance(item, str) or not item.strip():
                continue
            try:
                path = resolve_artifact_path(submission_dir, item)
            except ValueError:
                continue
            if path.exists() and path.is_file():
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                if text.strip():
                    texts.append(f"[{item}]\n{text}")
        if texts:
            result[name] = "\n\n".join(texts)
    return result


def _metric_contract(metric_ids: list[str], registry: MetricRegistry) -> list[dict[str, Any]]:
    judge_rubric = _load_judge_rubric()["metrics"]
    contracts: list[dict[str, Any]] = []
    for metric_id in metric_ids:
        metric = registry.get(metric_id)
        contract: dict[str, Any] = {
            "metric_id": metric_id,
            "family_id": metric.family_id,
            "name": metric.name,
            "native_scale": metric.scale,
            "score_rule": SCORE_RULES[metric.scale],
            "direction": metric.direction,
            "rubric": metric.rubric,
        }
        if metric_id in judge_rubric:
            # The registry field remains the short public description.  This
            # complete rule is authoritative for a machine-judge score.
            contract["machine_rubric"] = judge_rubric[metric_id]
        contracts.append(contract)
    return contracts


def _metric_scale_groups(metric_ids: list[str], registry: MetricRegistry) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for metric_id in metric_ids:
        groups[registry.get(metric_id).scale].append(metric_id)
    return dict(groups)


def _validate_native_score(metric_id: str, value: Any, registry: MetricRegistry) -> float:
    score = float(value)
    scale = registry.get(metric_id).scale
    if scale in {"0_1", "rate", "pass_fail"} and not 0 <= score <= 1:
        raise ValueError(f"{metric_id} score must be in [0,1]")
    if scale == "1_5" and (not 1 <= score <= 5 or not score.is_integer()):
        raise ValueError(f"{metric_id} score must be an integer in [1,5]")
    if scale == "seconds" and score < 0:
        raise ValueError(f"{metric_id} seconds must be non-negative")
    return score



def _normalize_time_ranges(value: Any) -> list[list[float]]:
    """Convert common remote-model timestamp formats into the judgment schema."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(part, (int, float)) and not isinstance(part, bool) for part in value):
            values = [value]
        else:
            values = list(value)
    else:
        raise ValueError("time_ranges must be an array of ranges")
    normalized: list[list[float]] = []
    for index, candidate in enumerate(values):
        if isinstance(candidate, str):
            match = re.fullmatch(r"\s*(\d+(?:\.\d*)?|\.\d+)\s*(?:seconds?|secs?|s|秒)?\s*(?:-|–|—|~|至|到)\s*(\d+(?:\.\d*)?|\.\d+)\s*(?:seconds?|secs?|s|秒)?\s*", candidate, re.IGNORECASE)
            if match is None:
                raise ValueError(f"time_ranges/{index}: invalid time range {candidate!r}")
            start, end = float(match.group(1)), float(match.group(2))
        elif isinstance(candidate, (list, tuple)):
            if len(candidate) != 2 or any(isinstance(part, bool) for part in candidate):
                raise ValueError(f"time_ranges/{index}: a range must contain two numeric timestamps")
            try:
                start, end = (float(part) for part in candidate)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"time_ranges/{index}: a range must contain two numeric timestamps") from exc
        else:
            raise ValueError(f"time_ranges/{index}: a range must contain both start and end timestamps")
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < 0:
            raise ValueError(f"time_ranges/{index}: timestamps must be finite non-negative numbers")
        if end < start:
            raise ValueError(f"time_ranges/{index}: end precedes start")
        normalized.append([start, end])
    return normalized

def _convert_payload(
    payload: dict[str, Any],
    *,
    allowed_metrics: set[str],
    registry: MetricRegistry,
    sample_id: str,
    system_id: str,
    track: str,
    setting: str,
    judge_type: str,
    provider: str,
    model: str,
    prompt_version: str,
    force_review: bool,
    prompt_metadata: dict[str, str] | None = None,
    default_unit: str = "sample",
    default_time_range: list[float] | None = None,
    unit_namespace: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    judgments: list[dict[str, Any]] = []
    errors: list[str] = []
    rejected_metric_ids: set[str] = set()
    rows = payload.get("judgments")
    if not isinstance(rows, list):
        return [], ["Remote response is missing a judgments array"], set()
    metric_counts = Counter(
        str(item.get("metric_id") or "")
        for item in rows
        if isinstance(item, dict)
    )
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            errors.append(f"judgments/{index} is not an object")
            continue
        metric_id = str(item.get("metric_id") or "")
        if metric_id not in allowed_metrics:
            errors.append(f"judgments/{index} returned disallowed metric {metric_id!r}")
            if metric_id:
                rejected_metric_ids.add(metric_id)
            continue
        not_applicable = bool(item.get("not_applicable", False))
        score: float | None = None
        if not not_applicable and item.get("score") is not None:
            try:
                score = _validate_native_score(metric_id, item["score"], registry)
            except (TypeError, ValueError) as exc:
                errors.append(f"judgments/{index}: {exc}")
                rejected_metric_ids.add(metric_id)
                continue
        source_unit = str(item.get("unit_id") or default_unit)
        if unit_namespace is None:
            unit_id = source_unit
        elif metric_counts[metric_id] == 1:
            unit_id = unit_namespace
        else:
            unit_id = f"{unit_namespace}:{source_unit}:row_{index:04d}"
        label = str(item.get("label") or ("not_applicable" if not_applicable else "model_judgment"))
        try:
            time_ranges = _normalize_time_ranges(item.get("time_ranges"))
        except ValueError as exc:
            errors.append(f"judgments/{index}: {exc}")
            rejected_metric_ids.add(metric_id)
            continue
        if not time_ranges and default_time_range is not None:
            time_ranges = [default_time_range]
        judgments.append(
            {
                "judgment_id": "j_"
                + hashlib.sha256(
                    "|".join(
                        (
                            sample_id,
                            system_id,
                            metric_id,
                            unit_id,
                            judge_type,
                            provider,
                            model,
                        )
                    ).encode("utf-8")
                ).hexdigest()[:20],
                "sample_id": sample_id,
                "system_id": system_id,
                "track": track,
                "setting": setting,
                "metric_id": metric_id,
                "unit_id": unit_id,
                "score": score,
                "label": label,
                "weight": max(0.0001, float(item.get("weight") or 1.0)),
                "not_applicable": not_applicable,
                "critical": bool(item.get("critical", False)),
                "evidence": {
                    "text_spans": [str(value) for value in item.get("evidence_spans") or []],
                    "time_ranges": time_ranges,
                    "gold_ids": [str(value) for value in item.get("gold_ids") or []],
                },
                "judge": {
                    "type": judge_type,
                    "name": f"{provider}:{model}",
                    "version": "api-fixed-at-run",
                    "prompt_version": prompt_version,
                    "role": "primary",
                },
                "confidence": (
                    min(1.0, max(0.0, float(item["confidence"])))
                    if item.get("confidence") is not None
                    else None
                ),
                "needs_review": force_review or bool(item.get("needs_review", False)),
                "notes": str(item.get("reason") or ""),
                "raw": (
                    {"response_item": item, "prompt": dict(prompt_metadata)}
                    if prompt_metadata is not None
                    else item
                ),
            }
        )
    return judgments, errors, rejected_metric_ids


def _conversion_metadata(
    requested: Iterable[str],
    judgments: Iterable[dict[str, Any]],
    rejected_metric_ids: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Describe accepted coverage without treating a partial response as OK."""
    requested_ids = {str(value) for value in requested}
    returned_ids = {
        str(item.get("metric_id"))
        for item in judgments
        if str(item.get("metric_id")) in requested_ids
    }
    return {
        "requested_metric_ids": sorted(requested_ids),
        "returned_metric_ids": sorted(returned_ids),
        "missing_metric_ids": sorted(requested_ids - returned_ids),
        "rejected_metric_ids": sorted({str(value) for value in rejected_metric_ids if str(value)}),
    }


def judge_text(
    task_input_path: Path,
    gold_path: Path,
    submission_path: Path,
    output_dir: Path,
    *,
    only: set[str] | None = None,
    system_text_override: str | None = None,
    router: TextProviderRouter | None = None,
    registry: MetricRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or MetricRegistry()
    task = _load_json(task_input_path)
    gold = _load_json(gold_path)
    submission = _load_json(submission_path)
    plan = registry.build_evaluation_plan(submission)
    planned = {
        item.metric_id
        for item in plan.activations
        if item.active
        and item.stage == "text"
        and item.prelabel_channel not in {"none", "rule"}
    }
    available = TEXT_METRICS & planned
    metric_ids = _selected_metrics(available, only)
    requested_metric_ids = list(metric_ids)
    emit_progress(
        "text",
        "info",
        "文本指标已选择",
        metrics=len(metric_ids),
        q_filter=",".join(sorted(only)) if only else "all",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not metric_ids:
        output_path = output_dir / "judgments.text.json"
        output_path.write_text(
                json.dumps({"schema_version": "0.4", "judgments": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stage = {
            "ok": True,
            "output": str(output_path),
            "judgment_count": 0,
            "errors": [],
            "providers": [],
            "requested_metric_ids": [],
            "returned_metric_ids": [],
            "missing_metric_ids": [],
            "rejected_metric_ids": [],
            "repair_attempted": False,
        }
        (output_dir / "stage_report.json").write_text(json.dumps(stage, ensure_ascii=False, indent=2), encoding="utf-8")
        return stage
    artifacts = _artifact_texts(submission, submission_path.resolve().parent)
    system_text = (
        system_text_override
        or artifacts.get("narration")
        or artifacts.get("asr_transcript")
        or artifacts.get("planning_output")
        or ""
    )
    if not system_text:
        raise ValueError("No narration, ASR transcript, planning output, or text override is available")
    if system_text_override:
        text_source = "evaluator_transcript"
    elif artifacts.get("narration"):
        text_source = "narration"
    elif artifacts.get("asr_transcript"):
        text_source = "submission_asr_transcript"
    else:
        text_source = "planning_output"
    emit_progress(
        "text",
        "info",
        "文本裁判输入已准备",
        evaluation_text=text_source,
        characters=len(system_text),
        artifact_texts=len(artifacts),
    )
    router = router or TextProviderRouter()
    judgments: list[dict[str, Any]] = []
    errors: list[str] = []
    rejection_errors: list[str] = []
    rejected_metric_ids: set[str] = set()
    providers_used: list[dict[str, str]] = []
    prompt_snapshots: list[dict[str, str]] = []
    repair_attempted = False
    if metric_ids:
        judge_rubric_context = _judge_rubric_context(metric_ids)
        content_contract = _load_prompt_contract("content_judge.md")
        system_prompt = _render_prompt_contract(
            content_contract,
            runtime_additions=(SCORE_CHECK_INSTRUCTION,),
        )
        content_prompt_metadata = {
            **_write_prompt_snapshot(
                output_dir,
                name="content_judge",
                contract=content_contract,
                rendered_prompt=system_prompt,
            ),
            **_judge_rubric_metadata(),
        }
        prompt_snapshots.append(content_prompt_metadata)
        user_payload = {
            "task_input": task,
            "gold": gold,
            "system_artifacts": artifacts,
            "evaluation_text": system_text,
            "metrics": _metric_contract(metric_ids, registry),
            "judge_rubric": judge_rubric_context,
            "metric_scale_groups": _metric_scale_groups(metric_ids, registry),
            "mandatory_score_check": SCORE_CHECK_INSTRUCTION,
        }
        request_started = time.monotonic()
        emit_progress(
            "text",
            "progress",
            "正在请求主文本评分",
            metrics=len(metric_ids),
        )
        result = router.complete_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        emit_progress(
            "text",
            "progress",
            "主文本评分已返回",
            provider=result.provider,
            model=result.model,
            elapsed_seconds=elapsed_seconds(request_started),
        )
        (output_dir / "text_judge.raw.txt").write_text(result.raw_text, encoding="utf-8")
        converted, conversion_errors, conversion_rejected_ids = _convert_payload(
            result.payload,
            allowed_metrics=set(metric_ids),
            registry=registry,
            sample_id=str(submission["sample_id"]),
            system_id=str(submission["system_id"]),
            track=str(submission["track"]),
            setting=str(submission["setting"]),
            judge_type="llm",
            provider=result.provider,
            model=result.model,
            prompt_version=content_contract.prompt_version,
            force_review=False,
            prompt_metadata=content_prompt_metadata,
        )
        judgments.extend(converted)
        rejection_errors.extend(conversion_errors)
        rejected_metric_ids.update(conversion_rejected_ids)
        providers_used.append({"provider": result.provider, "model": result.model})
        coverage = _conversion_metadata(metric_ids, converted, conversion_rejected_ids)
        if coverage["missing_metric_ids"]:
            repair_attempted = True
            missing_ids = coverage["missing_metric_ids"]
            repair_payload = {
                "task_input": task,
                "gold": gold,
                "system_artifacts": artifacts,
                "evaluation_text": system_text,
                "metrics": _metric_contract(missing_ids, registry),
                "judge_rubric": _judge_rubric_context(missing_ids),
                "mandatory_score_check": SCORE_CHECK_INSTRUCTION,
                "repair_instruction": "只补回下列遗漏或非法指标；不要返回任何其他指标。",
            }
            try:
                repaired = router.complete_json(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"},
                )
                (output_dir / "text_judge.repair.raw.txt").write_text(repaired.raw_text, encoding="utf-8")
                repaired_rows, repair_errors, repair_rejected_ids = _convert_payload(
                    repaired.payload,
                    allowed_metrics=set(missing_ids),
                    registry=registry,
                    sample_id=str(submission["sample_id"]),
                    system_id=str(submission["system_id"]),
                    track=str(submission["track"]),
                    setting=str(submission["setting"]),
                    judge_type="llm",
                    provider=repaired.provider,
                    model=repaired.model,
                    prompt_version=content_contract.prompt_version,
                    force_review=False,
                    prompt_metadata=content_prompt_metadata,
                )
                judgments.extend(repaired_rows)
                rejection_errors.extend(repair_errors)
                rejected_metric_ids.update(repair_rejected_ids)
                providers_used.append({"provider": repaired.provider, "model": repaired.model})
            except Exception as exc:
                errors.append(f"repair: {type(exc).__name__}: {exc}")
        emit_progress(
            "text",
            "progress",
            "文本评分转换完成",
            accepted=len(converted),
            rejected=len(conversion_errors),
        )
    payload = {"schema_version": "0.4", "judgments": judgments}
    output_path = output_dir / "judgments.text.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage = _conversion_metadata(requested_metric_ids, judgments, rejected_metric_ids)
    if coverage["missing_metric_ids"]:
        errors.append("missing metrics after repair: " + ", ".join(coverage["missing_metric_ids"]))
    stage = {
        "ok": not errors,
        "output": str(output_path),
        "judgment_count": len(judgments),
        "errors": errors,
        "providers": providers_used,
        "prompt_snapshots": prompt_snapshots,
        **coverage,
        "repair_attempted": repair_attempted,
    }
    (output_dir / "stage_report.json").write_text(json.dumps(stage, ensure_ascii=False, indent=2), encoding="utf-8")
    return stage


def _image_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _frame_segments(manifest_path: Path) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    manifest = _load_json(manifest_path)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in manifest.get("frames") or []:
        grouped[int(item["segment_index"])].append(item)
    return manifest, grouped


def _frame_prompt_name(kind: str) -> str:
    return "visual_judge.md" if kind == "visual" else "av_alignment_judge.md"


def _frame_checkpoint_contract(
    *,
    kind: str,
    task_input_path: Path,
    gold_path: Path,
    submission_path: Path,
    manifest_path: Path,
    metric_ids: list[str],
    registry: MetricRegistry,
    provider: OpenAIProvider,
    context: dict[str, Any],
    prompt_metadata: dict[str, str],
) -> str:
    return canonical_json_hash(
        {
            "checkpoint_version": FRAME_CHECKPOINT_VERSION,
            "kind": kind,
            "inputs": {
                "task_input": sha256_file(task_input_path),
                "gold": sha256_file(gold_path),
                "submission": sha256_file(submission_path),
                "frame_manifest": sha256_file(manifest_path),
            },
            "metrics": _metric_contract(metric_ids, registry),
            "prompt": prompt_metadata,
            "provider": {
                "name": provider.config.name,
                "base_url": provider.config.base_url,
                "model": provider.config.model,
            },
            "context_sha256": canonical_json_hash(context),
        }
    )


def _frame_segment_hash(
    manifest_path: Path,
    *,
    segment_index: int,
    start: float,
    end: float,
    frames: list[dict[str, Any]],
) -> str:
    entries: list[dict[str, Any]] = []
    for frame in frames:
        frame_path = manifest_path.resolve().parent / str(frame["path"])
        entries.append(
            {
                "frame_id": str(frame["frame_id"]),
                "timestamp_seconds": float(frame["timestamp_seconds"]),
                "path": str(frame["path"]),
                "sha256": sha256_file(frame_path),
            }
        )
    return canonical_json_hash(
        {
            "segment_index": segment_index,
            "start_seconds": start,
            "end_seconds": end,
            "frames": entries,
        }
    )


def _load_frame_checkpoint(
    path: Path,
    *,
    contract_sha256: str,
    segment_sha256: str,
    segment_index: int,
    metric_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        payload.get("checkpoint_version") != FRAME_CHECKPOINT_VERSION
        or payload.get("contract_sha256") != contract_sha256
        or payload.get("segment_sha256") != segment_sha256
        or payload.get("segment_index") != segment_index
    ):
        return None
    rows = payload.get("judgments")
    report = payload.get("report")
    if not isinstance(rows, list) or not all(isinstance(item, dict) for item in rows):
        return None
    if not isinstance(report, dict):
        return None
    namespace = f"segment_{segment_index:04d}"
    if any(
        str(item.get("metric_id")) not in metric_ids
        or not str(item.get("unit_id") or "").startswith(namespace)
        or not bool(item.get("needs_review", False))
        for item in rows
    ):
        return None
    coverage = _conversion_metadata(
        metric_ids,
        rows,
        report.get("rejected_metric_ids") or [],
    )
    if coverage["missing_metric_ids"]:
        return None
    coverage.update(
        {
            "segment_index": str(segment_index),
            "repair_attempted": bool(report.get("repair_attempted", False)),
            "resumed": True,
        }
    )
    return rows, coverage


def _frame_stage_payload(
    *,
    kind: str,
    output_path: Path,
    judgments: list[dict[str, Any]],
    errors: list[str],
    provider: OpenAIProvider,
    metric_ids: list[str],
    segment_reports: list[dict[str, Any]],
    all_segment_indices: list[int],
    complete: bool,
) -> dict[str, Any]:
    completed = sorted(
        int(report["segment_index"])
        for report in segment_reports
        if not report.get("missing_metric_ids")
    )
    resumed = sorted(
        int(report["segment_index"])
        for report in segment_reports
        if report.get("resumed")
    )
    returned_union = sorted(
        {
            metric_id
            for report in segment_reports
            for metric_id in report.get("returned_metric_ids") or []
        }
    )
    missing_union = sorted(
        {
            metric_id
            for report in segment_reports
            for metric_id in report.get("missing_metric_ids") or []
        }
    )
    rejected_union = sorted(
        {
            metric_id
            for report in segment_reports
            for metric_id in report.get("rejected_metric_ids") or []
        }
    )
    return {
        "ok": complete and not errors,
        "complete": complete,
        "output": str(output_path),
        "judgment_count": len(judgments),
        "errors": list(errors),
        "provider": provider.config.name,
        "model": provider.config.model,
        "requested_metric_ids": sorted(set(metric_ids)),
        "returned_metric_ids": returned_union,
        "missing_metric_ids": missing_union,
        "rejected_metric_ids": rejected_union,
        "completed_segment_indices": completed,
        "pending_segment_indices": sorted(set(all_segment_indices) - set(completed)),
        "resumed_segment_indices": resumed,
        "segments": segment_reports,
    }


def _write_frame_progress(
    output_dir: Path,
    *,
    kind: str,
    judgments: list[dict[str, Any]],
    stage: dict[str, Any],
) -> None:
    write_json(
        output_dir / f"judgments.{kind}.json",
            {"schema_version": "0.4", "judgments": judgments},
    )
    write_json(output_dir / "stage_report.json", stage)


def _judge_frames(
    kind: str,
    task_input_path: Path,
    gold_path: Path,
    submission_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    only: set[str] | None,
    transcript_override: str | None,
    provider: OpenAIProvider | None,
    resume: bool,
    registry: MetricRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or MetricRegistry()
    task = _load_json(task_input_path)
    gold = _load_json(gold_path)
    submission = _load_json(submission_path)
    planned = set(
        registry.build_evaluation_plan(submission).active_metric_ids
    )
    available = (
        VISUAL_METRICS if kind == "visual" else AV_METRICS
    ) & planned
    metric_ids = _selected_metrics(available, only)
    artifact_texts = _artifact_texts(submission, submission_path.resolve().parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not metric_ids:
        output_path = output_dir / f"judgments.{kind}.json"
        output_path.write_text(
                json.dumps({"schema_version": "0.4", "judgments": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stage = {
            "ok": True,
            "complete": True,
            "output": str(output_path),
            "judgment_count": 0,
            "errors": [],
            "provider": None,
            "model": None,
            "requested_metric_ids": [],
            "returned_metric_ids": [],
            "missing_metric_ids": [],
            "rejected_metric_ids": [],
            "completed_segment_indices": [],
            "pending_segment_indices": [],
            "resumed_segment_indices": [],
            "segments": [],
        }
        write_json(output_dir / "stage_report.json", stage)
        return stage
    manifest, segments = _frame_segments(manifest_path)
    provider = provider or OpenAIProvider(provider_config("multimodal_judge"))
    if kind == "visual":
        instruction = "评价时间戳帧序列的视觉忠实度、视觉质量和隐喻表现。稀疏帧无法可靠判断的运动细节必须标为 not_applicable。LLM/MLLM 评分与人类专家评分独立进行；需要专家正式评分的指标保留 needs_review=true。"
        context = {
            "visual_requirements": gold.get("visual_requirements"),
            "generation_requirements": gold.get("generation_requirements"),
            "metaphor_maps": gold.get("metaphor_maps"),
        }
    else:
        instruction = "评价当前帧序列与旁白/字幕在局部语义和关键事件上的对齐。输入没有原始音频，禁止评价音色、真实音效或口型毫秒级同步。LLM/MLLM 评分与人类专家评分独立进行；需要专家正式评分的指标保留 needs_review=true。"
        context = {
            "key_events": gold.get("key_events"),
            "alignment_relations": gold.get("alignment_relations"),
            "narration_or_asr": transcript_override or artifact_texts.get("asr_transcript") or artifact_texts.get("narration"),
            "timed_narration": artifact_texts.get("timed_narration"),
            "subtitles": artifact_texts.get("subtitles"),
        }
    frame_contract = _load_prompt_contract(_frame_prompt_name(kind))
    judge_rubric_context = _judge_rubric_context(metric_ids)
    frame_system_prompt = _render_prompt_contract(
        frame_contract,
        runtime_additions=(SCORE_CHECK_INSTRUCTION,),
    )
    frame_prompt_metadata = {
        **_write_prompt_snapshot(
            output_dir,
            name=f"{kind}_judge",
            contract=frame_contract,
            rendered_prompt=frame_system_prompt,
        ),
        **_judge_rubric_metadata(),
    }
    contract_sha256 = _frame_checkpoint_contract(
        kind=kind,
        task_input_path=task_input_path,
        gold_path=gold_path,
        submission_path=submission_path,
        manifest_path=manifest_path,
        metric_ids=metric_ids,
        registry=registry,
        provider=provider,
        context=context,
        prompt_metadata=frame_prompt_metadata,
    )
    emit_progress(
        kind,
        "info",
        "分段裁判输入已准备",
        segments=len(segments),
        metrics=len(metric_ids),
        provider=provider.config.name,
        model=provider.config.model,
    )
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    judgments: list[dict[str, Any]] = []
    errors: list[str] = []
    segment_reports: list[dict[str, Any]] = []
    ordered_segments = sorted(segments.items())
    all_segment_indices = [segment_index for segment_index, _frames in ordered_segments]
    output_path = output_dir / f"judgments.{kind}.json"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for segment_position, (segment_index, frames) in enumerate(ordered_segments, start=1):
        segment_started = time.monotonic()
        start = float(segment_index) * float(manifest["sampling"]["segment_seconds"])
        end = start + float(manifest["sampling"]["segment_seconds"])
        duration = manifest.get("duration_seconds")
        if isinstance(duration, (int, float)) and float(duration) > start:
            end = min(end, float(duration))
        unit_namespace = f"segment_{segment_index:04d}"
        segment_sha256 = _frame_segment_hash(
            manifest_path,
            segment_index=segment_index,
            start=start,
            end=end,
            frames=frames,
        )
        checkpoint_path = checkpoint_dir / f"{unit_namespace}.json"
        checkpoint = (
            _load_frame_checkpoint(
                checkpoint_path,
                contract_sha256=contract_sha256,
                segment_sha256=segment_sha256,
                segment_index=segment_index,
                metric_ids=metric_ids,
            )
            if resume
            else None
        )
        if checkpoint is not None:
            checkpoint_rows, checkpoint_report = checkpoint
            judgments.extend(checkpoint_rows)
            segment_reports.append(checkpoint_report)
            emit_progress(
                kind,
                "progress",
                "已复用视频分段 checkpoint",
                segment=f"{segment_position}/{len(ordered_segments)}",
                segment_index=segment_index,
                judgments=len(checkpoint_rows),
            )
            partial_stage = _frame_stage_payload(
                kind=kind,
                output_path=output_path,
                judgments=judgments,
                errors=errors,
                provider=provider,
                metric_ids=metric_ids,
                segment_reports=segment_reports,
                all_segment_indices=all_segment_indices,
                complete=False,
            )
            _write_frame_progress(
                output_dir,
                kind=kind,
                judgments=judgments,
                stage=partial_stage,
            )
            continue
        emit_progress(
            kind,
            "progress",
            "正在评判视频分段",
            segment=f"{segment_position}/{len(ordered_segments)}",
            segment_index=segment_index,
            frames=len(frames),
            time_range=f"{start:.1f}-{end:.1f}s",
        )
        frame_meta = [
            {"frame_id": item["frame_id"], "timestamp_seconds": item["timestamp_seconds"]}
            for item in frames
        ]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "instruction": instruction,
                        "task": task,
                        "gold_context": context,
                        "segment": {"index": segment_index, "start_seconds": start, "end_seconds": end, "frames": frame_meta},
                        "metrics": _metric_contract(metric_ids, registry),
                        "judge_rubric": judge_rubric_context,
                        "metric_scale_groups": _metric_scale_groups(metric_ids, registry),
                        "mandatory_score_check": SCORE_CHECK_INSTRUCTION,
                        "output_schema": frame_contract.output_schema,
                    },
                    ensure_ascii=False,
                ),
            }
        ]
        for frame in frames:
            frame_path = manifest_path.resolve().parent / str(frame["path"])
            content.append({"type": "image_url", "image_url": {"url": _image_url(frame_path), "detail": "high"}})
        try:
            result = provider.complete_json(
                [
                    {
                        "role": "system",
                        "content": frame_system_prompt,
                    },
                    {"role": "user", "content": content},
                ],
                max_tokens=10000,
                response_format={"type": "json_object"},
            )
            (raw_dir / f"segment_{segment_index:04d}.txt").write_text(result.raw_text, encoding="utf-8")
            converted, conversion_errors, conversion_rejected_ids = _convert_payload(
                result.payload,
                allowed_metrics=set(metric_ids),
                registry=registry,
                sample_id=str(submission["sample_id"]),
                system_id=str(submission["system_id"]),
                track=str(submission["track"]),
                setting=str(submission["setting"]),
                judge_type="mllm",
                provider=result.provider,
                model=result.model,
                prompt_version=frame_contract.prompt_version,
                force_review=True,
                prompt_metadata=frame_prompt_metadata,
                default_unit=unit_namespace,
                default_time_range=[start, end],
                unit_namespace=unit_namespace,
            )
            judgments.extend(converted)
            segment_judgments = list(converted)
            rejected_metric_ids = set(conversion_rejected_ids)
            coverage = _conversion_metadata(metric_ids, segment_judgments, rejected_metric_ids)
            repair_attempted = False
            if coverage["missing_metric_ids"]:
                repair_attempted = True
                repair_request = json.loads(content[0]["text"])
                repair_request["metrics"] = _metric_contract(coverage["missing_metric_ids"], registry)
                repair_request["judge_rubric"] = _judge_rubric_context(
                    coverage["missing_metric_ids"]
                )
                repair_request["repair_instruction"] = "只返回遗漏或非法指标，禁止返回其它指标。"
                repair_content = [
                    {"type": "text", "text": json.dumps(repair_request, ensure_ascii=False)},
                    *content[1:],
                ]
                try:
                    repaired = provider.complete_json(
                        [
                            {
                                "role": "system",
                                "content": frame_system_prompt,
                            },
                            {"role": "user", "content": repair_content},
                        ],
                        max_tokens=10000,
                        response_format={"type": "json_object"},
                    )
                    (raw_dir / f"segment_{segment_index:04d}.repair.txt").write_text(repaired.raw_text, encoding="utf-8")
                    repaired_rows, repair_errors, repair_rejected_ids = _convert_payload(
                        repaired.payload,
                        allowed_metrics=set(coverage["missing_metric_ids"]),
                        registry=registry,
                        sample_id=str(submission["sample_id"]),
                        system_id=str(submission["system_id"]),
                        track=str(submission["track"]),
                        setting=str(submission["setting"]),
                        judge_type="mllm",
                        provider=repaired.provider,
                        model=repaired.model,
                        prompt_version=frame_contract.prompt_version,
                        force_review=True,
                        prompt_metadata=frame_prompt_metadata,
                        default_unit=unit_namespace,
                        default_time_range=[start, end],
                        unit_namespace=unit_namespace,
                    )
                    judgments.extend(repaired_rows)
                    segment_judgments.extend(repaired_rows)
                    conversion_errors.extend(repair_errors)
                    rejected_metric_ids.update(repair_rejected_ids)
                except Exception as exc:
                    conversion_errors.append(f"repair: {type(exc).__name__}: {exc}")
            coverage = _conversion_metadata(metric_ids, segment_judgments, rejected_metric_ids)
            coverage["segment_index"] = str(segment_index)
            coverage["repair_attempted"] = repair_attempted
            coverage["resumed"] = False
            segment_reports.append(coverage)
            if coverage["missing_metric_ids"]:
                errors.append(f"segment_{segment_index}: missing metrics after repair: {', '.join(coverage['missing_metric_ids'])}")
            else:
                write_json(
                    checkpoint_path,
                    {
                        "checkpoint_version": FRAME_CHECKPOINT_VERSION,
                        "contract_sha256": contract_sha256,
                        "segment_sha256": segment_sha256,
                        "segment_index": segment_index,
                        "judgments": segment_judgments,
                        "report": coverage,
                    },
                )
            partial_stage = _frame_stage_payload(
                kind=kind,
                output_path=output_path,
                judgments=judgments,
                errors=errors,
                provider=provider,
                metric_ids=metric_ids,
                segment_reports=segment_reports,
                all_segment_indices=all_segment_indices,
                complete=False,
            )
            _write_frame_progress(
                output_dir,
                kind=kind,
                judgments=judgments,
                stage=partial_stage,
            )
            emit_progress(
                kind,
                "progress" if not conversion_errors else "warn",
                "视频分段评判完成",
                segment=f"{segment_position}/{len(ordered_segments)}",
                accepted=len(converted),
                rejected=len(conversion_errors),
                elapsed_seconds=elapsed_seconds(segment_started),
            )
        except Exception as exc:
            errors.append(f"segment_{segment_index}: {type(exc).__name__}: {exc}")
            segment_reports.append(
                {
                    "requested_metric_ids": sorted(set(metric_ids)),
                    "returned_metric_ids": [],
                    "missing_metric_ids": sorted(set(metric_ids)),
                    "rejected_metric_ids": [],
                    "segment_index": str(segment_index),
                    "repair_attempted": False,
                    "resumed": False,
                }
            )
            partial_stage = _frame_stage_payload(
                kind=kind,
                output_path=output_path,
                judgments=judgments,
                errors=errors,
                provider=provider,
                metric_ids=metric_ids,
                segment_reports=segment_reports,
                all_segment_indices=all_segment_indices,
                complete=False,
            )
            _write_frame_progress(
                output_dir,
                kind=kind,
                judgments=judgments,
                stage=partial_stage,
            )
            emit_progress(
                kind,
                "warn",
                "视频分段评判失败，继续下一段",
                segment=f"{segment_position}/{len(ordered_segments)}",
                error_type=type(exc).__name__,
                elapsed_seconds=elapsed_seconds(segment_started),
            )
    stage = _frame_stage_payload(
        kind=kind,
        output_path=output_path,
        judgments=judgments,
        errors=errors,
        provider=provider,
        metric_ids=metric_ids,
        segment_reports=segment_reports,
        all_segment_indices=all_segment_indices,
        complete=True,
    )
    _write_frame_progress(
        output_dir,
        kind=kind,
        judgments=judgments,
        stage=stage,
    )
    return stage


def judge_visual(
    task_input_path: Path,
    gold_path: Path,
    submission_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    only: set[str] | None = None,
    provider: OpenAIProvider | None = None,
    resume: bool = False,
    registry: MetricRegistry | None = None,
) -> dict[str, Any]:
    return _judge_frames("visual", task_input_path, gold_path, submission_path, manifest_path, output_dir, only=only, transcript_override=None, provider=provider, resume=resume, registry=registry)


def judge_av(
    task_input_path: Path,
    gold_path: Path,
    submission_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    only: set[str] | None = None,
    transcript_override: str | None = None,
    provider: OpenAIProvider | None = None,
    resume: bool = False,
    registry: MetricRegistry | None = None,
) -> dict[str, Any]:
    return _judge_frames("av", task_input_path, gold_path, submission_path, manifest_path, output_dir, only=only, transcript_override=transcript_override, provider=provider, resume=resume, registry=registry)
