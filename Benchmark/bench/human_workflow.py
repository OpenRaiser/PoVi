"""Run-scoped export, import, status, and finalization for human evaluation."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import shutil
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .artifacts import resolve_artifact_path
from .metrics.registry import MetricRegistry, MetricSpec
from .pipeline.orchestrator import merge_judgment_payloads
from .pipeline.preparation import gate_passed, verify_preparation_contract
from .runs import HumanStatus, MachineStatus, RunState, RunStore, utc_now
from .scoring.aggregate import aggregate_judgments
from .reporting import render_text_report
from .scoring.reliability import agreement_statistics
from .human_protocol import scale_anchors
from .storage.atomic import atomic_write_text, canonical_json_hash, sha256_file
from .storage.json_files import write_json
from .validation import load_json, validate_judgments
from .resources.loader import resource_path


HUMAN_SCHEMA_VERSION = "1.0"
SCHEMAS = resource_path("schemas")
CSV_FIELDS = (
    "run_id",
    "item_id",
    "assignment_id",
    "annotator_role",
    "assignment_role",
    "rubric_version",
    "annotator_id",
    "scores",
    "comment",
    "confidence",
    "timestamp",
    "not_applicable",
    "critical",
)
SIMPLE_FORM_FIELDS = ("指标", "评估内容", "量表", "参考依据", "分数")
NOT_APPLICABLE_SCORE_TOKENS = frozenset({"n", "n/a"})
REFERENCE_ROOT_NAME = "参考资料"
REFERENCE_COLLECTION_LABELS = {
    "evidence_claims": "证据主张",
    "required_information": "必要教学信息",
    "metaphor_maps": "隐喻映射",
    "visual_requirements": "视觉要求",
    "key_events": "关键事件",
    "pronunciation_lexicon": "发音词典",
    "learning_items": "学习测验与评分规则",
    "consistency_groups": "连续性分组",
    "event_relations": "事件关系",
    "generation_requirements": "生成要求",
    "alignment_relations": "对齐关系",
}
REFERENCE_COLLECTIONS_BY_FAMILY = {
    "S": ("required_information", "metaphor_maps", "evidence_claims"),
    "D_narr": ("key_events", "event_relations"),
    "R_vis": (
        "visual_requirements",
        "generation_requirements",
        "metaphor_maps",
        "consistency_groups",
        "key_events",
    ),
    "D_cross": ("visual_requirements", "key_events", "generation_requirements"),
    "R_cross": ("alignment_relations", "visual_requirements", "key_events"),
}
REQUIRED_REFERENCE_FIELDS_BY_FAMILY = {
    "S": ("required_information", "metaphor_maps"),
    "D_narr": (),
    "R_vis": ("visual_requirements",),
    "D_cross": (),
    "R_cross": (),
}
NONEMPTY_REFERENCE_FIELDS_BY_FAMILY = {
    "S": ("required_information", "metaphor_maps"),
    "R_vis": ("visual_requirements",),
}
NONEMPTY_REFERENCE_FIELDS_BY_METRIC: dict[str, tuple[str, ...]] = {}
APPLICABILITY_GUIDANCE = {
    "delayed_test_conducted": (
        "实际执行了延迟测试并有对应结果时适用；未执行时填 N 或 N/A。"
    ),
    "recurring_character": (
        "同一角色在多个需要比较的画面或镜头中重复出现时适用；否则填 N 或 N/A。"
    ),
}


def _stable_id(prefix: str, salt: str, *parts: str) -> str:
    digest = hmac.new(
        bytes.fromhex(salt),
        "|".join(parts).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    atomic_write_text(path, "\ufeff" + buffer.getvalue())


def _write_simple_form(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SIMPLE_FORM_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in SIMPLE_FORM_FIELDS})
    atomic_write_text(path, "\ufeff" + buffer.getvalue())


def _required_annotations(spec: MetricSpec) -> int:
    if spec.applicability:
        return 2
    return 1 if spec.formal_channel == "learner" else 2


def _display_scale(scale: str) -> str:
    label = (
        "0–1"
        if scale == "0_1"
        else "1–5"
        if scale == "1_5"
        else scale
    )
    return f"{label}；N/N/A=不适用"


def _display_rubric(spec: MetricSpec) -> str:
    guidance = APPLICABILITY_GUIDANCE.get(spec.applicability or "")
    if not guidance:
        return spec.rubric
    return f"{spec.rubric} 适用性判断：{guidance}"


def _is_not_applicable_score(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.strip().lower() in NOT_APPLICABLE_SCORE_TOKENS
    )


def _rubric_projection(
    registry: MetricRegistry, metric_ids: Iterable[str]
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric_id in sorted(set(metric_ids)):
        spec = registry.get(metric_id)
        role = "learner" if spec.formal_channel == "learner" else "expert"
        metrics[metric_id] = {
            "name": spec.name,
            "family_id": spec.family_id,
            "annotator_role": role,
            "required_annotations": _required_annotations(spec),
            "scale": spec.scale,
            "direction": spec.direction,
            "rubric": _display_rubric(spec),
            "anchors": scale_anchors(spec.scale),
        }
    return {
        "schema_version": HUMAN_SCHEMA_VERSION,
        "registry_version": registry.version,
        "protocol": {
            "expert_independent_raters": 2,
            "learner_minimum_raters": 1,
            "disagreement_1_5": "absolute_difference>=2",
            "disagreement_0_1": "absolute_difference>0.25",
            "critical_or_na_disagreement": "always_adjudicate",
            "conditional_applicability_independent_raters": 2,
        },
        "metrics": metrics,
    }


def _artifact_sources(
    submission: dict[str, Any], submission_dir: Path, unit_id: str
) -> list[tuple[str, Path]]:
    artifacts = submission.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        return []
    result: list[tuple[str, Path]] = []
    pairs = artifacts.get("prompt_media_pairs")
    if isinstance(pairs, list) and unit_id != "sample":
        for pair in pairs:
            if not isinstance(pair, dict) or str(pair.get("pair_id")) != unit_id:
                continue
            for field in ("prompt", "media"):
                value = pair.get(field)
                if isinstance(value, str):
                    try:
                        path = resolve_artifact_path(submission_dir, value)
                    except ValueError:
                        continue
                    if path.is_file():
                        result.append((f"prompt_media_{field}", path))
        return result
    for name, value in artifacts.items():
        if name == "prompt_media_pairs":
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, str) or not item.strip():
                continue
            try:
                path = resolve_artifact_path(submission_dir, item)
            except ValueError:
                continue
            if path.is_file():
                result.append((str(name), path))
    return result


_IDENTITY_KEYS = {
    "system_id",
    "system_name",
    "submitter",
    "submission_path",
    "model",
    "model_name",
    "model_names",
    "provider",
    "base_url",
}


def _sanitize_json(value: Any, redactions: Iterable[str]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_json(item, redactions)
            for key, item in value.items()
            if str(key).lower() not in _IDENTITY_KEYS
            and not str(key).lower().endswith("_path")
        }
    if isinstance(value, list):
        return [_sanitize_json(item, redactions) for item in value]
    if isinstance(value, str):
        for redaction in redactions:
            if redaction:
                value = value.replace(redaction, "[REDACTED]")
    return value


def _copy_blinded_asset(
    source: Path,
    target: Path,
    *,
    redactions: Iterable[str],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        write_json(target, _sanitize_json(payload, redactions))
        return
    if suffix in {".txt", ".md", ".csv", ".jsonl", ".srt", ".vtt"}:
        text = source.read_text(encoding="utf-8")
        for value in redactions:
            if value:
                text = text.replace(value, "[REDACTED]")
        atomic_write_text(target, text)
        return
    shutil.copyfile(source, target)


def _safe_metric_component(metric_id: str) -> str:
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if not metric_id or any(character not in allowed for character in metric_id):
        raise ValueError(f"Unsafe metric id for reference directory: {metric_id!r}")
    return metric_id


def _reference_index_relative_path(metric_id: str) -> str:
    safe_metric_id = _safe_metric_component(metric_id)
    return (
        Path(REFERENCE_ROOT_NAME)
        / safe_metric_id
        / f"{safe_metric_id}_00_资料索引.json"
    ).as_posix()


def _validate_reference_inputs(
    spec: MetricSpec,
    task: dict[str, Any],
    gold: dict[str, Any],
) -> None:
    errors: list[str] = []
    if not isinstance(task.get("poem"), dict):
        errors.append("task/poem must be an object")
    if not isinstance(task.get("task"), dict):
        errors.append("task/task must be an object")
    for field in REQUIRED_REFERENCE_FIELDS_BY_FAMILY.get(spec.family_id, ()):
        if field not in gold or not isinstance(gold.get(field), list):
            errors.append(f"gold/{field} must be an array")
    nonempty_fields = (
        NONEMPTY_REFERENCE_FIELDS_BY_FAMILY.get(spec.family_id, ())
        + NONEMPTY_REFERENCE_FIELDS_BY_METRIC.get(spec.metric_id, ())
    )
    for field in dict.fromkeys(nonempty_fields):
        value = gold.get(field)
        if isinstance(value, list) and not value:
            errors.append(f"gold/{field} must contain at least one item")
    if errors:
        raise ValueError(
            f"Cannot export complete references for {spec.metric_id}: "
            + "; ".join(errors)
        )


def _reference_documents(
    spec: MetricSpec,
    task: dict[str, Any],
    gold: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    _validate_reference_inputs(spec, task, gold)
    poem = dict(task.get("poem") or {})
    task_requirements = dict(task.get("task") or {})
    condition_inputs = dict(task.get("condition_inputs") or {})
    documents: list[tuple[str, str, dict[str, Any]]] = [
        (
            "诗词原文",
            "待评任务对应的完整诗词原文、逐行文本和行号。",
            {"poem": poem},
        ),
        (
            "任务与条件",
            "目标受众、语言、时长、模态、任务指令和公开条件。",
            {
                "track": task.get("track"),
                "setting": task.get("setting"),
                "task": task_requirements,
                "condition_inputs": condition_inputs,
            },
        ),
    ]
    if spec.family_id == "A1":
        target_narration = str(
            condition_inputs.get("narration_text") or poem.get("text") or ""
        )
        documents.append(
            (
                "目标旁白",
                "用于核对实际语音内容的目标旁白。",
                {"target_narration": target_narration},
            )
        )
    for field in REFERENCE_COLLECTIONS_BY_FAMILY.get(spec.family_id, ()):
        documents.append(
            (
                REFERENCE_COLLECTION_LABELS[field],
                f"指标 {spec.metric_id} 所需的结构化{REFERENCE_COLLECTION_LABELS[field]}。",
                {field: list(gold.get(field) or [])},
            )
        )
    return documents


def _export_metric_references(
    reviewer_dir: Path,
    metric_ids: Iterable[str],
    registry: MetricRegistry,
    task: dict[str, Any],
    gold: dict[str, Any],
    *,
    redactions: Iterable[str],
) -> dict[str, str]:
    reference_root = reviewer_dir / REFERENCE_ROOT_NAME
    if reference_root.exists():
        shutil.rmtree(reference_root)
    reference_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        reference_root / "00_参考资料说明.txt",
        "人工评审参考资料\n\n"
        "目录规则：每个指标一个目录；00 文件为资料索引，其余文件按序号排列。\n"
        "评分表中的“参考依据”指向对应指标的 00_资料索引.json。\n"
        "参考资料只用于当前指标，不应据此推断提交系统身份。\n",
    )
    result: dict[str, str] = {}
    for metric_id in sorted(set(metric_ids)):
        safe_metric_id = _safe_metric_component(metric_id)
        spec = registry.get(metric_id)
        metric_dir = reference_root / safe_metric_id
        metric_dir.mkdir(parents=True, exist_ok=False)
        entries: list[dict[str, Any]] = []
        for order, (label, description, payload) in enumerate(
            _reference_documents(spec, task, gold),
            start=1,
        ):
            filename = f"{safe_metric_id}_{order:02d}_{label}.json"
            sanitized = _sanitize_json(payload, redactions)
            write_json(metric_dir / filename, sanitized)
            entries.append(
                {
                    "order": order,
                    "file": filename,
                    "description": description,
                    "sha256": sha256_file(metric_dir / filename),
                    "item_count": sum(
                        len(value)
                        for value in sanitized.values()
                        if isinstance(value, list)
                    ),
                }
            )
        index_filename = f"{safe_metric_id}_00_资料索引.json"
        write_json(
            metric_dir / index_filename,
            {
                "schema_version": "1.0",
                "metric_id": metric_id,
                "metric_name": spec.name,
                "family_id": spec.family_id,
                "reference_complete": True,
                "files": entries,
            },
        )
        result[metric_id] = _reference_index_relative_path(metric_id)
    return result


def _export_visible_assets(
    reviewer_dir: Path,
    submission: dict[str, Any],
    submission_dir: Path,
    metric_ids: set[str],
    registry: MetricRegistry,
    *,
    redactions: Iterable[str],
) -> None:
    domains = {registry.get(metric_id).domain for metric_id in metric_ids}
    artifacts = submission.get("artifacts") or {}
    if "content" in domains:
        for name in ("planning_output", "narration", "asr_transcript"):
            value = artifacts.get(name)
            if isinstance(value, str) and value.strip():
                source = resolve_artifact_path(submission_dir, value)
                if source.is_file():
                    _copy_blinded_asset(source, reviewer_dir / "待评文本" / source.name, redactions=redactions)
                    break
    if domains & {"visual", "av"}:
        value = artifacts.get("final_video")
        if isinstance(value, str) and value.strip():
            source = resolve_artifact_path(submission_dir, value)
            if source.is_file():
                _copy_blinded_asset(source, reviewer_dir / "待评视频" / f"video{source.suffix}", redactions=redactions)


def export_human_eval(
    store: RunStore, run_id: str, registry: MetricRegistry | None = None
) -> dict[str, Any]:
    registry = registry or MetricRegistry()
    record = store.load(run_id)
    human_status = HumanStatus(
        str((record.get("human") or {}).get("status", HumanStatus.PENDING.value))
    )
    reexporting = human_status == HumanStatus.EXPORTED
    if reexporting:
        human = record.get("human") or {}
        if int(human.get("validated_annotation_count") or 0) or human.get("imports"):
            raise ValueError("human-export cannot replace forms after annotations were imported")
        existing_human_dir = store.run_dir(run_id) / "human_eval"
        for form_path in existing_human_dir.glob("reviewer_*/评分表.csv"):
            with form_path.open("r", encoding="utf-8-sig", newline="") as handle:
                if any(
                    str(row.get("分数") or "").strip()
                    for row in csv.DictReader(handle)
                ):
                    raise ValueError(
                        "human-export cannot replace a form containing unimported scores: "
                        f"{form_path}"
                    )
    elif human_status != HumanStatus.PENDING:
        raise ValueError(f"human-export is not allowed in human state {human_status.value}")
    if not bool((record.get("preparation") or {}).get("human_ready")):
        raise ValueError("human-export requires completed core preparation")
    run_dir = store.run_dir(run_id)
    plan = load_json(run_dir / "evaluation_plan.json")
    submission_path = store.input_path(record, "submission")
    task_path = store.input_path(record, "task_input")
    gold_path = store.input_path(record, "gold")
    verify_preparation_contract(
        run_dir, task_path, gold_path, submission_path, registry
    )
    submission = load_json(submission_path)
    task = load_json(task_path)
    gold = load_json(gold_path)
    active_ids = set(plan.get("selected_active_metric_ids") or [])
    disabled_active = sorted(
        metric_id
        for metric_id in active_ids
        if metric_id in registry.metrics and not registry.get(metric_id).enabled
    )
    if disabled_active:
        raise ValueError(
            "human-export cannot create ordinary score forms for experimental/"
            "disabled metrics: " + ", ".join(disabled_active)
        )
    gates_path = run_dir / "01_deterministic" / "gates.json"
    if gates_path.exists():
        gates = load_json(gates_path)
    else:
        gates = load_json(run_dir / "01_deterministic" / "report.json")
    text_ok = gate_passed(gates, "G2.1")
    video_ok = gate_passed(gates, "G2.2")
    audio_ok = gate_passed(gates, "G2.3")
    route_track = str((plan.get("route") or {}).get("track") or submission["track"])
    required_ids: set[str] = set()
    for metric_id in active_ids:
        if metric_id not in registry.metrics:
            continue
        spec = registry.get(metric_id)
        if not spec.enabled or spec.gate or spec.formal_channel == "learner":
            continue
        if spec.human_policy not in {"required", "review"} and spec.formal_channel != "human":
            continue
        if spec.domain == "content":
            if route_track == "text_planning" and not text_ok:
                continue
            if route_track == "full_task_teaching" and not audio_ok:
                continue
        if spec.domain == "visual" and not video_ok:
            continue
        if spec.domain == "av" and not (video_ok and audio_ok):
            continue
        required_ids.add(metric_id)
    rubric = _rubric_projection(registry, required_ids)
    rubric_version = "rubric_" + canonical_json_hash(rubric)[:20]
    rubric["rubric_version"] = rubric_version
    salt = str(record["blinding_salt"])
    human_dir = run_dir / "human_eval"
    human_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    internal_items: dict[str, Any] = {}
    internal_assignments: dict[str, Any] = {}
    redactions = [
        str(submission.get("system_id") or ""),
        str(submission.get("system_name") or ""),
        str(submission_path.parent),
        str(submission_path),
    ]
    for metric_id in sorted(required_ids):
        spec = registry.get(metric_id)
        role = "learner" if spec.formal_channel == "learner" else "expert"
        item_id = _stable_id("item", salt, metric_id, "sample", role)
        items.append({"item_id": item_id, "metric_id": metric_id})
        internal_items[item_id] = {
            "metric_id": metric_id,
            "unit_id": "sample",
            "annotator_role": role,
            "required_annotations": _required_annotations(spec),
        }
        count = int(internal_items[item_id]["required_annotations"])
        for slot in range(1, count + 1):
            assignment_id = _stable_id(
                "assignment", salt, item_id, "primary", str(slot)
            )
            internal_assignments[assignment_id] = {
                "item_id": item_id,
                "annotator_role": role,
                "assignment_role": "primary",
                "slot": slot,
                "adjudicates": [],
            }
    atomic_write_text(
        human_dir / "评审指南.txt",
        "人工评审指南\n\n"
        "只观看或阅读本目录中“待评”内容，并在评分表的“分数”列填写结果。\n"
        "每个指标只填写一个针对整个样本的总体结果。\n"
        "若指标不适用于待评内容，在分数列填写 N 或 N/A；不适用项不计分。\n"
        "数字 0 始终是有效零分，不能用于表示不适用。\n"
        "0–1：0=完全不满足，0.25=少量满足，0.5=部分满足，0.75=大部分满足，1=完全满足。\n"
        "1–5：1=严重失败，2=重大问题，3=部分满足，4=基本满足，5=充分满足。\n"
        "两名评审独立判断；一人填 N/N/A、另一人给分时必须由第三人仲裁。\n"
        "两人均给分时，1–5 相差至少 2 分、0–1 相差超过 0.25 时进入仲裁。\n",
    )
    reviewer_forms: dict[str, list[dict[str, str]]] = {}
    for slot in (1, 2):
        reviewer_name = f"reviewer_{slot}"
        reviewer_dir = human_dir / reviewer_name
        rows: list[dict[str, Any]] = []
        mappings: list[dict[str, str]] = []
        metric_ids: set[str] = set()
        for assignment_id, assignment in internal_assignments.items():
            if assignment["assignment_role"] != "primary" or assignment["slot"] != slot:
                continue
            item_id = str(assignment["item_id"])
            item = internal_items[item_id]
            metric_id = str(item["metric_id"])
            spec = registry.get(metric_id)
            metric_ids.add(metric_id)
            rows.append(
                {
                    "指标": f"{metric_id}｜{spec.name}",
                    "评估内容": _display_rubric(spec),
                    "量表": _display_scale(spec.scale),
                    "参考依据": _reference_index_relative_path(metric_id),
                    "分数": "",
                }
            )
            mappings.append({"item_id": item_id, "assignment_id": assignment_id, "metric_id": metric_id})
        _export_metric_references(
            reviewer_dir,
            metric_ids,
            registry,
            task,
            gold,
            redactions=redactions,
        )
        if not reexporting:
            _export_visible_assets(
                reviewer_dir,
                submission,
                submission_path.parent,
                metric_ids,
                registry,
                redactions=redactions,
            )
        _write_simple_form(reviewer_dir / "评分表.csv", rows)
        reviewer_forms[reviewer_name] = mappings
    record["human"].update(
        {
            "rubric_version": rubric_version,
            "item_count": len(items),
            "required_annotation_count": len(internal_assignments),
            "validated_annotation_count": 0,
            "pending_item_count": len(items),
            "pending_adjudication_count": 0,
            "items": internal_items,
            "assignments": internal_assignments,
            "reviewer_forms": reviewer_forms,
            "exported_at": utc_now(),
        }
    )
    human_target = HumanStatus.COMPLETED if not items else HumanStatus.EXPORTED
    store.set_phase_status(
        record,
        "human",
        human_target.value,
        operation="human-reexport" if reexporting else "human-export",
    )
    current = store.load(run_id)
    return {
        "run_id": run_id,
        "state": current["state"],
        "human_eval_dir": str(human_dir),
        "item_count": len(items),
        "assignment_count": len(internal_assignments),
        "rubric_version": rubric_version,
    }


def _load_annotation_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(dict(value))
        return rows
    raise ValueError("human-import input must be .csv or .jsonl")


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean {value!r}")


def _parse_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is required")
    datetime.fromisoformat(text.replace("Z", "+00:00"))
    return text


def _disagrees(left: dict[str, Any], right: dict[str, Any], scale: str) -> bool:
    if bool(left["critical"]) != bool(right["critical"]):
        return True
    if bool(left["not_applicable"]) != bool(right["not_applicable"]):
        return True
    if left["not_applicable"] and right["not_applicable"]:
        return False
    difference = abs(float(left["score"]) - float(right["score"]))
    if scale == "1_5":
        return difference >= 2
    if scale in {"0_1", "pass_fail", "rate"}:
        return difference > 0.25
    return False


def _annotation_to_judgment(
    annotation: dict[str, Any],
    item: dict[str, Any],
    assignment: dict[str, Any],
    submission: dict[str, Any],
    *,
    needs_review: bool,
) -> dict[str, Any]:
    seed = f"{annotation['assignment_id']}|{annotation['annotator_id']}"
    judge_type = "learner" if item["annotator_role"] == "learner" else "human"
    return {
        "judgment_id": "j_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20],
        "assignment_id": annotation["assignment_id"],
        "adjudicates": list(assignment.get("adjudicates") or []),
        "sample_id": str(submission["sample_id"]),
        "system_id": str(submission["system_id"]),
        "track": str(submission["track"]),
        "setting": str(submission["setting"]),
        "metric_id": str(item["metric_id"]),
        "unit_id": str(item["unit_id"]),
        "score": annotation["score"],
        "label": "human_annotation",
        "weight": 1.0,
        "not_applicable": bool(annotation["not_applicable"]),
        "critical": bool(annotation["critical"]),
        "evidence": {
            "text_spans": [annotation["comment"]],
            "time_ranges": [],
            "gold_ids": [],
        },
        "judge": {
            "type": judge_type,
            "name": "offline_human_eval",
            "version": HUMAN_SCHEMA_VERSION,
            "role": str(assignment["assignment_role"]),
            "annotator_id": annotation["annotator_id"],
        },
        "confidence": annotation["confidence"],
        "needs_review": needs_review,
        "notes": annotation["comment"],
        "raw": {
            "run_id": annotation["run_id"],
            "item_id": annotation["item_id"],
            "rubric_version": annotation["rubric_version"],
            "timestamp": annotation["timestamp"],
        },
    }


def import_human_annotations(
    store: RunStore,
    run_id: str,
    input_path: Path,
    registry: MetricRegistry | None = None,
    *,
    annotator_id: str | None = None,
) -> dict[str, Any]:
    registry = registry or MetricRegistry()
    record = store.load(run_id)
    human_status = HumanStatus(
        str((record.get("human") or {}).get("status", HumanStatus.PENDING.value))
    )
    if human_status not in {
        HumanStatus.EXPORTED,
        HumanStatus.PARTIAL,
        HumanStatus.COMPLETED,
    }:
        raise ValueError(
            f"human-import is not allowed in human state {human_status.value}"
        )
    human_dir = store.run_dir(run_id) / "human_eval"
    input_path = (
        input_path.resolve()
        if input_path.is_absolute()
        else (human_dir / input_path).resolve()
    )
    if not input_path.is_file():
        raise ValueError(f"Annotation input does not exist: {input_path}")
    batch_id = "batch_" + uuid.uuid4().hex[:20]
    batch_dir = human_dir / "imports" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    raw_path = batch_dir / f"raw{input_path.suffix.lower()}"
    shutil.copyfile(input_path, raw_path)
    existing_path = human_dir / "annotations.validated.jsonl"
    existing = _load_annotation_rows(existing_path) if existing_path.exists() else []
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    try:
        rows = _load_annotation_rows(raw_path)
    except Exception as exc:
        rows = []
        errors.append(f"input: {type(exc).__name__}: {exc}")
    human = record.get("human") or {}
    items = human.get("items") or {}
    assignments = human.get("assignments") or {}
    if rows and "指标" in rows[0]:
        reviewer_name = input_path.parent.name
        mappings = (human.get("reviewer_forms") or {}).get(reviewer_name)
        if not isinstance(mappings, list):
            errors.append("简洁评分表必须保留在原 reviewer_1 或 reviewer_2 目录中")
        elif not annotator_id:
            errors.append("简洁评分表导入必须提供 annotator_id")
        elif len(rows) != len(mappings):
            errors.append("简洁评分表行数与原始评分任务不一致")
        else:
            simple_rows: list[dict[str, Any]] = []
            for row, mapping in zip(rows, mappings):
                metric_id = str(mapping["metric_id"])
                score_text = str(row.get("分数") or "").strip()
                if not score_text:
                    errors.append(f"{metric_id}: 分数不能为空")
                    continue
                item_id = str(mapping["item_id"])
                assignment_id = str(mapping["assignment_id"])
                assignment = assignments.get(assignment_id) or {}
                not_applicable = _is_not_applicable_score(score_text)
                simple_rows.append(
                    {
                        "run_id": run_id,
                        "item_id": item_id,
                        "assignment_id": assignment_id,
                        "annotator_role": assignment.get("annotator_role", "expert"),
                        "assignment_role": assignment.get("assignment_role", "primary"),
                        "rubric_version": human.get("rubric_version", ""),
                        "annotator_id": annotator_id,
                        "scores": {
                            metric_id: None if not_applicable else score_text
                        },
                        "comment": (
                            "简洁评分表：不适用"
                            if not_applicable
                            else "简洁评分表"
                        ),
                        "confidence": "1.0",
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "not_applicable": str(not_applicable).lower(),
                        "critical": "false",
                    }
                )
            rows = simple_rows
    existing_assignment_ids = {str(row.get("assignment_id")) for row in existing}
    existing_annotators = {
        (
            str(row.get("item_id")),
            str(row.get("annotator_id")),
            str(row.get("assignment_role")),
        )
        for row in existing
    }
    batch_assignment_ids: set[str] = set()
    batch_annotators: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        prefix = f"rows/{index}"
        try:
            row_run_id = str(row.get("run_id") or "").strip()
            item_id = str(row.get("item_id") or "").strip()
            assignment_id = str(row.get("assignment_id") or "").strip()
            rubric_version = str(row.get("rubric_version") or "").strip()
            annotator_id = str(row.get("annotator_id") or "").strip()
            comment = str(row.get("comment") or "").strip()
            if row_run_id != run_id:
                raise ValueError(f"invalid run_id {row_run_id!r}")
            if item_id not in items:
                raise ValueError(f"unknown item_id {item_id!r}")
            assignment = assignments.get(assignment_id)
            if not isinstance(assignment, dict) or assignment.get("item_id") != item_id:
                raise ValueError(f"invalid assignment_id {assignment_id!r}")
            if rubric_version != human.get("rubric_version"):
                raise ValueError("rubric_version mismatch")
            if not annotator_id:
                raise ValueError("annotator_id is required")
            if not comment:
                raise ValueError("comment is required")
            assignment_role = str(row.get("assignment_role") or "").strip()
            annotator_role = str(row.get("annotator_role") or "").strip()
            if assignment_role != assignment.get("assignment_role"):
                raise ValueError("assignment_role mismatch")
            if annotator_role != assignment.get("annotator_role"):
                raise ValueError("annotator_role mismatch")
            scores_value = row.get("scores")
            scores = (
                json.loads(scores_value)
                if isinstance(scores_value, str)
                else scores_value
            )
            if not isinstance(scores, dict):
                raise ValueError("scores must be an object")
            metric_id = str(items[item_id]["metric_id"])
            if set(scores) != {metric_id}:
                raise ValueError(f"scores must contain only {metric_id}")
            score_value = scores[metric_id]
            not_applicable = (
                _parse_bool(row.get("not_applicable"))
                or _is_not_applicable_score(score_value)
            )
            if not_applicable:
                score = None
            else:
                score = registry.get(metric_id).validate_score(score_value)
            confidence = float(row.get("confidence"))
            if not 0 <= confidence <= 1:
                raise ValueError("confidence must be in [0,1]")
            timestamp = _parse_timestamp(row.get("timestamp"))
            duplicate_key = (item_id, annotator_id, assignment_role)
            if assignment_id in existing_assignment_ids or assignment_id in batch_assignment_ids:
                raise ValueError(f"duplicate assignment_id {assignment_id!r}")
            if duplicate_key in existing_annotators or duplicate_key in batch_annotators:
                raise ValueError("duplicate annotation for item/annotator/role")
            normalized.append(
                {
                    "run_id": run_id,
                    "item_id": item_id,
                    "assignment_id": assignment_id,
                    "annotator_role": annotator_role,
                    "assignment_role": assignment_role,
                    "rubric_version": rubric_version,
                    "annotator_id": annotator_id,
                    "scores": {metric_id: score},
                    "score": score,
                    "comment": comment,
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "not_applicable": not_applicable,
                    "critical": _parse_bool(row.get("critical")),
                }
            )
            batch_assignment_ids.add(assignment_id)
            batch_annotators.add(duplicate_key)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}: {exc}")
    all_annotations = existing + normalized
    annotations_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in all_annotations:
        annotations_by_item[str(annotation.get("item_id"))].append(annotation)
    for item_id, item_annotations in annotations_by_item.items():
        primary_annotators = {
            str(row.get("annotator_id"))
            for row in item_annotations
            if row.get("assignment_role") == "primary"
        }
        adjudicator_annotators = {
            str(row.get("annotator_id"))
            for row in item_annotations
            if row.get("assignment_role") == "adjudicator"
        }
        overlap = sorted(primary_annotators & adjudicator_annotators)
        if overlap:
            errors.append(
                f"item {item_id}: adjudicator must be independent from primary raters: {overlap}"
            )
    report = {
        "schema_version": HUMAN_SCHEMA_VERSION,
        "run_id": run_id,
        "batch_id": batch_id,
        "valid": not errors,
        "row_count": len(rows),
        "validated_count": len(normalized) if not errors else 0,
        "errors": errors,
    }
    write_json(batch_dir / "validation_report.json", report)
    import_entry = {
        "batch_id": batch_id,
        "imported_at": utc_now(),
        "raw": str(raw_path.relative_to(store.run_dir(run_id))),
        "valid": not errors,
        "row_count": len(rows),
        "validated_count": len(normalized) if not errors else 0,
    }
    record["human"].setdefault("imports", []).append(import_entry)
    if errors:
        store.record_error(
            record,
            operation="human-import",
            error="; ".join(errors),
        )
        raise ValueError("Invalid human annotations: " + "; ".join(errors))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in all_annotations:
        grouped[str(annotation["item_id"])].append(annotation)
    pending_items = 0
    pending_adjudication = 0
    disagreement_by_item: dict[str, bool] = {}
    reliability_values: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"pairs": [], "disagreements": 0}
    )
    adjudication_rows: list[dict[str, Any]] = []
    for item_id, item in items.items():
        annotations = grouped.get(item_id, [])
        primaries = [row for row in annotations if row["assignment_role"] == "primary"]
        adjudicators = [row for row in annotations if row["assignment_role"] == "adjudicator"]
        unique_primary = {row["annotator_id"] for row in primaries}
        required = int(item["required_annotations"])
        if len(unique_primary) < required:
            pending_items += 1
            disagreement_by_item[item_id] = False
            continue
        disagrees = False
        if len(primaries) >= 2:
            pair = sorted(
                primaries[:2],
                key=lambda row: int(
                    assignments[str(row["assignment_id"])].get("slot", 0)
                ),
            )
            metric_id = str(item["metric_id"])
            disagrees = _disagrees(
                pair[0], pair[1], registry.get(metric_id).scale
            )
            stats = reliability_values[metric_id]
            if pair[0]["score"] is not None and pair[1]["score"] is not None:
                stats["pairs"].append(
                    (float(pair[0]["score"]), float(pair[1]["score"]))
                )
            stats["disagreements"] += int(disagrees)
        disagreement_by_item[item_id] = disagrees
        if disagrees and not adjudicators:
            pending_adjudication += 1
            assignment_id = _stable_id(
                "assignment", str(record["blinding_salt"]), item_id, "adjudicator", "1"
            )
            primary_ids = [row["assignment_id"] for row in primaries[:2]]
            assignments[assignment_id] = {
                "item_id": item_id,
                "annotator_role": str(item["annotator_role"]),
                "assignment_role": "adjudicator",
                "slot": 1,
                "adjudicates": primary_ids,
            }
            adjudication_rows.append(
                {
                    "run_id": run_id,
                    "item_id": item_id,
                    "assignment_id": assignment_id,
                    "annotator_role": str(item["annotator_role"]),
                    "assignment_role": "adjudicator",
                    "rubric_version": human["rubric_version"],
                }
            )
    reliability = {
        metric_id: {
            **agreement_statistics(
                values["pairs"], registry.get(metric_id).scale
            ),
            "disagreement_rate": values["disagreements"] / len(values["pairs"])
            if values["pairs"]
            else None,
            "adjudication_count": values["disagreements"],
        }
        for metric_id, values in reliability_values.items()
    }
    submission = load_json(store.input_path(record, "submission"))
    judgments: list[dict[str, Any]] = []
    for annotation in all_annotations:
        item = items[annotation["item_id"]]
        assignment = assignments[annotation["assignment_id"]]
        judgments.append(
            _annotation_to_judgment(
                annotation,
                item,
                assignment,
                submission,
                needs_review=(
                    annotation["assignment_role"] != "adjudicator"
                    and disagreement_by_item.get(annotation["item_id"], False)
                ),
            )
        )
    judgments_payload = {"schema_version": "0.4", "judgments": judgments}
    judgment_errors = validate_judgments(
        judgments_payload, SCHEMAS / "judgment.schema.json", registry
    )
    if judgment_errors:
        raise ValueError("Generated human judgments are invalid: " + "; ".join(judgment_errors))
    _write_jsonl(batch_dir / "validated.jsonl", normalized)
    _write_jsonl(existing_path, all_annotations)
    adjudication_form: Path | None = None
    if adjudication_rows:
        task = load_json(store.input_path(record, "task_input"))
        gold = load_json(store.input_path(record, "gold"))
        reviewer_dir = human_dir / "reviewer_adjudicator"
        form_rows: list[dict[str, Any]] = []
        mappings: list[dict[str, str]] = []
        metric_ids: set[str] = set()
        for row in adjudication_rows:
            item_id = str(row["item_id"])
            metric_id = str(items[item_id]["metric_id"])
            spec = registry.get(metric_id)
            metric_ids.add(metric_id)
            form_rows.append(
                {
                    "指标": f"{metric_id}｜{spec.name}",
                    "评估内容": _display_rubric(spec),
                    "量表": _display_scale(spec.scale),
                    "参考依据": _reference_index_relative_path(metric_id),
                    "分数": "",
                }
            )
            mappings.append(
                {
                    "item_id": item_id,
                    "assignment_id": str(row["assignment_id"]),
                    "metric_id": metric_id,
                }
            )
        redactions = [
            str(submission.get("system_id") or ""),
            str(submission.get("system_name") or ""),
            str(store.input_path(record, "submission").parent),
        ]
        _export_metric_references(
            reviewer_dir,
            metric_ids,
            registry,
            task,
            gold,
            redactions=redactions,
        )
        _export_visible_assets(
            reviewer_dir,
            submission,
            store.input_path(record, "submission").parent,
            metric_ids,
            registry,
            redactions=redactions,
        )
        adjudication_form = reviewer_dir / "评分表.csv"
        _write_simple_form(adjudication_form, form_rows)
        human.setdefault("reviewer_forms", {})["reviewer_adjudicator"] = mappings
    write_json(human_dir / "agreement.json", reliability)
    write_json(human_dir / "judgments.human.json", judgments_payload)
    human["assignments"] = assignments
    human["validated_annotation_count"] = len(all_annotations)
    human["pending_item_count"] = pending_items
    human["pending_adjudication_count"] = pending_adjudication
    complete = pending_items == 0 and pending_adjudication == 0
    human_report_path: Path | None = None
    if complete:
        required_ids = {str(item["metric_id"]) for item in items.values()}
        expected_units: dict[str, set[str]] = defaultdict(set)
        for item in items.values():
            expected_units[str(item["metric_id"])].add(str(item["unit_id"]))
        human_report = aggregate_judgments(
            judgments_payload["judgments"],
            registry,
            track=str(submission["track"]),
            setting=str(submission["setting"]),
            include_diagnostics=any(
                registry.get(metric_id).diagnostic for metric_id in required_ids
            ),
            active_metric_ids=sorted(required_ids),
            required_review_metric_ids=required_ids,
            expected_units_by_metric=expected_units,
        )
        human_report_path = human_dir / "report.json"
        write_json(human_report_path, human_report)
        (human_dir / "report.txt").write_text(
            render_text_report(human_report, judgments_payload["judgments"], registry),
            encoding="utf-8",
        )
    target = HumanStatus.COMPLETED if complete else HumanStatus.PARTIAL
    store.set_phase_status(
        record,
        "human",
        target.value,
        operation="human-import",
        report=str(human_report_path) if human_report_path else None,
    )
    current = store.load(run_id)
    return {
        **report,
        "state": current["state"],
        "pending_item_count": pending_items,
        "pending_adjudication_count": pending_adjudication,
        "adjudication_form": str(adjudication_form) if adjudication_form else None,
    }


def finalize_run(
    store: RunStore, run_id: str, registry: MetricRegistry | None = None
) -> dict[str, Any]:
    registry = registry or MetricRegistry()
    record = store.load(run_id)
    state = RunState(record["state"])
    if state == RunState.FINALIZED:
        return dict(record.get("final") or {})
    human_status = HumanStatus(
        str((record.get("human") or {}).get("status", HumanStatus.PENDING.value))
    )
    machine_status = MachineStatus(
        str((record.get("machine") or {}).get("status", MachineStatus.PENDING.value))
    )
    if human_status != HumanStatus.COMPLETED:
        raise ValueError(
            f"finalize requires completed human evaluation, got {human_status.value}"
        )
    if machine_status != MachineStatus.COMPLETED:
        raise ValueError(
            f"finalize requires completed machine evaluation, got {machine_status.value}"
        )
    task_path = store.input_path(record, "task_input")
    gold_path = store.input_path(record, "gold")
    submission_path = store.input_path(record, "submission")
    run_dir = store.run_dir(run_id)
    verify_preparation_contract(
        run_dir,
        task_path,
        gold_path,
        submission_path,
        registry,
    )
    automatic = load_json(run_dir / "08_merged" / "judgments.json")
    human_path = run_dir / "human_eval" / "judgments.human.json"
    human = load_json(human_path) if human_path.exists() else {
        "schema_version": "0.4",
        "judgments": [],
    }
    merged = merge_judgment_payloads([automatic, human])
    errors = validate_judgments(
        merged, SCHEMAS / "judgment.schema.json", registry
    )
    if errors:
        raise ValueError("Cannot finalize invalid judgments: " + "; ".join(errors))
    plan = load_json(run_dir / "evaluation_plan.json")
    submission = load_json(submission_path)
    deterministic_path = run_dir / "01_deterministic" / "report.json"
    deterministic = load_json(deterministic_path) if deterministic_path.exists() else {}
    items = (record.get("human") or {}).get("items") or {}
    required_ids = {str(item["metric_id"]) for item in items.values()}
    expected_units: dict[str, set[str]] = defaultdict(set)
    for item in items.values():
        expected_units[str(item["metric_id"])].add(str(item["unit_id"]))
    report = aggregate_judgments(
        merged["judgments"],
        registry,
        track=str(submission["track"]),
        setting=str(submission["setting"]),
        include_diagnostics=any(
            registry.get(metric_id).diagnostic
            for metric_id in plan.get("selected_active_metric_ids") or []
        ),
        active_metric_ids=plan.get("selected_active_metric_ids") or [],
        required_review_metric_ids=required_ids,
        expected_units_by_metric=expected_units,
        auxiliary_reports=deterministic.get("auxiliary") or {},
    )
    if not bool((report.get("review_status") or {}).get("complete")):
        raise ValueError("Human evaluation is incomplete according to aggregation")
    agreement_path = run_dir / "human_eval" / "agreement.json"
    annotations_path = run_dir / "human_eval" / "annotations.validated.jsonl"
    annotations = _load_annotation_rows(annotations_path) if annotations_path.exists() else []
    report["human_evaluation"] = {
        "rubric_version": (record.get("human") or {}).get("rubric_version"),
        "annotation_count": len(annotations),
        "annotator_count": len({row["annotator_id"] for row in annotations}),
        "agreement": load_json(agreement_path) if agreement_path.exists() else {},
        "adjudications": [
            row for row in annotations if row["assignment_role"] == "adjudicator"
        ],
    }
    final_dir = run_dir / "final"
    write_json(final_dir / "judgments.json", merged)
    write_json(final_dir / "report.json", report)
    (final_dir / "report.txt").write_text(
        render_text_report(report, merged["judgments"], registry), encoding="utf-8"
    )
    final_record = {
        "run_id": run_id,
        "state": RunState.FINALIZED.value,
        "report": str(final_dir / "report.json"),
        "report_txt": str(final_dir / "report.txt"),
        "judgments": str(final_dir / "judgments.json"),
        "report_sha256": sha256_file(final_dir / "report.json"),
        "finalized_at": utc_now(),
    }
    record["final"] = final_record
    store.transition(
        record,
        RunState.FINALIZED,
        allowed_from={RunState.HUMAN_COMPLETED},
        operation="finalize",
    )
    return final_record


def run_status(store: RunStore, run_id: str) -> dict[str, Any]:
    record = store.load(run_id)
    state = RunState(record["state"])
    preparation = record.get("preparation") or {}
    machine = record.get("machine") or {}
    human = record.get("human") or {}
    machine_status = MachineStatus(
        str(machine.get("status", MachineStatus.PENDING.value))
    )
    human_status = HumanStatus(
        str(human.get("status", HumanStatus.PENDING.value))
    )
    next_commands: list[str] = []
    if state != RunState.FINALIZED:
        if not bool(preparation.get("human_ready")):
            next_commands.append("prepare --run-id")
        elif human_status == HumanStatus.PENDING:
            next_commands.append("human-export")
        elif human_status in {HumanStatus.EXPORTED, HumanStatus.PARTIAL}:
            next_commands.append("human-import")
        if machine_status != MachineStatus.COMPLETED:
            if bool(preparation.get("machine_ready")):
                next_commands.append("machine-evaluate")
            else:
                next_commands.append("prepare --run-id --force-stage asr")
        if (
            machine_status == MachineStatus.COMPLETED
            and human_status == HumanStatus.COMPLETED
        ):
            next_commands.append("finalize")
    return {
        "run_id": run_id,
        "state": state.value,
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "preparation": preparation,
        "machine": machine,
        "human": {
            "status": human_status.value,
            "rubric_version": human.get("rubric_version"),
            "item_count": human.get("item_count", 0),
            "required_annotation_count": human.get("required_annotation_count", 0),
            "validated_annotation_count": human.get("validated_annotation_count", 0),
            "pending_item_count": human.get("pending_item_count", 0),
            "pending_adjudication_count": human.get("pending_adjudication_count", 0),
            "import_count": len(human.get("imports") or []),
            "report": human.get("report"),
        },
        "last_error": record.get("last_error"),
        "next_commands": next_commands,
        "final": record.get("final"),
    }
