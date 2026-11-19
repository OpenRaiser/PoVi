from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


VGEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = VGEN_ROOT / "data" / "examples" / "qingyu_an_yuanxi.json"
DEFAULT_SCHEMA = VGEN_ROOT / "schema" / "poetic_video_director_graph.schema.json"
DEFAULT_RULE_VALIDATOR = VGEN_ROOT / "schema" / "validate_director_graph.py"
DEFAULT_OUTPUT_DIR = VGEN_ROOT / "outputs" / "director_graphs"
DEFAULT_BASE_URL = ""
DEFAULT_MODEL = "claude-opus-4.8"

TRUNCATION_MARKER = "展开阅读全文"
@dataclass(frozen=True)
class CompletionResult:
    content: str
    usage: dict[str, Any]


class GraphCompletionClient(Protocol):
    model: str
    base_url: str

    def complete(self, messages: list[dict[str, str]]) -> CompletionResult:
        ...


class OpenAIGraphCompletionClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_tokens: int,
        temperature: float,
    ) -> None:
        from openai import OpenAI

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=3,
        )

    def complete(self, messages: list[dict[str, str]]) -> CompletionResult:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError("LLM returned an empty response")
        usage = response.usage.model_dump() if response.usage is not None else {}
        return CompletionResult(content=content, usage=usage)


def load_corpus(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        records = data["records"]
        metadata = {key: value for key, value in data.items() if key != "records"}
    elif isinstance(data, dict) and all(key in data for key in ("title", "author", "content")):
        records = [data]
        metadata = {"count": 1, "source": str(path)}
    elif isinstance(data, list):
        records = data
        metadata = {"count": len(records)}
    else:
        raise ValueError(f"Unsupported poetry source structure: {path}")
    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Poetry records must be JSON objects: {path}")
    normalized_records = []
    for position, item in enumerate(records, start=1):
        record = dict(item)
        record.setdefault("index", position)
        identity_text = "\n".join(
            str(record.get(key) or "") for key in ("title", "author", "content")
        )
        record.setdefault("poem_id", hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:12])
        record.setdefault("dynasty", "现有材料未提供")
        record.setdefault("form", "shi")
        record.setdefault("url", str(record.get("source") or "user-provided"))
        normalized_records.append(record)
    return normalized_records, metadata


def select_records(
    records: list[dict[str, Any]],
    *,
    select_all: bool,
    indices: list[int] | None,
    poem_ids: list[str] | None,
    titles: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if select_all:
        selected = list(records)
    elif indices:
        wanted = set(indices)
        selected = [item for item in records if int(item.get("index", -1)) in wanted]
        missing = wanted - {int(item["index"]) for item in selected}
        if missing:
            raise ValueError(f"Corpus indices not found: {sorted(missing)}")
    elif poem_ids:
        wanted = set(poem_ids)
        selected = [item for item in records if str(item.get("poem_id", "")) in wanted]
        missing = wanted - {str(item["poem_id"]) for item in selected}
        if missing:
            raise ValueError(f"Poem ids not found: {sorted(missing)}")
    elif titles:
        wanted = set(titles)
        selected = [item for item in records if str(item.get("title", "")) in wanted]
        missing = wanted - {str(item["title"]) for item in selected}
        if missing:
            raise ValueError(f"Exact poem titles not found: {sorted(missing)}")
    else:
        raise ValueError("Select records with --all, --index, --poem-id, or --title")

    selected.sort(key=lambda item: int(item.get("index", 0)))
    return selected[:limit] if limit is not None else selected


def _clean_source_text(value: Any) -> tuple[str, bool]:
    text = str(value or "").strip()
    truncated = TRUNCATION_MARKER in text
    text = text.replace(TRUNCATION_MARKER, "[原始抓取内容在此处截断]")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, truncated


def _take_with_budget(text: str, remaining: int) -> tuple[str, int, bool]:
    if len(text) <= remaining:
        return text, remaining - len(text), False
    if remaining <= 0:
        return "", 0, True
    suffix = "\n[因源材料长度限制在此截断]"
    keep = max(0, remaining - len(suffix))
    return text[:keep] + suffix, 0, True


def build_source_packet(record: dict[str, Any], max_source_chars: int = 24000) -> dict[str, Any]:
    required = ["index", "poem_id", "title", "author", "dynasty", "content", "url"]
    missing = [key for key in required if not str(record.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Corpus record is missing required fields: {missing}")

    remaining = max_source_chars
    was_truncated = False
    canonical_materials: dict[str, str] = {}
    for key in ["background", "appreciation", "translation_annotation"]:
        text, source_truncated = _clean_source_text(record.get(key))
        text, remaining, budget_truncated = _take_with_budget(text, remaining)
        canonical_materials[key] = text
        was_truncated = was_truncated or source_truncated or budget_truncated

    sections: list[dict[str, str]] = []
    canonical_section_titles = {"创作背景", "简析", "译文及注释"}
    for section in record.get("sections", []):
        if not isinstance(section, dict):
            continue
        title = str(section.get("title", "未命名材料")).strip() or "未命名材料"
        if title in canonical_section_titles:
            continue
        text, source_truncated = _clean_source_text(section.get("text"))
        if not text:
            continue
        text, remaining, budget_truncated = _take_with_budget(text, remaining)
        was_truncated = was_truncated or source_truncated or budget_truncated
        if text:
            sections.append({"title": title, "text": text})
        if remaining == 0:
            break

    background = canonical_materials["background"]
    appreciation = canonical_materials["appreciation"]
    translation = canonical_materials["translation_annotation"]

    missing_materials = []
    if not background:
        missing_materials.append("creation_background")
    if not translation:
        missing_materials.append("translation_annotation")
    if not appreciation:
        missing_materials.append("appreciation")

    return {
        "record_identity": {
            "index": int(record["index"]),
            "poem_id": str(record["poem_id"]),
            "title": str(record["title"]),
            "author": str(record["author"]),
            "dynasty": str(record["dynasty"]),
            "category": str(record.get("category", "")),
            "tags": [str(item) for item in record.get("tags", [])],
            "source_url": str(record["url"]),
        },
        "poem_text": str(record["content"]).strip(),
        "creation_background": background,
        "appreciation": appreciation,
        "translation_annotation": translation,
        "source_sections": sections,
        "source_completeness": {
            "missing_materials": missing_materials,
            "contains_truncated_source": was_truncated,
            "policy": "缺失或截断处可由Knowledge Planner补充模型知识，但必须标为model_supplied且不得升级为已核验事实。",
        },
    }


def split_poem_lines(
    content: str,
    supplied_lines: Any = None,
) -> list[dict[str, Any]]:
    if isinstance(supplied_lines, list) and supplied_lines:
        normalized = []
        for index, item in enumerate(supplied_lines, start=1):
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                raise ValueError(f"Invalid supplied poem line at position {index}")
            normalized.append(
                {
                    "line_id": str(item.get("line_id") or f"l{index:02d}"),
                    "order": int(item.get("order") or index),
                    "text": str(item["text"]).strip(),
                }
            )
        if len({item["line_id"] for item in normalized}) != len(normalized):
            raise ValueError("Supplied poem line IDs must be unique")
        return normalized

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Poem content contains no non-empty lines")
    return [
        {"line_id": f"l{index:02d}", "order": index, "text": line}
        for index, line in enumerate(lines, start=1)
    ]


def make_doc_id(record: dict[str, Any]) -> str:
    if str(record.get("doc_id") or "").strip():
        return str(record["doc_id"]).strip()
    corpus = re.sub(r"[^a-z0-9_.-]+", "_", str(record.get("corpus") or "tangshi300").lower())
    return f"{corpus}_{int(record['index']):03d}_{str(record['poem_id']).lower()}"


def build_evidence_sources(record: dict[str, Any], packet: dict[str, Any]) -> list[dict[str, Any]]:
    title = str(record["title"])
    author = str(record["author"])
    excerpt_parts = [f"诗文：\n{packet['poem_text']}"]
    if packet["creation_background"]:
        excerpt_parts.append(f"创作背景：\n{packet['creation_background']}")
    if packet["appreciation"]:
        excerpt_parts.append(f"简析：\n{packet['appreciation']}")
    if packet["translation_annotation"]:
        excerpt_parts.append(f"译文及注释：\n{packet['translation_annotation']}")
    excerpt = "\n\n".join(excerpt_parts)[:16000]

    claims = [f"输入材料包含{author}《{title}》的诗文与作者信息"]
    if packet["creation_background"]:
        claims.append("页面提供该诗的创作背景材料")
    if packet["translation_annotation"]:
        claims.append("页面提供该诗的译文与注释材料")
    if packet["appreciation"]:
        claims.append("页面提供该诗的简析材料")

    source_name = str(record.get("source_name") or "输入诗词材料")
    source_type = str(record.get("source_type") or "web_article")
    return [
        {
            "source_id": "e01",
            "source_type": source_type,
            "citation": f"{source_name}《{title}》",
            "authority_level": "medium",
            "relevant_claims": claims,
            "source_excerpt": excerpt,
            "locator": str(record["url"]),
            "origin": "provided",
        }
    ]


def build_locked_graph_fields(
    record: dict[str, Any],
    packet: dict[str, Any],
    target_duration_seconds: float,
) -> dict[str, Any]:
    poem = {
        "title": str(record["title"]),
        "author": str(record["author"]),
        "dynasty": str(record["dynasty"]),
        "form": str(record.get("form") or "shi"),
        "text": str(record["content"]).strip(),
        "lines": split_poem_lines(str(record["content"]), record.get("lines")),
        "source": str(record["url"]),
    }
    # The five-agent realizer now owns the card narration. Keep only the
    # insertion/duration contract in the graph; source appreciation/background
    # remains available through evidence_sources and interpretation.
    poem_card: dict[str, Any] = {"minimum_seconds": 8}

    locked_fields = {
        "schema_version": "0.5",
        "doc_id": make_doc_id(record),
        "poem": poem,
        "task_profile": {
            "video_type": "hybrid_explainer_reenactment",
            "target_audience": str(record.get("target_audience") or "初中及以上学习者"),
            "target_duration_seconds": target_duration_seconds,
            "narration_style": "先讲有证据约束的前因后果，再逐句精读；明确区分史实、推断与争议",
            "visual_mode": "历史或创作背景前传加诗境演绎，使用视觉回顾连接背景与诗句",
            "constraints": [
                "单镜头3到9秒，按旁白语义自然分镜",
                "进入bridge_to_poem之前不得直接复演诗句字面场景",
                "来源未提供的时间、地点、人物心理或因果关系不得写成documented事实",
                "模型补充必须明确标注未外部核验并保留不确定性",
                "人物、时代服饰和环境不得出现现代元素",
                *[
                    str(instruction)
                    for instruction in record.get("task_instructions", [])
                    if str(instruction).strip()
                ],
            ],
        },
        "evidence_sources": build_evidence_sources(record, packet),
        "evaluation_hooks": {
            "v1_text_planning": [
                "causal_grounding",
                "biography_poem_bridge",
                "evidence_faithfulness",
                "uncertainty_calibration",
                "metaphor_correctness",
                "cultural_grounding",
                "teaching_coherence",
                "shot_decomposition",
                "prompt_executability",
            ],
            "v2_video_quality": [
                "semantic_alignment",
                "asset_consistency",
                "style_consistency",
                "camera_reasonableness",
                "transition_continuity",
                "aesthetic_quality",
            ],
            "v3_av_sync": [
                "audio_cue_alignment",
                "duration_fit",
                "semantic_key_moment_sync",
                "narration_rhythm",
            ],
        },
    }
    locked_fields["poem_card"] = poem_card
    return locked_fields


def build_system_prompt(schema: dict[str, Any]) -> str:
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return f"""你是古诗词Knowledge Planner。输出后续五个执行Agent共享的语义 Director Graph JSON。

事实纪律：
1. 优先使用SOURCE_PACKET中的材料，e01的origin必须为provided。
2. SOURCE_PACKET缺少背景时，程序会提供MODEL_KNOWLEDGE_INVENTORY和ALLOWED_MODEL_SOURCES。只能使用其中已锁定的模型知识，不得在生成graph时临时补充其他传记细节。模型来源不能支持documented或scholarly_consensus，只能支持historically_plausible、interpretive_hypothesis、contested或unknown，并在caveat中明确“模型补充，未外部核验”。
3. 材料没有说明时，字段仍需填写，但必须写“现有材料未说明……”，claim_status 使用 unknown、contested 或 interpretive_hypothesis，并写清 caveat。
4. “可能”“一说”“推测”等限定词必须保留，不能升级为确定事实。
5. 不得虚构书名、论文、史料、页码或引文。提供材料使用e01；模型补充只能逐字使用ALLOWED_MODEL_SOURCES中的ID和claims，不得冒充具体文献来源。
    6. 诗文、作者、行号、task_profile、poem_card和evaluation_hooks必须逐字采用LOCKED_GRAPH_FIELDS。evidence_sources必须包含LOCKED_GRAPH_FIELDS中的provided来源，并逐字加入ALLOWED_MODEL_SOURCES；不得改写或新增来源。poem_card只锁定插入/时长契约，不提供可直接朗读的赏析；卡片教学旁白由后续CloseReadingAgent生成。这些字段会被程序强制覆盖。

知识粒度：
1. 先判断作品是否有可用的人物事件或历史过程。若有，model_supplied.relevant_claims必须拆成原子事实，每项只讲一个可叙事事件，至少包含可确定的时间范围、人物、触发、选择或行动、直接结果；不要用“一生心系家国”“长期失意”等总括句代替已知的起事、任职、迁徙、贬谪、冲突或创作前因。
2. event_based、historical_context或hybrid语境通常生成4到8个context_nodes，并按“初始处境或压力 -> 触发事件 -> 选择与行动 -> 直接结果 -> 长期目标或受阻 -> 与作品的解释桥梁”组织。链条按材料实际复杂度取舍，不强填不存在的阶段。
3. 最早的具体事件节点必须出现在抽象的persistent_goal或blocked_goal之前。每个事件节点的5W1H要讲清事件如何发生和改变了什么；多个事件不得压进一个空泛节点。
4. 若模型只记得概括而不确信事件细节，就明确写unknown，不编数字、路线、对话或心理。低权威不等于可以虚构，但允许把模型确实掌握的具体传记知识作为未核验候选材料输出。
5. writing_situation必须有独立的提供材料或模型知识依据。严禁从诗句中的节日、地点、动作、人物或物象反推出作者亲历了同一创作现场；没有依据时，时间、地点和现场经过明确写现有材料未说明。

职责边界：
1. 只生成causal_context、interpretation和visual_bible等知识与高层风格信息。不要输出pre_poem_story、teaching_script、assets或shots；后续Story、Narration、Close Reading、Character Asset和Visual Director分别负责这些内容。
2. causal_context只建立可供故事Agent使用的事实、5W1H、因果关系和不确定性，不写最终故事或旁白。
3. interpretation逐行覆盖原文，提供字面内容、文化知识、可接受的解释与争议边界，不写逐句教学旁白。
4. visual_bible只定义时代、总体美术和禁忌，不创建人物资产、镜头、运镜或video prompt。
5. 所有ID和交叉引用必须存在。只输出schema_version、doc_id、poem、task_profile、evidence_sources、causal_context、poem_card、interpretation、visual_bible、evaluation_hooks和metadata。

只返回一个 JSON 对象，不要 Markdown，不要解释。必须符合以下 JSON Schema：
{schema_text}"""


def build_generation_messages(
    schema: dict[str, Any],
    packet: dict[str, Any],
    locked_fields: dict[str, Any],
    model_knowledge_inventory: dict[str, Any] | None = None,
    model_sources: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    user_payload = {
        "SOURCE_PACKET": packet,
        "MODEL_KNOWLEDGE_INVENTORY": model_knowledge_inventory,
        "ALLOWED_MODEL_SOURCES": model_sources or [],
        "LOCKED_GRAPH_FIELDS": locked_fields,
        "instruction": "生成语义director graph。事实只能来自SOURCE_PACKET和已锁定的MODEL_KNOWLEDGE_INVENTORY。",
    }
    return [
        {"role": "system", "content": build_system_prompt(schema)},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def build_model_knowledge_messages(packet: dict[str, Any]) -> list[dict[str, str]]:
    prompt = f"""请仅使用你的内部知识，为古诗词Knowledge Planner建立一个未外部核验的知识清单。此阶段不生成director graph、故事、旁白、分镜或视觉prompt。

输出严格为：
{{
  "biographical_events":[{{
    "event_id":"mk_event_01","what":"一个具体事件","who":"参与者","when":"时间或明确未知", "where":"地点或明确未知",
    "trigger":"事件为何发生","action":"人物采取的具体行动及经过","immediate_result":"直接结果","long_term_result":"后续影响",
    "confidence":"high","caveat":"模型知识，未外部核验"
  }}],
  "historical_conditions":[{{
    "condition_id":"mk_condition_01","what":"具体历史条件","when":"时间范围","where":"作用范围","effect_on_subject":"怎样影响作者或作品语境",
    "confidence":"medium","caveat":"模型知识，未外部核验"
  }}],
  "writing_situation":{{
    "status":"unknown","time":"现有模型知识无法确定","place":"现有模型知识无法确定","event":"现有模型知识无法确定",
    "confidence":"low","caveat":"不得由诗句字面反推创作现场"
  }},
  "interpretation_claims":[{{
    "claim_id":"mk_interpretation_01","claim":"一种有边界的解释","status":"contested","caveat":"模型知识，未外部核验"
  }}]
}}

规则：
1. 若你确实知道与作者及作品相关的具体传记事件，按时间顺序列3到8个biographical_events。每个事件必须有触发、行动经过、直接结果和长期结果，不能用“一生爱国”“长期失意”“始终坚持”等概括代替事件。
2. 起事、参战、冲突、迁徙、任职、贬谪、上书等不同事件分别列项，不得合并成一句履历。时间、人数、路线或书名不确定时写不确定，禁止猜一个精确值。
3. 如果作品主要是山水、咏物或私人抒情，且你不知道相关人物事件，biographical_events可以为空；不得为满足数量虚构故事。
4. writing_situation只能写你独立知道的创作时间、地点或触发事件。严禁从诗句里的节日、舟船、楼台、人物动作或物象反推作者亲历同一现场；不确定就status=unknown。
5. 不虚构文献、论文、页码、引语或“学界主流”。所有内容都是模型知识，后续统一标为未外部核验。
6. ID从01连续编号。confidence只能是high、medium、low；interpretation status只能是plausible、contested、unknown。只输出JSON。

Source packet:
{json.dumps(packet, ensure_ascii=False, indent=2)}
""".strip()
    return [
        {
            "role": "system",
            "content": "你是古诗词传记与历史知识整理员。只提取具体知识，不写教学故事，只输出严格JSON。",
        },
        {"role": "user", "content": prompt},
    ]


def parse_model_knowledge(content: str) -> dict[str, Any]:
    data = extract_json_object(content)
    required = {
        "biographical_events",
        "historical_conditions",
        "writing_situation",
        "interpretation_claims",
    }
    if set(data) != required:
        raise ValueError(f"Unexpected model-knowledge fields: {sorted(data)}")

    events = data["biographical_events"]
    if not isinstance(events, list) or len(events) > 8:
        raise ValueError("biographical_events must contain at most 8 events")
    event_fields = {
        "event_id", "what", "who", "when", "where", "trigger", "action",
        "immediate_result", "long_term_result", "confidence", "caveat",
    }
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict) or set(event) != event_fields:
            raise ValueError("Invalid biographical event fields")
        if event["event_id"] != f"mk_event_{index:02d}":
            raise ValueError("Biographical event ids must be sequential")
        if event["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("Invalid event confidence")
        if any(not str(event[field]).strip() for field in event_fields - {"event_id"}):
            raise ValueError(f"Empty biographical event field: {event['event_id']}")

    conditions = data["historical_conditions"]
    condition_fields = {
        "condition_id", "what", "when", "where", "effect_on_subject", "confidence", "caveat",
    }
    if not isinstance(conditions, list) or len(conditions) > 6:
        raise ValueError("historical_conditions must contain at most 6 items")
    for index, condition in enumerate(conditions, start=1):
        if not isinstance(condition, dict) or set(condition) != condition_fields:
            raise ValueError("Invalid historical condition fields")
        if condition["condition_id"] != f"mk_condition_{index:02d}":
            raise ValueError("Historical condition ids must be sequential")
        if condition["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("Invalid condition confidence")

    writing = data["writing_situation"]
    writing_fields = {"status", "time", "place", "event", "confidence", "caveat"}
    if not isinstance(writing, dict) or set(writing) != writing_fields:
        raise ValueError("Invalid writing_situation fields")
    if writing["status"] not in {"known", "contested", "unknown"}:
        raise ValueError("Invalid writing situation status")
    if writing["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("Invalid writing situation confidence")

    interpretations = data["interpretation_claims"]
    interpretation_fields = {"claim_id", "claim", "status", "caveat"}
    if not isinstance(interpretations, list) or len(interpretations) > 8:
        raise ValueError("interpretation_claims must contain at most 8 items")
    for index, item in enumerate(interpretations, start=1):
        if not isinstance(item, dict) or set(item) != interpretation_fields:
            raise ValueError("Invalid interpretation claim fields")
        if item["claim_id"] != f"mk_interpretation_{index:02d}":
            raise ValueError("Interpretation claim ids must be sequential")
        if item["status"] not in {"plausible", "contested", "unknown"}:
            raise ValueError("Invalid interpretation status")
    return data


def model_knowledge_source(inventory: dict[str, Any]) -> dict[str, Any]:
    claims = []
    for event in inventory["biographical_events"]:
        claims.append(
            f"{event['when']}，{event['who']}在{event['where']}经历{event['what']}；"
            f"起因是{event['trigger']}；行动经过为{event['action']}；"
            f"直接结果是{event['immediate_result']}；后续影响是{event['long_term_result']}。"
        )
    for condition in inventory["historical_conditions"]:
        claims.append(
            f"{condition['when']}，{condition['where']}存在{condition['what']}；"
            f"它对作者或作品语境的影响是{condition['effect_on_subject']}。"
        )
    writing = inventory["writing_situation"]
    if writing["status"] != "unknown":
        claims.append(
            f"作品写作处境为：时间{writing['time']}，地点{writing['place']}，"
            f"事件{writing['event']}；状态为{writing['status']}。"
        )
    claims.extend(str(item["claim"]) for item in inventory["interpretation_claims"])
    return {
        "source_id": "m01",
        "source_type": "reference_work",
        "citation": "模型知识清单，未外部核验",
        "authority_level": "low",
        "relevant_claims": claims,
        "source_excerpt": json.dumps(inventory, ensure_ascii=False),
        "locator": "model-internal-knowledge",
        "origin": "model_supplied",
    }


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM response root must be a JSON object")
    return data


def normalize_graph(
    graph: dict[str, Any],
    locked_fields: dict[str, Any],
    *,
    record: dict[str, Any],
    model: str,
    source_hash: str,
    model_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = copy.deepcopy(graph)
    for key in ("pre_poem_story", "teaching_script", "assets", "shots"):
        normalized.pop(key, None)
    for key in [
        "schema_version",
        "doc_id",
        "poem",
        "task_profile",
        "poem_card",
        "evaluation_hooks",
    ]:
        if key in locked_fields:
            normalized[key] = copy.deepcopy(locked_fields[key])
    provided_sources = copy.deepcopy(locked_fields["evidence_sources"])
    normalized["evidence_sources"] = provided_sources + copy.deepcopy(model_sources or [])
    metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
    metadata.update(
        {
            "corpus": str(record.get("corpus") or "tangshi300"),
            "corpus_index": int(record["index"]),
            "poem_id": str(record["poem_id"]),
            "source_url": str(record["url"]),
            "source_sha256": source_hash,
            "generator_model": model,
            "grounding_policy": "provided_sources_plus_explicit_unverified_model_inventory",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    normalized["metadata"] = metadata
    return normalized


_RULE_VALIDATOR_CACHE: dict[Path, Any] = {}


def _load_rule_validator(path: Path) -> Any:
    path = path.resolve()
    if path in _RULE_VALIDATOR_CACHE:
        return _RULE_VALIDATOR_CACHE[path]
    spec = importlib.util.spec_from_file_location("poetic_video_rule_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import rule validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RULE_VALIDATOR_CACHE[path] = module
    return module


def validate_generated_graph(
    graph: dict[str, Any],
    schema: dict[str, Any],
    rule_validator_path: Path,
) -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    schema_errors = []
    for error in sorted(
        Draft202012Validator(schema).iter_errors(graph),
        key=lambda item: ".".join(str(part) for part in item.path),
    ):
        location = ".".join(str(item) for item in error.absolute_path) or "$"
        schema_errors.append(f"{location}: {error.message}")

    rule_module = _load_rule_validator(rule_validator_path)
    rule_errors, warnings = rule_module.validate_graph(graph)
    grounding_errors = validate_grounding_contract(graph)
    errors = schema_errors + list(rule_errors) + grounding_errors
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": list(warnings),
        "schema_error_count": len(schema_errors),
        "rule_error_count": len(rule_errors),
        "grounding_error_count": len(grounding_errors),
    }


def validate_grounding_contract(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    sources = {
        str(item.get("source_id")): item
        for item in graph.get("evidence_sources", [])
        if isinstance(item, dict) and item.get("source_id")
    }
    unknown_ids = {
        source_id
        for source_id, source in sources.items()
        if not (
            (source_id == "e01" and source.get("origin") == "provided")
            or (re.fullmatch(r"m\d+", source_id) and source.get("origin") == "model_supplied")
        )
    }
    if unknown_ids:
        errors.append(f"Unsupported evidence ids were invented: {sorted(unknown_ids)}")
    supplied_ids = {
        source_id for source_id, source in sources.items() if source.get("origin") == "provided"
    }
    high_authority_ids = {
        source_id
        for source_id, source in sources.items()
        if source.get("authority_level") in {"primary", "high"}
    }

    causal = graph.get("causal_context", {})
    claim_items = []
    if isinstance(causal, dict):
        claim_items.extend(("context_node", item) for item in causal.get("context_nodes", []))
        claim_items.extend(("causal_link", item) for item in causal.get("causal_links", []))
    for label, item in claim_items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("node_id") or item.get("link_id") or "?"
        evidence_ids = set(str(value) for value in item.get("evidence_ids", []))
        status = str(item.get("claim_status", ""))
        if status == "documented" and not evidence_ids.intersection(supplied_ids):
            errors.append(f"{label}[{item_id}] is documented but has no supplied evidence")
        if status in {"documented", "scholarly_consensus"} and any(
            sources.get(source_id, {}).get("origin") == "model_supplied"
            for source_id in evidence_ids
        ):
            errors.append(f"{label}[{item_id}] treats model-supplied knowledge as verified fact")
        if status == "scholarly_consensus" and not evidence_ids.intersection(high_authority_ids):
            errors.append(f"{label}[{item_id}] claims scholarly consensus without a high-authority source")

    interpretation = graph.get("interpretation", {})
    if isinstance(interpretation, dict):
        for node in interpretation.get("metaphor_nodes", []):
            if not isinstance(node, dict) or node.get("interpretation_status") != "scholarly_consensus":
                continue
            evidence_ids = set(str(value) for value in node.get("evidence_ids", []))
            if not evidence_ids.intersection(high_authority_ids):
                errors.append(
                    f"metaphor_node[{node.get('node_id', '?')}] claims scholarly consensus "
                    "without a high-authority source"
                )
    return errors


def _source_hash(packet: dict[str, Any]) -> str:
    payload = json.dumps(packet, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_output_dir(output_root: Path, record: dict[str, Any]) -> Path:
    return output_root / f"{int(record['index']):03d}_{record['poem_id']}"


def _repair_messages(
    schema: dict[str, Any],
    packet: dict[str, Any],
    locked_fields: dict[str, Any],
    model_knowledge_inventory: dict[str, Any] | None,
    model_sources: list[dict[str, Any]],
    graph: dict[str, Any] | None,
    errors: list[str],
) -> list[dict[str, str]]:
    payload = {
        "SOURCE_PACKET": packet,
        "MODEL_KNOWLEDGE_INVENTORY": model_knowledge_inventory,
        "ALLOWED_MODEL_SOURCES": model_sources,
        "LOCKED_GRAPH_FIELDS": locked_fields,
        "INVALID_GRAPH": graph,
        "VALIDATION_ERRORS": errors[:120],
        "instruction": "只修复列出的错误，仍须遵守证据纪律；返回完整 JSON 对象。",
    }
    return [
        {"role": "system", "content": build_system_prompt(schema)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_one_record(
    record: dict[str, Any],
    *,
    schema: dict[str, Any],
    rule_validator_path: Path,
    output_root: Path,
    client: GraphCompletionClient | None,
    target_duration_seconds: float,
    max_source_chars: int,
    repair_attempts: int,
    prepare_only: bool,
    overwrite: bool,
    always_model_inventory: bool,
) -> dict[str, Any]:
    output_dir = _record_output_dir(output_root, record)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = build_source_packet(record, max_source_chars=max_source_chars)
    locked_fields = build_locked_graph_fields(record, packet, target_duration_seconds)
    source_hash = _source_hash(packet)
    _write_json(output_dir / "source_packet.json", packet)
    _write_json(output_dir / "locked_graph_fields.json", locked_fields)

    result_base = {
        "index": int(record["index"]),
        "poem_id": str(record["poem_id"]),
        "title": str(record["title"]),
        "author": str(record["author"]),
        "output_dir": str(output_dir),
        "source_sha256": source_hash,
    }
    if prepare_only:
        return {**result_base, "status": "prepared"}

    graph_path = output_dir / "director_graph.json"
    if graph_path.exists() and not overwrite:
        existing = json.loads(graph_path.read_text(encoding="utf-8"))
        report = validate_generated_graph(existing, schema, rule_validator_path)
        if report["ok"] and existing.get("metadata", {}).get("source_sha256") == source_hash:
            _write_json(output_dir / "validation.json", report)
            return {**result_base, "status": "skipped_valid", "graph": str(graph_path)}

    if client is None:
        raise RuntimeError("An LLM client is required unless --prepare-only is used")

    model_knowledge_inventory: dict[str, Any] | None = None
    model_sources: list[dict[str, Any]] = []
    model_knowledge_usage: list[dict[str, Any]] = []
    missing_materials = set(packet.get("source_completeness", {}).get("missing_materials", []))
    if always_model_inventory or "creation_background" in missing_materials:
        knowledge_messages = build_model_knowledge_messages(packet)
        knowledge_content = ""
        knowledge_error: Exception | None = None
        for attempt in range(1, repair_attempts + 2):
            completion = client.complete(knowledge_messages)
            model_knowledge_usage.append(completion.usage)
            knowledge_content = completion.content
            (output_dir / f"model_knowledge_response.attempt_{attempt}.txt").write_text(
                knowledge_content, encoding="utf-8"
            )
            try:
                model_knowledge_inventory = parse_model_knowledge(knowledge_content)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                knowledge_error = exc
                if attempt <= repair_attempts:
                    knowledge_messages = build_model_knowledge_messages(packet) + [
                        {"role": "assistant", "content": knowledge_content},
                        {
                            "role": "user",
                            "content": f"知识清单无法使用：{exc}。请按原格式返回修复后的完整JSON。",
                        },
                    ]
        if model_knowledge_inventory is None:
            raise RuntimeError(f"Model knowledge inventory failed validation: {knowledge_error}")
        _write_json(output_dir / "model_knowledge_inventory.json", model_knowledge_inventory)
        model_sources = [model_knowledge_source(model_knowledge_inventory)]
        _write_json(output_dir / "model_knowledge_sources.json", model_sources)

    messages = build_generation_messages(
        schema,
        packet,
        locked_fields,
        model_knowledge_inventory=model_knowledge_inventory,
        model_sources=model_sources,
    )
    graph: dict[str, Any] | None = None
    validation = {"ok": False, "errors": ["generation did not run"], "warnings": []}
    usage_by_attempt: list[dict[str, Any]] = []
    for attempt in range(1, repair_attempts + 2):
        completion = client.complete(messages)
        usage_by_attempt.append(completion.usage)
        (output_dir / f"llm_response.attempt_{attempt}.txt").write_text(
            completion.content, encoding="utf-8"
        )
        try:
            parsed = extract_json_object(completion.content)
            graph = normalize_graph(
                parsed,
                locked_fields,
                record=record,
                model=client.model,
                source_hash=source_hash,
                model_sources=model_sources,
            )
            _write_json(output_dir / f"director_graph.attempt_{attempt}.json", graph)
            validation = validate_generated_graph(graph, schema, rule_validator_path)
        except (ValueError, json.JSONDecodeError) as exc:
            graph = None
            validation = {
                "ok": False,
                "errors": [f"response_parse_error: {exc}"],
                "warnings": [],
                "schema_error_count": 0,
                "rule_error_count": 0,
                "grounding_error_count": 0,
            }
        validation["attempt"] = attempt
        validation["usage_by_attempt"] = usage_by_attempt
        validation["model_knowledge_usage"] = model_knowledge_usage
        _write_json(output_dir / "validation.json", validation)
        if validation["ok"] and graph is not None:
            _write_json(graph_path, graph)
            return {
                **result_base,
                "status": "generated",
                "graph": str(graph_path),
                "attempts": attempt,
                "warning_count": len(validation["warnings"]),
            }
        if attempt <= repair_attempts:
            messages = _repair_messages(
                schema,
                packet,
                locked_fields,
                model_knowledge_inventory,
                model_sources,
                graph,
                validation["errors"],
            )

    if graph is not None:
        _write_json(output_dir / "director_graph.failed.json", graph)
    return {
        **result_base,
        "status": "failed",
        "attempts": repair_attempts + 1,
        "errors": validation["errors"][:20],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build complete director graphs from supplied poetry records."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--rule-validator", type=Path, default=DEFAULT_RULE_VALIDATOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--all", action="store_true", help="Process every corpus record.")
    selector.add_argument("--index", type=int, action="append", help="Corpus index; repeat for multiple poems.")
    selector.add_argument("--poem-id", action="append", help="Exact poem_id; repeat for multiple poems.")
    selector.add_argument("--title", action="append", help="Exact title; repeat for multiple poems.")
    parser.add_argument("--limit", type=int, default=None, help="Limit records after selection.")

    parser.add_argument("--prepare-only", action="store_true", help="Write source packets without calling the LLM.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate already valid outputs.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent LLM requests; start with 1.")
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--target-duration", type=float, default=120.0)
    parser.add_argument("--max-source-chars", type=int, default=24000)
    parser.add_argument(
        "--always-model-inventory",
        action="store_true",
        help="Run the model-knowledge inventory stage even when source background is present.",
    )

    parser.add_argument(
        "--base-url",
        default=(
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("TANGSHI_LLM_BASE_URL")  # legacy alias
            or DEFAULT_BASE_URL
        ),
    )
    parser.add_argument(
        "--model",
        default=(
            os.environ.get("LLM_MODEL")
            or os.environ.get("TANGSHI_LLM_MODEL")  # legacy alias
            or DEFAULT_MODEL
        ),
    )
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser


def _resolve_api_key() -> str:
    key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("TANGSHI_LLM_API_KEY")  # legacy alias
    )
    if not key:
        raise SystemExit(
            "Set LLM_API_KEY or OPENAI_API_KEY. "
            "The key is intentionally not accepted as a CLI argument or stored in output files."
        )
    return key


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.repair_attempts < 0:
        raise SystemExit("--repair-attempts cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    records, corpus_metadata = load_corpus(args.corpus)
    selected = select_records(
        records,
        select_all=args.all,
        indices=args.index,
        poem_ids=args.poem_id,
        titles=args.title,
        limit=args.limit,
    )
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    _load_rule_validator(args.rule_validator)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client: GraphCompletionClient | None = None
    if not args.prepare_only:
        client = OpenAIGraphCompletionClient(
            base_url=args.base_url,
            api_key=_resolve_api_key(),
            model=args.model,
            timeout_seconds=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    common = {
        "schema": schema,
        "rule_validator_path": args.rule_validator,
        "output_root": args.output_dir,
        "client": client,
        "target_duration_seconds": args.target_duration,
        "max_source_chars": args.max_source_chars,
        "repair_attempts": args.repair_attempts,
        "prepare_only": args.prepare_only,
        "overwrite": args.overwrite,
        "always_model_inventory": args.always_model_inventory,
    }
    results: list[dict[str, Any]] = []

    def record_kwargs(record: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(common)
        per_record_duration = record.get("target_duration_seconds")
        if per_record_duration is not None:
            duration = float(per_record_duration)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    f"Invalid target_duration_seconds for corpus index {record['index']}: "
                    f"{per_record_duration!r}"
                )
            kwargs["target_duration_seconds"] = duration
        return kwargs

    def persist_manifest() -> None:
        counts: dict[str, int] = {}
        for item in results:
            status = str(item["status"])
            counts[status] = counts.get(status, 0) + 1
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus": str(args.corpus),
            "corpus_metadata": corpus_metadata,
            "schema": str(args.schema),
            "model": None if args.prepare_only else args.model,
            "base_url": None if args.prepare_only else args.base_url,
            "grounding_policy": "provided_sources_plus_explicit_unverified_model_inventory",
            "always_model_inventory": args.always_model_inventory,
            "selected_count": len(selected),
            "completed_count": len(results),
            "status_counts": counts,
            "items": sorted(results, key=lambda item: int(item["index"])),
        }
        _write_json(args.output_dir / "manifest.json", manifest)

    if args.workers == 1:
        for record in selected:
            try:
                result = build_one_record(record, **record_kwargs(record))
            except Exception as exc:
                result = {
                    "index": int(record["index"]),
                    "poem_id": str(record["poem_id"]),
                    "title": str(record["title"]),
                    "author": str(record["author"]),
                    "status": "failed",
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            results.append(result)
            persist_manifest()
            print(f"[{result['status']}] {result['index']:03d} {result['author']}《{result['title']}》")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_record = {
                executor.submit(build_one_record, record, **record_kwargs(record)): record
                for record in selected
            }
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "index": int(record["index"]),
                        "poem_id": str(record["poem_id"]),
                        "title": str(record["title"]),
                        "author": str(record["author"]),
                        "status": "failed",
                        "errors": [f"{type(exc).__name__}: {exc}"],
                    }
                results.append(result)
                persist_manifest()
                print(f"[{result['status']}] {result['index']:03d} {result['author']}《{result['title']}》")

    persist_manifest()
    return 1 if any(item["status"] == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
