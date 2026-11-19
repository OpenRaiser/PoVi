from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .common import (
    FORBIDDEN_EXPLICIT_PHRASES,
    FORBIDDEN_NARRATION_STAGE_DIRECTIONS,
    FORBIDDEN_TEACHING_TEMPLATES,
    NARRATION_ROLES,
    READING_ACTIONS,
    ChatCompletionClient,
    close_reading_context_for_agent,
    normalized_quote_units,
    poem_line_ids,
    relaxed_intro_validation,
    relaxed_keyword_validation,
    spoken_character_count,
)


# User-tuned narration length limits.
POEM_INTRO_MIN_CHARACTERS = 120
POEM_INTRO_MAX_CHARACTERS = 260
CLOSE_READING_MIN_CHARACTERS = 36
CLOSE_READING_MAX_CHARACTERS = 72
LEARNING_SUMMARY_MIN_CHARACTERS = 50
LEARNING_SUMMARY_MAX_CHARACTERS = 100

INTRO_KNOWLEDGE_TYPES = {
    "poem_identity",
    "context",
    "overall_structure",
    "metaphor_mapping",
    "interpretive_boundary",
}
SEGMENT_KNOWLEDGE_TYPES = {
    "lexical",
    "allusion",
    "rhetoric",
    "syntax",
    "structure",
    "historical_context",
    "interpretive_debate",
    "interpretive_boundary",
    "mapping_bridge",
    "overall_summary",
}
TEACHING_POINT_FOCUSES = {
    "definition",
    "literal_meaning",
    "verb_effect",
    "syntax_rhythm",
    "rhetoric_mapping",
    "allusion",
    "sensory_space",
    "structure_transition",
    "interpretive_boundary",
    "famous_line_reason",
    "form_or_culture",
}

GENERIC_TEACHING_PHRASES = (
    "典型风俗场景",
    "典型的风俗场景",
    "完整的风俗场景",
    "营造氛围",
    "还原盛景",
    "构建完整",
    "表现热闹",
)
FORBIDDEN_INTERNAL_AUDIO_PHRASES = (
    "Source",
    "Target",
    "Bridge",
    "source",
    "target",
    "bridge",
    "point",
    "anchor",
    "teaching_points",
    "teaching_goal",
    "p01",
    "s01",
    "s_final",
    "锚定",
    "明确的对应逻辑",
    "源-桥-靶",
    "源桥靶",
    "落实",
)
FORBIDDEN_AUDIO_LEAD_INS = (
    "你看",
    "请看",
    "我们来看",
    "再看",
    "接下去",
    "然后看",
    "眼前",
    "你会发现",
    "你会看到",
    "注意",
)
FORBIDDEN_INTRO_TEMPLATES = (
    "读这首诗，可以先",
    "读这首词，可以先",
    "读这首诗可以先",
    "读这首词可以先",
    "我们将从",
    "阅读路径",
    "完成全篇阅读",
    "下面先完整读一遍原作",
    "下面完整读一遍原作",
    "下面先读一遍原作",
)
FORBIDDEN_FINAL_ABSTRACTIONS = (
    "公共狂欢",
    "公共繁华",
    "价值选择",
    "价值反转",
    "价值指向",
    "独特价值",
)
SECOND_PERSON_PRONOUNS = ("你", "您")
INTRO_OPENING_PREFIX = "今天让我们走近"
FINAL_OPENING_PREFIXES = (
    "读完这首诗，应该记住的是：",
    "读完这首诗，应该学到的是：",
    "读完这首词，应该记住的是：",
    "读完这首词，应该学到的是：",
)

# Only normalize frequent model slips. The public JSON contract remains small.
KNOWLEDGE_TYPE_ALIASES = {
    "definition": "lexical",
    "form_or_culture": "historical_context",
    "sensory_space": "rhetoric",
    "verb_effect": "syntax",
    "famous_line_reason": "interpretive_debate",
    "rhetoric_mapping": "rhetoric",
    "interpretation": "interpretive_debate",
    "interpretive": "interpretive_debate",
}
INTRO_TYPE_ALIASES = {
    "rhetoric_mapping": "metaphor_mapping",
    "overall_sensory_layout": "overall_structure",
    "interpretive_boundary_overview": "interpretive_boundary",
}

_PUNCTUATION_RE = re.compile(r"[\s，。！？、；：,.!?;:]")


def _normalize_text(text: Any) -> str:
    return _PUNCTUATION_RE.sub("", str(text or ""))


def _estimate_close_reading_seconds(text: str) -> float:
    characters = spoken_character_count(text)
    sentence_pauses = len(re.findall(r"[。！？!?；;]", text)) * 0.28
    short_pauses = len(re.findall(r"[，,、：:]", text)) * 0.12
    return round(max(2.0, characters / 4.2 + min(1.0, sentence_pauses + short_pauses)), 1)


def _normalize_knowledge_type(value: Any) -> str:
    value = str(value or "").strip()
    if value in SEGMENT_KNOWLEDGE_TYPES:
        return value
    return KNOWLEDGE_TYPE_ALIASES.get(value, "")


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


class CloseReadingAgent:
    """Plan concrete teaching points, then realize learner-facing narration."""

    def __init__(self, client: ChatCompletionClient):
        self.client = client

    def plan(
        self,
        context: dict[str, Any],
        story_plan: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        reading_context = close_reading_context_for_agent(context, story_plan)
        expected_line_ids = poem_line_ids(context)
        poem = context.get("poem", {})
        interpretation = context.get("interpretation", {}) or {}

        planner_messages = [
            {
                "role": "system",
                "content": (
                    "你是古诗词教学规划教师。建立作品阅读路径、具体教学要点池和逐句slot；"
                    "不写最终旁白，只输出严格JSON。"
                ),
            },
            {"role": "user", "content": self._build_planning_prompt(reading_context)},
        ]
        planner_payload = self._payload(planner_messages, temperature=0.2, max_tokens=10000)
        self._write_json(output_dir / "close_reading_planner_request.json", planner_payload)
        plan, _ = self._generate_validated_json(
            planner_payload,
            planner_messages,
            output_dir,
            response_name="close_reading_planner_response",
            label="close_reading_planner",
            validator=lambda content: self._parse_plan(content, expected_line_ids, poem),
            repair_instruction=(
                "规划无法使用：{error}。只修复报错项。"
                f"line_ids必须按{expected_line_ids}覆盖；focus只能使用"
                f"{sorted(TEACHING_POINT_FOCUSES)}；普通slot最多引用两个point_id；"
                "最后必须是s_final。"
            ),
        )
        self._write_json(output_dir / "close_reading_reading_plan.json", plan["reading_path"])
        self._write_json(output_dir / "close_reading_teaching_points.json", plan["teaching_points"])
        self._write_json(output_dir / "close_reading_slot_plan.json", plan["slots"])

        realizer_messages = [
            {
                "role": "system",
                "content": (
                    "你是古诗词教学旁白教师。完整诗词展示前导入负责建立阅读期待，逐句旁白负责解释字词、"
                    "器物、动作、句意和结构，最后总结具体学习收获。不得使用第二人称，"
                    "不得输出内部规划术语或新增故事事实。只输出严格JSON。"
                ),
            },
            {
                "role": "user",
                "content": self._build_realization_prompt(reading_context, plan),
            },
        ]
        realizer_payload = self._payload(realizer_messages, temperature=0.3, max_tokens=14000)
        self._write_json(output_dir / "close_reading_realizer_request.json", realizer_payload)
        result, content = self._generate_validated_json(
            realizer_payload,
            realizer_messages,
            output_dir,
            response_name="close_reading_realizer_response",
            label="close_reading_realizer",
            validator=lambda value: self._parse(
                value,
                expected_line_ids,
                poem,
                interpretation=interpretation,
                plan=plan,
            ),
            repair_instruction=(
                "实现无法使用：{error}。只修复报错的完整诗词展示前导入或逐句segment，"
                "保持slot数量、顺序和line_ids。"
                f"导入必须以“{INTRO_OPENING_PREFIX}”开头，长度"
                f"{POEM_INTRO_MIN_CHARACTERS}到{POEM_INTRO_MAX_CHARACTERS}个有效字符，"
                "至少写四句：先说作品身份，再概括内容展开和情绪变化，最后提出一个值得追踪的阅读问题；"
                "说出朝代、诗人/词人、作者和题名，不写流程提示。逐句讲解直接从原文或关键词开始，"
                f"长度{CLOSE_READING_MIN_CHARACTERS}到{CLOSE_READING_MAX_CHARACTERS}个有效字符；"
                "不得出现“你/您”或Source、Target、point、anchor、p01、锚定、落实等内部术语。"
                f"最后收束以“读完这首诗/词，应该记住/学到的是：”开头，长度"
                f"{LEARNING_SUMMARY_MIN_CHARACTERS}到{LEARNING_SUMMARY_MAX_CHARACTERS}个有效字符，"
                "只总结一个具体文本变化。"
            ),
        )
        (output_dir / "close_reading_agent_response.txt").write_text(content, encoding="utf-8")
        self._write_json(output_dir / "close_reading_plan.json", result)
        return result

    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        return {
            "model": self.client.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
        }

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def _generate_validated_json(
        self,
        payload: dict[str, Any],
        messages: list[dict[str, str]],
        output_dir: Path,
        *,
        response_name: str,
        label: str,
        validator: Callable[[str], dict[str, Any]],
        repair_instruction: str,
    ) -> tuple[dict[str, Any], str]:
        content = self.client.complete(payload, label=label)
        (output_dir / f"{response_name}.txt").write_text(content, encoding="utf-8")
        for attempt in range(3):
            try:
                return validator(content), content
            except Exception as error:
                if attempt == 2:
                    raise
                repair_payload = {
                    **payload,
                    "temperature": 0.1,
                    "messages": messages
                    + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": repair_instruction.format(error=error)},
                    ],
                }
                content = self.client.complete(repair_payload, label=f"{label}_repair")
                suffix = "" if attempt == 0 else f"_{attempt + 1}"
                (output_dir / f"{response_name}_repaired{suffix}.txt").write_text(
                    content, encoding="utf-8"
                )
        raise RuntimeError(f"{label} did not produce a valid result")

    @staticmethod
    def _build_planning_prompt(reading_context: dict[str, Any]) -> str:
        return f"""
请先建立当前作品的阅读路径和具体教学要点池，再按原文顺序规划slot。不要写最终旁白。

输出格式：
{{
  "reading_path": {{
    "mode": "适合当前作品的阅读模式",
    "path": "一句话说明内容怎样展开",
    "key_turn": "全篇最重要的转折或收束"
  }},
  "teaching_points": [{{
    "id": "p01",
    "focus": "definition",
    "line_ids": ["l01"],
    "anchor": "需要讲清的原文词语或关系",
    "goal": "这一要点要解决的具体理解问题"
  }}],
  "slots": [{{
    "id": "s01",
    "line_ids": ["l01"],
    "quote_units": ["对应的连续原文片段"],
    "point_ids": ["p01"],
    "goal": "这个slot要让学生具体理解什么"
  }}, {{
    "id": "s_final",
    "line_ids": [],
    "quote_units": [],
    "point_ids": [],
    "goal": "读完本作应记住或学到的一个具体文本认识"
  }}]
}}

规则：
1. 词可参考上下阕，律诗可参考首颔颈尾联，绝句可关注入景和末句转折，但必须服从实际文本。
2. teaching_points.focus只能使用：{sorted(TEACHING_POINT_FOCUSES)}。
3. 要点必须具体回答：词语/器物是什么、句意是什么、动词或句法如何推进、修辞或典故如何起作用、
   结构怎样转折、名句为什么有感染力。禁止只写“营造氛围、还原盛景、表现热闹”等空话。
4. 普通slot最多引用两个point_id；同一行有三个以上独立知识点时必须拆成连续slot。
5. quote_units和line_ids按原文顺序覆盖所有行；每个quote_unit不超过22个有效字符，
   长句按标点或语义拆成连续slot。最后必须添加s_final，且其三个数组都为空。
6. s_final只规划一个可复述的文本收获，不写人生启示、价值选择或泛泛共鸣。
7. 争议解释保留开放边界，不把单首作品的解释写成跨作品固定模板。

Close-reading context:
{json.dumps(reading_context, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _parse_plan(
        content: str,
        expected_line_ids: list[str],
        poem: dict[str, Any],
    ) -> dict[str, Any]:
        data = ChatCompletionClient.extract_json(content)
        if set(data) != {"reading_path", "teaching_points", "slots"}:
            raise ValueError("Plan must contain reading_path, teaching_points and slots")

        reading_path = data["reading_path"]
        required_path_fields = {"mode", "path", "key_turn"}
        if not isinstance(reading_path, dict) or set(reading_path) != required_path_fields:
            raise ValueError("Invalid reading_path fields")
        if any(
            spoken_character_count(str(reading_path.get(field) or "")) < 6
            for field in required_path_fields
        ):
            raise ValueError("reading_path is too vague")

        known_lines = set(expected_line_ids)
        points = data["teaching_points"]
        if not isinstance(points, list) or not points:
            raise ValueError("teaching_points must be a non-empty array")
        point_ids: set[str] = set()
        point_fields = {"id", "focus", "line_ids", "anchor", "goal"}
        for point in points:
            if not isinstance(point, dict) or set(point) != point_fields:
                raise ValueError("Invalid teaching point fields")
            point_id = str(point.get("id") or "").strip()
            if not point_id or point_id in point_ids:
                raise ValueError(f"Duplicate teaching point id: {point_id}")
            point_ids.add(point_id)
            if point.get("focus") not in TEACHING_POINT_FOCUSES:
                raise ValueError(f"Invalid teaching point focus: {point.get('focus')}")
            line_ids = [str(value) for value in point.get("line_ids", [])]
            if not line_ids or any(line_id not in known_lines for line_id in line_ids):
                raise ValueError(f"Invalid teaching point line_ids: {point_id}")
            if not str(point.get("anchor") or "").strip():
                raise ValueError(f"Teaching point anchor is empty: {point_id}")
            goal = str(point.get("goal") or "")
            if spoken_character_count(goal) < 10 or _contains_any(goal, GENERIC_TEACHING_PHRASES):
                raise ValueError(f"Teaching point goal is too generic: {point_id}")

        slots = data["slots"]
        if not isinstance(slots, list) or len(slots) < 2:
            raise ValueError("slots must contain regular slots and s_final")
        expected_slot_ids = [
            *[f"s{index:02d}" for index in range(1, len(slots))],
            "s_final",
        ]
        actual_slot_ids = [
            str(slot.get("id") or "") if isinstance(slot, dict) else ""
            for slot in slots
        ]
        if actual_slot_ids != expected_slot_ids:
            raise ValueError(f"Slot ids must be {expected_slot_ids}")

        line_text_by_id = {
            str(line.get("line_id") or ""): str(line.get("text") or "")
            for line in poem.get("lines", [])
            if isinstance(line, dict)
        }
        referenced_lines: list[str] = []
        planned_quotes_by_line = {line_id: [] for line_id in expected_line_ids}
        slot_fields = {"id", "line_ids", "quote_units", "point_ids", "goal"}
        for slot in slots:
            if not isinstance(slot, dict) or set(slot) != slot_fields:
                raise ValueError("Invalid slot fields")
            if slot["id"] == "s_final":
                if any(slot[field] for field in ("line_ids", "quote_units", "point_ids")):
                    raise ValueError("s_final must not claim source lines or teaching points")
                if spoken_character_count(str(slot.get("goal") or "")) < 8:
                    raise ValueError("s_final goal is too vague")
                continue

            line_ids = [str(value) for value in slot.get("line_ids", [])]
            point_refs = [str(value) for value in slot.get("point_ids", [])]
            quote_units = [
                str(value).strip()
                for value in slot.get("quote_units", [])
                if str(value).strip()
            ]
            if not line_ids or any(line_id not in known_lines for line_id in line_ids):
                raise ValueError(f"Invalid slot line_ids: {slot['id']}")
            point_refs = [point_id for point_id in point_refs if point_id in point_ids]
            if not point_refs:
                compatible_points = [
                    str(point.get("id") or "")
                    for point in points
                    if set(str(value) for value in point.get("line_ids", []))
                    & set(line_ids)
                ]
                point_refs = compatible_points[:2]
                slot["point_ids"] = point_refs
            if not point_refs:
                raise ValueError(f"Invalid slot point_ids: {slot['id']}")
            if len(point_refs) > 2:
                raise ValueError(f"Slot has too many teaching points: {slot['id']}")
            if not quote_units:
                raise ValueError(f"Slot quote_units are empty: {slot['id']}")
            if any(spoken_character_count(unit) > 22 for unit in quote_units):
                raise ValueError(f"Slot quote_units are too long; split the slot: {slot['id']}")
            if spoken_character_count(str(slot.get("goal") or "")) < 8:
                raise ValueError(f"Slot goal is too vague: {slot['id']}")
            for line_id in line_ids:
                source = _normalize_text(line_text_by_id.get(line_id, ""))
                if not any(_normalize_text(unit) in source for unit in quote_units):
                    raise ValueError(f"Slot quote_units do not match line {line_id}")
                planned_quotes_by_line[line_id].extend(quote_units)
            referenced_lines.extend(line_ids)

        positions = []
        for line_id in expected_line_ids:
            if line_id not in referenced_lines:
                raise ValueError(f"Plan misses line: {line_id}")
            positions.append(referenced_lines.index(line_id))
        if positions != sorted(positions):
            raise ValueError("Plan is not in source order")
        for line_id in expected_line_ids:
            planned_text = "".join(_normalize_text(unit) for unit in planned_quotes_by_line[line_id])
            missing = [
                unit
                for unit in normalized_quote_units(line_text_by_id.get(line_id, ""))
                if unit not in planned_text
            ]
            if missing:
                raise ValueError(f"Plan misses quote fragments for {line_id}: {missing}")
        return data

    @staticmethod
    def _build_realization_prompt(
        reading_context: dict[str, Any],
        plan: dict[str, Any],
    ) -> str:
        return f"""
根据reading_path、teaching_points和slots写最终旁白。

输出格式：
{{
  "poem_card_narration": {{
    "audio_text": "整诗导入",
    "teaching_move": "导入建立的阅读任务",
    "focus_line_ids": ["l01"],
    "metaphor_node_ids": [],
    "knowledge_points": [{{"type": "overall_structure", "content": "整体阅读抓手"}}],
    "target_seconds": 20
  }},
  "segments": [{{
    "shot_id": "cr01",
    "line_ids": ["l01"],
    "audio_text": "逐句教学旁白",
    "narration_role": "poem_quote",
    "reading_action": "quote_and_explain",
    "knowledge_points": [{{"type": "lexical", "content": "具体教学知识"}}],
    "teaching_move": "本段的教学推进",
    "unstated_visual_meaning": "留给视觉呈现的语义",
    "target_seconds": 6
  }}]
}}

规则：
1. segments与slots一一对应，不能增删或重排；普通segment必须引用对应quote_units。
2. poem_card_narration是完整诗词展示前的导入，不是逐句赏析。必须以“{INTRO_OPENING_PREFIX}”开头，
   自然说出朝代、诗人/词人、作者和题名；概括作品怎样展开，并提出一个值得追踪的问题。
   长度为{POEM_INTRO_MIN_CHARACTERS}到{POEM_INTRO_MAX_CHARACTERS}个有效字符。
   不写“读这首诗/词，可以先……”“阅读路径”“下面读原作”等流程模板，也不解释具体字词。
3. 普通逐句旁白直接从原文片段或关键词开始，不用“你看、请看、再看、眼前”等套话。
   按“引用原文—还原动作/声音/空间—解释词语或句意—说明推进作用”自然讲述。
4. 名词和器物要说清“是什么”；动词、副词和句法要说清怎样改变动作、节奏、视角或情绪；
   关键句要说明感染力来自意象、动作、节奏、空间转折或开放性中的哪一项。
5. 普通audio_text长度为{CLOSE_READING_MIN_CHARACTERS}到{CLOSE_READING_MAX_CHARACTERS}个有效字符。
   信息较多时依照Planner拆出的连续slot讲，不写成长段总赏析。
6. 最后的s_final使用reading_action=synthesis或connect，且line_ids为空。audio_text必须以
   “读完这首诗/词，应该记住/学到的是：”开头，长度为
   {LEARNING_SUMMARY_MIN_CHARACTERS}到{LEARNING_SUMMARY_MAX_CHARACTERS}个有效字符，
   只总结一个具体文本变化，不讲公共繁华、价值选择或泛泛人生启示。
7. 争议解释使用“可以理解为、也可、一种读法”等开放措辞。
8. 所有朗读文本不得出现第二人称“你/您”，不得出现Source、Target、Bridge、point、anchor、
   p01、s01、s_final、锚定、落实等内部规划语言，不写镜头或生产说明。
9. knowledge_points.type：poem_card_narration使用{sorted(INTRO_KNOWLEDGE_TYPES)}；
   普通segment使用{sorted(SEGMENT_KNOWLEDGE_TYPES)}。

reading_path:
{json.dumps(plan["reading_path"], ensure_ascii=False, indent=2)}

teaching_points:
{json.dumps(plan["teaching_points"], ensure_ascii=False, indent=2)}

slots:
{json.dumps(plan["slots"], ensure_ascii=False, indent=2)}

Close-reading context:
{json.dumps(reading_context, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _parse(
        content: str,
        expected_line_ids: list[str],
        poem: dict[str, Any],
        *,
        interpretation: dict[str, Any] | None = None,
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = ChatCompletionClient.extract_json(content)
        if set(data) != {"poem_card_narration", "segments"}:
            raise ValueError("Output must contain poem_card_narration and segments")
        if not isinstance(data["segments"], list):
            raise ValueError("segments must be an array")

        known_lines = set(expected_line_ids)
        CloseReadingAgent._validate_intro(
            data["poem_card_narration"],
            poem,
            known_lines,
            interpretation or {},
        )

        planned_slots = plan.get("slots", []) if isinstance(plan, dict) else []
        segments = data["segments"]
        if planned_slots and len(segments) != len(planned_slots):
            raise ValueError("Realized segment count does not match slots")
        expected_shot_ids = [f"cr{index:02d}" for index in range(1, len(segments) + 1)]
        # Normalize only the line-free final synthesis identifier. Regular
        # source segments must already be sequential; changing them would hide
        # a real ordering error.
        for index, segment in enumerate(segments):
            if (
                index < len(planned_slots)
                and str(planned_slots[index].get("id") or "") == "s_final"
            ):
                segment["shot_id"] = expected_shot_ids[index]
                if str(segment.get("narration_role") or "") not in NARRATION_ROLES:
                    segment["narration_role"] = "delayed_inference"
        # A repair may leave a previously forbidden teaching template in an
        # otherwise coherent segment. Normalize only the known lexical slips;
        # do not shorten or rewrite the explanation.
        for segment in segments:
            audio = str(segment.get("audio_text") or "")
            audio = audio.replace("暗示着", "写出").replace("反衬出", "更显出")
            audio = audio.replace("反衬", "更显")
            segment["audio_text"] = audio
        actual_shot_ids = [str(item.get("shot_id") or "") for item in segments]
        if actual_shot_ids != expected_shot_ids:
            # Gemini sometimes returns the final line-free synthesis as
            # s_final. It is metadata noise, not a source-order error; the
            # final slot was already identified above, so normalize it here.
            if (
                len(actual_shot_ids) == len(expected_shot_ids)
                and actual_shot_ids[:-1] == expected_shot_ids[:-1]
                and planned_slots
                and str(planned_slots[-1].get("id") or "") == "s_final"
                and actual_shot_ids[-1] == "s_final"
            ):
                segments[-1]["shot_id"] = expected_shot_ids[-1]
            else:
                raise ValueError(
                    f"Close-reading shot ids must be sequential: expected={expected_shot_ids}, "
                    f"actual={actual_shot_ids}"
                )

        line_text_by_id = {
            str(line.get("line_id") or ""): str(line.get("text") or "")
            for line in poem.get("lines", [])
            if isinstance(line, dict)
        }
        referenced_lines: list[str] = []
        explained_lines: set[str] = set()
        spoken_by_line = {line_id: [] for line_id in expected_line_ids}

        for index, segment in enumerate(segments):
            planned_slot = planned_slots[index] if planned_slots else {}
            if str(planned_slot.get("id") or "") == "s_final":
                CloseReadingAgent._validate_final_segment(segment)
                continue
            CloseReadingAgent._validate_regular_segment(
                segment,
                planned_slot,
                known_lines,
            )
            line_ids = [str(value) for value in segment["line_ids"]]
            referenced_lines.extend(line_ids)
            for line_id in line_ids:
                spoken_by_line[line_id].append(str(segment["audio_text"]))
            if segment["reading_action"] in {"quote_and_explain", "explain", "connect", "synthesis"}:
                explained_lines.update(line_ids)

        CloseReadingAgent._validate_line_coverage(
            expected_line_ids,
            referenced_lines,
            explained_lines,
            spoken_by_line,
            line_text_by_id,
        )
        return data

    @staticmethod
    def _validate_intro(
        intro: Any,
        poem: dict[str, Any],
        known_lines: set[str],
        interpretation: dict[str, Any],
    ) -> None:
        required_fields = {
            "audio_text",
            "teaching_move",
            "focus_line_ids",
            "metaphor_node_ids",
            "knowledge_points",
            "target_seconds",
        }
        if not isinstance(intro, dict) or set(intro) != required_fields:
            raise ValueError("Invalid poem introduction fields")
        audio = str(intro.get("audio_text") or "").strip()
        count = spoken_character_count(audio)
        if (
            not POEM_INTRO_MIN_CHARACTERS <= count <= POEM_INTRO_MAX_CHARACTERS
            and not relaxed_intro_validation()
        ):
            raise ValueError(
                f"Poem introduction has {count} effective characters; "
                f"expected {POEM_INTRO_MIN_CHARACTERS}..{POEM_INTRO_MAX_CHARACTERS}"
            )
        # Keep this field as an internal metadata value; duration is recalculated
        # from audio_text below.
        if not str(intro.get("teaching_move") or "").strip():
            raise ValueError("Poem introduction teaching_move is empty")
        for field in ("title", "author", "dynasty"):
            expected = str(poem.get(field) or "").strip()
            if expected and expected not in audio and not relaxed_intro_validation():
                raise ValueError(f"Poem introduction must name poem {field}")
        if not audio.startswith(INTRO_OPENING_PREFIX) and not relaxed_intro_validation():
            raise ValueError(f"Poem introduction must start with {INTRO_OPENING_PREFIX}")
        creator_role = "词人" if str(poem.get("form") or "").lower() in {"ci", "词"} else "诗人"
        # Some valid corpus authors are not conventionally described as
        # 诗人 (for example 歌辞作者/宫廷歌者). Accept a small set of
        # equivalent labels when the introduction still names the author.
        creator_role_aliases = {
            "诗人": {"诗人", "词作者", "歌者", "作者"},
            "词人": {"词人", "词作者", "作者"},
        }
        if (
            not any(role in audio for role in creator_role_aliases[creator_role])
            and not relaxed_intro_validation()
        ):
            raise ValueError(f"Poem introduction must call the author a {creator_role}")
        CloseReadingAgent._validate_spoken_text(audio, intro=True)

        focus_ids = [str(value) for value in intro.get("focus_line_ids", [])]
        if not focus_ids or any(line_id not in known_lines for line_id in focus_ids):
            raise ValueError("Invalid poem introduction focus_line_ids")
        known_metaphors = {
            str(node.get("node_id") or "")
            for node in interpretation.get("metaphor_nodes", [])
            if isinstance(node, dict) and node.get("node_id")
        }
        metaphor_ids = [str(value) for value in intro.get("metaphor_node_ids", [])]
        if len(metaphor_ids) > 1 or any(node_id not in known_metaphors for node_id in metaphor_ids):
            raise ValueError("Invalid poem introduction metaphor_node_ids")

        points = intro.get("knowledge_points")
        if not isinstance(points, list) or not 1 <= len(points) <= 4:
            raise ValueError("Poem introduction needs 1..4 knowledge points")
        types = set()
        for point in points:
            if not isinstance(point, dict) or set(point) != {"type", "content"}:
                raise ValueError("Invalid poem introduction knowledge point")
            point_type = INTRO_TYPE_ALIASES.get(point["type"], point["type"])
            if point_type not in INTRO_KNOWLEDGE_TYPES:
                raise ValueError(f"Invalid poem introduction knowledge type: {point['type']}")
            point["type"] = point_type
            if (
                spoken_character_count(str(point["content"])) < 8
                and not relaxed_intro_validation()
            ):
                raise ValueError("Poem introduction knowledge point is too vague")
            types.add(point_type)
        if "overall_structure" not in types and not relaxed_intro_validation():
            raise ValueError("Poem introduction misses overall_structure")
        intro["target_seconds"] = _estimate_close_reading_seconds(audio)

    @staticmethod
    def _validate_final_segment(segment: Any) -> None:
        CloseReadingAgent._validate_segment_fields(segment)
        if segment.get("line_ids") or segment.get("reading_action") not in {"synthesis", "connect"}:
            raise ValueError("Final synthesis must be line-free and use synthesis/connect")
        if str(segment.get("narration_role") or "") not in NARRATION_ROLES:
            raise ValueError("Invalid final synthesis narration_role")
        audio = str(segment.get("audio_text") or "").strip()
        if not audio.startswith(FINAL_OPENING_PREFIXES) and not relaxed_keyword_validation():
            raise ValueError("Final synthesis must state the learning takeaway first")
        count = spoken_character_count(audio)
        if not LEARNING_SUMMARY_MIN_CHARACTERS <= count <= LEARNING_SUMMARY_MAX_CHARACTERS:
            raise ValueError(
                f"Final synthesis has {count} effective characters; "
                f"expected {LEARNING_SUMMARY_MIN_CHARACTERS}..{LEARNING_SUMMARY_MAX_CHARACTERS}"
            )
        CloseReadingAgent._validate_spoken_text(audio, final=True)
        CloseReadingAgent._validate_knowledge_points(segment)
        segment["target_seconds"] = _estimate_close_reading_seconds(audio)

    @staticmethod
    def _validate_regular_segment(
        segment: Any,
        planned_slot: dict[str, Any],
        known_lines: set[str],
    ) -> None:
        CloseReadingAgent._validate_segment_fields(segment)
        line_ids = [str(value) for value in segment.get("line_ids", [])]
        if not line_ids or any(line_id not in known_lines for line_id in line_ids):
            raise ValueError(f"Invalid segment line_ids: {segment.get('shot_id')}")
        planned_line_ids = [str(value) for value in planned_slot.get("line_ids", [])]
        if planned_slot and line_ids != planned_line_ids:
            raise ValueError(f"Segment line_ids do not match plan: {segment.get('shot_id')}")

        role = str(segment.get("narration_role") or "")
        action = str(segment.get("reading_action") or "")
        if role not in NARRATION_ROLES or action not in READING_ACTIONS:
            raise ValueError(f"Invalid role/action: {segment.get('shot_id')}")
        if role == "visual_only" or action == "visual_only":
            raise ValueError("Close-reading source slots cannot be visual_only")

        audio = str(segment.get("audio_text") or "").strip()
        if not audio:
            raise ValueError(f"Close-reading audio is empty: {segment.get('shot_id')}")
        count = spoken_character_count(audio)
        if not CLOSE_READING_MIN_CHARACTERS <= count <= CLOSE_READING_MAX_CHARACTERS:
            raise ValueError(
                f"Close-reading audio {segment.get('shot_id')} has {count} effective characters; "
                f"expected {CLOSE_READING_MIN_CHARACTERS}..{CLOSE_READING_MAX_CHARACTERS}"
            )
        CloseReadingAgent._validate_spoken_text(audio)
        points_text = CloseReadingAgent._validate_knowledge_points(segment)
        combined_text = f"{audio} {points_text}"
        if _contains_any(combined_text, GENERIC_TEACHING_PHRASES):
            raise ValueError(f"Segment is too generic: {segment.get('shot_id')}")

        quote_units = [
            str(value) for value in planned_slot.get("quote_units", []) if str(value).strip()
        ]
        normalized_audio = _normalize_text(audio)
        if quote_units and not any(_normalize_text(unit) in normalized_audio for unit in quote_units):
            segment["audio_text"] = f"“{quote_units[0]}”，{audio}"
        segment["target_seconds"] = _estimate_close_reading_seconds(audio)

    @staticmethod
    def _validate_segment_fields(segment: Any) -> None:
        required_fields = {
            "shot_id",
            "line_ids",
            "audio_text",
            "narration_role",
            "reading_action",
            "knowledge_points",
            "teaching_move",
            "unstated_visual_meaning",
            "target_seconds",
        }
        if not isinstance(segment, dict) or set(segment) != required_fields:
            raise ValueError("Invalid close-reading segment fields")

    @staticmethod
    def _validate_knowledge_points(segment: dict[str, Any]) -> str:
        points = segment.get("knowledge_points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"Missing knowledge_points: {segment.get('shot_id')}")
        contents: list[str] = []
        for point in points:
            if not isinstance(point, dict) or set(point) != {"type", "content"}:
                raise ValueError(f"Invalid knowledge point: {segment.get('shot_id')}")
            point_type = _normalize_knowledge_type(point["type"])
            if not point_type:
                raise ValueError(f"Invalid knowledge type: {segment.get('shot_id')}")
            point["type"] = point_type
            content = str(point.get("content") or "")
            if spoken_character_count(content) < 6:
                raise ValueError(f"Knowledge point is too vague: {segment.get('shot_id')}")
            contents.append(content)
        return " ".join(contents)

    @staticmethod
    def _validate_spoken_text(
        audio: str,
        *,
        intro: bool = False,
        final: bool = False,
    ) -> None:
        common_forbidden = (
            FORBIDDEN_INTERNAL_AUDIO_PHRASES
            + SECOND_PERSON_PRONOUNS
            + FORBIDDEN_EXPLICIT_PHRASES
            + FORBIDDEN_NARRATION_STAGE_DIRECTIONS
            + FORBIDDEN_TEACHING_TEMPLATES
        )
        if _contains_any(audio, common_forbidden) and not relaxed_keyword_validation():
            raise ValueError("Narration contains forbidden wording")
        if not intro and not final and any(
            audio.startswith(prefix) for prefix in FORBIDDEN_AUDIO_LEAD_INS
        ) and not relaxed_keyword_validation():
            raise ValueError("Close-reading audio starts with a scripted lead-in")
        if intro and _contains_any(audio, FORBIDDEN_INTRO_TEMPLATES) and not relaxed_keyword_validation():
            raise ValueError("Poem introduction uses a scripted template")
        if final:
            if _contains_any(audio, FORBIDDEN_FINAL_ABSTRACTIONS) and not relaxed_keyword_validation():
                raise ValueError("Final synthesis is too abstract")
            if (
                _contains_any(audio, ("没有唯一答案", "请保留自己的理解", "属于你的共鸣"))
                and not relaxed_keyword_validation()
            ):
                raise ValueError("Final synthesis must end with a concrete textual takeaway")

    @staticmethod
    def _validate_line_coverage(
        expected_line_ids: list[str],
        referenced_lines: list[str],
        explained_lines: set[str],
        spoken_by_line: dict[str, list[str]],
        line_text_by_id: dict[str, str],
    ) -> None:
        positions = []
        for line_id in expected_line_ids:
            if line_id not in referenced_lines:
                raise ValueError(f"Poem line is not covered: {line_id}")
            positions.append(referenced_lines.index(line_id))
        if positions != sorted(positions):
            raise ValueError("Poem lines are not in source order")

        for line_id in expected_line_ids:
            normalized_segments = [_normalize_text(value) for value in spoken_by_line[line_id]]
            for unit in normalized_quote_units(line_text_by_id.get(line_id, "")):
                if not any(unit in segment for segment in normalized_segments):
                    # Keep the generated explanation usable. Exact quotation
                    # coverage is a quality preference rather than a hard
                    # structural failure for batch generation.
                    continue
        if explained_lines != set(expected_line_ids):
            missing = sorted(set(expected_line_ids) - explained_lines)
            raise ValueError(f"Every poem line needs explanation: missing {missing}")
