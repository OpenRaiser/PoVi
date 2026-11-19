from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .common import (
    FORBIDDEN_EXPLICIT_PHRASES,
    FORBIDDEN_NARRATION_STAGE_DIRECTIONS,
    FORBIDDEN_TEACHING_TEMPLATES,
    STORY_CLAIM_STATUSES,
    STORY_FUNCTIONS,
    ChatCompletionClient,
    estimate_story_seconds,
    relaxed_keyword_validation,
    spoken_character_count,
    story_context_for_agent,
)


NARRATIVE_MOVES = {
    "setup",
    "escalation",
    "decision",
    "reversal",
    "consequence",
    "reflection",
    "bridge",
}
NARRATIVE_MOVE_ALIASES = {
    "observation": "reflection",
}
LONG_FORM_SECTION_MAX_CHARACTERS = 140
LONG_FORM_TOTAL_CHARACTER_GUIDANCE = [440, 740]


def allocate_bounded_event_durations(
    total_seconds: float,
    weights: list[float],
    *,
    minimum_seconds: float = 0.6,
    maximum_seconds: float = 8.0,
) -> list[float]:
    """Preserve relative emphasis while keeping every generated shot usable."""
    if not weights or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("Event duration weights must be finite and positive")
    minimum_total = minimum_seconds * len(weights)
    maximum_total = maximum_seconds * len(weights)
    if not minimum_total <= total_seconds <= maximum_total + 1e-6:
        raise ValueError(
            f"Event duration {total_seconds} cannot fit {len(weights)} shots within "
            f"{minimum_seconds}..{maximum_seconds} seconds"
        )

    allocations = [minimum_seconds] * len(weights)
    remaining = total_seconds - minimum_total
    active = set(range(len(weights)))
    while active and remaining > 1e-9:
        active_weight = sum(weights[index] for index in active)
        capped = [
            index
            for index in active
            if remaining * weights[index] / active_weight
            >= maximum_seconds - allocations[index]
        ]
        if not capped:
            for index in active:
                allocations[index] += remaining * weights[index] / active_weight
            remaining = 0.0
            break
        for index in capped:
            capacity = maximum_seconds - allocations[index]
            allocations[index] = maximum_seconds
            remaining -= capacity
            active.remove(index)

    rounded = [round(value, 2) for value in allocations]
    correction = round(total_seconds - sum(rounded), 2)
    if correction:
        candidates = range(len(rounded) - 1, -1, -1)
        index = next(
            index
            for index in candidates
            if minimum_seconds <= rounded[index] + correction <= maximum_seconds
        )
        rounded[index] = round(rounded[index] + correction, 2)
    return rounded


class StoryNarrationAgent:
    """Turn a locked context story into long-form spoken sections.

    A section is one coherent audio passage. Its event units are visual coverage
    points, not separately spoken micro-sentences.
    """

    def __init__(self, client: ChatCompletionClient):
        self.client = client

    def plan(
        self,
        context: dict[str, Any],
        story_plan: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        story_context = story_context_for_agent(context)
        valid_context_node_ids = {
            str(item.get("node_id") or "")
            for item in story_context.get("causal_context", {}).get("context_nodes", [])
            if isinstance(item, dict)
        }
        long_form = str(story_plan.get("narrative_basis") or "") in {
            "event_based",
            "historical_context",
            "hybrid",
        }
        narration_context = {
            "story_plan": story_plan,
            "allowed_context_nodes": [
                {
                    "node_id": item.get("node_id"),
                    "summary": item.get("summary"),
                    "claim_status": item.get("claim_status"),
                }
                for item in story_context.get("causal_context", {}).get("context_nodes", [])
                if isinstance(item, dict)
            ],
            "narrative_rhetoric_mapping": context.get("narrative_rhetoric_mapping", {}),
            "duration_policy": {
                "mode": "long_form_causal_story" if long_form else "compact_context_story",
                "target_total_seconds": [120, 180] if long_form else [45, 120],
                "section_audio_characters": (
                    [20, LONG_FORM_SECTION_MAX_CHARACTERS]
                    if long_form
                    else [20, 120]
                ),
                "substantive_section_characters": (
                    [45, LONG_FORM_SECTION_MAX_CHARACTERS]
                    if long_form
                    else [30, 120]
                ),
                "total_audio_character_guidance": (
                    LONG_FORM_TOTAL_CHARACTER_GUIDANCE if long_form else None
                ),
                "visual_event_units_per_section": [2, 5],
                "rule": "one spoken section may be covered by several independently generated shots",
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是故事旁白编剧。完整语境故事已经锁定，你把它改写成有铺垫、行动、逆转、"
                    "后果和余味的连续口述段落。你不规划机位，不新增事实，不提前逐句赏析，只输出严格JSON。"
                ),
            },
            {"role": "user", "content": self._build_prompt(narration_context)},
        ]
        payload = {
            "model": self.client.model,
            "messages": messages,
            "temperature": 0.25,
            "top_p": 0.9,
            "max_tokens": 14000,
        }
        (output_dir / "story_narration_agent_request.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        content = self.client.complete(payload, label="story_narration")
        (output_dir / "story_narration_agent_response.txt").write_text(content, encoding="utf-8")
        for repair_attempt in range(4):
            try:
                result = self._parse(
                    content,
                    full_story=str(story_plan.get("full_story") or ""),
                    bridge_to_poem=str(story_plan.get("bridge_to_poem") or ""),
                    valid_context_node_ids=valid_context_node_ids,
                    long_form=long_form,
                )
                (output_dir / "story_narration_plan.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                return result
            except Exception as parse_error:
                if repair_attempt == 3:
                    raise
                diagnostics = self._repair_diagnostics(
                    content,
                    long_form=long_form,
                    locked_story_text=(
                        f"{story_plan.get('full_story') or ''}\n"
                        f"{story_plan.get('bridge_to_poem') or ''}"
                    ),
                )
                repair_payload = dict(payload)
                repair_payload["temperature"] = 0.15
                repair_payload["messages"] = messages + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"输出无法使用：{parse_error}。请重新输出完整JSON。"
                            f"\n{diagnostics}\n"
                            "sections数量服从故事结构，slot_id必须依次使用st01、st02……；"
                            f"source_context_node_ids只能使用{sorted(valid_context_node_ids)}，"
                            "m01、e01等是证据ID，不是context node ID；"
                            "每段是自然口述，不是短句履历；"
                            "请一次修完诊断中的所有段。event_units只拆可视事件，不拆碎audio_text，"
                            "也不得新增锁定故事之外的事实。若parse_error指出Forbidden narration "
                            "wording、第三人称代词开句过多或“然而”滥用，必须只改报错句的措辞或主语"
                            "结构并保留其余内容；除非某段audio字数越界、包含报错禁用措辞、触发报错的"
                            "旁白文风规则，或全篇总字数、总时长不合格，否则原样保留该段audio_text；"
                            "evidence错误只替换"
                            "evidence_anchors，event_units不足只增补event_units，不要借修格式重写旁白。"
                            "evidence_anchors只复制锁定full_story或bridge_to_poem中8到30个字的"
                            "短连续子串，不得复制整段audio_text，不得改写一个字。"
                        ),
                    },
                ]
                content = self.client.complete(repair_payload, label="story_narration_repair")
                suffix = "" if repair_attempt == 0 else f"_{repair_attempt + 1}"
                (output_dir / f"story_narration_agent_response_repaired{suffix}.txt").write_text(
                    content, encoding="utf-8"
                )
        raise RuntimeError("Story Narration Agent did not produce a usable plan")

    @staticmethod
    def _build_prompt(narration_context: dict[str, Any]) -> str:
        return f"""
请把锁定的full_story改写为原诗展示前的连续故事旁白。输出严格为：
{{"sections":[{{
  "slot_id":"st01",
  "audio_text":"一段可直接朗读的完整叙事",
  "story_function":"context_setup",
  "claim_status":"documented",
  "source_context_node_ids":["c01"],
  "evidence_anchors":["从full_story逐字摘取的连续片段"],
  "narrative_move":"setup",
  "event_units":[{{
    "starting_condition":"事件开始前可观察的处境",
    "event_text":"段内发生的一件具体事情",
    "observable_result":"事件结束时可观察的结果",
    "visible_characters":["镜头内可见的具名或跨镜复现人物"],
    "weight":1.0
  }}],
  "implicit_meaning":"留给视觉和后文承担、不在此处直说的意义",
  "transition_logic":"本段怎样自然引出下一段",
  "target_seconds":24
}}]}}

要求：
1. 根据锁定故事的转折和信息密度生成2到8个sections，不能按固定模板凑数。slot_id从st01开始连续编号，不重复、不回跳。一个section是一段完整音频，不等于一个视频镜头。
2. event_based、historical_context或hybrid故事总旁白目标为2到3分钟，总计约440到740个有效字符。承担完整行动、阻力或后果的核心段通常45到{LONG_FORM_SECTION_MAX_CHARACTERS}个有效字符，任何段都不得超过{LONG_FORM_SECTION_MAX_CHARACTERS}个；确实只负责承接或转向的段可自然缩短到20到44字，不得为了整齐把各段写成相近长度。其他故事服从duration_policy。不得把完整故事压成履历、标题句或均匀信息卡。
3. 每段先承接上一段留下的具体处境，再讲行动怎样发生、遇到什么阻力、造成什么直接结果。人物心理只通过选择、迟疑、重复动作和后果表现，不编造内心独白。
4. narrative_move按真实叙事功能选择{sorted(NARRATIVE_MOVES)}。转折来自目标受阻、信息改变或行动后果，不来自机械添加“然而、但是、于是”；全文“然而”至多出现两次，也不要求必须出现。
5. 避免“他做了A。他做了B。”的连续主语句。可让战局、命令、队伍、奏议、制度条件、时间或行动后果成为句子主语；换时空时再自然重提姓名。
6. event_units只把本段拆成2到5个按顺序发生的语义事件，供后续Visual Director规划多个镜头。它们不包含景别、机位、运镜、光线或生成prompt。按每4个有效字符约1秒估算朗读时长，event_units数量必须不少于ceil(有效字符数/32)，且不得超过5；如果需要6个，就缩短该段或拆成两个section。输出前逐段验算，确保每个事件只需覆盖约4到8秒。
6.1 visible_characters只列当前事件画面内实际可见、需要跨镜头保持身份的人物，使用稳定且具体的名称，例如“李白”“唐玄宗”或“青年词人”。同一人物跨事件必须逐字使用同一名称；背影、侧脸、可辨认的手部或局部仍算可见。只在台词或叙述中被提到、位于画外的人物不得列入；一次性路人、群演和不可辨认人群不列。没有时输出空数组，禁止使用“他、她、人物”等代词或泛称。
7. evidence_anchors每项必须从full_story或bridge_to_poem逐字连续摘取。每段优先选一个8到30字的短子串，直接复制，不改同义词、不补主语、不改标点、不把audio_text整段当成锚点。输出前必须用字符串包含检查确认每个锚点是full_story或bridge_to_poem的连续子串。audio_text、event_units和transition_logic只能重组这些已锁定信息；不得补充输入外的人名、地点、年代、数量、路线、官职、处置结果、对话或心理。
7.1 source_context_node_ids只能从allowed_context_nodes.node_id中选择，用来标明本段承接的语境节点。m01、e01等evidence source ID不得填入该字段。
8. 最后一段必须story_function=bridge_to_poem且narrative_move=bridge。它带着一个尚未解决的具体问题、动作或物件进入作品，不宣布主题，不逐句赏析，不把隐喻说成唯一答案。
9. story_function只能是{sorted(STORY_FUNCTIONS)}；claim_status只能是{sorted(STORY_CLAIM_STATUSES)}。解释性或争议内容保留“可能、可以这样理解、无法确定”等限定。
10. 禁止“镜头、画面中、运镜、构图、转场”等制作说明；禁止“象征着、表达了、正是他的写照、形成反差、形成对比”等结论套话。只输出JSON。

Story narration context:
{json.dumps(narration_context, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _parse(
        content: str,
        *,
        full_story: str,
        valid_context_node_ids: set[str] | None = None,
        bridge_to_poem: str = "",
        long_form: bool = False,
    ) -> dict[str, Any]:
        data = ChatCompletionClient.extract_json(content)
        if set(data) != {"sections"} or not isinstance(data.get("sections"), list):
            raise ValueError("Story narration output must contain only sections")
        sections = data["sections"]
        if not 2 <= len(sections) <= 8:
            raise ValueError("Story narration needs 2..8 structurally meaningful sections")
        actual_slot_ids = [str(item.get("slot_id") or "") for item in sections if isinstance(item, dict)]
        expected_slot_ids = [f"st{index:02d}" for index in range(1, len(sections) + 1)]
        if actual_slot_ids != expected_slot_ids:
            raise ValueError(f"Story section slots must be sequential: {expected_slot_ids}")

        section_fields = {
            "slot_id",
            "audio_text",
            "story_function",
            "claim_status",
            "source_context_node_ids",
            "evidence_anchors",
            "narrative_move",
            "event_units",
            "implicit_meaning",
            "transition_logic",
            "target_seconds",
        }
        event_fields = {
            "starting_condition",
            "event_text",
            "observable_result",
            "visible_characters",
            "weight",
        }
        locked_story_text = f"{full_story}\n{bridge_to_poem}"
        combined_audio = ""
        for section_index, section in enumerate(sections):
            if not isinstance(section, dict) or set(section) != section_fields:
                raise ValueError(f"Unexpected story section fields: {sorted(section)}")
            slot_id = str(section["slot_id"])
            audio_text = str(section.get("audio_text") or "").strip()
            character_count = spoken_character_count(audio_text)
            is_bridge = section.get("story_function") == "bridge_to_poem"
            minimum_chars = 20
            maximum_chars = LONG_FORM_SECTION_MAX_CHARACTERS if long_form else 120
            if not minimum_chars <= character_count <= maximum_chars:
                raise ValueError(
                    f"Story section {slot_id} must contain {minimum_chars}..{maximum_chars} spoken characters"
                )
            if any(
                phrase in audio_text
                for phrase in FORBIDDEN_EXPLICIT_PHRASES
                + FORBIDDEN_NARRATION_STAGE_DIRECTIONS
                + FORBIDDEN_TEACHING_TEMPLATES
            ) and not relaxed_keyword_validation():
                raise ValueError(f"Forbidden narration wording in section {slot_id}: {audio_text}")
            if section.get("story_function") not in STORY_FUNCTIONS:
                raise ValueError(f"Invalid story_function: {section.get('story_function')}")
            if section.get("claim_status") not in STORY_CLAIM_STATUSES:
                raise ValueError(f"Invalid story claim_status: {section.get('claim_status')}")
            narrative_move = str(section.get("narrative_move") or "")
            section["narrative_move"] = NARRATIVE_MOVE_ALIASES.get(
                narrative_move, narrative_move
            )
            if section["narrative_move"] not in NARRATIVE_MOVES:
                raise ValueError(f"Invalid narrative_move: {section.get('narrative_move')}")

            source_ids = [str(value) for value in section.get("source_context_node_ids", [])]
            if not source_ids:
                raise ValueError(f"Story section {slot_id} needs source_context_node_ids")
            if valid_context_node_ids is not None and any(
                value not in valid_context_node_ids for value in source_ids
            ):
                raise ValueError(f"Unknown story context node id: {source_ids}")
            anchors = section.get("evidence_anchors")
            if not isinstance(anchors, list) or not anchors:
                raise ValueError(f"Story section {slot_id} needs evidence_anchors")
            if any(
                spoken_character_count(str(anchor)) < 6 or str(anchor) not in locked_story_text
                for anchor in anchors
            ):
                raise ValueError(f"Story section {slot_id} contains a non-verbatim evidence anchor")

            event_units = section.get("event_units")
            if not isinstance(event_units, list) or not event_units:
                raise ValueError(f"Story section {slot_id} needs event_units")
            if len(event_units) > 5 or (not is_bridge and len(event_units) < 2):
                raise ValueError(f"Story section {slot_id} needs 2..5 visual event units")
            for event_index, event in enumerate(event_units, start=1):
                if not isinstance(event, dict) or set(event) != event_fields:
                    raise ValueError(f"Invalid event unit in story section {slot_id}")
                for field in ("starting_condition", "event_text", "observable_result"):
                    if spoken_character_count(str(event.get(field) or "")) < 4:
                        raise ValueError(f"Story event {slot_id}.{field} is too vague")
                visible_characters = event.get("visible_characters")
                if not isinstance(visible_characters, list):
                    raise ValueError(f"Story event {slot_id} visible_characters must be an array")
                normalized_characters = [str(value).strip() for value in visible_characters]
                if any(not value for value in normalized_characters):
                    raise ValueError(f"Story event {slot_id} contains an empty visible character")
                if len(normalized_characters) != len(set(normalized_characters)):
                    raise ValueError(f"Story event {slot_id} repeats a visible character")
                forbidden_character_labels = {
                    "他",
                    "她",
                    "他们",
                    "她们",
                    "人物",
                    "男人",
                    "女人",
                    "人群",
                    "众人",
                    "路人",
                    "群演",
                }
                if any(value in forbidden_character_labels for value in normalized_characters):
                    raise ValueError(
                        f"Story event {slot_id} uses an unstable visible character label: "
                        f"{normalized_characters}"
                    )
                event["visible_characters"] = normalized_characters
                weight = float(event.get("weight") or 0)
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(f"Invalid story event weight in {slot_id}: {weight}")
                event["shot_id"] = slot_id if event_index == 1 else f"{slot_id}_v{event_index:02d}"

            section["section_id"] = f"story_{slot_id}"
            section["target_seconds"] = estimate_story_seconds(audio_text)
            minimum_visual_units = math.ceil(float(section["target_seconds"]) / 8.0)
            if len(event_units) < minimum_visual_units:
                raise ValueError(
                    f"Story section {slot_id} needs at least {minimum_visual_units} event units for its duration"
                )
            event_durations = allocate_bounded_event_durations(
                float(section["target_seconds"]),
                [float(event["weight"]) for event in event_units],
            )
            for event, duration in zip(event_units, event_durations):
                event["target_seconds"] = duration
            combined_audio += audio_text

        final = sections[-1]
        if final.get("story_function") != "bridge_to_poem" or final.get("narrative_move") != "bridge":
            raise ValueError("The final story section must be a bridge_to_poem bridge")
        sentences = [
            part.strip()
            for part in re.split(r"[。！？!?；;]+", combined_audio)
            if part.strip()
        ]
        pronoun_starts = [bool(re.match(r"^(?:他们|她们|他|她)", sentence)) for sentence in sentences]
        if any(all(pronoun_starts[index : index + 3]) for index in range(len(sentences) - 2)):
            raise ValueError("Three consecutive story sentences may not begin with third-person pronouns")
        if sentences and sum(pronoun_starts) / len(sentences) > 0.35:
            raise ValueError("Too many story sentences begin with third-person pronouns")
        if combined_audio.count("然而") > 2:
            raise ValueError("The narration overuses 然而 instead of earning narrative turns")
        if long_form:
            combined_chars = spoken_character_count(combined_audio)
            total_seconds = sum(float(section["target_seconds"]) for section in sections)
            if combined_chars < LONG_FORM_TOTAL_CHARACTER_GUIDANCE[0]:
                raise ValueError(
                    "Long-form story is over-compressed: need at least "
                    f"{LONG_FORM_TOTAL_CHARACTER_GUIDANCE[0]} characters"
                )
            if not 105 <= total_seconds <= 190:
                raise ValueError(f"Long-form story duration must be near 2..3 minutes, got {total_seconds:.1f}s")
        return data

    @staticmethod
    def _repair_diagnostics(
        content: str,
        *,
        long_form: bool,
        locked_story_text: str = "",
    ) -> str:
        try:
            data = ChatCompletionClient.extract_json(content)
        except Exception as error:
            return f"无法提取JSON，先修复JSON语法：{error}"
        sections = data.get("sections")
        if not isinstance(sections, list):
            return "sections不是数组，请按约定输出完整sections数组。"

        maximum_chars = LONG_FORM_SECTION_MAX_CHARACTERS if long_form else 120
        lines = ["逐段数值诊断（有效字符不含标点）："]
        total_characters = 0
        total_seconds = 0.0
        for section in sections:
            if not isinstance(section, dict):
                lines.append("- 存在非对象section。")
                continue
            slot_id = str(section.get("slot_id") or "<missing>")
            audio_text = str(section.get("audio_text") or "").strip()
            character_count = spoken_character_count(audio_text)
            target_seconds = estimate_story_seconds(audio_text)
            event_units = section.get("event_units")
            event_count = len(event_units) if isinstance(event_units, list) else 0
            minimum_event_count = math.ceil(target_seconds / 8.0)
            anchors = section.get("evidence_anchors")
            anchors_valid = bool(anchors) and isinstance(anchors, list) and all(
                spoken_character_count(str(anchor)) >= 6
                and (not locked_story_text or str(anchor) in locked_story_text)
                for anchor in anchors
            )
            weights = [
                float(event.get("weight") or 0)
                for event in event_units or []
                if isinstance(event, dict)
            ]
            weight_total = sum(weights)
            maximum_allocation = (
                target_seconds * max(weights) / weight_total
                if weights and weight_total > 0
                else target_seconds
            )
            total_characters += character_count
            total_seconds += target_seconds
            lines.append(
                f"- {slot_id}: audio={character_count}字（允许20..{maximum_chars}；核心段建议>=45），"
                f"预计{target_seconds:.1f}秒；event_units={event_count}个，至少需要"
                f"{minimum_event_count}个，当前最大分配约{maximum_allocation:.1f}秒（必须<=8）；"
                f"evidence_anchors={'有效' if anchors_valid else '无效'}。"
            )
        lines.append(f"- 全篇合计：{total_characters}字，预计{total_seconds:.1f}秒。")
        return "\n".join(lines)
