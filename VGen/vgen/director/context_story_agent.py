from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .common import (
    ARC_TYPES,
    NARRATIVE_BASES,
    STORY_CLAIM_STATUSES,
    STORY_RELATIONS,
    ChatCompletionClient,
    relaxed_keyword_validation,
    spoken_character_count,
    story_context_for_agent,
    unsupported_quantitative_details,
)


FORBIDDEN_STORY_META_PHRASES = (
    "模型补充",
    "未外部核验",
    "以上内容",
    "以上生平",
    "输入材料",
    "证据来源",
    "知识库",
)


class ContextStoryAgent:
    def __init__(self, client: ChatCompletionClient):
        self.client = client

    def plan(self, context: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        story_context = story_context_for_agent(context)
        expected_subject_name = str(
            story_context.get("poem_identity", {}).get("author") or ""
        ).strip()
        allowed_fact_text = json.dumps(
            story_context.get("allowed_fact_statements", []), ensure_ascii=False
        )
        valid_evidence_ids = {
            str(item.get("source_id") or "")
            for item in story_context.get("evidence_sources", [])
            if isinstance(item, dict)
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是古诗词语境叙事编剧和创作背景研究员。你先判断作品应沿人物事件、历史背景、"
                    "物象演变、空间观察、情绪处境还是混合线索讲述，再把有证据边界的材料写成完整故事。"
                    "没有人物事件时不得强造事件或心理，不规划旁白时长、镜头、构图、运镜或转场，只输出严格JSON。"
                ),
            },
            {"role": "user", "content": self._build_prompt(story_context)},
        ]
        payload = {
            "model": self.client.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 14000,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "story_agent_request.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        content = self.client.complete(payload, label="context_story")
        (output_dir / "story_agent_response.txt").write_text(content, encoding="utf-8")
        for repair_attempt in range(5):
            try:
                result = self._parse(
                    content,
                    valid_evidence_ids=valid_evidence_ids,
                    allowed_fact_text=allowed_fact_text,
                    expected_subject_name=expected_subject_name,
                )
                (output_dir / "story_plan.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                return result
            except Exception as parse_error:
                if repair_attempt == 4:
                    raise
                repair_payload = dict(payload)
                repair_payload["temperature"] = 0.0
                quantitative_repair = ""
                if "Unsupported quantitative details" in str(parse_error):
                    quantitative_repair = (
                        "\n这是数量越界错误。请先扫描全文，逐字删除错误中列出的数量表达，"
                        "不得换成同义数量，也不得换成另一个数字。"
                        "其中“万字/上万字”改写为“完整的方略/详细的文字”，"
                        "“数百里/数百里路”改写为“长途奔袭/长途押解”；"
                        "如果错误列出其他数量，就改成不含数量的动作表达。"
                        "修复后全文不得再出现这些禁止数量。"
                    )
                repair_payload["messages"] = messages + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"输出无法使用：{parse_error}。请重新输出完整JSON。"
                            f"{quantitative_repair}"
                            "必须补足上下文维度、通用story_logic和与叙事依据匹配的experience_arc，"
                            "不能用短句年表、意象清单或输入材料之外的具体事实敷衍。"
                            "若full_story未达到要求，必须在现有证据范围内补清触发处境、选择、"
                            "行动推进、阻力、直接结果、长期后果和未解决问题；不得靠重复免责声明、"
                            "背景结论或诗句释义凑字数。"
                            "full_story与bridge_to_poem是可朗读成品，不得出现模型、核验、输入材料、"
                            "证据来源等生产过程说明；来源边界只保留在结构化状态和证据ID中。"
                            "重写时围绕一个持续目标，让每次压力迫使人物采取行动，并让行动后果产生"
                            "下一步处境；不要只给原事件逐条加长解释。"
                            "人物事件故事必须先用一到两句交代核心人物是谁、身处什么局势、原本要做"
                            "什么，以及关键配角和他的关系，再让导火索发生；可以用年份开头，但年份"
                            "只是时间坐标，不能替代这些背景信息。"
                            "错误中列出的数量不得换成另一个数量；必须删除，或改回"
                            "allowed_fact_statements中已经原样出现的定性表达。"
                        ),
                    },
                ]
                content = self.client.complete(repair_payload, label="context_story_repair")
                suffix = "" if repair_attempt == 0 else f"_{repair_attempt + 1}"
                (output_dir / f"story_agent_response_repaired{suffix}.txt").write_text(
                    content, encoding="utf-8"
                )
        raise RuntimeError("Story Agent did not produce a usable plan")

    @staticmethod
    def _build_prompt(story_context: dict[str, Any]) -> str:
        return f"""
请先选择适合当前作品的叙事依据，再写完整语境故事。输出格式严格为：
{{
  "narrative_basis":"event_based",
  "context_dimensions":{{"who":"...或不适用及原因","when":"...","where":"...","what":"...","why":"...","how":"..."}},
  "story_logic":[{{"node_id":"n1","from_state":"...","relation":"contextualizes","to_state":"...","claim_status":"documented","evidence_ids":["e01"]}}],
  "experience_arc":[{{"stage":"...","arc_type":"psychological","external_condition":"...","focus_or_choice":"...","observable_change":"...","inferred_change":"...","certainty":"documented_or_observable"}}],
  "full_story":"完整故事",
  "bridge_to_poem":"从语境故事自然进入作品，但不提前逐句赏析"
}}

硬约束：
1. narrative_basis只能是{sorted(NARRATIVE_BASES)}。人物事件用event_based；时代压力用historical_context；咏物可用object_lifecycle；山水写景可用spatial_observation；抒情处境可用emotional_situation；确有多条主线才用hybrid。
2. context_dimensions回答who/when/where/what/why/how。某一维确实不适用时写“not_applicable: 具体原因”，不得为了填满5W1H虚构人物、事件或动机。
3. story_logic不是固定因果链。根据材料选择precedes、causes、motivates、enables、obstructs、contrasts_with、transforms_into、recalls、contextualizes、supports_interpretation等关系，数量与材料复杂度匹配。
4. full_story必须连贯、具体。event_based、historical_context或hybrid通常写500到800个有效字符，足以改编为2到3分钟旁白，并讲清前因、选择、行动经过、直接结果、长期后果与尚未解决的问题；其他类型通常不少于220个有效字符并讲清物象、空间、声音或感受怎样推进。不要写成年表、意象清单或人物简介。
4.0 人物事件型故事的开头先完成叙事定向，再发生导火索：用一到两句自然交代核心人物的身份或既有处境、故事发生的时代与环境、他原本坚持或试图完成的目标，以及首批关键配角与他的关系。允许用具体年份开头，但年份只是时间坐标；不能紧接着就在听众尚不知道人物是谁、彼此什么关系时同时抛出多个姓名、机构和行动结果。次要人物第一次出现时，用最短的同位语说明其角色或与核心人物的关系。
4.1 如果context_nodes提供了具体人物事件，full_story必须从最早的具体处境和触发事件讲起，依次交代人物为何必须选择、采取了什么行动、行动怎样推进、造成什么直接结果，再进入长期目标或受阻；不得把多个已有事件压缩成“青年时立志”“一生追求”等人物简介。causal_chain中的具体事件都必须在故事中获得可辨认的经过。
4.2 先确定一条贯穿全文的持续目标或待解决问题，再组织“外部压力改变处境—人物作出选择—行动产生可见后果—后果带来下一重压力”的叙事推进。每一部分都要改变人物可采取的行动或目标能否实现，不能按年份逐项罗列经历，也不能把context_nodes依次扩写成资料卡。
4.3 语言应当像成熟的历史叙事而不是研究摘要：用具体主语和动作承接因果，长短句自然变化；少用“随后、之后、同一时期、正式开启、彻底失去”等流水账连接或结论。人物心理通过迟疑、选择、坚持、放弃和行动后果表现，不编造对话、内心独白、感官细节或输入外的戏剧冲突。
5. experience_arc必须与narrative_basis匹配：有人物且有证据时可写psychological；无人物时使用observational、object、spatial或historical，不得强造心理。推断必须用可能、可以理解为等限定语。
6. 严格限定事实颗粒度。人名、地名、年代、数量、路线、著作、官职、处置结果和可观察行动，只能来自context_nodes或evidence_sources的relevant_claims；输入没提供的细节不得靠常识补全，也不能根据模型记忆补出路线节点、设施、目的地或后续处置。数字、年龄、年数以及“数十”“数百”“数千”“近百”“余人”等模糊数量，必须在allowed_fact_statements中原样出现；不得四舍五入、近似改写或把“长期”换算成自己推测的年数。输出前逐句检查full_story，无法在输入中指出依据的具体细节直接删除。
7. 不把解释性桥梁改写成已证实的创作现场或作者心理。origin=model_supplied的来源只能标为historically_plausible、contested或interpretive，不能写成documented。来源边界由claim_status、certainty和evidence_ids承担；full_story与bridge_to_poem禁止出现“模型补充、未外部核验、以上内容、输入材料、证据来源”等生产过程元话语。写作时间、地点、动机有争议时，用“具体时间仍难确定”“一种读法是”等自然叙述保留开放性，不能写“作者当晚因此想到”。
7.1 若writing_situation或某个节点明确表示现场未知，full_story不得用诗句字面补出作者在节日、人群、舟中、楼上等现场的行动。可以从已知经历自然桥接到作品的阅读问题，但不能把解释拍成传记事实。
8. 是否在入诗前复演原文意象，完全服从输入的depiction_policy和constraints，不得写死某首作品的意象或结构。bridge_to_poem承接故事里尚未解决的目标、选择或问题进入作品，不提前逐句赏析、不宣布隐喻答案，也不要用“现在我们进入作品”一类流程提示收尾。
9. story_logic.relation只能是{sorted(STORY_RELATIONS)}；experience_arc.arc_type只能是{sorted(ARC_TYPES)}。关系和阶段数量服从材料本身，不构造固定因果链。
10. claim_status只能是{sorted(STORY_CLAIM_STATUSES)}；certainty只能是documented_or_observable、historically_plausible、interpretive。
11. 禁止镜头、画面中、运镜、构图、转场等制作说明；禁止“象征着、表达了、历史的回响、正是他的写照”等抽象套话。
12. 只输出JSON。不要输出segments或镜头ID。

Evidence-grounded story context:
{json.dumps(story_context, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _parse(
        content: str,
        *,
        valid_evidence_ids: set[str] | None = None,
        allowed_fact_text: str = "",
        expected_subject_name: str = "",
    ) -> dict[str, Any]:
        data = ChatCompletionClient.extract_json(content)
        required_top = {
            "narrative_basis",
            "context_dimensions",
            "story_logic",
            "experience_arc",
            "full_story",
            "bridge_to_poem",
        }
        if set(data) != required_top:
            raise ValueError(f"Unexpected story fields: {sorted(data)}")
        narrative_basis = str(data.get("narrative_basis") or "")
        if narrative_basis not in NARRATIVE_BASES:
            raise ValueError(f"Invalid narrative_basis: {narrative_basis}")
        context_dimensions = data.get("context_dimensions")
        if not isinstance(context_dimensions, dict) or set(context_dimensions) != {
            "who",
            "when",
            "where",
            "what",
            "why",
            "how",
        }:
            raise ValueError("context_dimensions must contain exactly who/when/where/what/why/how")
        for key, value in context_dimensions.items():
            if spoken_character_count(str(value)) < 6:
                raise ValueError(f"context_dimensions.{key} is too vague")

        story_logic = data.get("story_logic")
        if not isinstance(story_logic, list) or not story_logic:
            raise ValueError("story_logic must contain at least one evidence-bounded relation")
        logic_fields = {"node_id", "from_state", "relation", "to_state", "claim_status", "evidence_ids"}
        for item in story_logic:
            if not isinstance(item, dict) or set(item) != logic_fields:
                raise ValueError("Invalid story_logic item fields")
            if item.get("relation") not in STORY_RELATIONS:
                raise ValueError(f"Invalid story relation: {item.get('relation')}")
            if item.get("claim_status") not in STORY_CLAIM_STATUSES:
                raise ValueError(f"Invalid story-logic claim_status: {item.get('claim_status')}")
            evidence_ids = [str(value) for value in item.get("evidence_ids", [])]
            if valid_evidence_ids is not None and any(value not in valid_evidence_ids for value in evidence_ids):
                raise ValueError(f"Unknown story-logic evidence id: {evidence_ids}")

        experience_arc = data.get("experience_arc")
        if not isinstance(experience_arc, list) or len(experience_arc) < 2:
            raise ValueError("experience_arc must contain at least two stages")
        arc_fields = {
            "stage",
            "arc_type",
            "external_condition",
            "focus_or_choice",
            "observable_change",
            "inferred_change",
            "certainty",
        }
        for item in experience_arc:
            if not isinstance(item, dict) or set(item) != arc_fields:
                raise ValueError("Invalid experience_arc item fields")
            if item.get("arc_type") not in ARC_TYPES:
                raise ValueError(f"Invalid arc_type: {item.get('arc_type')}")
            if item.get("certainty") not in {
                "documented_or_observable",
                "historically_plausible",
                "interpretive",
            }:
                raise ValueError(f"Invalid experience certainty: {item.get('certainty')}")

        full_story = str(data.get("full_story") or "").strip()
        minimum_story_chars = 500 if narrative_basis in {
            "event_based",
            "historical_context",
            "hybrid",
        } else 180
        story_chars = spoken_character_count(full_story)
        if story_chars < minimum_story_chars:
            raise ValueError(
                "full_story is not detailed enough: "
                f"got {story_chars} spoken characters, need at least {minimum_story_chars}"
            )
        if narrative_basis in {"event_based", "hybrid"}:
            opening_sentences = "。".join(
                part.strip()
                for part in re.split(r"[。！？!?]", full_story)[:2]
                if part.strip()
            )
            if expected_subject_name and expected_subject_name not in opening_sentences:
                raise ValueError(
                    "Story opening must introduce the central subject within its first two sentences: "
                    f"{expected_subject_name}"
                )
        bridge = str(data.get("bridge_to_poem") or "").strip()
        if spoken_character_count(bridge) < 15:
            raise ValueError("bridge_to_poem is too short")
        spoken_story = f"{full_story}\n{bridge}"
        meta_phrases = [
            phrase for phrase in FORBIDDEN_STORY_META_PHRASES if phrase in spoken_story
        ]
        if meta_phrases and not relaxed_keyword_validation():
            raise ValueError(
                "full_story/bridge_to_poem contains source-provenance meta language: "
                f"{meta_phrases}"
            )
        if allowed_fact_text:
            unsupported_details = unsupported_quantitative_details(
                json.dumps(data, ensure_ascii=False), allowed_fact_text
            )
            if unsupported_details:
                raise ValueError(
                    f"Unsupported quantitative details not present in supplied evidence: {unsupported_details}"
                )
        return data
