from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import CHARACTER_REFERENCE_MODES, ChatCompletionClient, spoken_character_count


class CharacterAssetAgent:
    def __init__(self, client: ChatCompletionClient):
        self.client = client

    def plan(
        self,
        context: dict[str, Any],
        story_plan: dict[str, Any],
        narrative_plan: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        shot_summaries = [
            {
                key: item.get(key)
                for key in (
                    "shot_id",
                    "narrative_phase",
                    "story_function",
                    "event_text",
                    "visible_characters",
                    "line_ids",
                    "visual_delegation",
                )
            }
            for item in narrative_plan.get("segments", [])
            if isinstance(item, dict)
        ]
        valid_shot_ids = [str(item.get("shot_id") or "") for item in shot_summaries]
        visible_characters_by_shot = {
            str(item.get("shot_id") or ""): [
                str(value).strip()
                for value in item.get("visible_characters", [])
                if str(value).strip()
            ]
            for item in shot_summaries
        }
        asset_context = {
            "poem": context.get("poem", {}),
            "causal_context": context.get("causal_context", {}),
            "interpretation": context.get("interpretation", {}),
            "visual_bible": context.get("visual_bible", {}),
            "story_context_dimensions": story_plan.get("context_dimensions", {}),
            "experience_arc": story_plan.get("experience_arc", []),
            "planned_shots": shot_summaries,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是历史影像角色资产设计师。你只把已有角色证据和视觉设定整理成可生成、可复用的"
                    "合成角色参考图规范，不补写历史长相，不使用明星脸，不规划故事、旁白或镜头，只输出严格JSON。"
                ),
            },
            {"role": "user", "content": self._build_prompt(asset_context)},
        ]
        payload = {
            "model": self.client.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 8000,
        }
        (output_dir / "character_asset_agent_request.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        content = self.client.complete(payload, label="character_asset")
        (output_dir / "character_asset_agent_response.txt").write_text(content, encoding="utf-8")
        for repair_attempt in range(3):
            try:
                result = self._parse(
                    content,
                    valid_shot_ids,
                    visible_characters_by_shot=visible_characters_by_shot,
                )
                (output_dir / "character_asset_plan.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                return result
            except Exception as parse_error:
                if repair_attempt == 2:
                    raise
                repair_payload = dict(payload)
                repair_payload["temperature"] = 0.1
                repair_payload["messages"] = messages + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"输出无法使用：{parse_error}。请重新输出完整JSON。"
                            "只为多个镜头中确实重复出现、需要身份一致性的角色创建资产；"
                            f"used_by_shot_ids只能引用{valid_shot_ids}。"
                            "对planned_shots.visible_characters中重复出现的每个名称，必须创建且只创建"
                            "一个同名资产，并穷举它出现的全部shot_id；"
                            "参考图必须是单角色、单视图、纯视觉身份锚点，不得虚构史实长相。"
                        ),
                    },
                ]
                content = self.client.complete(repair_payload, label="character_asset_repair")
                suffix = "" if repair_attempt == 0 else f"_{repair_attempt + 1}"
                (output_dir / f"character_asset_agent_response_repaired{suffix}.txt").write_text(
                    content, encoding="utf-8"
                )
        raise RuntimeError("Character Asset Agent did not produce a usable plan")

    @staticmethod
    def _build_prompt(asset_context: dict[str, Any]) -> str:
        return f"""
请从锁定故事和逐句讲解的镜头语义中识别需要保持一致的重复人物，并生成参考图规范。输出格式严格为：
{{
  "character_assets":[{{
    "asset_id":"char_01",
    "name":"角色名称",
    "identity_group":"同一角色跨年龄版本共用的稳定组ID",
    "reference_mode":"fixed_identity",
    "canonical_description":"不冒充史实肖像的合成角色总描述",
    "face_anchor":"眉骨、脸型、鼻梁等少量稳定视觉锚点；身份需含混时写保持不可辨认",
    "hair_headwear":"发式与头冠",
    "wardrobe_palette":"服装轮廓、材质与主色",
    "body_silhouette":"体态和身形轮廓",
    "reference_prompt":"用于生成单角色、单视图参考图的完整prompt",
    "negative_prompt":"负面约束",
    "used_by_shot_ids":["s01"]
  }}],
  "continuity_links":[{{
    "from_asset_id":"a_young",
    "to_asset_id":"a_adult",
    "shared_identity_anchors":["稳定骨相"],
    "allowed_changes":["年龄纹理","头冠与服制"]
  }}]
}}

硬约束：
1. 只为确实会在多个镜头中重复出现、需要身份一致性的角色创建资产；路人、群像、一次性剪影不要建资产。planned_shots.visible_characters中同一名称出现至少两次时，必须创建且只创建一个name与该名称逐字一致的资产。asset_id从char_01开始连续编号。没有这类人物时character_assets和continuity_links都返回空数组。
2. reference_mode只能是{sorted(CHARACTER_REFERENCE_MODES)}。确定的重复角色用fixed_identity；同一角色年龄版本用age_variant；争议人物或匿名侧影用anonymous_silhouette。
3. 历史人物没有可靠肖像时，canonical_description必须明确这是为视频一致性设计的合成角色，不声称还原真实容貌。
4. reference_prompt只生成一个角色、一个三分之四视角或侧面视角、完整可辨的服饰轮廓、简洁中性背景。禁止三视图、拼图、分屏、多人群像、文字、logo和明星脸。
5. 同一identity_group的年龄变体共享face_anchor与基本身形，只允许年龄纹理、发冠、服制、泥尘和伤损等有动机的变化。
6. anonymous_silhouette不得补充性别、正脸和唯一身份；它用于维持轮廓、服饰时代和光线一致，而不是锁死解释。
7. used_by_shot_ids只能引用planned_shots中的shot_id，并必须穷举该角色出现在visible_characters中的全部镜头：可见就必须挂载，画外提及则不得挂载，不允许只选择代表镜头。negative_prompt必须合并视觉圣经中的时代禁忌。continuity_links只连接确属同一身份的资产；没有跨版本身份时可为空。
8. 只输出JSON。

Character asset context:
{json.dumps(asset_context, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _parse(
        content: str,
        valid_shot_ids: list[str],
        *,
        visible_characters_by_shot: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        data = ChatCompletionClient.extract_json(content)
        if set(data) != {"character_assets", "continuity_links"}:
            raise ValueError(f"Unexpected character asset fields: {sorted(data)}")
        assets = data.get("character_assets")
        links = data.get("continuity_links")
        if not isinstance(assets, list) or not isinstance(links, list):
            raise ValueError("character_assets and continuity_links must be arrays")
        if not assets and links:
            raise ValueError("continuity_links must be empty when there are no character assets")
        actual_ids = [str(item.get("asset_id") or "") for item in assets if isinstance(item, dict)]
        expected_asset_ids = [f"char_{index:02d}" for index in range(1, len(assets) + 1)]
        if actual_ids != expected_asset_ids:
            raise ValueError(f"Character asset ids must be sequential: {expected_asset_ids}")
        asset_fields = {
            "asset_id",
            "name",
            "identity_group",
            "reference_mode",
            "canonical_description",
            "face_anchor",
            "hair_headwear",
            "wardrobe_palette",
            "body_silhouette",
            "reference_prompt",
            "negative_prompt",
            "used_by_shot_ids",
        }
        visible_characters_by_shot = visible_characters_by_shot or {}
        required_shots_by_character: dict[str, list[str]] = {}
        for shot_id in valid_shot_ids:
            for character_name in visible_characters_by_shot.get(shot_id, []):
                required_shots_by_character.setdefault(str(character_name), []).append(shot_id)
        required_shots_by_character = {
            name: shot_ids
            for name, shot_ids in required_shots_by_character.items()
            if len(shot_ids) >= 2
        }
        assets_by_name: dict[str, list[dict[str, Any]]] = {}
        for item in assets:
            if not isinstance(item, dict) or set(item) != asset_fields:
                raise ValueError(f"Invalid character asset fields: {sorted(item)}")
            asset_id = str(item["asset_id"])
            asset_name = str(item.get("name") or "").strip()
            assets_by_name.setdefault(asset_name, []).append(item)
            if item.get("reference_mode") not in CHARACTER_REFERENCE_MODES:
                raise ValueError(f"Invalid character reference mode: {item.get('reference_mode')}")
            used_by = [str(value) for value in item.get("used_by_shot_ids", [])]
            if len(used_by) < 2 or len(used_by) != len(set(used_by)):
                raise ValueError(f"Character asset {asset_id} must be reused by at least two distinct shots")
            if any(shot_id not in valid_shot_ids for shot_id in used_by):
                raise ValueError(f"Character asset {asset_id} references an unknown shot")
            expected_shots = required_shots_by_character.get(asset_name)
            if expected_shots is not None and used_by != expected_shots:
                raise ValueError(
                    f"Character asset {asset_id} must cover every visible shot for {asset_name}: "
                    f"expected {expected_shots}, got {used_by}"
                )
            if spoken_character_count(str(item.get("name") or "")) < 1:
                raise ValueError(f"Character asset {asset_id}.name is empty")
            for field in asset_fields - {"asset_id", "name", "reference_mode", "used_by_shot_ids"}:
                if spoken_character_count(str(item.get(field) or "")) < 3:
                    raise ValueError(f"Character asset {asset_id}.{field} is too vague")
            reference_prompt = str(item.get("reference_prompt") or "")
            if any(phrase in reference_prompt for phrase in ("三视图", "多视图", "拼图", "分屏")):
                raise ValueError(f"Reference prompt must request one view: {asset_id}")

        for character_name, expected_shots in required_shots_by_character.items():
            matching_assets = assets_by_name.get(character_name, [])
            if len(matching_assets) != 1:
                raise ValueError(
                    f"Visible recurring character {character_name} needs exactly one same-name asset "
                    f"for shots {expected_shots}"
                )

        link_fields = {"from_asset_id", "to_asset_id", "shared_identity_anchors", "allowed_changes"}
        known_ids = set(expected_asset_ids)
        for link in links:
            if not isinstance(link, dict) or set(link) != link_fields:
                raise ValueError(f"Invalid character continuity link: {link}")
            if str(link.get("from_asset_id") or "") not in known_ids or str(
                link.get("to_asset_id") or ""
            ) not in known_ids:
                raise ValueError(f"Unknown asset in continuity link: {link}")
            if not link.get("shared_identity_anchors") or not link.get("allowed_changes"):
                raise ValueError(f"Continuity link lacks anchors or allowed changes: {link}")
        return data
