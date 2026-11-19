from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .common import (
    EDIT_RELATIONS,
    FORBIDDEN_MULTI_SCENE_PROMPT_PHRASES,
    VISUAL_MODES,
    ChatCompletionClient,
    expected_edit_relation,
    relaxed_keyword_validation,
    visual_context_for_agent,
)

MAX_VISUAL_SEQUENCE_SECONDS = 10.0


class VisualDirectorAgent:
    def __init__(self, client: ChatCompletionClient):
        self.client = client

    def plan(
        self,
        context: dict[str, Any],
        narrative_plan: dict[str, Any],
        output_dir: Path,
        story_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        shot_ids = [str(shot["shot_id"]) for shot in context["shots"]]
        allowed_asset_refs_by_shot = {
            str(shot["shot_id"]): [
                str(asset.get("asset_id") or "")
                for asset in shot.get("assets", [])
                if isinstance(asset, dict) and str(asset.get("asset_id") or "")
            ]
            for shot in context["shots"]
        }
        character_asset_ids = {
            str(item.get("asset_id") or "")
            for item in context.get("character_asset_plan", {}).get("character_assets", [])
            if isinstance(item, dict)
        }
        required_character_refs_by_shot = {
            shot_id: [asset_id for asset_id in asset_ids if asset_id in character_asset_ids]
            for shot_id, asset_ids in allowed_asset_refs_by_shot.items()
        }
        callback_specs = (context.get("interpretation", {}) or {}).get("visual_callbacks", [])
        require_callback = isinstance(callback_specs, list) and bool(callback_specs)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是古诗词短片的视觉导演和剪辑设计师。旁白已经锁定，你只规划视觉、回顾镜头和共享时间轴。"
                    "你不得改写旁白或史实，不加入评价型 critic。只输出严格 JSON。"
                ),
            },
            {"role": "user", "content": self._build_prompt(context, narrative_plan, story_plan)},
        ]
        payload = {
            "model": self.client.model,
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.9,
            # Expanded close-reading plans contain one visual contract per
            # subshot. Keep enough output budget for all segments and groups;
            # truncation here otherwise produces an invalid JSON tail before
            # media generation even begins.
            "max_tokens": 30000,
        }
        (output_dir / "visual_director_request.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        content = self.client.complete(payload, label="visual_director")
        (output_dir / "visual_director_response.txt").write_text(content, encoding="utf-8")
        result = None
        for repair_attempt in range(3):
            try:
                result = self._parse(
                    content,
                    shot_ids,
                    narrative_plan,
                    require_callback=require_callback,
                    allowed_asset_refs_by_shot=allowed_asset_refs_by_shot,
                    required_character_refs_by_shot=required_character_refs_by_shot,
                )
                break
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
                            f"输出无法使用：{parse_error}。请重新检查所有约束并输出完整 JSON。"
                            f"segments 的 shot_id 顺序严格为 {shot_ids}；"
                            "edit_groups 展开后也必须严格覆盖该顺序。"
                            "每个 video_prompt 必须是单场景单主运镜；每组 relation、旁白目标时长、"
                            "visual_sequence 秒数总和必须同时匹配。"
                        ),
                    },
                ]
                content = self.client.complete(repair_payload, label="visual_director_repair")
                repair_name = (
                    "visual_director_response_repaired.txt"
                    if repair_attempt == 0
                    else f"visual_director_response_repaired_{repair_attempt + 1}.txt"
                )
                (output_dir / repair_name).write_text(content, encoding="utf-8")
        if result is None:
            raise RuntimeError("Visual Director did not produce a usable plan")
        (output_dir / "visual_plan.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    def _build_prompt(
        self,
        context: dict[str, Any],
        narrative_plan: dict[str, Any],
        story_plan: dict[str, Any] | None = None,
    ) -> str:
        visual_context = visual_context_for_agent(context)
        narrative_by_id = {
            str(item.get("shot_id") or ""): item
            for item in narrative_plan.get("segments", [])
            if isinstance(item, dict)
        }
        locked_shot_contracts = []
        for shot in visual_context.get("shots", []):
            shot_id = str(shot.get("shot_id") or "")
            narrative = narrative_by_id.get(shot_id, {})
            locked_shot_contracts.append(
                {
                    "shot": shot,
                    "locked_event": {
                        key: narrative.get(key, "")
                        for key in (
                            "audio_text",
                            "narrative_phase",
                            "section_id",
                            "section_narration_anchor",
                            "story_function",
                            "state_before",
                            "event_text",
                            "state_after",
                            "tension_delta",
                            "narrative_move",
                            "visual_delegation",
                            "target_seconds",
                            "visual_target_seconds",
                            "knowledge_points",
                        )
                    },
                }
            )
        global_visual_context = dict(visual_context)
        global_visual_context.pop("shots", None)
        story_reference = story_plan or {}
        callback_policy = (
            "输入提供了visual_callbacks，至少实现一次与其setup/payoff一致的callback。"
            if (context.get("interpretation", {}) or {}).get("visual_callbacks")
            else "输入没有要求callback；仅在确有前后视觉母题时使用，不为满足形式强行添加。"
        )
        return f"""
请为锁定的旁白规划视频生成 prompt 和 edit_groups。

输出格式严格为：
{{
  "segments":[{{
    "shot_id":"s01",
    "depicts_event":"原样复制locked_event.event_text",
    "ends_at_state":"原样复制locked_event.state_after",
    "video_prompt":"...",
    "visual_role":"setup",
    "asset_refs":["a_character"]
  }}],
  "edit_groups":[{{
    "group_id":"g01",
    "source_shot_ids":["s01"],
    "narration_shot_ids":["s01"],
    "visual_sequence":[{{"shot_id":"s01","seconds":5.5,"mode":"primary","start_seconds":0}}],
    "relation":"one_to_one",
    "target_seconds":5.5,
    "transition":"..."
  }}]
}}

要求：
1. segments 的 shot_id 数量和顺序与输入 shots 完全一致；每个 video_prompt 只描述一个可独立生成的连续镜头。
2. depicts_event和ends_at_state必须逐字复制当前locked_event的event_text与state_after，用于程序核对。video_prompt必须先直译当前事件及其直接可见后果，不能提前拍相邻事件，也不能用旧模板动作代替；故事镜头从state_before建立可见处境并停在state_after。若抽象状态无法直接拍摄，用当前事件造成的动作、物件或空间变化表现，但不得跳到多年后的后果。
3. video_prompt 必须自包含时代、人物、地点、动作、景别、机位、单一主要运镜、光线和段尾状态。tension_delta结合narrative_rhetoric_mapping.rhythm_profiles决定空间收紧、转向、释放或衔接。禁止在一个生成 prompt 内要求闪回、多次切镜或复杂蒙太奇。
4. edit_groups 是最终共享时间轴。source_shot_ids 展开后必须严格等于输入 shot_id 顺序，每个 shot 只出现一次。
5. narration_shot_ids引用该组需要说出的旁白来源；每个非空旁白来源必须且只能出现一次。独立留白组使用空narration_shot_ids，source_shot_ids只能包含visual_only事件，relation必须为visual_only。
6. visual_sequence 可一对多、多对一或多对多。mode=primary 时 shot_id 应来自当前 source_shot_ids；mode=callback 时可引用更早已生成的 shot，并用 start_seconds 指定回顾片段起点。
   同一个edit_group内，visual_sequence的shot_id绝对不能重复；不能靠重复同一素材补齐旁白时长。
7. seconds 为单个视频素材在最终剪辑中的分配，范围 0.6 到 10 秒；超过8秒的分配仅用于允许8秒素材做有限慢放，不能用静帧补齐。visual_sequence 的 seconds 总和必须等于 target_seconds；target_seconds 必须等于本组旁白 target_seconds 总和，误差不超过 0.75 秒。
8. causal_story中，相同section_id的连续事件必须组成一个edit_group：source_shot_ids覆盖该section全部事件，narration_shot_ids只引用section_narration_anchor=true的首事件，visual_sequence把每个事件作为primary使用一次。因此一段完整旁白天然对应多个镜头。
   close_reading中，同一原始讲解段已经被展开为同一section_id下的多个视觉子镜头；必须把这些子镜头全部放在同一个edit_group，
   narration_shot_ids只引用第一个有旁白的anchor，visual_sequence把每个子镜头作为primary使用一次。
   close_reading默认禁止callback，不能把前一个精读镜头拿来填当前精读段的时长；只有输入明确要求visual_callbacks时才可回看。{callback_policy}
   callback不加入source_shot_ids，也不重复旁白。
9. narrative_move记录本段真实叙事功能，cinematic_rhetoric_mapping规定可用的视觉转译。需要回顾时可使用匹配剪辑、声音桥、重复物件或相同构图；不要把任何作品的主题线索设为全局默认。
10. 对任何claim_status=contested的解释保持开放；若争议对象不是人物，也要用物象、空间或结尾状态保留多种读法，不得套用固定人物剪影方案。
11. 只有输入语义确实包含发现、显现或认知反转时才使用delayed_reveal；正式揭示前不得提前暴露目标。其他作品不要强加发现结构。
12. asset_refs不是可选创作项：必须逐项复制当前shot.assets内的全部asset_id，不得选择、省略或增加。把每个资产的canonical_description、face_anchor、服饰和体态锚点自然写进video_prompt；年龄变体服从continuity_links。匿名争议人物只使用anonymous_silhouette，不补正脸、性别或唯一身份。
13. video_prompt中仍禁止“参考图、上图、上一镜”等依赖上下文的措辞；禁止字幕、题字、logo、水印、现代物品、与作品时代不符的服饰器物、明星脸和清晰正脸写真。书页、卷轴、碑刻只能作为不可读的墨迹纹理，不要求视频模型生成标题、诗句或其他可读文字。
14. relation 方向固定为“旁白来源数量 -> 视觉片段数量”：1段旁白配多个visual_sequence是one_to_many，多段旁白配1个视觉是many_to_one，两侧都多个是many_to_many；无旁白是visual_only。relation只能是{sorted(EDIT_RELATIONS)}；visual_sequence.mode只能是{sorted(VISUAL_MODES)}。
15. 只输出 JSON。

Global visual context（不含旧分镜动作）:
{json.dumps(global_visual_context, ensure_ascii=False, indent=2)}

Locked event contracts（每个镜头与自己的section和事件相邻提供，不能借用相邻事件）:
{json.dumps(locked_shot_contracts, ensure_ascii=False, indent=2)}

Detailed context story reference（只用于理解叙事连续性和状态变化，不得改写旁白）:
{json.dumps(story_reference, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _parse(
        content: str,
        expected_shot_ids: list[str],
        narrative_plan: dict[str, Any],
        *,
        require_callback: bool = False,
        allowed_asset_refs_by_shot: dict[str, list[str]] | None = None,
        required_character_refs_by_shot: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        data = ChatCompletionClient.extract_json(content)
        segments = data.get("segments")
        groups = data.get("edit_groups")
        if not isinstance(segments, list) or not isinstance(groups, list):
            raise ValueError("Missing segments or edit_groups array")
        actual_ids = [str(item.get("shot_id", "")) for item in segments if isinstance(item, dict)]
        if actual_ids != expected_shot_ids:
            raise ValueError(f"visual shot_id mismatch: expected {expected_shot_ids}, got {actual_ids}")
        narrative_by_id = {
            str(item["shot_id"]): item
            for item in narrative_plan.get("segments", [])
            if isinstance(item, dict)
        }
        causal_sources_by_section: dict[str, list[str]] = {}
        close_reading_sources_by_section: dict[str, list[str]] = {}
        for shot_id in expected_shot_ids:
            narrative = narrative_by_id.get(shot_id, {})
            if narrative.get("narrative_phase") != "causal_story":
                if narrative.get("narrative_phase") == "close_reading":
                    section_id = str(narrative.get("section_id") or "")
                    close_reading_sources_by_section.setdefault(section_id, []).append(shot_id)
                continue
            section_id = str(narrative.get("section_id") or "")
            causal_sources_by_section.setdefault(section_id, []).append(shot_id)
        for item in segments:
            if set(item) != {
                "shot_id",
                "depicts_event",
                "ends_at_state",
                "video_prompt",
                "visual_role",
                "asset_refs",
            }:
                raise ValueError(f"Unexpected visual fields for {item.get('shot_id')}: {sorted(item)}")
            shot_id = str(item.get("shot_id") or "")
            narrative = narrative_by_id.get(shot_id, {})
            if str(item.get("depicts_event") or "") != str(narrative.get("event_text") or ""):
                item["depicts_event"] = str(narrative.get("event_text") or "")
            if str(item.get("ends_at_state") or "") != str(narrative.get("state_after") or ""):
                item["ends_at_state"] = str(narrative.get("state_after") or "")
            asset_refs = item.get("asset_refs")
            if not isinstance(asset_refs, list) or any(not str(value) for value in asset_refs):
                raise ValueError(f"asset_refs must be an array for {shot_id}")
            if len(asset_refs) != len(set(str(value) for value in asset_refs)):
                raise ValueError(f"Duplicate asset_refs for {shot_id}")
            if allowed_asset_refs_by_shot is not None:
                allowed_refs = set(allowed_asset_refs_by_shot.get(shot_id, []))
                unknown_refs = [str(value) for value in asset_refs if str(value) not in allowed_refs]
                if unknown_refs:
                    raise ValueError(f"Unknown asset_refs for {shot_id}: {unknown_refs}")
            if required_character_refs_by_shot is not None:
                character_refs = set(required_character_refs_by_shot.get(shot_id, []))
                missing_refs = character_refs.difference(str(value) for value in asset_refs)
                if missing_refs:
                    raise ValueError(
                        f"Shot {shot_id} must reference every assigned character asset: "
                        f"missing {sorted(missing_refs)}"
                    )
            prompt = str(item.get("video_prompt") or "").strip()
            if not prompt:
                raise ValueError(f"Empty video_prompt: {item.get('shot_id')}")
            if any(phrase in prompt for phrase in ("参考图", "上图", "上一镜")) and not relaxed_keyword_validation():
                raise ValueError(f"Reference wording is not allowed: {item.get('shot_id')}")
            if re.search(r"(?:露出|呈现|显示|写有).{0,24}(?:诗句|词句|字句|标题|篇名|文字)", prompt) and not relaxed_keyword_validation():
                raise ValueError(f"Readable generated text is not allowed: {item.get('shot_id')}")
            if any(phrase in prompt for phrase in FORBIDDEN_MULTI_SCENE_PROMPT_PHRASES) and not relaxed_keyword_validation():
                raise ValueError(f"Multi-scene editing leaked into video prompt: {item.get('shot_id')}")

        flattened_sources: list[str] = []
        used_narration: list[str] = []
        non_one_to_one = 0
        callback_count = 0
        seen_source_ids: set[str] = set()
        seen_group_ids: set[str] = set()
        required_group_fields = {
            "group_id",
            "source_shot_ids",
            "narration_shot_ids",
            "visual_sequence",
            "relation",
            "target_seconds",
            "transition",
        }
        for group in groups:
            if set(group) != required_group_fields:
                raise ValueError(f"Unexpected edit_group fields for {group.get('group_id')}: {sorted(group)}")
            group_id = str(group.get("group_id") or "").strip()
            if not group_id or group_id in seen_group_ids:
                raise ValueError(f"Missing or duplicate edit group id: {group_id}")
            seen_group_ids.add(group_id)
            source_ids = [str(value) for value in group.get("source_shot_ids", [])]
            narration_ids = [str(value) for value in group.get("narration_shot_ids", [])]
            visual_sequence = group.get("visual_sequence")
            if not source_ids or not isinstance(visual_sequence, list) or not visual_sequence:
                raise ValueError(f"Empty mapping in edit group {group.get('group_id')}")
            flattened_sources.extend(source_ids)
            relation = expected_edit_relation(len(narration_ids), len(visual_sequence))
            group["relation"] = relation
            if relation != "one_to_one":
                non_one_to_one += 1
            target_seconds = float(group.get("target_seconds") or 0)
            causal_section_ids = {
                str(narrative_by_id.get(shot_id, {}).get("section_id") or "")
                for shot_id in source_ids
                if narrative_by_id.get(shot_id, {}).get("narrative_phase") == "causal_story"
            }
            if causal_section_ids:
                if len(causal_section_ids) != 1 or any(
                    narrative_by_id.get(shot_id, {}).get("narrative_phase") != "causal_story"
                    for shot_id in source_ids
                ):
                    raise ValueError(f"Causal story section is split or mixed in {group_id}")
                section_id = next(iter(causal_section_ids))
                expected_sources = causal_sources_by_section[section_id]
                if source_ids != expected_sources:
                    raise ValueError(
                        f"Causal story section {section_id} must stay in one edit group: {expected_sources}"
                    )
                expected_anchor_ids = [
                    shot_id
                    for shot_id in expected_sources
                    if narrative_by_id[shot_id].get("section_narration_anchor")
                ]
                if narration_ids != expected_anchor_ids:
                    raise ValueError(
                        f"Causal story section {section_id} narration must use {expected_anchor_ids}"
                    )
            close_section_ids = {
                str(narrative_by_id.get(shot_id, {}).get("section_id") or "")
                for shot_id in source_ids
                if narrative_by_id.get(shot_id, {}).get("narrative_phase") == "close_reading"
            }
            if close_section_ids:
                if len(close_section_ids) != 1 or any(
                    narrative_by_id.get(shot_id, {}).get("narrative_phase") != "close_reading"
                    for shot_id in source_ids
                ):
                    raise ValueError(f"Close-reading section is split or mixed in {group_id}")
                close_section_id = next(iter(close_section_ids))
                expected_sources = close_reading_sources_by_section[close_section_id]
                if source_ids != expected_sources:
                    raise ValueError(
                        f"Close-reading section {close_section_id} must stay in one edit group: "
                        f"{expected_sources}"
                    )
                expected_anchor_ids = [
                    shot_id
                    for shot_id in expected_sources
                    if narrative_by_id[shot_id].get("section_narration_anchor")
                ]
                if narration_ids != expected_anchor_ids:
                    raise ValueError(
                        f"Close-reading section {close_section_id} narration must use "
                        f"{expected_anchor_ids}"
                    )
                if not require_callback and any(
                    str(entry.get("mode") or "") == "callback"
                    for entry in visual_sequence
                ):
                    raise ValueError(
                        f"Close-reading group {group_id} must not use callback visuals"
                    )
            narration_target = 0.0
            if narration_ids:
                for narration_id in narration_ids:
                    if narration_id not in narrative_by_id:
                        raise ValueError(f"Unknown narration shot_id in {group.get('group_id')}: {narration_id}")
                    if not str(narrative_by_id[narration_id].get("audio_text") or "").strip():
                        raise ValueError(f"visual-only event used as narration in {group.get('group_id')}: {narration_id}")
                    narration_target += float(narrative_by_id[narration_id].get("target_seconds") or 0)
            else:
                non_visual_only_sources = [
                    shot_id
                    for shot_id in source_ids
                    if str(narrative_by_id.get(shot_id, {}).get("audio_text") or "").strip()
                ]
                if non_visual_only_sources:
                    raise ValueError(
                        f"Silent group contains narrated sources in {group.get('group_id')}: {non_visual_only_sources}"
                    )
                narration_target = sum(
                    float(narrative_by_id.get(shot_id, {}).get("target_seconds") or 0)
                    for shot_id in source_ids
                )
            if not 2 <= target_seconds <= 45:
                raise ValueError(f"target_seconds out of range in {group.get('group_id')}: {target_seconds}")
            if abs(target_seconds - narration_target) > 0.75:
                raise ValueError(
                    f"Target mismatch in {group.get('group_id')}: group={target_seconds}, narration={narration_target}"
                )
            for narration_id in narration_ids:
                used_narration.append(narration_id)
            has_primary = False
            primary_ids: list[str] = []
            visual_ids: list[str] = []
            visual_seconds_total = 0.0
            for entry in visual_sequence:
                if set(entry) != {"shot_id", "seconds", "mode", "start_seconds"}:
                    raise ValueError(f"Unexpected visual_sequence fields in {group.get('group_id')}: {sorted(entry)}")
                visual_id = str(entry.get("shot_id") or "")
                visual_ids.append(visual_id)
                mode = str(entry.get("mode") or "")
                seconds = float(entry.get("seconds") or 0)
                start_seconds = float(entry.get("start_seconds") or 0)
                if visual_id not in expected_shot_ids:
                    raise ValueError(f"Unknown visual shot_id in {group.get('group_id')}: {visual_id}")
                if mode not in VISUAL_MODES:
                    raise ValueError(f"Invalid visual mode in {group.get('group_id')}: {mode}")
                if not 0.6 <= seconds <= MAX_VISUAL_SEQUENCE_SECONDS:
                    raise ValueError(f"Visual seconds out of range in {group.get('group_id')}: {seconds}")
                if not 0 <= start_seconds < 8:
                    raise ValueError(f"start_seconds out of range in {group.get('group_id')}: {start_seconds}")
                visual_seconds_total += seconds
                if mode == "primary":
                    if visual_id not in source_ids:
                        raise ValueError(f"Primary visual must belong to source_shot_ids: {visual_id}")
                    has_primary = True
                    primary_ids.append(visual_id)
                else:
                    if visual_id not in seen_source_ids:
                        raise ValueError(f"Callback must reference an earlier shot: {visual_id}")
                    callback_count += 1
            if not has_primary:
                raise ValueError(f"Edit group has no primary visual: {group.get('group_id')}")
            if len(visual_ids) != len(set(visual_ids)):
                raise ValueError(
                    f"An edit group cannot repeat the same visual shot: {group.get('group_id')}"
                )
            if (causal_section_ids or close_section_ids) and primary_ids != source_ids:
                raise ValueError(
                    f"Every source event must be used once as a primary visual in {group_id}"
                )
            if abs(visual_seconds_total - target_seconds) > 0.35:
                raise ValueError(
                    f"Visual allocation mismatch in {group.get('group_id')}: "
                    f"visual={visual_seconds_total}, target={target_seconds}"
                )
            seen_source_ids.update(source_ids)

        if flattened_sources != expected_shot_ids:
            raise ValueError(f"edit_groups do not cover source shots in order: {flattened_sources}")
        expected_narration = [
            shot_id
            for shot_id in expected_shot_ids
            if str(narrative_by_id.get(shot_id, {}).get("audio_text") or "").strip()
        ]
        if sorted(used_narration) != sorted(expected_narration) or len(used_narration) != len(set(used_narration)):
            raise ValueError(
                f"Narration sections must be used exactly once: expected={expected_narration}, used={used_narration}"
            )
        if require_callback and callback_count == 0:
            raise ValueError("The graph defines visual_callbacks but the plan contains no callback visual")
        return {"segments": segments, "edit_groups": groups}
