from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .character_asset_agent import CharacterAssetAgent
from .close_reading_agent import CloseReadingAgent
from .common import ChatCompletionClient, build_execution_context, compact_graph_for_agents
from .context_story_agent import ContextStoryAgent
from .story_narration_agent import StoryNarrationAgent
from .visual_director_agent import VisualDirectorAgent


def align_character_assets_to_visual_prompts(
    character_asset_plan: dict[str, Any],
    narrative_plan: dict[str, Any],
    visual_plan: dict[str, Any],
) -> None:
    """Propagate validated asset IDs from the visual plan.

    ``asset_refs`` are stable IDs, not character names.  The visual prompt is
    deliberately not parsed here: prompts may use any language, aliases, or
    descriptions, and recovering identity from prompt text is both brittle
    and unnecessary.  If an older visual plan omitted the IDs, fall back to
    the character agent's per-shot assignment.
    """
    assets = [
        item
        for item in character_asset_plan.get("character_assets", [])
        if isinstance(item, dict)
    ]
    asset_by_id = {
        str(item.get("asset_id") or ""): item
        for item in assets
        if str(item.get("asset_id") or "")
    }
    declared_refs_by_shot: dict[str, list[str]] = {}
    for asset in assets:
        asset_id = str(asset.get("asset_id") or "")
        if not asset_id:
            continue
        for shot_id in asset.get("used_by_shot_ids", []):
            declared_refs_by_shot.setdefault(str(shot_id), []).append(asset_id)
    used_by_asset: dict[str, list[str]] = {
        str(item.get("asset_id") or ""): [] for item in assets
    }
    visible_names_by_shot: dict[str, list[str]] = {}

    for segment in visual_plan.get("segments", []):
        if not isinstance(segment, dict):
            continue
        shot_id = str(segment.get("shot_id") or "")
        raw_refs = segment.get("asset_refs", [])
        if not isinstance(raw_refs, list):
            raise ValueError(f"asset_refs must be a list for visual shot {shot_id}")
        unknown_refs = [
            str(value)
            for value in raw_refs
            if str(value) not in asset_by_id
        ]
        if unknown_refs:
            raise ValueError(f"Unknown asset_refs for visual shot {shot_id}: {unknown_refs}")
        existing_refs = list(dict.fromkeys(str(value) for value in raw_refs))
        # The Visual Director's IDs are authoritative.  For legacy plans that
        # left the field empty, recover the assignment already produced by the
        # Character Asset Agent; never infer identity from free-form prompts.
        selected_refs = existing_refs or list(declared_refs_by_shot.get(shot_id, []))
        selected_refs = list(dict.fromkeys(selected_refs))
        segment["asset_refs"] = selected_refs
        visible_names_by_shot[shot_id] = [
            str(asset_by_id[asset_id].get("name") or "")
            for asset_id in selected_refs
            if asset_id in asset_by_id
            and asset_by_id[asset_id].get("reference_mode") != "anonymous_silhouette"
        ]
        for asset_id in selected_refs:
            if asset_id in used_by_asset:
                used_by_asset[asset_id].append(shot_id)

    for asset in assets:
        asset["used_by_shot_ids"] = used_by_asset[str(asset.get("asset_id") or "")]
    for segment in narrative_plan.get("segments", []):
        if isinstance(segment, dict):
            segment["visible_characters"] = visible_names_by_shot.get(
                str(segment.get("shot_id") or ""), []
            )


class PoetryDirectorGraphRealizer:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: int = 600, max_tokens: int | None = None):
        client = ChatCompletionClient(base_url, api_key, model, timeout_seconds, max_tokens=max_tokens)
        self.client = client
        self.story_agent = ContextStoryAgent(client)
        self.character_asset_agent = CharacterAssetAgent(client)
        self.story_narration_agent = StoryNarrationAgent(client)
        self.close_reading_agent = CloseReadingAgent(client)
        self.visual_director = VisualDirectorAgent(client)

    def realize(self, graph: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        context = compact_graph_for_agents(graph)
        story_plan = self.story_agent.plan(context, output_dir)
        story_narration_plan = self.story_narration_agent.plan(context, story_plan, output_dir)
        close_reading_plan = self.close_reading_agent.plan(context, story_plan, output_dir)
        poem_card_narration = close_reading_plan["poem_card_narration"]
        if not isinstance(poem_card_narration, dict) or not str(
            poem_card_narration.get("audio_text") or ""
        ).strip():
            raise ValueError("CloseReadingAgent must return poem_card_narration.audio_text")
        narrative_segments: list[dict[str, Any]] = []
        narrative_tension = {
            "setup": "hold",
            "escalation": "rise",
            "decision": "rise",
            "reversal": "turn",
            "consequence": "release",
            "reflection": "hold",
            "bridge": "bridge",
        }
        last_story_shot_id = ""
        for section in story_narration_plan["sections"]:
            claim_status = str(section.get("claim_status") or "")
            if section.get("story_function") == "bridge_to_poem" or claim_status in {
                "contested",
                "interpretive",
            }:
                narration_role = "uncertainty"
            elif claim_status == "documented":
                narration_role = "historical_evidence"
            else:
                narration_role = "scene_action"
            for event_index, event in enumerate(section["event_units"]):
                shot_id = str(event["shot_id"])
                last_story_shot_id = shot_id
                narrative_segments.append(
                    {
                        "shot_id": shot_id,
                        "template_shot_id": shot_id,
                        "section_id": section["section_id"],
                        "section_narration_anchor": event_index == 0,
                        "audio_text": section["audio_text"] if event_index == 0 else "",
                        "narration_role": narration_role if event_index == 0 else "visual_only",
                        "visual_delegation": section["implicit_meaning"],
                        "target_seconds": section["target_seconds"] if event_index == 0 else 0,
                        "visual_target_seconds": event["target_seconds"],
                        "narrative_phase": "causal_story",
                        "story_function": section["story_function"],
                        "state_before": event["starting_condition"],
                        "event_text": event["event_text"],
                        "state_after": event["observable_result"],
                        "visible_characters": list(event["visible_characters"]),
                        "tension_delta": narrative_tension[section["narrative_move"]],
                        "narrative_move": section["narrative_move"],
                        "line_ids": [],
                        "reading_action": "",
                        "knowledge_points": [],
                        "context_node_ids": section["source_context_node_ids"],
                    }
                )
        for item in close_reading_plan["segments"]:
            narrative_segments.extend(
                self._expand_close_reading_segment(item)
            )
        narrative_plan = {"segments": narrative_segments}
        (output_dir / "narrative_plan.json").write_text(
            json.dumps(narrative_plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        character_asset_plan = self.character_asset_agent.plan(
            context, story_plan, narrative_plan, output_dir
        )
        execution_context = build_execution_context(
            context, narrative_plan, character_asset_plan
        )
        visual_plan = self.visual_director.plan(execution_context, narrative_plan, output_dir, story_plan)
        align_character_assets_to_visual_prompts(
            character_asset_plan,
            narrative_plan,
            visual_plan,
        )
        (output_dir / "character_asset_plan.json").write_text(
            json.dumps(character_asset_plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "narrative_plan.json").write_text(
            json.dumps(narrative_plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "visual_plan.json").write_text(
            json.dumps(visual_plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        narrative_by_id = {str(item["shot_id"]): item for item in narrative_plan["segments"]}
        visual_by_id = {str(item["shot_id"]): item for item in visual_plan["segments"]}
        shot_ids = [str(shot["shot_id"]) for shot in execution_context["shots"]]
        result = {
            "planning_mode": "context_story_character_assets_narration_close_reading_card_visual_agents",
            "character_asset_plan": character_asset_plan,
            "poem_card_insert_after_shot_id": last_story_shot_id,
            "poem_card_narration": poem_card_narration,
            "segments": [
                {
                    "shot_id": shot_id,
                    "template_shot_id": str(narrative_by_id[shot_id]["template_shot_id"]),
                    "section_id": str(narrative_by_id[shot_id]["section_id"]),
                    "section_narration_anchor": bool(
                        narrative_by_id[shot_id]["section_narration_anchor"]
                    ),
                    "audio_text": str(narrative_by_id[shot_id]["audio_text"]),
                    "video_prompt": str(visual_by_id[shot_id]["video_prompt"]),
                    "narration_role": narrative_by_id[shot_id]["narration_role"],
                    "visual_delegation": narrative_by_id[shot_id]["visual_delegation"],
                    "narration_target_seconds": narrative_by_id[shot_id]["target_seconds"],
                    "visual_target_seconds": narrative_by_id[shot_id]["visual_target_seconds"],
                    "narrative_phase": narrative_by_id[shot_id]["narrative_phase"],
                    "story_function": narrative_by_id[shot_id]["story_function"],
                    "state_before": narrative_by_id[shot_id]["state_before"],
                    "event_text": narrative_by_id[shot_id]["event_text"],
                    "state_after": narrative_by_id[shot_id]["state_after"],
                    "visible_characters": narrative_by_id[shot_id]["visible_characters"],
                    "tension_delta": narrative_by_id[shot_id]["tension_delta"],
                    "narrative_move": narrative_by_id[shot_id]["narrative_move"],
                    "line_ids": narrative_by_id[shot_id]["line_ids"],
                    "reading_action": narrative_by_id[shot_id]["reading_action"],
                    "knowledge_points": narrative_by_id[shot_id]["knowledge_points"],
                    "visual_role": visual_by_id[shot_id]["visual_role"],
                    "asset_refs": visual_by_id[shot_id]["asset_refs"],
                }
                for shot_id in shot_ids
            ],
            "edit_groups": visual_plan["edit_groups"],
        }
        (output_dir / "realized_text_plan.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        usage = self.client.usage_records
        totals = {
            key: sum(int(item.get("usage", {}).get(key) or 0) for item in usage)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        (output_dir / "llm_usage.json").write_text(
            json.dumps(
                {"model": self.client.model, "calls": usage, "totals": totals},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    @staticmethod
    def _expand_close_reading_segment(item: dict[str, Any]) -> list[dict[str, Any]]:
        """Create distinct visual subshots without changing the narration.

        A close-reading narration can naturally take 12–20 seconds, while a
        remote video request should usually produce one short clip. The old
        fallback solved this by repeating the same shot. Instead, keep one
        narration anchor and create several visual-only subshots, each with a
        different teaching focus. Visual Director then generates one prompt
        and one primary clip for each subshot.
        """
        target_seconds = float(item.get("target_seconds") or 0)
        # Keep each planned subshot conservative because measured TTS duration
        # can be longer than the text estimate.
        subshot_count = max(1, min(4, math.ceil(target_seconds / 6.5)))
        if subshot_count == 1:
            subshot_count = 1

        focuses = [
            "先呈现原文中的具体人物、物象、声音或动作",
            "再呈现本段关键字词、器物或局部动作的变化",
            "继续呈现由本段带出的空间、节奏或感官推进",
            "最后呈现本段落到下一处的情绪或结构作用",
        ]
        base_id = str(item["shot_id"])
        section_id = f"reading_{base_id}"
        visual_seconds = target_seconds / subshot_count if subshot_count else target_seconds
        result: list[dict[str, Any]] = []
        for index in range(subshot_count):
            focus = focuses[min(index, len(focuses) - 1)]
            is_anchor = index == 0
            shot_id = base_id if subshot_count == 1 else f"{base_id}_v{index + 1:02d}"
            result.append(
                {
                    "shot_id": shot_id,
                    "template_shot_id": shot_id,
                    "section_id": section_id,
                    "section_narration_anchor": is_anchor,
                    "audio_text": item["audio_text"] if is_anchor else "",
                    "narration_role": item["narration_role"] if is_anchor else "visual_only",
                    "visual_delegation": f"{item['unstated_visual_meaning']}；视觉重点：{focus}",
                    "target_seconds": item["target_seconds"] if is_anchor else 0,
                    "visual_target_seconds": round(visual_seconds, 3),
                    "narrative_phase": "close_reading",
                    "story_function": "",
                    "state_before": "",
                    "event_text": f"{item['teaching_move']}；视觉重点：{focus}",
                    "state_after": f"{item['unstated_visual_meaning']}；本段推进：{focus}",
                    "visible_characters": [],
                    "tension_delta": "hold",
                    "narrative_move": "",
                    "line_ids": item["line_ids"],
                    "reading_action": item["reading_action"] if is_anchor else "visual_only",
                    "knowledge_points": item["knowledge_points"] if is_anchor else [],
                    "context_node_ids": [],
                }
            )
        return result
