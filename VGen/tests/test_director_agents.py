from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


VGEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VGEN_ROOT))

from vgen.director_agents import (
    CharacterAssetAgent,
    CloseReadingAgent,
    ContextStoryAgent,
    StoryNarrationAgent,
    VisualDirectorAgent,
    compact_graph_for_agents,
    build_execution_context,
    unsupported_quantitative_details,
    visual_context_for_agent,
)
from vgen.director_graph_pipeline import build_group_audio_segments, edit_group_statistics, graph_to_segments
from vgen.poetry_teaching import SegmentPlan
from vgen.director.story_narration_agent import allocate_bounded_event_durations
from vgen.director.realizer import align_character_assets_to_visual_prompts


class SplitDirectorAgentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shot_ids = ["s01", "s02", "s03", "s04"]
        self.narrative_plan = {
            "segments": [
                {
                    "shot_id": "s01",
                    "audio_text": "他从书案起身，走进北方的风雪。",
                    "narration_role": "scene_action",
                    "visual_delegation": "让身后聚拢的人群承担选择的重量",
                    "event_text": "人物离开书案走入风雪",
                    "state_after": "人物已经进入风雪中的行动",
                    "target_seconds": 3,
                },
                {
                    "shot_id": "s02",
                    "audio_text": "一次选择，后来成了他一生没有放下的方向。",
                    "narration_role": "delayed_inference",
                    "visual_delegation": "用相同背影构图连接青年与中年",
                    "event_text": "早先选择成为长期方向",
                    "state_after": "行动方向跨越时间仍然持续",
                    "target_seconds": 3,
                },
                {
                    "shot_id": "s03",
                    "audio_text": "",
                    "narration_role": "visual_only",
                    "visual_delegation": "只保留关闭宫门和案头奏议",
                    "event_text": "宫门关闭而奏议留在案头",
                    "state_after": "行动被阻挡在关闭的门外",
                    "target_seconds": 3,
                },
                {
                    "shot_id": "s04",
                    "audio_text": "这样的背影，对他并不陌生。",
                    "narration_role": "delayed_inference",
                    "visual_delegation": "回看早先的奏议，再回到灯火边缘",
                    "event_text": "当下背影唤回先前受阻的处境",
                    "state_after": "过去和当下建立可见联系",
                    "target_seconds": 3,
                },
            ]
        }

    def _visual_segments(self) -> list[dict[str, object]]:
        narrative_by_id = {
            item["shot_id"]: item for item in self.narrative_plan["segments"]
        }
        return [
            {
                "shot_id": shot_id,
                "depicts_event": narrative_by_id[shot_id]["event_text"],
                "ends_at_state": narrative_by_id[shot_id]["state_after"],
                "video_prompt": f"历史写实连续镜头 {shot_id}",
                "visual_role": "setup",
                "asset_refs": [],
            }
            for shot_id in self.shot_ids
        ]

    def test_visual_context_removes_draft_prompt_transition_and_camera_path(self) -> None:
        graph = {
            "assets": [],
            "teaching_script": {"beats": []},
            "shots": [
                {
                    "shot_id": "s01",
                    "order": 1,
                    "video_prompt": "匹配切到另一场景",
                    "transition_to_next": "叠化",
                    "camera": {
                        "shot_size": "medium",
                        "movement": "pan",
                        "path": "分三段扫过人群",
                    },
                    "script_beat_ids": [],
                }
            ],
        }
        compact = compact_graph_for_agents(graph)
        visual_context = visual_context_for_agent(compact)
        shot = visual_context["shots"][0]
        self.assertNotIn("video_prompt_draft", shot)
        self.assertNotIn("transition_to_next", shot)
        self.assertNotIn("path", shot["camera_options"])
        self.assertEqual(shot["camera_options"]["movement"], "pan")

    def test_event_duration_allocation_caps_dominant_weight_without_losing_time(self) -> None:
        durations = allocate_bounded_event_durations(20.0, [8.0, 1.0, 1.0])

        self.assertEqual(durations, [8.0, 6.0, 6.0])
        self.assertAlmostEqual(sum(durations), 20.0)
        self.assertTrue(all(0.6 <= value <= 8.0 for value in durations))

    def test_final_visual_prompts_remove_offscreen_character_references(self) -> None:
        character_plan = {
            "character_assets": [
                {
                    "asset_id": "char_01",
                    "name": "李白",
                    "reference_mode": "fixed_identity",
                    "used_by_shot_ids": ["s01", "s02", "s03"],
                },
                {
                    "asset_id": "char_02",
                    "name": "唐玄宗",
                    "reference_mode": "fixed_identity",
                    "used_by_shot_ids": ["s01", "s03"],
                },
            ],
            "continuity_links": [],
        }
        narrative_plan = {
            "segments": [
                {"shot_id": shot_id, "visible_characters": ["李白", "唐玄宗"]}
                for shot_id in ("s01", "s02", "s03")
            ]
        }
        visual_plan = {
            "segments": [
                {"shot_id": "s01", "video_prompt": "李白接过诏书", "asset_refs": ["char_01", "char_02"]},
                {"shot_id": "s02", "video_prompt": "几名官员在廊下议论", "asset_refs": ["char_01"]},
                {
                    "shot_id": "s03",
                    "video_prompt": "李白在殿内拜见唐玄宗",
                    "asset_refs": ["char_01", "char_02"],
                },
            ]
        }

        align_character_assets_to_visual_prompts(character_plan, narrative_plan, visual_plan)

        self.assertEqual(visual_plan["segments"][0]["asset_refs"], ["char_01"])
        self.assertEqual(visual_plan["segments"][1]["asset_refs"], [])
        self.assertEqual(visual_plan["segments"][2]["asset_refs"], ["char_01", "char_02"])
        self.assertEqual(character_plan["character_assets"][0]["used_by_shot_ids"], ["s01", "s03"])
        self.assertEqual(narrative_plan["segments"][1]["visible_characters"], [])

    def test_character_asset_plan_locks_age_variants(self) -> None:
        plan = {
            "character_assets": [
                {
                    "asset_id": "char_01",
                    "name": "青年词人",
                    "identity_group": "poet_identity",
                    "reference_mode": "fixed_identity",
                    "canonical_description": "为视频一致性设计的青年合成角色，不声称复原历史容貌",
                    "face_anchor": "平直眉骨、窄长脸型和清晰鼻梁",
                    "hair_headwear": "束发，不戴现代头饰",
                    "wardrobe_palette": "青灰短袍与低饱和轻甲",
                    "body_silhouette": "清瘦挺直、行动利落",
                    "reference_prompt": "单个青年古代人物，三分之四侧面，全身，青灰短袍，中性背景",
                    "negative_prompt": "明星脸、现代发型、多人、文字、拼图",
                    "used_by_shot_ids": ["s01", "s02"],
                },
                {
                    "asset_id": "char_02",
                    "name": "中年词人",
                    "identity_group": "poet_identity",
                    "reference_mode": "age_variant",
                    "canonical_description": "为视频一致性设计的中年合成角色，不声称复原历史容貌",
                    "face_anchor": "保持平直眉骨、窄长脸型和清晰鼻梁",
                    "hair_headwear": "正式发冠与整齐束发",
                    "wardrobe_palette": "青灰南宋常服与低饱和织物",
                    "body_silhouette": "保持清瘦挺直，动作更克制",
                    "reference_prompt": "单个中年古代人物，三分之四侧面，全身，青灰常服，中性背景",
                    "negative_prompt": "明星脸、清代官服、多人、文字、拼图",
                    "used_by_shot_ids": ["s03", "s04"],
                },
            ],
            "continuity_links": [
                {
                    "from_asset_id": "char_01",
                    "to_asset_id": "char_02",
                    "shared_identity_anchors": ["眉骨", "脸型", "鼻梁"],
                    "allowed_changes": ["年龄纹理", "头冠", "服制"],
                }
            ],
        }
        parsed = CharacterAssetAgent._parse(
            json.dumps(plan, ensure_ascii=False),
            ["s01", "s02", "s03", "s04"],
            visible_characters_by_shot={
                "s01": ["青年词人"],
                "s02": ["青年词人"],
                "s03": ["中年词人"],
                "s04": ["中年词人"],
            },
        )
        self.assertEqual(parsed["character_assets"][1]["reference_mode"], "age_variant")
        self.assertEqual(parsed["continuity_links"][0]["to_asset_id"], "char_02")

        incomplete = json.loads(json.dumps(plan, ensure_ascii=False))
        incomplete["character_assets"][0]["used_by_shot_ids"] = ["s01", "s03"]
        with self.assertRaisesRegex(ValueError, "cover every visible shot"):
            CharacterAssetAgent._parse(
                json.dumps(incomplete, ensure_ascii=False),
                ["s01", "s02", "s03", "s04"],
                visible_characters_by_shot={
                    "s01": ["青年词人"],
                    "s02": ["青年词人"],
                    "s03": ["中年词人"],
                    "s04": ["中年词人"],
                },
            )

    def test_character_asset_plan_requires_every_recurring_visible_character(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs exactly one same-name asset"):
            CharacterAssetAgent._parse(
                '{"character_assets":[],"continuity_links":[]}',
                ["s01", "s02"],
                visible_characters_by_shot={"s01": ["李白"], "s02": ["李白"]},
            )

    def test_media_plan_preserves_full_character_coverage_and_human_labels(self) -> None:
        asset = {
            "asset_id": "char_01",
            "name": "李白",
            "used_by_shot_ids": ["s01", "s02"],
        }
        realized = {
            "character_asset_plan": {"character_assets": [asset], "continuity_links": []},
            "segments": [
                {
                    "shot_id": shot_id,
                    "video_prompt": f"李白出现在{shot_id}",
                    "audio_text": "旁白" if index == 1 else "",
                    "visible_characters": ["李白"],
                    "asset_refs": ["char_01"],
                    "narration_target_seconds": 4,
                    "visual_target_seconds": 4,
                }
                for index, shot_id in enumerate(("s01", "s02"), start=1)
            ],
        }

        segments = graph_to_segments({"shots": [], "assets": []}, realized)

        self.assertEqual(segments[0].reference_asset_labels, ["李白"])
        self.assertEqual(segments[1].asset_refs, ["char_01"])
        realized["segments"][1]["asset_refs"] = []
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            graph_to_segments({"shots": [], "assets": []}, realized)

    def test_visual_plan_supports_many_to_one_and_callback(self) -> None:
        visual_plan = {
            "segments": self._visual_segments(),
            "edit_groups": [
                {
                    "group_id": "g01",
                    "source_shot_ids": ["s01", "s02"],
                    "narration_shot_ids": ["s01", "s02"],
                    "visual_sequence": [
                        {"shot_id": "s01", "seconds": 6, "mode": "primary", "start_seconds": 0}
                    ],
                    "relation": "many_to_one",
                    "target_seconds": 6,
                    "transition": "背影匹配",
                },
                {
                    "group_id": "g02",
                    "source_shot_ids": ["s03", "s04"],
                    "narration_shot_ids": ["s04"],
                    "visual_sequence": [
                        {"shot_id": "s04", "seconds": 2.2, "mode": "primary", "start_seconds": 0},
                        {"shot_id": "s01", "seconds": 0.8, "mode": "callback", "start_seconds": 1.2},
                    ],
                    "relation": "one_to_many",
                    "target_seconds": 3,
                    "transition": "短闪回后回到现实",
                },
            ],
        }
        parsed = VisualDirectorAgent._parse(
            json.dumps(visual_plan, ensure_ascii=False), self.shot_ids, self.narrative_plan
        )
        self.assertEqual(parsed["edit_groups"][0]["relation"], "many_to_one")
        self.assertEqual(parsed["edit_groups"][1]["visual_sequence"][1]["mode"], "callback")

    def test_visual_plan_requires_all_assigned_character_assets(self) -> None:
        visual_plan = {
            "segments": self._visual_segments(),
            "edit_groups": [
                {
                    "group_id": "g01",
                    "source_shot_ids": self.shot_ids,
                    "narration_shot_ids": self.shot_ids,
                    "visual_sequence": [
                        {"shot_id": shot_id, "seconds": 3, "mode": "primary", "start_seconds": 0}
                        for shot_id in self.shot_ids
                    ],
                    "relation": "many_to_many",
                    "target_seconds": 12,
                    "transition": "连续动作",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "every assigned character asset"):
            VisualDirectorAgent._parse(
                json.dumps(visual_plan, ensure_ascii=False),
                self.shot_ids,
                self.narrative_plan,
                allowed_asset_refs_by_shot={"s01": ["char_01", "char_02"]},
                required_character_refs_by_shot={"s01": ["char_01", "char_02"]},
            )

    def test_causal_story_section_uses_one_audio_across_multiple_visuals(self) -> None:
        narrative_plan = {
            "segments": [
                {
                    "shot_id": shot_id,
                    "section_id": "story_s01" if index < 3 else "reading_s04",
                    "section_narration_anchor": index in {0, 3},
                    "narrative_phase": "causal_story" if index < 3 else "close_reading",
                    "audio_text": "一段完整故事旁白。" if index == 0 else ("逐句讲解。" if index == 3 else ""),
                    "event_text": f"事件{index + 1}",
                    "state_after": f"结果{index + 1}",
                    "target_seconds": 18 if index == 0 else (3 if index == 3 else 0),
                }
                for index, shot_id in enumerate(self.shot_ids)
            ]
        }
        visual_segments = [
            {
                "shot_id": item["shot_id"],
                "depicts_event": item["event_text"],
                "ends_at_state": item["state_after"],
                "video_prompt": f"历史连续事件 {item['shot_id']}",
                "visual_role": "action",
                "asset_refs": [],
            }
            for item in narrative_plan["segments"]
        ]
        visual_plan = {
            "segments": visual_segments,
            "edit_groups": [
                {
                    "group_id": "g01",
                    "source_shot_ids": ["s01", "s02", "s03"],
                    "narration_shot_ids": ["s01"],
                    "visual_sequence": [
                        {"shot_id": shot_id, "seconds": 6, "mode": "primary", "start_seconds": 0}
                        for shot_id in ["s01", "s02", "s03"]
                    ],
                    "relation": "one_to_many",
                    "target_seconds": 18,
                    "transition": "continuous_action",
                },
                {
                    "group_id": "g02",
                    "source_shot_ids": ["s04"],
                    "narration_shot_ids": ["s04"],
                    "visual_sequence": [
                        {"shot_id": "s04", "seconds": 3, "mode": "primary", "start_seconds": 0}
                    ],
                    "relation": "one_to_one",
                    "target_seconds": 3,
                    "transition": "cut",
                },
            ],
        }

        parsed = VisualDirectorAgent._parse(
            json.dumps(visual_plan, ensure_ascii=False), self.shot_ids, narrative_plan
        )

        self.assertEqual(parsed["edit_groups"][0]["relation"], "one_to_many")
        self.assertEqual(parsed["edit_groups"][0]["narration_shot_ids"], ["s01"])

    def test_visual_plan_supports_independent_visual_only_group(self) -> None:
        visual_plan = {
            "segments": self._visual_segments(),
            "edit_groups": [
                {
                    "group_id": "g01",
                    "source_shot_ids": ["s01", "s02"],
                    "narration_shot_ids": ["s01", "s02"],
                    "visual_sequence": [
                        {"shot_id": "s01", "seconds": 6, "mode": "primary", "start_seconds": 0}
                    ],
                    "relation": "many_to_one",
                    "target_seconds": 6,
                    "transition": "背影匹配",
                },
                {
                    "group_id": "g02",
                    "source_shot_ids": ["s03"],
                    "narration_shot_ids": [],
                    "visual_sequence": [
                        {"shot_id": "s03", "seconds": 3, "mode": "primary", "start_seconds": 0}
                    ],
                    "relation": "visual_only",
                    "target_seconds": 3,
                    "transition": "静默停顿",
                },
                {
                    "group_id": "g03",
                    "source_shot_ids": ["s04"],
                    "narration_shot_ids": ["s04"],
                    "visual_sequence": [
                        {"shot_id": "s04", "seconds": 2.2, "mode": "primary", "start_seconds": 0},
                        {"shot_id": "s01", "seconds": 0.8, "mode": "callback", "start_seconds": 1.2},
                    ],
                    "relation": "one_to_many",
                    "target_seconds": 3,
                    "transition": "短闪回后回到现实",
                },
            ],
        }
        parsed = VisualDirectorAgent._parse(
            json.dumps(visual_plan, ensure_ascii=False), self.shot_ids, self.narrative_plan
        )
        self.assertEqual(parsed["edit_groups"][1]["relation"], "visual_only")
        self.assertEqual(parsed["edit_groups"][1]["narration_shot_ids"], [])

    def test_visual_plan_does_not_force_callback_without_graph_requirement(self) -> None:
        visual_plan = {
            "segments": self._visual_segments(),
            "edit_groups": [
                {
                    "group_id": f"g{index:02d}",
                    "source_shot_ids": [shot_id],
                    "narration_shot_ids": [] if shot_id == "s03" else [shot_id],
                    "visual_sequence": [
                        {"shot_id": shot_id, "seconds": 3, "mode": "primary", "start_seconds": 0}
                    ],
                    "relation": "visual_only" if shot_id == "s03" else "one_to_one",
                    "target_seconds": 3,
                    "transition": "hard_cut",
                }
                for index, shot_id in enumerate(self.shot_ids, start=1)
            ],
        }
        parsed = VisualDirectorAgent._parse(
            json.dumps(visual_plan, ensure_ascii=False), self.shot_ids, self.narrative_plan
        )
        self.assertTrue(all(
            item["mode"] == "primary"
            for group in parsed["edit_groups"]
            for item in group["visual_sequence"]
        ))

    def test_group_audio_combines_multiple_narration_beats(self) -> None:
        source_segments = [
            SegmentPlan(
                index=index,
                title=item["shot_id"],
                story_beat="",
                teaching_goal="",
                image_prompt="",
                video_prompt="prompt",
                audio_text=item["audio_text"],
                duration=int(item["target_seconds"]),
            )
            for index, item in enumerate(self.narrative_plan["segments"], start=1)
        ]
        groups = [
            {
                "group_id": "g01",
                "source_shot_ids": ["s01", "s02"],
                "narration_shot_ids": ["s01", "s02"],
                "relation": "many_to_one",
                "target_seconds": 6,
            }
        ]
        result = build_group_audio_segments(source_segments, groups)
        self.assertEqual(len(result), 1)
        self.assertIn("他从书案起身", result[0].audio_text)
        self.assertIn("一次选择", result[0].audio_text)

    def test_group_audio_supports_visual_only_silence(self) -> None:
        source_segments = [
            SegmentPlan(
                index=1,
                title="s03",
                story_beat="",
                teaching_goal="",
                image_prompt="",
                video_prompt="prompt",
                audio_text="",
                duration=3,
            )
        ]
        groups = [
            {
                "group_id": "g01",
                "source_shot_ids": ["s03"],
                "narration_shot_ids": [],
                "relation": "visual_only",
                "target_seconds": 2.75,
            }
        ]
        result = build_group_audio_segments(source_segments, groups)
        self.assertEqual(result[0].audio_text, "")
        self.assertEqual(result[0].duration, 2.75)

    def test_normalizes_relation_label_from_mapping_cardinality(self) -> None:
        visual_plan = {
            "segments": self._visual_segments(),
            "edit_groups": [
                {
                    "group_id": "g01",
                    "source_shot_ids": self.shot_ids,
                    "narration_shot_ids": ["s01", "s02", "s04"],
                    "visual_sequence": [
                        {"shot_id": "s01", "seconds": 3, "mode": "primary", "start_seconds": 0},
                        {"shot_id": "s03", "seconds": 3, "mode": "primary", "start_seconds": 0},
                        {"shot_id": "s04", "seconds": 3, "mode": "primary", "start_seconds": 0},
                    ],
                    "relation": "one_to_one",
                    "target_seconds": 9,
                    "transition": "动作匹配",
                }
            ],
        }
        parsed = VisualDirectorAgent._parse(
            json.dumps(visual_plan, ensure_ascii=False), self.shot_ids, self.narrative_plan
        )
        self.assertEqual(parsed["edit_groups"][0]["relation"], "many_to_many")

    def test_edit_group_statistics_counts_callbacks_and_relations(self) -> None:
        groups = [
            {
                "relation": "one_to_many",
                "narration_shot_ids": ["s01"],
                "visual_sequence": [
                    {"shot_id": "s01", "mode": "primary"},
                    {"shot_id": "s00", "mode": "callback"},
                ],
                "target_seconds": 4,
            },
            {
                "relation": "many_to_one",
                "narration_shot_ids": ["s02", "s03"],
                "visual_sequence": [{"shot_id": "s02", "mode": "primary"}],
                "target_seconds": 6,
            },
        ]
        statistics = edit_group_statistics(groups)
        self.assertEqual(statistics["callback_count"], 1)
        self.assertEqual(statistics["non_one_to_one_count"], 2)
        self.assertEqual(statistics["narration_source_count"], 3)
        self.assertEqual(statistics["visual_clip_count"], 3)

    def test_graph_to_segments_uses_dynamic_story_template(self) -> None:
        graph = {
            "shots": [
                {
                    "shot_id": "s01",
                    "order": 1,
                    "phase": "context_setup",
                    "duration_target": 6,
                    "teaching_function": "建立语境",
                },
                {
                    "shot_id": "s02",
                    "order": 2,
                    "phase": "poem_scene",
                    "duration_target": 6,
                    "teaching_function": "进入原文",
                },
            ]
        }
        realized = {
            "segments": [
                {
                    "shot_id": "s01",
                    "template_shot_id": "s01",
                    "video_prompt": "古代庭院中的连续观察",
                    "audio_text": "第一段语境旁白。",
                    "narration_target_seconds": 5.2,
                },
                {
                    "shot_id": "s01_b02",
                    "template_shot_id": "s01",
                    "video_prompt": "同一语境中的下一次状态变化",
                    "audio_text": "第二段语境旁白。",
                    "narration_target_seconds": 6.1,
                },
                {
                    "shot_id": "s02",
                    "template_shot_id": "s02",
                    "video_prompt": "进入原诗场景",
                    "audio_text": "原文讲解。",
                    "narration_target_seconds": 4.4,
                },
            ]
        }
        segments = graph_to_segments(graph, realized)
        self.assertEqual([item.title for item in segments], ["s01", "s01_b02", "s02"])
        self.assertEqual([item.index for item in segments], [1, 2, 3])
        self.assertEqual(segments[1].duration, 7)


class StoryAndCloseReadingAgentsTest(unittest.TestCase):
    def test_story_fact_guard_rejects_unsupported_quantities(self) -> None:
        unsupported = unsupported_quantitative_details(
            "1161年，二十一岁的主人公率五十余名骑兵出发。",
            "事情发生在1161年前后，当事人二十一岁左右，只记载率少数骑兵出发。",
        )
        self.assertEqual(unsupported, ["五十余名"])

    def test_story_fact_guard_allows_discourse_person_groups(self) -> None:
        unsupported = unsupported_quantitative_details(
            "二人此前并无深交，后来两人再次相见。",
            "两位人物的姓名和各自经历均有记载。",
        )
        self.assertEqual(unsupported, [])

    def test_story_fact_guard_still_rejects_material_headcounts(self) -> None:
        unsupported = unsupported_quantitative_details(
            "主人公带着两名随从出发。",
            "主人公独自出发，未记载随从人数。",
        )
        self.assertEqual(unsupported, ["两名"])

    def test_story_fact_guard_allows_indefinite_year_reference(self) -> None:
        unsupported = unsupported_quantitative_details(
            "故事发生在嘉祐五年之后的某一年早春。",
            "故事发生在嘉祐五年之后，确切年份不详。",
        )
        self.assertEqual(unsupported, [])

    def test_story_fact_guard_still_rejects_material_duration(self) -> None:
        unsupported = unsupported_quantitative_details(
            "这项工作历时一年完成。",
            "这项工作的持续时间没有记载。",
        )
        self.assertEqual(unsupported, ["一年"])

    def test_story_plan_supports_non_event_object_narrative(self) -> None:
        segment_texts = [
            "梅枝最初藏在严寒和积雪里，花苞没有开放，颜色与周围冷寂的庭院几乎融在一起。",
            "天气仍冷时，花瓣先于群芳展开，清淡香气沿着风进入庭院，物象的处境由隐藏转为显现。",
            "花开并不需要虚构人物事件，季节、位置和开放过程已经为进入原诗建立了一条清楚主线。",
        ]
        plan = {
            "narrative_basis": "object_lifecycle",
            "context_dimensions": {
                "who": "not_applicable: 主体是梅花物象而非人物",
                "when": "冬末春初、群芳尚未开放的寒冷时节",
                "where": "积雪未消的中国古代庭院或驿外空间",
                "what": "梅花从含苞、开放到香气被感知的过程",
                "why": "物象的季节处境和独自开放构成讲述依据",
                "how": "沿温度、花瓣、香气和观察距离的变化推进",
            },
            "story_logic": [
                {
                    "node_id": "n1",
                    "from_state": "严寒中含苞",
                    "relation": "precedes",
                    "to_state": "花瓣开始开放",
                    "claim_status": "documented",
                    "evidence_ids": [],
                },
                {
                    "node_id": "n2",
                    "from_state": "花瓣开放",
                    "relation": "enables",
                    "to_state": "香气进入观察者感知",
                    "claim_status": "historically_plausible",
                    "evidence_ids": [],
                },
            ],
            "experience_arc": [
                {
                    "stage": "含苞",
                    "arc_type": "object",
                    "external_condition": "低温与积雪",
                    "focus_or_choice": "观察花苞和枝干状态",
                    "observable_change": "花苞保持收拢",
                    "inferred_change": "不推断人物心理",
                    "certainty": "documented_or_observable",
                },
                {
                    "stage": "开放",
                    "arc_type": "object",
                    "external_condition": "寒意尚未退去",
                    "focus_or_choice": "观察花瓣与香气变化",
                    "observable_change": "花瓣展开并出现香气",
                    "inferred_change": "可进入物象品格联想但不宣布唯一象征",
                    "certainty": "interpretive",
                },
            ],
            "full_story": "".join(segment_texts)
            + "这条讲述不依赖作者轶事，也不需要添加战争或宫廷背景。物象自己的季节位置、材质变化和被看见的方式，足以构成进入作品前的完整语境。"
            + "观察者最初只能辨认积雪中的深色枝干，后来才看见花瓣边缘和花蕊，最后由香气确认花已经开放；观察顺序本身就带来情绪变化。",
            "bridge_to_poem": "沿着梅花从隐藏到显现的过程，可以进入原诗怎样安排观察和感受。",
        }
        parsed = ContextStoryAgent._parse(json.dumps(plan, ensure_ascii=False), valid_evidence_ids=set())
        self.assertEqual(parsed["narrative_basis"], "object_lifecycle")
        self.assertTrue(all(item["arc_type"] == "object" for item in parsed["experience_arc"]))

        plan["full_story"] += "以上内容均为模型补充，未外部核验。"
        with self.assertRaisesRegex(ValueError, "source-provenance meta language"):
            ContextStoryAgent._parse(json.dumps(plan, ensure_ascii=False), valid_evidence_ids=set())

    def test_story_plan_requires_detailed_causal_structure(self) -> None:
        segment_texts = [
            "二十一岁的辛弃疾身处金人统治的北方，战事逼近家乡，他没有南逃，而是开始联络愿意抗金的人。",
            "耿京义军逐渐壮大后，辛弃疾带着自己聚拢的队伍加入，第一次把个人选择放进更大的抗金行动。",
            "叛徒杀害耿京并带走军印，原来的队伍顿时失去中心，辛弃疾必须决定退走还是冒险追回叛徒。",
            "他选择带少数骑兵闯入敌营，趁守军没有防备控制叛将，再连夜押解南下，行动中没有退路可走。",
            "擒叛南归让他完成一次高风险选择，也让恢复故土从青年行动变成此后反复坚持的长期方向。",
            "到了南宋，他把战场经验写进奏议，希望推动恢复中原，可最重要的主张长期没有变成真正的行动。",
            "多年以后再面对元夕人潮，他过去的选择和未竟目标仍在身后，这段经历为进入词中寻找提供一种读法。",
        ]
        story = {
            "narrative_basis": "event_based",
            "context_dimensions": {
                "who": "青年辛弃疾、耿京义军与杀害耿京的叛徒",
                "when": "南宋绍兴三十一年前后的北方战乱时期",
                "where": "金人统治下的山东、敌军营地与南归渡口",
                "what": "辛弃疾聚众抗金、闯营擒叛并在南归后持续谋求恢复中原",
                "why": "家乡受占和义军领袖遇害迫使他作出承担风险的选择",
                "how": "先加入义军，再率少数骑兵突袭，南归后通过奏议延续恢复主张",
            },
            "story_logic": [
                {
                    "node_id": f"n{index}",
                    "from_state": "前一步造成新的压力",
                    "relation": "causes",
                    "to_state": "人物作出具体选择并产生后续影响",
                    "claim_status": "documented",
                    "evidence_ids": ["e01"],
                }
                for index in range(1, 6)
            ],
            "experience_arc": [
                {
                    "stage": f"阶段{index}",
                    "arc_type": "psychological",
                    "external_condition": "外部局势不断收紧",
                    "focus_or_choice": "通过行动承担风险",
                    "observable_change": "从联络队伍转向亲自闯营并长期上奏",
                    "inferred_change": "可以理解为目标逐渐固定下来",
                    "certainty": "historically_plausible",
                }
                for index in range(1, 4)
            ],
            "full_story": "".join(segment_texts)
            + "南归改变了他行动的环境，却没有消除最初的问题：北方故土仍未恢复，他熟悉军事却难以让主张进入决策。"
            + "这些事件不能直接证明词中那人的身份，却能解释一种有传记根据的阅读为什么会出现。"
            + "青年时期能够立即付诸行动的目标，进入新的政治环境后变成长期等待；每一次上奏都延续原先方向，每一次搁置也让行动与现实之间的距离更加清楚。"
            + "因此，故事的重点不是把后来的词句逐字套回生平，而是让观众先理解一种持续多年、始终没有完成的公共目标。"
            + "这种变化有明确的外部过程：起初是战乱中的集结和突袭，后来是朝廷与地方任职环境中的反复陈说。行动方式变了，想要解决的问题却没有消失。"
            + "等到故事进入节日场景，观众已经知道人物为何不会只看见表面的热闹，也能理解作品中的寻找为何可能获得超出爱情之外的解释空间。",
            "bridge_to_poem": "当长期目标仍未完成，元夕人潮中的一次回望便可能带上过去几十年的重量。",
        }
        parsed = ContextStoryAgent._parse(json.dumps(story, ensure_ascii=False), valid_evidence_ids={"e01"})
        self.assertGreaterEqual(len(parsed["story_logic"]), 5)
        self.assertGreaterEqual(len(parsed["full_story"]), 500)

        story["full_story"] = (
            "1162年，辛弃疾已经投身北方义军，恢复故土是他正在用行动承担的目标。"
            + story["full_story"]
        )
        dated_opening = ContextStoryAgent._parse(
            json.dumps(story, ensure_ascii=False),
            valid_evidence_ids={"e01"},
            expected_subject_name="辛弃疾",
        )
        self.assertTrue(dated_opening["full_story"].startswith("1162年"))

    def test_story_narration_adapter_can_expand_semantic_slots(self) -> None:
        section_specs = [
            (
                "st01",
                "寒意未退，枝头花苞仍裹在积雪里，观察者最初只看见深色枝干。天色稍亮，花瓣边缘才从积雪旁显出来，原本隐藏的形态开始变得清楚。",
                "situation",
                "c01",
                "setup",
            ),
            (
                "st02",
                "风经过庭院时，最先抵达的不是颜色，而是一缕很淡的香气。观察随后从远处枝影移到花蕊，物象由看不清转为可以辨认，感受也随之改变。",
                "observation",
                "c02",
                "reflection",
            ),
            (
                "st03",
                "沿着隐藏、显现和被感知的次序，我们再进入原诗，看它怎样安排这次相遇。",
                "bridge_to_poem",
                "c03",
                "bridge",
            ),
        ]
        payload = {
            "sections": [
                {
                    "slot_id": slot_id,
                    "audio_text": audio_text,
                    "story_function": story_function,
                    "claim_status": "historically_plausible",
                    "source_context_node_ids": [context_id],
                    "evidence_anchors": [audio_text],
                    "narrative_move": narrative_move,
                    "event_units": [
                        {
                            "starting_condition": "物象仍处于上一阶段",
                            "event_text": "观察位置或物象状态发生变化",
                            "observable_result": "新的细节变得可以辨认",
                            "visible_characters": [],
                            "weight": 1,
                        }
                        for _ in range(1 if story_function == "bridge_to_poem" else 3)
                    ],
                    "implicit_meaning": "让物象状态的变化先于抽象解释",
                    "transition_logic": "沿观察次序进入下一阶段",
                    "target_seconds": 12,
                }
                for slot_id, audio_text, story_function, context_id, narrative_move in section_specs
            ]
        }
        full_story = "".join(item[1] for item in section_specs) + "观察过程本身构成了进入作品的语境。"
        parsed = StoryNarrationAgent._parse(
            json.dumps(payload, ensure_ascii=False),
            full_story=full_story,
            valid_context_node_ids={"c01", "c02", "c03"},
        )
        self.assertEqual(
            [event["shot_id"] for section in parsed["sections"] for event in section["event_units"]],
            ["st01", "st01_v02", "st01_v03", "st02", "st02_v02", "st02_v03", "st03"],
        )

        narrative_plan = {
            "segments": [
                {
                    "shot_id": event["shot_id"],
                    "section_id": section["section_id"],
                    "narrative_phase": "causal_story",
                    "story_function": section["story_function"],
                    "visual_delegation": section["implicit_meaning"],
                    "visual_target_seconds": event["target_seconds"],
                    "line_ids": [],
                    "context_node_ids": section["source_context_node_ids"],
                }
                for section in parsed["sections"]
                for event in section["event_units"]
            ]
        }
        expanded = build_execution_context({}, narrative_plan, {"character_assets": []})
        self.assertEqual(
            [item["shot_id"] for item in expanded["shots"]],
            ["st01", "st01_v02", "st01_v03", "st02", "st02_v02", "st02_v03", "st03"],
        )

    def test_story_narration_allows_short_transition_and_reports_allocation(self) -> None:
        short_transition = "队伍已经渡江，行动的环境变了，原先的问题却仍留在身后。"
        bridge = "那个没有消失的问题，仍在等待原诗给出自己的回应。"
        payload = {
            "sections": [
                {
                    "slot_id": "st01",
                    "audio_text": short_transition,
                    "story_function": "consequence",
                    "claim_status": "documented",
                    "source_context_node_ids": ["c01"],
                    "evidence_anchors": [short_transition],
                    "narrative_move": "consequence",
                    "event_units": [
                        {
                            "starting_condition": "队伍仍在江边集结",
                            "event_text": "队伍依次渡过江面",
                            "observable_result": "众人抵达另一侧江岸",
                            "visible_characters": [],
                            "weight": 0.21,
                        },
                        {
                            "starting_condition": "众人已经完成渡江",
                            "event_text": "队伍回望来时方向",
                            "observable_result": "江水隔开原来的行动环境",
                            "visible_characters": [],
                            "weight": 0.79,
                        },
                    ],
                    "implicit_meaning": "环境改变没有自动解决目标",
                    "transition_logic": "从行动结果进入未解决问题",
                    "target_seconds": 8,
                },
                {
                    "slot_id": "st02",
                    "audio_text": bridge,
                    "story_function": "bridge_to_poem",
                    "claim_status": "documented",
                    "source_context_node_ids": ["c01"],
                    "evidence_anchors": [bridge],
                    "narrative_move": "bridge",
                    "event_units": [
                        {
                            "starting_condition": "问题仍然没有解决",
                            "event_text": "叙述停在未完成的问题上",
                            "observable_result": "原诗接续这个问题展开",
                            "visible_characters": [],
                            "weight": 1,
                        }
                    ],
                    "implicit_meaning": "保留问题而不提前解释",
                    "transition_logic": "进入原诗",
                    "target_seconds": 6,
                },
            ]
        }
        parsed = StoryNarrationAgent._parse(
            json.dumps(payload, ensure_ascii=False),
            full_story=short_transition,
            bridge_to_poem=bridge,
            valid_context_node_ids={"c01"},
        )
        self.assertEqual(parsed["sections"][0]["audio_text"], short_transition)

        payload["sections"][0]["narrative_move"] = "observation"
        parsed = StoryNarrationAgent._parse(
            json.dumps(payload, ensure_ascii=False),
            full_story=short_transition,
            bridge_to_poem=bridge,
            valid_context_node_ids={"c01"},
        )
        self.assertEqual(parsed["sections"][0]["narrative_move"], "reflection")

        diagnostics = StoryNarrationAgent._repair_diagnostics(
            json.dumps(payload, ensure_ascii=False),
            long_form=True,
        )
        self.assertIn("st01: audio=", diagnostics)
        self.assertIn("至少需要", diagnostics)

    def test_story_narration_prompt_has_compatible_length_and_event_limits(self) -> None:
        prompt = StoryNarrationAgent._build_prompt({"duration_policy": {}})
        self.assertIn("任何段都不得超过140个", prompt)
        self.assertIn("ceil(有效字符数/32)", prompt)
        self.assertIn("如果需要6个，就缩短该段或拆成两个section", prompt)

    def test_close_reading_covers_each_line_with_quote_and_explanation(self) -> None:
        poem = {
            "lines": [
                {"line_id": "l01", "order": 1, "text": "东风夜放花千树。更吹落、星如雨。"},
                {"line_id": "l02", "order": 2, "text": "宝马雕车香满路。凤箫声动，玉壶光转，一夜鱼龙舞。"},
                {"line_id": "l03", "order": 3, "text": "蛾儿雪柳黄金缕。笑语盈盈暗香去。"},
                {"line_id": "l04", "order": 4, "text": "众里寻他千百度。蓦然回首，那人却在，灯火阑珊处。"},
            ]
        }
        segments = []
        shot_index = 1
        explanations = {
            "l01": "花千树和星如雨都是比喻，把灯火与烟火写成铺满视野的动态盛景。",
            "l02": "车马、凤箫、玉壶和鱼龙连续出现，调动嗅觉、听觉与视觉。",
            "l03": "笑语和暗香随着人影离去，去字把热闹写成无法停留的流动。",
            "l04": "千百度写反复寻找，蓦然回首却突然转向；那人身份仍可开放解释。",
        }
        for line in poem["lines"]:
            line_id = line["line_id"]
            segments.append(
                {
                    "shot_id": f"cr{shot_index:02d}",
                    "line_ids": [line_id],
                    "audio_text": line["text"],
                    "narration_role": "poem_quote",
                    "reading_action": "quote",
                    "knowledge_points": [
                        {"type": "lexical", "content": "准确朗读本行原文及其停顿"}
                    ],
                    "teaching_move": "先让原文进入听觉",
                    "unstated_visual_meaning": "让动作和空间变化先发生",
                    "target_seconds": 5,
                }
            )
            shot_index += 1
            segments.append(
                {
                    "shot_id": f"cr{shot_index:02d}",
                    "line_ids": [line_id],
                    "audio_text": explanations[line_id],
                    "narration_role": "delayed_inference",
                    "reading_action": "explain",
                    "knowledge_points": [
                        {"type": "structure", "content": "说明本行在全篇推进中的具体作用"}
                    ],
                    "teaching_move": "从现场推进到开放理解",
                    "unstated_visual_meaning": "用人物位置和人流方向承担未直说的关系",
                    "target_seconds": 6,
                }
            )
            shot_index += 1
        card = {
            "audio_text": (
                "《青玉案·元夕》是宋代辛弃疾的词。先从元宵夜的繁华读到灯火稀疏处，"
                "“花千树”把灯火写成繁花，密集的光亮为后面的寻找转折铺路；"
                "“那人”也可以保留字面和寄托两种读法，读者还可结合全词结构理解，"
                "不必急着下唯一结论，再回看前后语气的变化。"
            ),
            "teaching_move": "建立由繁入寂、先景后情的整体阅读路径",
            "focus_line_ids": ["l01", "l04"],
            "metaphor_node_ids": ["m01"],
            "knowledge_points": [
                {"type": "poem_identity", "content": "明确题名、作者、时代与体裁"},
                {"type": "overall_structure", "content": "由繁华铺陈转向灯火稀疏处"},
                {"type": "metaphor_mapping", "content": "花千树以繁花形态说明灯火密集明亮"},
                {"type": "interpretive_boundary", "content": "那人可保留字面与寄托两种读法"},
            ],
            "target_seconds": 20,
        }
        parsed = CloseReadingAgent._parse(
            json.dumps({"poem_card_narration": card, "segments": segments}, ensure_ascii=False),
            ["l01", "l02", "l03", "l04"],
            poem,
            interpretation={
                "metaphor_nodes": [
                    {
                        "node_id": "m01",
                        "interpretation_status": "contested_reading",
                    }
                ]
            },
        )
        self.assertEqual(parsed["segments"][0]["line_ids"], ["l01"])
        self.assertEqual(parsed["segments"][-1]["line_ids"], ["l04"])
        self.assertTrue(all(item["knowledge_points"] for item in parsed["segments"]))

    def test_close_reading_can_quote_and_explain_each_line_in_one_slot(self) -> None:
        poem = {
            "lines": [
                {"line_id": "l01", "order": 1, "text": "第一行。"},
                {"line_id": "l02", "order": 2, "text": "第二行。"},
                {"line_id": "l03", "order": 3, "text": "第三行。"},
                {"line_id": "l04", "order": 4, "text": "第四行。"},
            ]
        }
        shot_ids = ["cr01", "cr02", "cr03", "cr04"]
        segments = [
            {
                "shot_id": shot_id,
                "line_ids": [line["line_id"]],
                "audio_text": line["text"] + "原文中的变化继续向下一行推进，理解落在具体动作和感受上。",
                "narration_role": "delayed_inference",
                "reading_action": "quote_and_explain",
                "knowledge_points": [
                    {"type": "structure", "content": "原文变化与下一行的承接关系"}
                ],
                "teaching_move": "同段完成原文进入和理解",
                "unstated_visual_meaning": "让原文中的主体变化先发生",
                "target_seconds": 6,
            }
            for shot_id, line in zip(shot_ids, poem["lines"])
        ]
        card = {
            "audio_text": (
                "《样例》是宋代作者的词。先从眼前景物读到情绪转折，"
                "“第一行”中的景物形态怎样支持后面的心绪，需要在全篇关系里理解；"
                "具体寄托仍可保持开放，也可结合后文转折继续体会，"
                "不必把一种读法当成固定答案，还要留意语气怎样变化。"
            ),
            "teaching_move": "建立景物与心绪之间的整体关系",
            "focus_line_ids": ["l01"],
            "metaphor_node_ids": ["m01"],
            "knowledge_points": [
                {"type": "poem_identity", "content": "明确作品和作者身份"},
                {"type": "overall_structure", "content": "景物推进到情绪转折"},
                {"type": "metaphor_mapping", "content": "景物属性支持心绪理解"},
            ],
            "target_seconds": 20,
        }
        parsed = CloseReadingAgent._parse(
            json.dumps({"poem_card_narration": card, "segments": segments}, ensure_ascii=False),
            ["l01", "l02", "l03", "l04"],
            poem,
            interpretation={"metaphor_nodes": [{"node_id": "m01", "interpretation_status": "textually_explicit"}]},
        )
        self.assertTrue(all(item["reading_action"] == "quote_and_explain" for item in parsed["segments"]))

    def test_close_reading_can_split_one_source_line_across_beats(self) -> None:
        poem = {
            "lines": [
                {"line_id": "l01", "order": 1, "text": "前句先发生。后句再转向。"},
            ]
        }
        plan = {
            "segments": [
                {
                    "shot_id": "cr01",
                    "line_ids": ["l01"],
                    "audio_text": "前句先发生。这个动作把注意力留在当前处境里。",
                    "narration_role": "historical_evidence",
                    "reading_action": "connect",
                    "knowledge_points": [
                        {"type": "structure", "content": "前句建立当前动作和注意方向"}
                    ],
                    "teaching_move": "先连接前一阶段",
                    "unstated_visual_meaning": "让动作继续发生",
                    "target_seconds": 5,
                },
                {
                    "shot_id": "cr02",
                    "line_ids": ["l01"],
                    "audio_text": "后句再转向。转折改变了刚刚建立的感受，但没有把答案说死。",
                    "narration_role": "poem_quote",
                    "reading_action": "quote_and_explain",
                    "knowledge_points": [
                        {"type": "structure", "content": "后句通过转向改变前句建立的感受"}
                    ],
                    "teaching_move": "完成原文和理解",
                    "unstated_visual_meaning": "由位置变化承担转折",
                    "target_seconds": 6,
                },
            ]
        }
        plan["poem_card_narration"] = {
            "audio_text": (
                "《样例》是宋代作者的词。先看前句如何建立处境，再看后句怎样把情绪推向转折；"
                "其中的关系可以结合具体字词理解，不必预设唯一结论，也可保留不同的阅读方向，"
                "慢慢回看前后承接，并比较词语选择带来的语气变化，体会情绪如何落下。"
            ),
            "teaching_move": "建立前句到后句的整体转折",
            "focus_line_ids": ["l01"],
            "metaphor_node_ids": ["m01"],
            "knowledge_points": [
                {"type": "poem_identity", "content": "明确作品题名作者与体裁身份"},
                {"type": "overall_structure", "content": "前句建立处境、后句转向"},
                {"type": "metaphor_mapping", "content": "具体字词关系支撑情绪转折"},
            ],
            "target_seconds": 20,
        }
        parsed = CloseReadingAgent._parse(
            json.dumps(plan, ensure_ascii=False),
            ["l01"],
            poem,
            interpretation={"metaphor_nodes": [{"node_id": "m01", "interpretation_status": "textually_explicit"}]},
        )
        self.assertEqual(len(parsed["segments"]), 2)

    def test_close_reading_ignores_editorial_variant_annotations(self) -> None:
        poem = {
            "lines": [
                {
                    "line_id": "l01",
                    "order": 1,
                    "text": "上有青冥之长天，下有渌水之波澜。(长天 一作：高天)",
                }
            ]
        }
        segment = {
            "shot_id": "cr01",
            "line_ids": ["l01"],
            "audio_text": "上有青冥之长天，下有渌水之波澜。天地对举放大了阻隔。",
            "narration_role": "poem_quote",
            "reading_action": "quote_and_explain",
            "knowledge_points": [
                {"type": "rhetoric", "content": "上下对举铺开遥远空间"}
            ],
            "teaching_move": "理解空间如何推进情绪",
            "unstated_visual_meaning": "长空与水面先建立距离",
            "target_seconds": 6,
        }
        parsed = CloseReadingAgent._parse(
            json.dumps(
                {
                    "poem_card_narration": {
                        "audio_text": (
                            "《样例》是宋代作者的词。先从开头的空间铺陈读到情绪收束，"
                            "具体字词的关系支持整体理解；不同版本仍可保留合理差异，也可比较其语气变化，"
                            "体会文字选择的分寸，并观察它如何影响最后的情绪落点和朗读语气变化，"
                            "让理解回到原文。"
                        ),
                        "teaching_move": "建立空间与情绪的整体阅读路径",
                        "focus_line_ids": ["l01"],
                        "metaphor_node_ids": ["m01"],
                        "knowledge_points": [
                            {"type": "poem_identity", "content": "明确作品题名作者与体裁身份"},
                            {"type": "overall_structure", "content": "空间铺陈转向情绪收束"},
                            {"type": "metaphor_mapping", "content": "具体字词关系支撑整体理解"},
                        ],
                        "target_seconds": 20,
                    },
                    "segments": [segment],
                },
                ensure_ascii=False,
            ),
            ["l01"],
            poem,
            interpretation={"metaphor_nodes": [{"node_id": "m01", "interpretation_status": "textually_explicit"}]},
        )
        self.assertEqual(parsed["segments"][0]["shot_id"], "cr01")


if __name__ == "__main__":
    unittest.main()
