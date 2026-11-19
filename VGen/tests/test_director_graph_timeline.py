from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


VGEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VGEN_ROOT))

from vgen.director_graph_pipeline import (
    DirectorGraphPipeline,
    add_close_reading_timing_callbacks,
    load_existing_text_plan,
    plan_audio_speedups,
    plan_video_generation_seconds,
    realize_edit_group_timings,
)
from vgen.image_backends import GeminiImageGenerator, OpenAIImageGenerator
from vgen.poetry_teaching import MediaComposer
from vgen.poetry_teaching import SegmentPlan
from vgen.reference_assets import ReferenceImageServer, generate_character_reference_images
from vgen.video_backends import Seedance20VideoGenerator, Veo31TextVideoGenerator


class DirectorGraphTimelineTest(unittest.TestCase):
    def setUp(self) -> None:
        config = SimpleNamespace(overwrite=True, fps=10)
        try:
            self.ffmpeg = MediaComposer(config)._ffmpeg_executable()
        except RuntimeError as exc:
            self.skipTest(str(exc))

    def _make_color_video(self, path: Path, color: str) -> None:
        subprocess.run(
            [
                self.ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=160x90:r=10:d=1.2",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
        )

    def _sample_rgb(self, path: Path, seconds: float) -> tuple[int, int, int]:
        result = subprocess.run(
            [
                self.ffmpeg,
                "-v",
                "error",
                "-ss",
                str(seconds),
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                "scale=1:1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        return tuple(result.stdout[:3])

    def test_compose_edit_group_reuses_callback_and_matches_audio_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            red_path = output_dir / "red.mp4"
            blue_path = output_dir / "blue.mp4"
            self._make_color_video(red_path, "red")
            self._make_color_video(blue_path, "blue")

            config = SimpleNamespace(overwrite=True, fps=10)
            pipeline = DirectorGraphPipeline.__new__(DirectorGraphPipeline)
            pipeline.output_dir = output_dir
            pipeline.config = config
            pipeline.composer = MediaComposer(config)
            group = {
                "group_id": "g_callback",
                "visual_sequence": [
                    {"shot_id": "s01", "seconds": 1.5, "mode": "primary", "start_seconds": 0},
                    {"shot_id": "s02", "seconds": 1.0, "mode": "primary", "start_seconds": 0},
                    {"shot_id": "s01", "seconds": 0.5, "mode": "callback", "start_seconds": 0.2},
                ],
            }
            result = pipeline._compose_edit_group_video(
                group,
                {"s01": red_path, "s02": blue_path},
                audio_seconds=3.0,
            )

            self.assertAlmostEqual(pipeline.composer.media_duration(result), 3.0, delta=0.25)
            first = self._sample_rgb(result, 0.4)
            middle = self._sample_rgb(result, 1.9)
            last = self._sample_rgb(result, 2.8)
            self.assertGreater(first[0], first[2] + 80)
            self.assertGreater(middle[2], middle[0] + 80)
            self.assertGreater(last[0], last[2] + 80)

    def test_write_silent_audio_has_exact_planned_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "silent.wav"
            pipeline = DirectorGraphPipeline.__new__(DirectorGraphPipeline)
            pipeline.config = SimpleNamespace(overwrite=True)
            pipeline._write_silent_audio(output_path, 2.75)
            with wave.open(str(output_path), "rb") as audio_file:
                duration = audio_file.getnframes() / audio_file.getframerate()
                self.assertEqual(audio_file.getnchannels(), 1)
            self.assertAlmostEqual(duration, 2.75, places=3)

    def test_trim_or_extend_rejects_long_frozen_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            source_path = output_dir / "source.mp4"
            self._make_color_video(source_path, "red")

            config = SimpleNamespace(overwrite=True, fps=10)
            pipeline = DirectorGraphPipeline.__new__(DirectorGraphPipeline)
            pipeline.config = config
            pipeline.composer = MediaComposer(config)

            with self.assertRaisesRegex(ValueError, "Split the edit group"):
                pipeline._trim_or_extend_video(
                    source_path=source_path,
                    start_seconds=0,
                    duration_seconds=1.6,
                    output_path=output_dir / "too_long.mp4",
                )

    def test_realizes_edit_groups_from_measured_audio(self) -> None:
        groups = [
            {
                "group_id": "g01",
                "target_seconds": 6,
                "visual_sequence": [
                    {"shot_id": "s01", "seconds": 4, "mode": "primary", "start_seconds": 0},
                    {"shot_id": "s02", "seconds": 2, "mode": "primary", "start_seconds": 0},
                ],
            }
        ]
        audio_segment = SegmentPlan(
            index=1,
            title="g01",
            story_beat="",
            teaching_goal="",
            image_prompt="",
            video_prompt="",
            audio_text="旁白",
            duration=6,
        )

        realized = realize_edit_group_timings(groups, [audio_segment], {1: 9.0})

        self.assertEqual(realized[0]["planned_target_seconds"], 6)
        self.assertEqual(realized[0]["target_seconds"], 9)
        self.assertEqual([item["seconds"] for item in realized[0]["visual_sequence"]], [6, 3])

    def test_plans_variable_veo_durations_from_realized_timeline(self) -> None:
        groups = [
            {
                "visual_sequence": [
                    {"shot_id": "s01", "seconds": 4.2, "start_seconds": 0},
                    {"shot_id": "s02", "seconds": 5.8, "start_seconds": 0},
                    {"shot_id": "s03", "seconds": 7.5, "start_seconds": 0},
                    {"shot_id": "s04", "seconds": 9.8, "start_seconds": 0},
                ]
            }
        ]

        durations = plan_video_generation_seconds(groups, ["s01", "s02", "s03", "s04"])

        self.assertEqual(durations, {"s01": 4, "s02": 6, "s03": 8, "s04": 8})

    def test_plans_bounded_audio_speedup_for_single_visual(self) -> None:
        groups = [
            {
                "group_id": "g01",
                "visual_sequence": [
                    {"shot_id": "s01", "seconds": 8, "start_seconds": 0, "mode": "primary"}
                ],
            }
        ]
        segments = [SimpleNamespace(index=1, title="g01")]
        speedups = plan_audio_speedups(groups, segments, {1: 11.12})
        self.assertGreater(speedups[1], 1.11)
        self.assertLess(speedups[1], 1.15)

        with self.assertRaisesRegex(ValueError, "Split the narration"):
            plan_audio_speedups(groups, segments, {1: 12.0})

    def test_adds_close_reading_callback_before_excessive_audio_speedup(self) -> None:
        source_segments = [
            SimpleNamespace(title="cr01", story_beat="close_reading"),
            SimpleNamespace(title="cr02", story_beat="close_reading"),
        ]
        groups = [
            {
                "group_id": "g01",
                "source_shot_ids": ["cr01"],
                "narration_shot_ids": ["cr01"],
                "visual_sequence": [
                    {"shot_id": "cr01", "seconds": 8.5, "mode": "primary", "start_seconds": 0}
                ],
                "relation": "one_to_one",
                "target_seconds": 8.5,
            },
            {
                "group_id": "g02",
                "source_shot_ids": ["cr02"],
                "narration_shot_ids": ["cr02"],
                "visual_sequence": [
                    {"shot_id": "cr02", "seconds": 9.1, "mode": "primary", "start_seconds": 0}
                ],
                "relation": "one_to_one",
                "target_seconds": 9.1,
            },
        ]
        audio_segments = [
            SimpleNamespace(index=1, title="g01"),
            SimpleNamespace(index=2, title="g02"),
        ]

        adjusted = add_close_reading_timing_callbacks(
            groups, source_segments, audio_segments, {1: 9.0, 2: 12.0}
        )

        self.assertEqual(adjusted[1]["relation"], "one_to_many")
        self.assertEqual(adjusted[1]["visual_sequence"][0]["shot_id"], "cr01")
        self.assertEqual(adjusted[1]["visual_sequence"][0]["mode"], "callback")
        speedups = plan_audio_speedups(adjusted, audio_segments, {1: 9.0, 2: 12.0})
        self.assertEqual(speedups[2], 1.0)

    def test_audio_speedup_is_idempotent_from_preserved_original(self) -> None:
        config = SimpleNamespace(overwrite=True, fps=10)
        pipeline = DirectorGraphPipeline.__new__(DirectorGraphPipeline)
        pipeline.config = config
        pipeline.composer = MediaComposer(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "group.wav"
            pipeline._write_silent_audio(audio_path, 11.12)
            speedup = 11.12 / 9.9
            pipeline._speed_up_audio(audio_path, speedup)
            first_duration = pipeline.composer.audio_duration(audio_path)
            pipeline._speed_up_audio(audio_path, speedup)
            second_duration = pipeline.composer.audio_duration(audio_path)
            self.assertAlmostEqual(first_duration, 9.9, delta=0.05)
            self.assertAlmostEqual(second_duration, first_duration, delta=0.01)
            self.assertAlmostEqual(pipeline._original_audio_duration(audio_path), 11.12, delta=0.01)

    def test_rejects_timeline_that_exceeds_longest_veo_clip(self) -> None:
        groups = [
            {
                "visual_sequence": [
                    {"shot_id": "s01", "seconds": 2, "start_seconds": 7},
                ]
            }
        ]
        with self.assertRaisesRegex(ValueError, "split the edit group"):
            plan_video_generation_seconds(groups, ["s01"])

    def test_veo_backend_uses_segment_duration(self) -> None:
        generator = Veo31TextVideoGenerator(SimpleNamespace(veo_duration_seconds=8))
        requested = [4, 5, 7, 10]
        actual = [
            generator._request_duration_seconds(SimpleNamespace(duration=duration))
            for duration in requested
        ]
        self.assertEqual(actual, [4, 6, 8, 8])

    def test_veo_payload_includes_character_reference_and_prompt_binding(self) -> None:
        generator = Veo31TextVideoGenerator(SimpleNamespace(veo_duration_seconds=8))
        segment = SimpleNamespace(
            duration=6,
            video_prompt="青年人物走入营地",
            reference_asset_labels=["青年辛弃疾"],
            reference_image_urls=["http://example.test/a_xin_young.png"],
        )

        payload = generator._build_payload(segment)

        content = payload["messages"][0]["content"]
        self.assertEqual(payload["params"]["durationSeconds"], 8)
        self.assertEqual(content[1]["type"], "reference_image_url")
        self.assertEqual(content[1]["reference_type"], "asset")
        self.assertIn("青年辛弃疾的人物形象如图", content[0]["value"])

    def test_veo_ignores_reference_labels_without_reference_urls(self) -> None:
        generator = Veo31TextVideoGenerator(SimpleNamespace(veo_duration_seconds=8))
        segment = SimpleNamespace(
            duration=5,
            video_prompt="青年人物走入营地",
            reference_asset_labels=["青年辛弃疾"],
            reference_image_urls=[],
        )

        payload = generator._build_payload(segment)

        content = payload["messages"][0]["content"]
        self.assertEqual(payload["params"]["durationSeconds"], 6)
        self.assertEqual(len(content), 1)
        self.assertNotIn("人物形象如图", content[0]["value"])

    def test_seedance_payload_uses_custom_i2v_protocol(self) -> None:
        generator = Seedance20VideoGenerator(
            SimpleNamespace(
                seedance_duration_seconds=8,
                seedance_timeout_seconds=600,
                width=1280,
                height=720,
                fps=24,
            )
        )
        segment = SimpleNamespace(
            duration=6,
            video_prompt="青年人物走入营地",
            reference_asset_labels=["青年辛弃疾"],
            reference_image_urls=["http://example.test/a_xin_young.png"],
        )

        payload = generator._build_payload(segment)

        self.assertEqual(
            payload["model_marker"], "api_doubao_doubao-seedance-2-0-260128"
        )
        self.assertEqual(payload["params"]["duration"], "6")
        self.assertEqual(payload["params"]["ratio"], "16:9")
        self.assertEqual(payload["params"]["resolution"], "720p")
        self.assertEqual(payload["params"]["framespersecond"], "24")
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[-1]["type"], "text")
        self.assertIn("青年辛弃疾的人物形象如图", content[-1]["value"])

    def test_seedance_auth_has_no_api_version_header(self) -> None:
        headers = Seedance20VideoGenerator._auth_headers(
            "app-id:app-key",
            "test-source",
            "Mon, 20 Apr 2026 08:00:00 GMT",
        )

        self.assertNotIn("Apiversion", headers)
        self.assertEqual(headers["Source"], "test-source")
        self.assertIn('hmac id="app-id"', headers["Authorization"])

    def test_seedance_payload_can_inline_local_reference(self) -> None:
        generator = Seedance20VideoGenerator(
            SimpleNamespace(
                seedance_duration_seconds=8,
                seedance_timeout_seconds=600,
                width=1280,
                height=720,
                fps=24,
            )
        )
        segment = SimpleNamespace(
            duration=4,
            video_prompt="人物转身",
            reference_asset_labels=["辛弃疾"],
            reference_image_urls=["http://unreachable.test/person.png"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "person.png"
            reference_path.write_bytes(b"png-bytes")
            with mock.patch.dict(os.environ, {"SEEDANCE_REFERENCE_MODE": "data_uri"}):
                payload = generator._build_payload(
                    segment,
                    reference_images=[reference_path],
                )

        image_value = payload["messages"][0]["content"][0]["value"]
        self.assertTrue(image_value.startswith("data:image/png;base64,"))

    def test_reference_image_server_maps_output_file_to_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "character_assets" / "person.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"test")
            server = ReferenceImageServer.__new__(ReferenceImageServer)
            server.root = root.resolve()
            server.advertised_host = "10.0.0.8"
            server.server = SimpleNamespace(server_port=8123)

            url = server.url_for(image_path)

            self.assertEqual(url, "http://10.0.0.8:8123/character_assets/person.png")

    def test_character_age_variant_derives_from_first_reference(self) -> None:
        class FakeImageGenerator:
            def __init__(self) -> None:
                self.calls = []

            def generate_prompt(self, *, output_path: Path, reference_image: Path | None, **kwargs: object) -> Path:
                output_path.write_bytes(b"image")
                self.calls.append((output_path.name, reference_image.name if reference_image else ""))
                return output_path

        plan = {
            "character_assets": [
                {
                    "asset_id": "young",
                    "identity_group": "poet",
                    "reference_mode": "age_variant",
                    "reference_prompt": "青年人物",
                    "negative_prompt": "现代服饰",
                },
                {
                    "asset_id": "adult",
                    "identity_group": "poet",
                    "reference_mode": "age_variant",
                    "reference_prompt": "中年人物",
                    "negative_prompt": "现代服饰",
                },
            ],
            "continuity_links": [
                {
                    "from_asset_id": "young",
                    "to_asset_id": "adult",
                    "shared_identity_anchors": ["脸型"],
                    "allowed_changes": ["年龄"],
                }
            ],
        }
        generator = FakeImageGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = generate_character_reference_images(plan, generator, Path(temp_dir))

        self.assertEqual(generator.calls, [("young.png", ""), ("adult.png", "young.png")])
        self.assertEqual(manifest["assets"][1]["derived_from_asset_id"], "young")

    def test_load_existing_text_plan_requires_matching_graph(self) -> None:
        graph = {"schema_version": "test", "poem": {"title": "sample"}}
        plan = {
            "planning_mode": "test",
            "character_asset_plan": {"character_assets": [], "continuity_links": []},
            "poem_card_narration": {"audio_text": "test"},
            "segments": [{"shot_id": "s01"}],
            "edit_groups": [{"group_id": "g01"}],
            "poem_card_insert_after_shot_id": "s01",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "director_graph.input.json").write_text(
                json.dumps(graph), encoding="utf-8"
            )
            (output_dir / "realized_text_plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            self.assertEqual(load_existing_text_plan(output_dir, graph), plan)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_existing_text_plan(output_dir, {"schema_version": "other"})

    def test_load_existing_text_plan_rejects_unexpected_fields(self) -> None:
        graph = {"schema_version": "test"}
        plan = {
            "planning_mode": "test",
            "character_asset_plan": {"character_assets": [], "continuity_links": []},
            "poem_card_narration": {"audio_text": "test"},
            "segments": [{"shot_id": "s01"}],
            "edit_groups": [{"group_id": "g01"}],
            "poem_card_insert_after_shot_id": "s01",
            "unexpected": True,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "director_graph.input.json").write_text(
                json.dumps(graph), encoding="utf-8"
            )
            (output_dir / "realized_text_plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Unexpected realized text plan fields"):
                load_existing_text_plan(output_dir, graph)

    def test_gemini_image_payload_supports_reference_image(self) -> None:
        generator = GeminiImageGenerator(SimpleNamespace(width=1280, height=720))
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.png"
            reference_path.write_bytes(b"reference-bytes")
            payload = generator._build_payload(
                "single historical character",
                "text, watermark",
                reference_path,
            )
        self.assertEqual(payload["model_marker"], "api_google_gemini-3-pro-image-preview")
        config = payload["params"]["generationConfig"]
        self.assertEqual(config["responseModalities"], ["TEXT", "IMAGE"])
        self.assertEqual(config["imageConfig"]["aspectRatio"], "16:9")
        reference = payload["messages"][0]["content"][1]
        self.assertEqual(reference["type"], "image_url")
        self.assertTrue(reference["value"].startswith("data:image/png;base64,"))

    def test_gemini_image_extracts_gateway_and_native_base64(self) -> None:
        encoded = "cG5nLWJ5dGVz"
        self.assertEqual(
            GeminiImageGenerator._extract_image_bytes(
                {"answer": [{"type": "image_base64", "value": encoded}]}
            ),
            b"png-bytes",
        )
        self.assertEqual(
            GeminiImageGenerator._extract_image_bytes(
                {
                    "candidates": [
                        {"content": {"parts": [{"inlineData": {"data": encoded}}]}}
                    ]
                }
            ),
            b"png-bytes",
        )

    def test_gemini_image_payload_supports_style_prefix(self) -> None:
        generator = GeminiImageGenerator(SimpleNamespace(width=1280, height=720))
        with mock.patch.dict(
            os.environ,
            {"GEMINI_IMAGE_PROMPT_PREFIX": "明确的二维动画角色，非照片"},
        ):
            payload = generator._build_payload("南宋人物", "文字")

        prompt = payload["messages"][0]["content"][0]["value"]
        self.assertTrue(prompt.startswith("明确的二维动画角色，非照片\n南宋人物"))

    def test_openai_image_extracts_and_scrubs_base64(self) -> None:
        payload = {"b64_json": "cG5nLWJ5dGVz", "revised_prompt": None}
        self.assertEqual(
            OpenAIImageGenerator._extract_image_bytes(payload), b"png-bytes"
        )
        summary = OpenAIImageGenerator._response_summary(payload)
        self.assertEqual(summary["b64_json"], "<binary payload omitted: 12 characters>")


if __name__ == "__main__":
    unittest.main()
