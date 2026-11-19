from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vgen.poetry_teaching import (
    CosyVoiceGenerator,
    MediaComposer,
    PipelineConfig,
    SegmentPlan,
)
from vgen.director_agents import PoetryDirectorGraphRealizer
from vgen.director.common import expected_edit_relation
from vgen.image_backends import build_image_generator
from vgen.poem_card import PoemCardRenderer, PoemCardSpec, build_poem_card_spec
from vgen.reference_assets import ReferenceImageServer, generate_character_reference_images
from vgen.video_backends import (
    MINIMAX_H3_SUPPORTED_DURATION_SECONDS,
    VEO_SUPPORTED_DURATION_SECONDS,
    MiniMaxH3VideoGenerator,
    Seedance20VideoGenerator,
    Veo31VideoGenerator,
)
from vgen.workspace import (
    DEFAULT_COSYVOICE_MODEL,
    DEFAULT_COSYVOICE_PROMPT_TEXT,
    DEFAULT_COSYVOICE_PROMPT_WAV,
    DEFAULT_COSYVOICE_REPO,
    DEFAULT_QWEN_IMAGE_MODEL,
    DEFAULT_ROOT,
)


VGEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = VGEN_ROOT / "schema" / "examples" / "qingyu_an_yuanxi.director_graph.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "outputs" / "director_graph"
DEFAULT_POEM_CARD_FONT = DEFAULT_ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
MAX_VIDEO_SLOWDOWN_FACTOR = 1.25
PREFERRED_VIDEO_SLOWDOWN_FACTOR = 1.10
MAX_END_FRAME_PADDING_SECONDS = 0.12
MAX_AUDIO_SPEEDUP_FACTOR = 2.50


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def load_existing_text_plan(output_dir: Path, graph: dict[str, Any]) -> dict[str, Any]:
    graph_path = output_dir / "director_graph.input.json"
    plan_path = output_dir / "realized_text_plan.json"
    if not graph_path.exists() or not plan_path.exists():
        raise FileNotFoundError(
            "--resume-media requires director_graph.input.json and realized_text_plan.json "
            f"in {output_dir}"
        )
    if load_json(graph_path) != graph:
        raise ValueError("The stored director graph does not match --director-graph")

    plan = load_json(plan_path)
    required_fields = {
        "planning_mode",
        "character_asset_plan",
        "poem_card_narration",
        "segments",
        "edit_groups",
        "poem_card_insert_after_shot_id",
    }
    if set(plan) != required_fields:
        raise ValueError(f"Unexpected realized text plan fields: {sorted(plan)}")
    if not isinstance(plan["segments"], list) or not plan["segments"]:
        raise ValueError("The realized text plan has no segments")
    if not isinstance(plan["edit_groups"], list) or not plan["edit_groups"]:
        raise ValueError("The realized text plan has no edit groups")
    card_narration = plan["poem_card_narration"]
    if not isinstance(card_narration, dict) or not str(
        card_narration.get("audio_text") or ""
    ).strip():
        raise ValueError("The realized text plan has no poem_card_narration.audio_text")
    return plan


def validate_graph(graph: dict[str, Any]) -> list[str]:
    validator_path = VGEN_ROOT / "schema" / "validate_director_graph.py"
    spec = importlib.util.spec_from_file_location("storytelling_director_graph_validator", validator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load validator: {validator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    errors, warnings = module.validate_graph(graph)
    if errors:
        formatted = "\n".join(f"- {item}" for item in errors)
        raise ValueError(f"Director graph validation failed:\n{formatted}")
    return [str(item) for item in warnings]


def validate_character_reference_coverage(realized: dict[str, Any]) -> None:
    """Reject text plans that would silently regenerate a recurring character."""
    segments = [item for item in realized.get("segments", []) if isinstance(item, dict)]
    assets = [
        item
        for item in realized.get("character_asset_plan", {}).get("character_assets", [])
        if isinstance(item, dict)
    ]
    asset_ids = [str(item.get("asset_id") or "") for item in assets]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("Character asset ids must be unique")

    expected_assets_by_shot: dict[str, list[str]] = {}
    for asset in assets:
        asset_id = str(asset.get("asset_id") or "")
        for shot_id in asset.get("used_by_shot_ids", []):
            expected_assets_by_shot.setdefault(str(shot_id), []).append(asset_id)

    segment_ids = [str(item.get("shot_id") or "") for item in segments]
    for item in segments:
        shot_id = str(item.get("shot_id") or "")
        expected_refs = expected_assets_by_shot.get(shot_id, [])
        actual_refs = [
            str(value)
            for value in item.get("asset_refs", [])
            if str(value) in asset_ids
        ]
        if set(actual_refs) != set(expected_refs):
            raise ValueError(
                f"Character reference coverage mismatch for {shot_id}: "
                f"expected {expected_refs}, got {actual_refs}"
            )
    unknown_shots = sorted(set(expected_assets_by_shot).difference(segment_ids))
    if unknown_shots:
        raise ValueError(f"Character assets reference unknown planned shots: {unknown_shots}")

    if not any("visible_characters" in item for item in segments):
        return
    visible_shots_by_character: dict[str, list[str]] = {}
    for item in segments:
        shot_id = str(item.get("shot_id") or "")
        visible_characters = item.get("visible_characters", [])
        if not isinstance(visible_characters, list):
            raise ValueError(f"visible_characters must be an array for {shot_id}")
        for character_name in visible_characters:
            visible_shots_by_character.setdefault(str(character_name), []).append(shot_id)

    assets_by_name: dict[str, list[dict[str, Any]]] = {}
    for asset in assets:
        assets_by_name.setdefault(str(asset.get("name") or ""), []).append(asset)
    for character_name, shot_ids in visible_shots_by_character.items():
        if len(shot_ids) < 2:
            continue
        matching_assets = assets_by_name.get(character_name, [])
        if len(matching_assets) != 1:
            raise ValueError(
                f"Recurring visible character {character_name} needs exactly one same-name asset"
            )
        used_by = [str(value) for value in matching_assets[0].get("used_by_shot_ids", [])]
        if used_by != shot_ids:
            raise ValueError(
                f"Recurring visible character {character_name} is not covered exhaustively: "
                f"expected {shot_ids}, got {used_by}"
            )


def graph_to_segments(graph: dict[str, Any], realized: dict[str, Any]) -> list[SegmentPlan]:
    validate_character_reference_coverage(realized)
    graph_shots = {
        str(shot.get("shot_id") or ""): shot
        for shot in graph.get("shots", [])
        if isinstance(shot, dict)
    }
    asset_names = {
        str(asset.get("asset_id") or ""): str(asset.get("name") or asset.get("asset_id") or "")
        for asset in graph.get("assets", [])
        if isinstance(asset, dict)
    }
    character_asset_ids = {
        str(asset.get("asset_id") or "")
        for asset in realized.get("character_asset_plan", {}).get("character_assets", [])
        if isinstance(asset, dict)
    }
    asset_names.update(
        {
            str(asset.get("asset_id") or ""): str(
                asset.get("name") or asset.get("asset_id") or ""
            )
            for asset in realized.get("character_asset_plan", {}).get("character_assets", [])
            if isinstance(asset, dict)
        }
    )
    segments: list[SegmentPlan] = []
    for index, item in enumerate(realized.get("segments", []), start=1):
        shot_id = str(item["shot_id"])
        template_shot_id = str(item.get("template_shot_id") or shot_id)
        shot = graph_shots.get(template_shot_id)
        if shot is None:
            shot = {
                "shot_id": shot_id,
                "phase": item.get("story_function") or item.get("narrative_phase") or "",
                "teaching_function": item.get("visual_delegation") or "",
                "duration_target": item.get("visual_target_seconds") or 6,
                "audio_cue": "",
                "sync": {},
            }
        sync = shot.get("sync") or {}
        planned_seconds = float(item.get("narration_target_seconds") or shot.get("duration_target", 6))
        segments.append(
            SegmentPlan(
                index=index,
                title=shot_id,
                story_beat=str(shot.get("phase", "")),
                teaching_goal=str(shot.get("teaching_function", "")),
                image_prompt="",
                video_prompt=str(item["video_prompt"]).strip(),
                audio_text=str(item["audio_text"]).strip(),
                duration=max(1, int(math.ceil(planned_seconds))),
                image_prompts=[],
                script_beat=str(shot.get("audio_cue", "")),
                av_sync_plan=[
                    {
                        "visual_key_moment": sync.get("visual_key_moment", ""),
                        "motion_key_moment": sync.get("motion_key_moment", ""),
                    }
                ],
                asset_refs=[str(value) for value in item.get("asset_refs", [])],
                reference_asset_labels=[
                    asset_names.get(str(value), str(value))
                    for value in item.get("asset_refs", [])
                    if str(value) in character_asset_ids
                ],
            )
        )
    return segments


def build_group_audio_segments(
    source_segments: list[SegmentPlan],
    edit_groups: list[dict[str, Any]],
) -> list[SegmentPlan]:
    source_by_id = {segment.title: segment for segment in source_segments}
    result: list[SegmentPlan] = []
    for index, group in enumerate(edit_groups, start=1):
        narration_ids = [str(value) for value in group.get("narration_shot_ids", [])]
        parts = [source_by_id[shot_id].audio_text.strip() for shot_id in narration_ids]
        parts = [part for part in parts if part]
        audio_text = "".join(
            part if part[-1:] in "。！？!?；;" else part + "。"
            for part in parts
        )
        result.append(
            SegmentPlan(
                index=index,
                title=str(group.get("group_id") or f"g{index:02d}"),
                story_beat=str(group.get("relation") or "edit_group"),
                teaching_goal="shared timeline narration for " + ",".join(group.get("source_shot_ids", [])),
                image_prompt="",
                video_prompt="",
                audio_text=audio_text,
                duration=max(0.05, float(group.get("target_seconds") or 1)),
                image_prompts=[],
                script_beat=" + ".join(narration_ids),
                av_sync_plan=[],
            )
        )
    return result


def realize_edit_group_timings(
    edit_groups: list[dict[str, Any]],
    group_segments: list[SegmentPlan],
    audio_seconds_by_index: dict[int, float],
) -> list[dict[str, Any]]:
    if len(edit_groups) != len(group_segments):
        raise ValueError("Edit groups and audio segments must have the same length")

    realized_groups: list[dict[str, Any]] = []
    for group, audio_segment in zip(edit_groups, group_segments):
        group_id = str(group.get("group_id") or "")
        if group_id != audio_segment.title:
            raise ValueError(f"Edit group/audio segment mismatch: {group_id} != {audio_segment.title}")
        audio_seconds = max(0.05, float(audio_seconds_by_index[audio_segment.index]))
        sequence = group.get("visual_sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"Edit group has no visual sequence: {group_id}")
        planned_allocations = [max(0.01, float(item.get("seconds") or 0)) for item in sequence]
        planned_total = sum(planned_allocations)
        allocations = [audio_seconds * value / planned_total for value in planned_allocations]
        allocations[-1] = max(0.05, audio_seconds - sum(allocations[:-1]))

        realized_sequence = []
        for item, seconds in zip(sequence, allocations):
            realized_item = dict(item)
            realized_item["seconds"] = round(seconds, 3)
            realized_sequence.append(realized_item)
        realized_sequence[-1]["seconds"] = round(
            audio_seconds - sum(float(item["seconds"]) for item in realized_sequence[:-1]),
            3,
        )

        realized_group = dict(group)
        realized_group["planned_target_seconds"] = float(group.get("target_seconds") or planned_total)
        realized_group["target_seconds"] = round(audio_seconds, 3)
        realized_group["audio_seconds"] = round(audio_seconds, 3)
        realized_group["visual_sequence"] = realized_sequence
        realized_groups.append(realized_group)
    return realized_groups


def plan_video_generation_seconds(
    edit_groups: list[dict[str, Any]],
    source_shot_ids: list[str],
    *,
    maximum_seconds: int = 8,
    supported_durations: tuple[int, ...] = VEO_SUPPORTED_DURATION_SECONDS,
) -> dict[str, int]:
    allowed = tuple(value for value in supported_durations if value <= maximum_seconds)
    if not allowed:
        raise ValueError(f"No supported Veo duration is <= {maximum_seconds}")

    requirements: dict[str, list[tuple[float, float]]] = {shot_id: [] for shot_id in source_shot_ids}
    for group in edit_groups:
        visual_ids = [
            str(item.get("shot_id") or "")
            for item in group.get("visual_sequence", [])
            if isinstance(item, dict)
        ]
        if len(visual_ids) != len(set(visual_ids)):
            raise ValueError(
                f"Edit group {group.get('group_id')} repeats a visual shot; "
                "create distinct close-reading subshots instead of replaying one clip."
            )
        for item in group.get("visual_sequence", []):
            shot_id = str(item.get("shot_id") or "")
            if shot_id not in requirements:
                raise ValueError(f"Unknown visual shot in edit group: {shot_id}")
            requirements[shot_id].append(
                (float(item.get("start_seconds") or 0), float(item.get("seconds") or 0))
            )

    result: dict[str, int] = {}
    for shot_id, uses in requirements.items():
        if not uses:
            continue
        selected = None
        for slowdown_limit in (PREFERRED_VIDEO_SLOWDOWN_FACTOR, MAX_VIDEO_SLOWDOWN_FACTOR):
            for duration in allowed:
                if all(
                    start_seconds < duration
                    and clip_seconds <= (duration - start_seconds) * slowdown_limit + 0.05
                    for start_seconds, clip_seconds in uses
                ):
                    selected = duration
                    break
            if selected is not None:
                break
        if selected is None:
            details = ", ".join(
                f"start={start:.2f}s clip={seconds:.2f}s" for start, seconds in uses
            )
            raise ValueError(
                f"Shot {shot_id} cannot cover its realized audio timeline with a "
                f"{allowed[-1]}s Veo clip ({details}); split the edit group."
            )
        result[shot_id] = selected
    return result


def plan_audio_speedups(
    edit_groups: list[dict[str, Any]],
    group_segments: list[SegmentPlan],
    audio_seconds_by_index: dict[int, float],
    *,
    maximum_video_seconds: int = 8,
) -> dict[int, float]:
    if len(edit_groups) != len(group_segments):
        raise ValueError("Edit groups and audio segments must have the same length")

    result: dict[int, float] = {}
    for group, segment in zip(edit_groups, group_segments):
        sequence = group.get("visual_sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"Edit group has no visual sequence: {group.get('group_id')}")
        capacity = sum(
            max(0.05, maximum_video_seconds - float(item.get("start_seconds") or 0))
            * MAX_VIDEO_SLOWDOWN_FACTOR
            for item in sequence
        )
        fitted_capacity = max(0.05, capacity - 0.05)
        audio_seconds = float(audio_seconds_by_index[segment.index])
        speedup = max(1.0, audio_seconds / fitted_capacity)
        if speedup > MAX_AUDIO_SPEEDUP_FACTOR:
            raise ValueError(
                f"Audio group {segment.title} needs {audio_seconds:.3f}s but its visuals cover at most "
                f"{capacity:.3f}s; required audio speedup {speedup:.3f}x exceeds "
                f"{MAX_AUDIO_SPEEDUP_FACTOR:.2f}x. Split the narration across more visuals."
            )
        result[segment.index] = round(speedup, 6)
    return result


def add_close_reading_timing_callbacks(
    edit_groups: list[dict[str, Any]],
    source_segments: list[SegmentPlan],
    group_segments: list[SegmentPlan],
    audio_seconds_by_index: dict[int, float],
    *,
    maximum_video_seconds: int = 8,
) -> list[dict[str, Any]]:
    """Add a short previous-line callback when one Veo clip cannot carry the narration."""
    if len(edit_groups) != len(group_segments):
        raise ValueError("Edit groups and audio segments must have the same length")

    source_by_id = {segment.title: segment for segment in source_segments}
    previous_source_id = ""
    for group, audio_segment in zip(edit_groups, group_segments):
        source_ids = [str(value) for value in group.get("source_shot_ids", [])]
        sequence = group.get("visual_sequence")
        audio_seconds = float(audio_seconds_by_index[audio_segment.index])
        is_close_reading = bool(source_ids) and all(
            source_by_id.get(shot_id)
            and source_by_id[shot_id].story_beat == "close_reading"
            for shot_id in source_ids
        )

        # Close-reading sections are expanded into distinct visual subshots
        # before reaching the Visual Director. Never add a previous-shot
        # callback here as a duration patch: that would replay a shot and
        # create the exact visual duplication this timeline is designed to
        # avoid. If a measured TTS duration still exceeds the planned
        # subshot capacity, let plan_audio_speedups report the real mismatch
        # instead of silently reusing footage.
        if is_close_reading:
            if source_ids:
                previous_source_id = source_ids[-1]
            continue

        if is_close_reading and isinstance(sequence, list) and len(sequence) == 1:
            capacity = (
                max(
                    0.05,
                    maximum_video_seconds - float(sequence[0].get("start_seconds") or 0),
                )
                * MAX_VIDEO_SLOWDOWN_FACTOR
            )
            required_speedup = audio_seconds / max(0.05, capacity - 0.05)
            if required_speedup > MAX_AUDIO_SPEEDUP_FACTOR and previous_source_id:
                planned_total = max(0.05, float(group.get("target_seconds") or 0))
                preferred_primary_seconds = maximum_video_seconds * PREFERRED_VIDEO_SLOWDOWN_FACTOR
                callback_audio_seconds = max(2.0, audio_seconds - preferred_primary_seconds)
                callback_weight = planned_total * callback_audio_seconds / audio_seconds
                callback_weight = min(planned_total - 0.6, max(0.6, callback_weight))

                primary = dict(sequence[0])
                primary["seconds"] = round(planned_total - callback_weight, 3)
                group["visual_sequence"] = [
                    {
                        "shot_id": previous_source_id,
                        "seconds": round(callback_weight, 3),
                        "mode": "callback",
                        "start_seconds": 0,
                    },
                    primary,
                ]
                group["relation"] = expected_edit_relation(
                    len(group.get("narration_shot_ids", [])),
                    len(group["visual_sequence"]),
                )

        if source_ids:
            previous_source_id = source_ids[-1]
    return edit_groups


def edit_group_statistics(edit_groups: list[dict[str, Any]]) -> dict[str, Any]:
    relation_counts = {
        relation: 0
        for relation in ("one_to_one", "one_to_many", "many_to_one", "many_to_many", "visual_only")
    }
    callback_count = 0
    visual_clip_count = 0
    narration_source_count = 0
    planned_seconds = 0.0
    for group in edit_groups:
        relation = str(group.get("relation") or "one_to_one")
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        sequence = group.get("visual_sequence", [])
        visual_clip_count += len(sequence)
        narration_source_count += len(group.get("narration_shot_ids", []))
        planned_seconds += float(group.get("target_seconds") or 0)
        callback_count += sum(
            1 for item in sequence if isinstance(item, dict) and item.get("mode") == "callback"
        )
    return {
        "group_count": len(edit_groups),
        "relation_counts": relation_counts,
        "non_one_to_one_count": len(edit_groups)
        - relation_counts.get("one_to_one", 0)
        - relation_counts.get("visual_only", 0),
        "narration_source_count": narration_source_count,
        "visual_clip_count": visual_clip_count,
        "callback_count": callback_count,
        "planned_seconds": round(planned_seconds, 3),
    }


class DirectorGraphPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_root) / args.run_name
        config = PipelineConfig(
            vgen_root=DEFAULT_ROOT,
            output_root=Path(args.output_root),
            qwen_image_model=Path(args.qwen_image_model),
            image_backend=str(args.image_backend),
            fps=int(args.fps),
            width=int(args.width),
            height=int(args.height),
            overwrite=bool(args.overwrite),
            cosyvoice_model=Path(args.cosyvoice_model),
            cosyvoice_repo=Path(args.cosyvoice_repo),
            prompt_wav=Path(args.prompt_wav),
            prompt_text=args.prompt_text,
            cosyvoice_python=Path(args.cosyvoice_python) if args.cosyvoice_python else None,
            audio_backend=str(getattr(args, "audio_backend", "") or os.environ.get("AUDIO_BACKEND", "local_cosyvoice")),
        )
        config.image_device = str(args.image_device)
        config.image_steps = int(args.image_steps)
        config.veo_api_key = args.veo_api_key
        config.veo_host = args.veo_host
        config.veo_api_version = args.veo_api_version
        config.veo_model_marker = args.veo_model_marker
        config.veo_duration_seconds = int(args.veo_duration_seconds)
        config.veo_timeout_seconds = int(args.veo_timeout_seconds)
        config.veo_retries = int(args.veo_retries)
        config.video_backend = str(args.video_backend)
        config.seedance_api_key = args.seedance_api_key
        config.seedance_host = args.seedance_host
        config.seedance_model_marker = args.seedance_model_marker
        config.seedance_duration_seconds = int(args.seedance_duration_seconds)
        config.seedance_resolution = args.seedance_resolution
        config.seedance_timeout_seconds = int(args.seedance_timeout_seconds)
        config.seedance_retries = int(args.seedance_retries)
        config.minimax_api_key = args.minimax_api_key
        config.minimax_host = args.minimax_host
        config.minimax_model = args.minimax_model
        config.minimax_duration_seconds = int(args.minimax_duration_seconds)
        config.minimax_resolution = args.minimax_resolution
        config.minimax_ratio = args.minimax_ratio
        config.minimax_timeout_seconds = int(args.minimax_timeout_seconds)
        config.minimax_poll_seconds = float(args.minimax_poll_seconds)
        config.minimax_retries = int(args.minimax_retries)
        if config.video_backend == "seedance20":
            config.veo_duration_seconds = config.seedance_duration_seconds
        elif config.video_backend == "minimax_h3":
            config.veo_duration_seconds = config.minimax_duration_seconds
        self.config = config
        self.use_reference_images = not bool(args.no_reference_images)
        self.image_generator = build_image_generator(config) if self.use_reference_images else None
        self.video_generator = (
            Seedance20VideoGenerator(config)
            if config.video_backend == "seedance20"
            else (
                MiniMaxH3VideoGenerator(config)
                if config.video_backend == "minimax_h3"
                else Veo31VideoGenerator(config)
            )
        )
        self.audio_generator = CosyVoiceGenerator(config)
        self.composer = MediaComposer(config)
        self.poem_card_renderer = PoemCardRenderer(
            width=config.width,
            height=config.height,
            font_path=Path(args.poem_card_font),
        )

    def run(self) -> Path | None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        graph_path = Path(self.args.director_graph or DEFAULT_GRAPH)
        graph = load_json(graph_path)
        warnings = validate_graph(graph)

        (self.output_dir / "director_graph.input.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.output_dir / "validation.json").write_text(
            json.dumps(
                {"ok": True, "input_mode": "director_graph", "warnings": warnings},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if self.args.resume_media:
            realized = load_existing_text_plan(self.output_dir, graph)
        else:
            realizer = PoetryDirectorGraphRealizer(
                base_url=self.args.llm_base_url,
                api_key=self.args.llm_api_key,
                model=self.args.llm_model,
                timeout_seconds=int(self.args.llm_timeout_seconds),
            )
            realized = realizer.realize(graph, self.output_dir)

        source_segments = graph_to_segments(graph, realized)
        if not self.use_reference_images:
            for segment in source_segments:
                segment.reference_asset_labels = []
                segment.reference_image_urls = []
        edit_groups = realized["edit_groups"]
        group_segments = build_group_audio_segments(source_segments, edit_groups)
        planned_group_statistics = edit_group_statistics(edit_groups)
        poem_card_spec = build_poem_card_spec(
            graph,
            realized=realized,
            minimum_seconds=float(self.args.poem_card_min_seconds),
        )
        if poem_card_spec is None:
            raise ValueError("Director graph must provide enough poem metadata to build the poem card")
        poem_card_spec.insert_after_shot_id = str(realized["poem_card_insert_after_shot_id"])
        (self.output_dir / "execution_plan.json").write_text(
            json.dumps(
                {
                    "artifact_type": "planned_execution_plan",
                    "status": "estimated_before_tts",
                    "planning_mode": realized["planning_mode"],
                    "character_asset_plan": realized["character_asset_plan"],
                    "poem_card_narration": realized["poem_card_narration"],
                    "segments": [asdict(item) for item in group_segments],
                    "source_segments": [asdict(item) for item in source_segments],
                    "edit_groups": edit_groups,
                    "edit_group_statistics": planned_group_statistics,
                    "poem_card": asdict(poem_card_spec),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.args.dry_run:
            return None

        for name in ["videos", "audio", "muxed", "character_assets"]:
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)
        if self.image_generator is None:
            reference_manifest = {"assets": [], "generation_mode": "disabled"}
        else:
            try:
                reference_manifest = generate_character_reference_images(
                    realized["character_asset_plan"],
                    self.image_generator,
                    self.output_dir,
                )
            finally:
                self.image_generator.close()
        reference_manifest_path = self.output_dir / "character_asset_manifest.json"
        reference_manifest_path.write_text(
            json.dumps(reference_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        reference_paths_by_id = {
            str(item["asset_id"]): Path(item["image_path"])
            for item in reference_manifest["assets"]
        }

        poem_card_segment = self._poem_card_segment(poem_card_spec)

        audio_segments = list(group_segments)
        audio_output_paths = {
            segment.index: self.output_dir / "audio" / f"group_{segment.index:02d}.wav"
            for segment in group_segments
        }
        audio_segments.append(poem_card_segment)
        audio_output_paths[poem_card_segment.index] = self.output_dir / "audio" / "poem_card.wav"
        audio_paths = self._generate_audio_batch(audio_segments, output_paths=audio_output_paths)
        raw_audio_seconds_by_index = {
            segment.index: self._original_audio_duration(audio_paths[segment.index])
            for segment in group_segments
        }
        edit_groups = add_close_reading_timing_callbacks(
            edit_groups,
            source_segments,
            group_segments,
            raw_audio_seconds_by_index,
            maximum_video_seconds=int(self.config.veo_duration_seconds),
        )
        planned_audio_speedups = plan_audio_speedups(
            edit_groups,
            group_segments,
            raw_audio_seconds_by_index,
            maximum_video_seconds=int(self.config.veo_duration_seconds),
        )
        for segment in group_segments:
            speedup = planned_audio_speedups[segment.index]
            if speedup > 1.0:
                self._speed_up_audio(audio_paths[segment.index], speedup)
        audio_seconds_by_index = {
            segment.index: self.composer.audio_duration(audio_paths[segment.index])
            for segment in group_segments
        }
        audio_speedup_factors = {
            segment.title: self._audio_speedup_factor(audio_paths[segment.index])
            for segment in group_segments
        }
        edit_groups = realize_edit_group_timings(
            edit_groups,
            group_segments,
            audio_seconds_by_index,
        )
        video_generation_seconds = plan_video_generation_seconds(
            edit_groups,
            [segment.title for segment in source_segments],
            maximum_seconds=int(self.config.veo_duration_seconds),
            supported_durations=(
                MINIMAX_H3_SUPPORTED_DURATION_SECONDS
                if self.config.video_backend == "minimax_h3"
                else VEO_SUPPORTED_DURATION_SECONDS
            ),
        )
        if self.config.video_backend == "veo31" and int(self.config.veo_duration_seconds) < 8 and any(
            segment.reference_asset_labels for segment in source_segments
        ):
            raise ValueError("Veo reference-to-video requires an 8-second maximum duration")
        for segment in source_segments:
            if (
                self.config.video_backend == "veo31"
                and segment.title in video_generation_seconds
                and segment.reference_asset_labels
            ):
                video_generation_seconds[segment.title] = 8
        for segment in source_segments:
            if segment.title in video_generation_seconds:
                segment.duration = video_generation_seconds[segment.title]
        group_statistics = edit_group_statistics(edit_groups)
        realized_group_segments = []
        for segment in group_segments:
            item = asdict(segment)
            item["planned_duration"] = item["duration"]
            item["duration"] = round(audio_seconds_by_index[segment.index], 3)
            item["audio_speedup_factor"] = audio_speedup_factors[segment.title]
            realized_group_segments.append(item)
        (self.output_dir / "realized_execution_plan.json").write_text(
            json.dumps(
                {
                    "artifact_type": "realized_execution_plan",
                    "status": "timed_after_tts",
                    "planning_mode": realized["planning_mode"],
                    "character_asset_plan": realized["character_asset_plan"],
                    "poem_card_narration": realized["poem_card_narration"],
                    "segments": realized_group_segments,
                    "source_segments": [asdict(item) for item in source_segments],
                    "video_generation_seconds": video_generation_seconds,
                    "audio_speedup_factors": audio_speedup_factors,
                    "edit_groups": edit_groups,
                    "edit_group_statistics": group_statistics,
                    "poem_card": asdict(poem_card_spec),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        server_context = (
            ReferenceImageServer(
                self.output_dir,
                advertised_host=str(self.args.reference_image_host),
                port=int(self.args.reference_image_port),
            )
            if reference_paths_by_id
            else contextlib.nullcontext(None)
        )
        with server_context as reference_server:
            if reference_server is not None:
                for item in reference_manifest["assets"]:
                    item["published_url"] = reference_server.url_for(Path(item["image_path"]))
                reference_manifest_path.write_text(
                    json.dumps(reference_manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            required_visual_ids = {
                str(item.get("shot_id") or "")
                for group in edit_groups
                for item in group.get("visual_sequence", [])
                if isinstance(item, dict)
            }
            video_jobs: list[tuple[SegmentPlan, Path, list[Path]]] = []
            for segment in source_segments:
                if segment.title not in required_visual_ids:
                    continue
                reference_paths = [
                    reference_paths_by_id[asset_id]
                    for asset_id in segment.asset_refs
                    if asset_id in reference_paths_by_id
                ]
                segment.reference_image_urls = [
                    reference_server.url_for(path) for path in reference_paths
                ] if reference_server is not None else []
                video_path = self.output_dir / "videos" / f"visual_{segment.title}.mp4"
                video_jobs.append((segment, video_path, reference_paths))

            source_video_paths: dict[str, Path] = {}
            video_workers = int(self.args.video_workers)
            if video_workers == 1:
                for segment, video_path, reference_paths in video_jobs:
                    source_video_paths[segment.title] = self.video_generator.generate(
                        segment,
                        None,
                        video_path,
                        reference_paths,
                    )
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=video_workers) as pool:
                    video_futures = {
                        pool.submit(
                            self.video_generator.generate,
                            segment,
                            None,
                            video_path,
                            reference_paths,
                        ): segment.title
                        for segment, video_path, reference_paths in video_jobs
                    }
                    try:
                        for future in concurrent.futures.as_completed(video_futures):
                            shot_id = video_futures[future]
                            source_video_paths[shot_id] = future.result()
                    except Exception:
                        for future in video_futures:
                            future.cancel()
                        raise

            muxed_paths: list[Path] = []
            report_segments: list[dict[str, Any]] = []
            for group_index, (group, audio_segment) in enumerate(zip(edit_groups, group_segments), start=1):
                audio_path = audio_paths[audio_segment.index]
                audio_seconds = self.composer.audio_duration(audio_path)
                group_video_path = self._compose_edit_group_video(
                    group,
                    source_video_paths,
                    audio_seconds,
                )
                muxed_path = self.output_dir / "muxed" / f"segment_{group_index:02d}.mp4"
                self.composer.mux(group_video_path, audio_path, muxed_path, shortest=True)
                muxed_paths.append(muxed_path)
                report_segments.append(
                    {
                        "kind": "edit_group",
                        "index": group_index,
                        "shot_id": audio_segment.title,
                        "source_shot_ids": group.get("source_shot_ids", []),
                        "narration_shot_ids": group.get("narration_shot_ids", []),
                        "visual_sequence": group.get("visual_sequence", []),
                        "relation": group.get("relation", "one_to_one"),
                        "planned_target_seconds": float(
                            group.get("planned_target_seconds") or audio_segment.duration
                        ),
                        "target_seconds": float(group.get("target_seconds") or audio_segment.duration),
                        "video_seconds": self.composer.media_duration(group_video_path),
                        "audio_seconds": self.composer.audio_duration(audio_path),
                        "muxed_seconds": self.composer.media_duration(muxed_path),
                    }
                )

        poem_card_path, poem_card_report = self._generate_poem_card_assets(
            poem_card_spec,
            audio_paths[poem_card_segment.index],
        )
        self._insert_poem_card(
            muxed_paths,
            report_segments,
            poem_card_path,
            poem_card_report,
            poem_card_spec.insert_after_shot_id,
        )

        final_path = self.output_dir / "final_director_graph.mp4"
        self.composer.concat(muxed_paths, final_path, reencode=True)
        report = {
            "director_graph": str(graph_path.resolve()),
            "run_name": self.args.run_name,
            "planning_mode": realized["planning_mode"],
            "character_asset_plan": realized["character_asset_plan"],
            "character_asset_manifest": reference_manifest,
            "reference_images_used": bool(reference_manifest["assets"]),
            "video_backend": (
                f"remote_{self.config.video_backend}_with_character_references"
                if reference_manifest["assets"]
                else f"remote_{self.config.video_backend}_text_to_video"
            ),
            "edit_group_statistics": group_statistics,
            "audio_speedup_factors": audio_speedup_factors,
            "segments": report_segments,
            "poem_card": poem_card_report,
            "final_video": str(final_path),
            "final_seconds": self.composer.media_duration(final_path),
        }
        report_path = self.output_dir / "run_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        subtitled_path = self._add_subtitles(final_path, report_path)
        report["subtitled_video"] = str(subtitled_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return subtitled_path

    def _compose_edit_group_video(
        self,
        group: dict[str, Any],
        source_video_paths: dict[str, Path],
        audio_seconds: float,
    ) -> Path:
        sequence = group.get("visual_sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"Edit group has no visual_sequence: {group.get('group_id')}")
        requested = [max(0.01, float(item.get("seconds") or 0)) for item in sequence]
        requested_total = sum(requested)
        allocations = [audio_seconds * value / requested_total for value in requested]
        if allocations:
            allocations[-1] = max(0.05, audio_seconds - sum(allocations[:-1]))

        group_id = "".join(
            char if char.isalnum() or char in "_-" else "_"
            for char in str(group.get("group_id") or "group")
        )
        parts_dir = self.output_dir / "videos" / "edit_parts" / group_id
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_paths: list[Path] = []
        for index, (item, duration) in enumerate(zip(sequence, allocations), start=1):
            shot_id = str(item.get("shot_id") or "")
            source_path = source_video_paths.get(shot_id)
            if source_path is None:
                raise ValueError(f"Unknown visual shot in edit group {group_id}: {shot_id}")
            part_path = parts_dir / f"part_{index:02d}_{shot_id}.mp4"
            self._trim_or_extend_video(
                source_path=source_path,
                start_seconds=float(item.get("start_seconds") or 0),
                duration_seconds=duration,
                output_path=part_path,
            )
            part_paths.append(part_path)

        if len(part_paths) == 1:
            return part_paths[0]
        output_path = self.output_dir / "videos" / "edit_groups" / f"{group_id}.mp4"
        if output_path.exists() and not self.config.overwrite:
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [self.composer._ffmpeg_executable(), "-y"]
        for part_path in part_paths:
            command.extend(["-i", str(part_path)])
        filter_parts = [
            f"[{index}:v:0]setpts=PTS-STARTPTS,setsar=1,format=yuv420p[v{index}]"
            for index in range(len(part_paths))
        ]
        filter_parts.append(
            "".join(f"[v{index}]" for index in range(len(part_paths)))
            + f"concat=n={len(part_paths)}:v=1:a=0[v]"
        )
        command.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[v]",
                "-r",
                str(self.config.fps),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                str(output_path),
            ]
        )
        self.composer._run_ffmpeg(command)
        return output_path

    def _trim_or_extend_video(
        self,
        *,
        source_path: Path,
        start_seconds: float,
        duration_seconds: float,
        output_path: Path,
    ) -> Path:
        if output_path.exists() and not self.config.overwrite:
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target_seconds = max(0.05, duration_seconds)
        source_seconds = self.composer.media_duration(source_path)
        available_seconds = source_seconds - max(0.0, start_seconds)
        if available_seconds <= 0.05:
            raise ValueError(
                f"Video slice starts beyond usable footage: {source_path} "
                f"(start={start_seconds:.3f}s, source={source_seconds:.3f}s)"
            )

        if target_seconds > available_seconds:
            slowdown_factor = target_seconds / available_seconds
            if slowdown_factor > MAX_VIDEO_SLOWDOWN_FACTOR:
                raise ValueError(
                    f"Narration needs {target_seconds:.3f}s but {source_path.name} has only "
                    f"{available_seconds:.3f}s after the requested start. Split the edit group; "
                    f"required video slowdown {slowdown_factor:.3f}x exceeds "
                    f"{MAX_VIDEO_SLOWDOWN_FACTOR:.2f}x."
                )
            timing_filter = (
                f"trim=duration={available_seconds:.6f},"
                f"setpts=(PTS-STARTPTS)*{slowdown_factor:.8f}"
            )
        else:
            timing_filter = f"trim=duration={target_seconds:.6f},setpts=PTS-STARTPTS"

        video_filter = (
            f"{timing_filter},"
            f"tpad=stop_mode=clone:stop_duration={MAX_END_FRAME_PADDING_SECONDS:.3f},"
            f"trim=duration={target_seconds:.6f},setsar=1,format=yuv420p"
        )
        self.composer._run_ffmpeg(
            [
                self.composer._ffmpeg_executable(),
                "-y",
                "-ss",
                f"{max(0.0, start_seconds):.3f}",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-vf",
                video_filter,
                "-r",
                str(self.config.fps),
                "-t",
                f"{target_seconds:.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                str(output_path),
            ]
        )
        return output_path

    def _add_subtitles(self, input_video: Path, report_path: Path) -> Path:
        output_video = input_video.with_name(input_video.stem + "_subtitled.mp4")
        ass_path = self.output_dir / "subtitles" / f"{input_video.stem}.ass"
        if output_video.exists() and ass_path.exists() and not self.config.overwrite:
            return output_video
        recover_partial_output = output_video.exists() or ass_path.exists()
        command = [
            sys.executable,
            "-m",
            "vgen.add_subtitles",
            str(self.output_dir),
            "--source-mode",
            "director-graph",
            "--input-video",
            str(input_video),
            "--output-video",
            str(output_video),
            "--plan",
            str(self.output_dir / "execution_plan.json"),
            "--run-report",
            str(report_path),
            "--ass-path",
            str(ass_path),
            "--font-file",
            str(self.args.poem_card_font),
            "--font-size",
            str(self.args.subtitle_font_size),
            "--subtitle-width-ratio",
            str(self.args.subtitle_width_ratio),
            "--max-lines",
            str(self.args.subtitle_max_lines),
            "--margin-v",
            str(self.args.subtitle_margin_v),
            "--crf",
            str(self.args.subtitle_crf),
            "--preset",
            str(self.args.subtitle_preset),
        ]
        if self.config.overwrite or recover_partial_output:
            command.append("--overwrite")
        subprocess.run(command, check=True)
        if not output_video.exists():
            raise RuntimeError(f"Subtitle stage did not create: {output_video}")
        return output_video

    @staticmethod
    def _poem_card_segment(spec: PoemCardSpec) -> SegmentPlan:
        return SegmentPlan(
            index=0,
            title="poem_card",
            story_beat="poem_text_interlude",
            teaching_goal="show the complete poem between causal background and close reading",
            image_prompt="",
            video_prompt="",
            audio_text=spec.audio_text,
            duration=max(1, int(math.ceil(spec.minimum_seconds))),
            script_beat=spec.audio_source,
        )

    def _generate_poem_card_assets(
        self,
        spec: PoemCardSpec,
        audio_path: Path,
    ) -> tuple[Path, dict[str, Any]]:
        render_result = self.poem_card_renderer.render(spec, self.output_dir / "poem_card")
        audio_seconds = self.composer.audio_duration(audio_path)
        card_seconds = max(spec.minimum_seconds, audio_seconds)
        mux_audio_path = self._pad_audio_to_duration(audio_path, card_seconds)
        silent_video_path = self.output_dir / "videos" / "poem_card_silent.mp4"
        muxed_path = self.output_dir / "muxed" / "poem_card.mp4"
        self.composer.image_to_video(render_result.png_path, card_seconds, silent_video_path)
        self.composer.mux(silent_video_path, mux_audio_path, muxed_path, shortest=True)
        report = {
            "kind": "poem_card",
            "index": 0,
            "shot_id": "poem_card",
            "insert_after_shot_id": spec.insert_after_shot_id,
            "audio_source": spec.audio_source,
            "audio_source_reference": spec.audio_source_reference,
            "audio_text": spec.audio_text,
            "html": str(render_result.html_path),
            "image": str(render_result.png_path),
            "audio": str(mux_audio_path),
            "target_seconds": spec.minimum_seconds,
            "audio_seconds": self.composer.audio_duration(mux_audio_path),
            "muxed_seconds": self.composer.media_duration(muxed_path),
        }
        (self.output_dir / "poem_card" / "run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return muxed_path, report

    def _pad_audio_to_duration(self, audio_path: Path, target_seconds: float) -> Path:
        audio_seconds = self.composer.audio_duration(audio_path)
        if audio_seconds + 0.05 >= target_seconds:
            return audio_path
        padded_path = self.output_dir / "audio" / "poem_card_padded.wav"
        if padded_path.exists() and not self.config.overwrite:
            if self.composer.audio_duration(padded_path) + 0.05 >= target_seconds:
                return padded_path
        self.composer._run_ffmpeg(
            [
                self.composer._ffmpeg_executable(),
                "-y",
                "-i",
                str(audio_path),
                "-af",
                "apad",
                "-t",
                f"{target_seconds:.3f}",
                "-ar",
                "24000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(padded_path),
            ]
        )
        return padded_path

    @staticmethod
    def _insert_poem_card(
        muxed_paths: list[Path],
        report_segments: list[dict[str, Any]],
        poem_card_path: Path,
        poem_card_report: dict[str, Any],
        insert_after_shot_id: str,
    ) -> None:
        insert_index = next(
            (
                index
                for index, item in enumerate(report_segments)
                if item.get("shot_id") == insert_after_shot_id
                or insert_after_shot_id in item.get("source_shot_ids", [])
            ),
            None,
        )
        if insert_index is None:
            raise ValueError(f"Poem-card insertion shot is not present: {insert_after_shot_id}")
        muxed_paths.insert(insert_index + 1, poem_card_path)
        report_segments.insert(insert_index + 1, poem_card_report)

    def _generate_audio_batch(
        self,
        segments: list[SegmentPlan],
        *,
        output_paths: dict[int, Path] | None = None,
    ) -> dict[int, Path]:
        if output_paths is None:
            output_paths = {
                segment.index: self.output_dir / "audio" / f"segment_{segment.index:02d}.wav"
                for segment in segments
            }
        missing_output_paths = [segment.index for segment in segments if segment.index not in output_paths]
        if missing_output_paths:
            raise ValueError(f"Missing audio output paths for segment indexes: {missing_output_paths}")
        silent_segments = [segment for segment in segments if not segment.audio_text.strip()]
        spoken_segments = [
            segment
            for segment in segments
            if segment.audio_text.strip()
            and (self.config.overwrite or not output_paths[segment.index].exists())
        ]
        for segment in silent_segments:
            self._write_silent_audio(output_paths[segment.index], float(segment.duration))
        if not spoken_segments:
            return output_paths
        if str(getattr(self.config, "audio_backend", "")) == "dashscope_cosyvoice":
            for segment in spoken_segments:
                self.audio_generator.generate(segment, output_paths[segment.index])
            return output_paths
        cosyvoice_python = self.config.cosyvoice_python
        if cosyvoice_python is None:
            for segment in spoken_segments:
                self.audio_generator.generate(segment, output_paths[segment.index])
            return output_paths

        request_path = self.output_dir / "audio" / "cosyvoice_batch_request.json"
        request = {
            "segments": [
                {"segment": asdict(segment), "output_path": str(output_paths[segment.index])}
                for segment in spoken_segments
            ],
            "config": {
                "vgen_root": str(self.config.vgen_root),
                "cosyvoice_model": str(self.config.cosyvoice_model),
                "cosyvoice_repo": str(self.config.cosyvoice_repo),
                "prompt_wav": str(self.config.prompt_wav) if self.config.prompt_wav is not None else "",
                "prompt_text": self.config.prompt_text,
                "overwrite": self.config.overwrite,
            },
        }
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        env = os.environ.copy()
        repo = self.config.cosyvoice_repo
        matcha = repo / "third_party" / "Matcha-TTS"
        python_paths = [str(self.config.vgen_root), str(repo), str(matcha)]
        env["PYTHONPATH"] = os.pathsep.join(python_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        subprocess.run(
            [
                str(cosyvoice_python),
                "-m",
                "vgen.director_graph_pipeline",
                "--cosyvoice-batch-worker-request",
                str(request_path),
            ],
            env=env,
            check=True,
        )
        missing = [str(path) for path in output_paths.values() if not path.exists()]
        if missing:
            raise RuntimeError(f"CosyVoice batch worker did not create: {missing}")
        return output_paths

    def _speed_up_audio(self, audio_path: Path, speedup: float) -> None:
        if speedup <= 1.0:
            return
        original_path = audio_path.with_name(audio_path.stem + "_original.wav")
        if not original_path.exists():
            audio_path.replace(original_path)
        temporary_path = audio_path.with_name(audio_path.stem + "_fitted.tmp.wav")
        self.composer._run_ffmpeg(
            [
                self.composer._ffmpeg_executable(),
                "-y",
                "-i",
                str(original_path),
                "-filter:a",
                f"atempo={speedup:.8f}",
                "-ar",
                "24000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(temporary_path),
            ]
        )
        temporary_path.replace(audio_path)

    def _audio_speedup_factor(self, audio_path: Path) -> float:
        original_path = audio_path.with_name(audio_path.stem + "_original.wav")
        if not original_path.exists():
            return 1.0
        original_seconds = self.composer.audio_duration(original_path)
        fitted_seconds = self.composer.audio_duration(audio_path)
        return round(original_seconds / max(0.001, fitted_seconds), 4)

    def _original_audio_duration(self, audio_path: Path) -> float:
        original_path = audio_path.with_name(audio_path.stem + "_original.wav")
        return self.composer.audio_duration(original_path if original_path.exists() else audio_path)

    def _write_silent_audio(self, output_path: Path, duration_seconds: float, sample_rate: int = 24000) -> Path:
        if output_path.exists() and not self.config.overwrite:
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_count = max(1, round(max(0.05, duration_seconds) * sample_rate))
        with wave.open(str(output_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(sample_rate)
            audio_file.writeframes(b"\x00\x00" * frame_count)
        return output_path


def run_cosyvoice_batch_worker(request_path: Path) -> None:
    request = load_json(request_path)
    payload = request["config"]
    prompt_wav = str(payload.get("prompt_wav", ""))
    config = PipelineConfig(
        vgen_root=Path(payload["vgen_root"]),
        cosyvoice_model=Path(payload["cosyvoice_model"]),
        cosyvoice_repo=Path(payload["cosyvoice_repo"]),
        prompt_wav=Path(prompt_wav) if prompt_wav else None,
        prompt_text=str(payload["prompt_text"]),
        overwrite=bool(payload.get("overwrite", False)),
        cosyvoice_python=None,
    )
    generator = CosyVoiceGenerator(config)
    for item in request.get("segments", []):
        segment = SegmentPlan(**item["segment"])
        generator.generate(segment, Path(item["output_path"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a causal director graph through a remote video backend.")
    parser.add_argument("--director-graph", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "doubao/doubao-seed-2-0-lite"))
    parser.add_argument(
        "--llm-timeout-seconds",
        type=int,
        default=int(os.environ.get("LLM_TIMEOUT_SECONDS", "1200")),
    )
    parser.add_argument(
        "--video-backend",
        choices=["veo31", "seedance20", "minimax_h3"],
        default=os.environ.get("VIDEO_BACKEND", "veo31"),
    )
    parser.add_argument("--veo-api-key", default=os.environ.get("VEO_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    parser.add_argument("--veo-host", default=os.environ.get("VEO_HOST", ""))
    parser.add_argument("--veo-api-version", default=os.environ.get("VEO_API_VERSION", "v2.03"))
    parser.add_argument("--veo-model-marker", default=os.environ.get("VEO_MODEL_MARKER", "api_google_veo-3.1-generate-001"))
    parser.add_argument(
        "--veo-duration-seconds",
        type=int,
        default=int(os.environ.get("VEO_DURATION_SECONDS", "8")),
        help="Maximum Veo clip duration; each request is selected dynamically from 4, 6, or 8 seconds.",
    )
    parser.add_argument("--veo-timeout-seconds", type=int, default=int(os.environ.get("VEO_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--veo-retries", type=int, default=int(os.environ.get("VEO_RETRIES", "2")))
    parser.add_argument("--seedance-api-key", default=os.environ.get("SEEDANCE_API_KEY", ""))
    parser.add_argument(
        "--seedance-host",
        default=os.environ.get("SEEDANCE_HOST", ""),
    )
    parser.add_argument(
        "--seedance-model-marker",
        default=os.environ.get(
            "SEEDANCE_MODEL_MARKER", "api_doubao_doubao-seedance-2-0-260128"
        ),
    )
    parser.add_argument(
        "--seedance-duration-seconds",
        type=int,
        default=int(os.environ.get("SEEDANCE_DURATION_SECONDS", "8")),
        help="Maximum Seedance clip duration; director timing currently selects 4, 6, or 8 seconds.",
    )
    parser.add_argument(
        "--seedance-resolution",
        default=os.environ.get("SEEDANCE_RESOLUTION", "720p"),
    )
    parser.add_argument(
        "--seedance-timeout-seconds",
        type=int,
        default=int(os.environ.get("SEEDANCE_TIMEOUT_SECONDS", "900")),
    )
    parser.add_argument(
        "--seedance-retries",
        type=int,
        default=int(os.environ.get("SEEDANCE_RETRIES", "2")),
    )
    parser.add_argument(
        "--minimax-api-key",
        default=os.environ.get("MINIMAX_API_KEY", ""),
        help="MiniMax-H3 app_id:app_key token.",
    )
    parser.add_argument(
        "--minimax-host",
        default=os.environ.get(
            "MINIMAX_HOST", ""
        ),
    )
    parser.add_argument(
        "--minimax-model",
        default=os.environ.get("MINIMAX_MODEL", "MiniMax-H3"),
    )
    parser.add_argument(
        "--minimax-duration-seconds",
        type=int,
        default=int(os.environ.get("MINIMAX_DURATION_SECONDS", "4")),
        help="Maximum MiniMax-H3 clip duration; defaults to the 4-second API example.",
    )
    parser.add_argument(
        "--minimax-resolution",
        default=os.environ.get("MINIMAX_RESOLUTION", "768P"),
    )
    parser.add_argument(
        "--minimax-ratio",
        default=os.environ.get("MINIMAX_RATIO", "adaptive"),
    )
    parser.add_argument(
        "--minimax-timeout-seconds",
        type=int,
        default=int(os.environ.get("MINIMAX_TIMEOUT_SECONDS", "1800")),
    )
    parser.add_argument(
        "--minimax-poll-seconds",
        type=float,
        default=float(os.environ.get("MINIMAX_POLL_SECONDS", "10")),
    )
    parser.add_argument(
        "--minimax-retries",
        type=int,
        default=int(os.environ.get("MINIMAX_RETRIES", "2")),
    )
    parser.add_argument("--video-workers", type=int, default=int(os.environ.get("VIDEO_WORKERS", "1")))
    parser.add_argument("--qwen-image-model", default=os.environ.get("QWEN_IMAGE_MODEL", str(DEFAULT_QWEN_IMAGE_MODEL)))
    parser.add_argument(
        "--image-backend",
        default=os.environ.get("IMAGE_BACKEND", "qwen_image"),
        choices=["qwen_image", "bernini_t2i", "gemini_image", "openai_image", "auto"],
    )
    parser.add_argument("--image-device", default=os.environ.get("IMAGE_DEVICE", "cuda:0"))
    parser.add_argument("--image-steps", type=int, default=int(os.environ.get("IMAGE_STEPS", "30")))
    parser.add_argument(
        "--reference-image-host",
        default=os.environ.get("VEO_REFERENCE_IMAGE_HOST", ""),
        help="Host or IP reachable by the remote video service; empty auto-detects the current node address.",
    )
    parser.add_argument(
        "--reference-image-port",
        type=int,
        default=int(os.environ.get("VEO_REFERENCE_IMAGE_PORT", "0")),
        help="Temporary HTTP port for generated references; 0 asks the OS for a free port.",
    )
    parser.add_argument(
        "--no-reference-images",
        action="store_true",
        help="Skip character image generation and send text-only video requests with dynamic 4/6/8-second durations.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--cosyvoice-model", default=os.environ.get("COSYVOICE_MODEL", str(DEFAULT_COSYVOICE_MODEL)))
    parser.add_argument("--cosyvoice-repo", default=os.environ.get("COSYVOICE_REPO", str(DEFAULT_COSYVOICE_REPO)))
    parser.add_argument("--cosyvoice-python", default=os.environ.get("COSYVOICE_PYTHON", ""))
    parser.add_argument(
        "--audio-backend",
        default=os.environ.get("AUDIO_BACKEND", "local_cosyvoice"),
        choices=["local_cosyvoice", "dashscope_cosyvoice"],
        help="local_cosyvoice uses the GPU worker; dashscope_cosyvoice uses official CosyVoice enrollment TTS.",
    )
    parser.add_argument("--prompt-wav", default=os.environ.get("PROMPT_WAV", str(DEFAULT_COSYVOICE_PROMPT_WAV)))
    parser.add_argument(
        "--prompt-text",
        default=os.environ.get("PROMPT_TEXT", DEFAULT_COSYVOICE_PROMPT_TEXT),
        help="Exact transcript of the zero-shot reference wav; do not replace it with narration copy.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-media",
        action="store_true",
        help="Continue media generation from this run's validated realized_text_plan.json.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--poem-card-min-seconds", type=float, default=8.0)
    parser.add_argument("--poem-card-font", default=str(DEFAULT_POEM_CARD_FONT))
    parser.add_argument("--subtitle-font-size", type=int, default=32)
    parser.add_argument("--subtitle-width-ratio", type=float, default=0.82)
    parser.add_argument("--subtitle-max-lines", type=int, default=1)
    parser.add_argument("--subtitle-margin-v", type=int, default=34)
    parser.add_argument("--subtitle-crf", type=int, default=18)
    parser.add_argument("--subtitle-preset", default="veryfast")
    parser.add_argument("--cosyvoice-batch-worker-request", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cosyvoice_batch_worker_request:
        run_cosyvoice_batch_worker(Path(args.cosyvoice_batch_worker_request))
        return
    if not args.run_name:
        raise ValueError("--run-name is required.")
    if not args.director_graph:
        args.director_graph = str(DEFAULT_GRAPH)
    if not args.resume_media and not args.llm_api_key:
        raise ValueError("Set OPENAI_API_KEY or pass --llm-api-key for text realization.")
    if not args.dry_run:
        if args.video_backend == "veo31" and not args.veo_api_key:
            raise ValueError("Set VEO_API_KEY or OPENAI_API_KEY for remote Veo generation.")
        if args.video_backend == "seedance20" and not (
            args.seedance_api_key
            or (os.environ.get("DOUBAO_TTS_USER") and os.environ.get("DOUBAO_TTS_KEY"))
        ):
            raise ValueError(
                "Set SEEDANCE_API_KEY or both DOUBAO_TTS_USER and DOUBAO_TTS_KEY."
            )
        if args.video_backend == "minimax_h3" and ":" not in (
            args.minimax_api_key
            or os.environ.get("MINIMAX_API_KEY", "")
            or (
                f"{os.environ.get('MINIMAX_APP_ID', '')}:"
                f"{os.environ.get('MINIMAX_APP_KEY', '')}"
            )
        ):
            raise ValueError(
                "Set MINIMAX_API_KEY in app_id:app_key format or both "
                "MINIMAX_APP_ID and MINIMAX_APP_KEY."
            )
    pipeline = DirectorGraphPipeline(args)
    result = pipeline.run()
    print(json.dumps({"output_dir": str(Path(args.output_root) / args.run_name), "final_video": str(result) if result else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
