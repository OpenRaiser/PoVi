from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def relaxed_keyword_validation() -> bool:
    return os.environ.get("VGEN_ALLOW_FORBIDDEN_KEYWORDS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def force_json_output() -> bool:
    return os.environ.get("VGEN_FORCE_JSON", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def relaxed_intro_validation() -> bool:
    return os.environ.get("VGEN_ALLOW_INCOMPLETE_INTRO", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


NARRATION_ROLES = {
    "scene_action",
    "historical_evidence",
    "poem_quote",
    "open_question",
    "delayed_inference",
    "uncertainty",
    "visual_only",
}
POEM_PHASES = {"poem_scene", "interpretive_turn", "summary"}
STORY_FUNCTIONS = {
    "context_setup",
    "situation",
    "inciting_incident",
    "observation",
    "encounter",
    "decision",
    "action",
    "dangerous_action",
    "immediate_result",
    "consequence",
    "psychological_turn",
    "historical_aftermath",
    "long_term_consequence",
    "creation_trigger",
    "bridge_to_poem",
}
NARRATIVE_BASES = {
    "event_based",
    "historical_context",
    "object_lifecycle",
    "spatial_observation",
    "emotional_situation",
    "hybrid",
}
STORY_RELATIONS = {
    "precedes",
    "causes",
    "motivates",
    "enables",
    "obstructs",
    "contrasts_with",
    "transforms_into",
    "recalls",
    "contextualizes",
    "supports_interpretation",
    "shapes",
}
STORY_CLAIM_STATUSES = {"documented", "historically_plausible", "contested", "interpretive"}
ARC_TYPES = {"psychological", "observational", "object", "spatial", "historical"}
READING_ACTIONS = {"quote", "quote_and_explain", "explain", "connect", "synthesis", "visual_only"}
EDIT_RELATIONS = {"one_to_one", "one_to_many", "many_to_one", "many_to_many", "visual_only"}
VISUAL_MODES = {"primary", "callback"}
TENSION_DELTAS = {"rise", "hold", "turn", "release", "bridge"}
NARRATION_TECHNIQUES = {
    "scene_entry",
    "action_chain",
    "pressure_escalation",
    "withheld_result",
    "reversal",
    "consequence_reveal",
    "temporal_compression",
    "behavioral_interiority",
    "motif_callback",
    "open_bridge",
}
CHARACTER_REFERENCE_MODES = {"fixed_identity", "age_variant", "anonymous_silhouette"}
FORBIDDEN_MULTI_SCENE_PROMPT_PHRASES = (
    "匹配切",
    "切到",
    "叠化",
    "闪回",
    "蒙太奇",
    "分三段",
    "每段",
    "多次切换",
)
FORBIDDEN_EXPLICIT_PHRASES = (
    "亮暗形成反差",
    "亮和暗形成反差",
    "形成鲜明对比",
    "形成强烈反差",
    "形成反差",
    "象征着",
    "寓意着",
    "暗示着",
    "代表着",
    "衬托出",
    "意在说明",
    "恰好重合",
    "正是他的写照",
    "真正的价值",
    "真正的人生价值",
    "这里说明",
    "这里表现了",
)
FORBIDDEN_NARRATION_STAGE_DIRECTIONS = (
    "镜头",
    "画面中",
    "画面里",
    "观众可以看到",
    "运镜",
    "构图",
    "转场",
)
FORBIDDEN_TEACHING_TEMPLATES = (
    "关键意象是",
    "动静结合",
    "冷暖对比",
    "主旨是",
    "表达了",
    "本段视频",
    "这一句写了",
    "视线拉远",
    "空间感逐渐开阔",
    "历史的回响",
    "承载着",
    "凸显了",
    "映照着",
)


def compact_graph_for_agents(graph: dict[str, Any]) -> dict[str, Any]:
    mapping_path = Path(__file__).resolve().parents[2] / "schema" / "cinematic_rhetoric_mapping.json"
    cinematic_mapping = {}
    if mapping_path.exists():
        cinematic_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    narrative_mapping_path = Path(__file__).resolve().parents[2] / "schema" / "narrative_rhetoric_mapping.json"
    narrative_mapping = {}
    if narrative_mapping_path.exists():
        narrative_mapping = json.loads(narrative_mapping_path.read_text(encoding="utf-8"))
    assets = {
        str(asset["asset_id"]): {
            "asset_id": asset.get("asset_id", ""),
            "type": asset.get("type", ""),
            "name": asset.get("name", ""),
            "description": asset.get("description", ""),
            "source_node_ids": asset.get("source_node_ids", []),
            "context_node_ids": asset.get("context_node_ids", []),
            "consistency_rules": asset.get("consistency_rules", ""),
            "negative_constraints": asset.get("negative_constraints", []),
            "generation_prompt": asset.get("generation_prompt", ""),
        }
        for asset in graph.get("assets", [])
    }
    script_beats = {
        str(beat["beat_id"]): {
            "text": beat.get("text", ""),
            "rhetorical_function": beat.get("rhetorical_function", ""),
            "estimated_seconds": beat.get("estimated_seconds", 0),
            "source_node_ids": beat.get("source_node_ids", []),
            "context_node_ids": beat.get("context_node_ids", []),
        }
        for beat in graph.get("teaching_script", {}).get("beats", [])
    }
    shots = []
    for shot in sorted(graph.get("shots", []), key=lambda item: int(item.get("order", 0))):
        shots.append(
            {
                "shot_id": shot.get("shot_id"),
                "template_shot_id": shot.get("template_shot_id", shot.get("shot_id")),
                "section_id": shot.get("section_id", ""),
                "order": shot.get("order"),
                "phase": shot.get("phase"),
                "duration_target": shot.get("duration_target"),
                "audio_cue_draft": shot.get("audio_cue"),
                "teaching_function": shot.get("teaching_function"),
                "visual_goal": shot.get("visual_goal"),
                "camera": shot.get("camera"),
                "subject_motion": shot.get("subject_motion"),
                "transition_to_next": shot.get("transition_to_next"),
                "video_prompt_draft": shot.get("video_prompt"),
                "source_node_ids": shot.get("source_node_ids", []),
                "context_node_ids": shot.get("context_node_ids", []),
                "assets": [assets.get(str(asset_id), {"asset_id": asset_id}) for asset_id in shot.get("asset_refs", [])],
                "script_beats": [
                    script_beats.get(str(beat_id), {"beat_id": beat_id})
                    for beat_id in shot.get("script_beat_ids", [])
                ],
            }
        )
    causal_context = graph.get("causal_context", {})
    return {
        "poem": graph.get("poem", {}),
        "task_profile": graph.get("task_profile", {}),
        "causal_context": {
            "central_question": causal_context.get("central_question", ""),
            "poem_causal_thesis": causal_context.get("poem_causal_thesis", ""),
            "alternative_explanations": causal_context.get("alternative_explanations", []),
            "uncertainty_statement": causal_context.get("uncertainty_statement", ""),
            "context_nodes": [
                {
                    "node_id": node.get("node_id"),
                    "narrative_role": node.get("narrative_role"),
                    "claim_status": node.get("claim_status"),
                    "five_w_one_h": node.get("five_w_one_h", {}),
                    "summary": node.get("summary"),
                    "causal_relevance": node.get("causal_relevance"),
                    "visual_anchor": node.get("visual_anchor", ""),
                    "evidence_ids": node.get("evidence_ids", []),
                    "caveat": node.get("caveat", ""),
                }
                for node in causal_context.get("context_nodes", [])
            ],
            "causal_links": causal_context.get("causal_links", []),
            "causal_chain_node_ids": causal_context.get("causal_chain_node_ids", []),
            "writing_situation": causal_context.get("writing_situation", {}),
        },
        "evidence_sources": graph.get("evidence_sources", []),
        "interpretation": graph.get("interpretation", {}),
        "visual_bible": graph.get("visual_bible", {}),
        "assets": list(assets.values()),
        "cinematic_rhetoric_mapping": cinematic_mapping,
        "narrative_rhetoric_mapping": narrative_mapping,
        "shots": shots,
    }


def visual_context_for_agent(context: dict[str, Any]) -> dict[str, Any]:
    """Remove draft edit instructions while preserving visual semantics and assets."""
    visual_shots = []
    for shot in context.get("shots", []):
        camera = shot.get("camera") if isinstance(shot.get("camera"), dict) else {}
        visual_shots.append(
            {
                "shot_id": shot.get("shot_id"),
                "template_shot_id": shot.get("template_shot_id", shot.get("shot_id")),
                "order": shot.get("order"),
                "phase": shot.get("phase"),
                "duration_target": shot.get("duration_target"),
                "teaching_function": shot.get("teaching_function"),
                "camera_options": {
                    key: camera.get(key)
                    for key in ("shot_size", "angle", "movement", "speed")
                    if camera.get(key) not in (None, "")
                },
                "subject_motion": shot.get("subject_motion"),
                "source_node_ids": shot.get("source_node_ids", []),
                "context_node_ids": shot.get("context_node_ids", []),
                "visible_characters": shot.get("visible_characters", []),
                "assets": shot.get("assets", []),
            }
        )
    return {
        "poem": context.get("poem", {}),
        "task_profile": context.get("task_profile", {}),
        "causal_context": context.get("causal_context", {}),
        "interpretation": context.get("interpretation", {}),
        "visual_bible": context.get("visual_bible", {}),
        "assets": context.get("assets", []),
        "character_asset_plan": context.get("character_asset_plan", {}),
        "cinematic_rhetoric_mapping": context.get("cinematic_rhetoric_mapping", {}),
        "narrative_rhetoric_mapping": context.get("narrative_rhetoric_mapping", {}),
        "shots": visual_shots,
    }


def story_context_for_agent(context: dict[str, Any]) -> dict[str, Any]:
    context_nodes = context.get("causal_context", {}).get("context_nodes", [])
    evidence_sources = [
        {
            key: source.get(key)
            for key in (
                "source_id",
                "source_type",
                "authority_level",
                "origin",
                "relevant_claims",
            )
        }
        for source in context.get("evidence_sources", [])
        if isinstance(source, dict)
    ]
    allowed_fact_statements = []
    for node in context_nodes:
        if not isinstance(node, dict):
            continue
        dimensions = node.get("five_w_one_h") if isinstance(node.get("five_w_one_h"), dict) else {}
        allowed_fact_statements.extend(str(value) for value in dimensions.values() if str(value).strip())
        allowed_fact_statements.extend(
            str(node.get(key) or "")
            for key in ("summary", "causal_relevance", "caveat")
            if str(node.get(key) or "").strip()
        )
    for source in evidence_sources:
        allowed_fact_statements.extend(
            str(value)
            for value in source.get("relevant_claims", [])
            if str(value).strip()
        )
    return {
        "poem_identity": {
            key: context.get("poem", {}).get(key)
            for key in ("title", "author", "dynasty")
        },
        "causal_context": context.get("causal_context", {}),
        "evidence_sources": evidence_sources,
        "allowed_fact_statements": allowed_fact_statements,
        "task_constraints": context.get("task_profile", {}).get("constraints", []),
    }


def close_reading_context_for_agent(
    context: dict[str, Any],
    story_plan: dict[str, Any],
) -> dict[str, Any]:
    interpretation = context.get("interpretation", {}) or {}
    semantic_interpretation = {
        "narrative_mode": interpretation.get("narrative_mode", ""),
        "thesis": interpretation.get("thesis", ""),
        "emotional_arc": interpretation.get("emotional_arc", []),
        "background_context": interpretation.get("background_context", ""),
        "metaphor_nodes": [
            {
                key: node.get(key)
                for key in (
                    "node_id",
                    "source_line_ids",
                    "source_text",
                    "literal_image",
                    "inferred_meaning",
                    "cultural_knowledge",
                    "teaching_point",
                    "difficulty",
                    "evidence",
                    "context_node_ids",
                    "evidence_ids",
                    "interpretation_status",
                )
            }
            for node in interpretation.get("metaphor_nodes", [])
            if isinstance(node, dict)
        ],
    }
    return {
        "poem": context.get("poem", {}),
        "task_profile": context.get("task_profile", {}),
        "interpretation": semantic_interpretation,
        "alternative_explanations": (
            context.get("causal_context", {}).get("alternative_explanations", [])
        ),
        "writing_situation": (
            context.get("causal_context", {}).get("writing_situation", {})
        ),
        "knowledge_background": interpretation.get("background_context", ""),
        "context_story_summary": story_plan.get("full_story", ""),
        "context_bridge": story_plan.get("bridge_to_poem", ""),
    }


def poem_line_ids(context: dict[str, Any]) -> list[str]:
    lines = context.get("poem", {}).get("lines", [])
    if not isinstance(lines, list):
        return []
    ordered = sorted(
        (line for line in lines if isinstance(line, dict)),
        key=lambda line: int(line.get("order", 0)),
    )
    return [str(line.get("line_id") or "") for line in ordered if str(line.get("line_id") or "")]


def build_execution_context(
    context: dict[str, Any],
    narrative_plan: dict[str, Any],
    character_asset_plan: dict[str, Any],
) -> dict[str, Any]:
    """Compile agent-produced narration into the shot contract used by Visual Director."""
    assets = [
        item
        for item in character_asset_plan.get("character_assets", [])
        if isinstance(item, dict)
    ]
    metaphor_nodes = [
        item
        for item in context.get("interpretation", {}).get("metaphor_nodes", [])
        if isinstance(item, dict)
    ]
    expanded_shots: list[dict[str, Any]] = []
    for segment in narrative_plan.get("segments", []):
        if not isinstance(segment, dict):
            continue
        shot_id = str(segment.get("shot_id") or "")
        line_ids = {str(value) for value in segment.get("line_ids", [])}
        source_node_ids = [
            str(node.get("node_id") or "")
            for node in metaphor_nodes
            if line_ids.intersection(str(value) for value in node.get("source_line_ids", []))
        ]
        shot_assets = [
            asset
            for asset in assets
            if shot_id in [str(value) for value in asset.get("used_by_shot_ids", [])]
        ]
        expanded_shots.append(
            {
                "shot_id": shot_id,
                "template_shot_id": shot_id,
                "section_id": str(segment.get("section_id") or ""),
                "order": len(expanded_shots) + 1,
                "phase": str(segment.get("story_function") or segment.get("narrative_phase") or ""),
                "duration_target": float(segment.get("visual_target_seconds") or 6),
                "teaching_function": str(segment.get("visual_delegation") or ""),
                "camera": {},
                "subject_motion": "",
                "source_node_ids": source_node_ids,
                "context_node_ids": list(segment.get("context_node_ids") or []),
                "visible_characters": list(segment.get("visible_characters") or []),
                "assets": shot_assets,
            }
        )
    expanded = dict(context)
    expanded["shots"] = expanded_shots
    expanded["assets"] = assets
    expanded["character_asset_plan"] = character_asset_plan
    return expanded


def normalized_quote_units(text: str) -> list[str]:
    # Source corpora sometimes append editorial variants such as
    # "(长天 一作：高天)". They are metadata, not poem text to be recited.
    text = re.sub(
        r"(?:（[^（）]*?一作\s*[：:]?[^（）]*?）|\([^()]*?一作\s*[：:]?[^()]*?\))",
        "",
        text,
    )
    units = [
        re.sub(r"[\s，。！？、；：,.!?;:]", "", part)
        for part in re.split(r"[，。！？、；：,.!?;:]", text)
    ]
    units = [unit for unit in units if unit]
    if units:
        return units
    normalized = re.sub(r"[\s，。！？、；：,.!?;:]", "", text)
    return [normalized] if normalized else []


def unsupported_quantitative_details(text: str, allowed_fact_text: str) -> list[str]:
    normalized_allowed = re.sub(r"\s+", "", allowed_fact_text)
    patterns = (
        r"\d+(?:\.\d+)?(?:余|多)?(?:年|岁|月|日|人|名|骑|里|公里|字)",
        r"(?<![数几])[零〇一二两三四五六七八九十百千万亿]+(?:余|多)?(?:年|岁|月|日|人|名|骑|里|公里|字)",
        r"(?:数|几)(?:百|千|万|十万)(?:余|多)?",
    )
    candidates: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = match.group(0)
            if candidate == "一年" and match.start() > 0 and text[match.start() - 1] == "某":
                continue
            candidates.append(candidate)
    # These forms commonly refer back to already introduced people rather than
    # asserting a new, independently sourced headcount.
    grammatical_quantifiers = {"一人", "一名", "二人", "两人"}
    return sorted(
        {
            candidate
            for candidate in candidates
            if candidate not in grammatical_quantifiers and candidate not in normalized_allowed
        }
    )


def spoken_character_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", text))


def estimate_narration_seconds(text: str) -> float:
    characters = spoken_character_count(text)
    sentence_pauses = len(re.findall(r"[。！？!?；;]", text)) * 0.28
    short_pauses = len(re.findall(r"[，,、：:]", text)) * 0.12
    seconds = characters / 4.2 + min(1.0, sentence_pauses + short_pauses)
    return round(min(10.0, max(2.0, seconds)), 1)


def estimate_story_seconds(text: str) -> float:
    characters = spoken_character_count(text)
    sentence_pauses = len(re.findall(r"[。！？!?；;]", text)) * 0.3
    short_pauses = len(re.findall(r"[，,、：:]", text)) * 0.12
    seconds = characters / 4.2 + min(1.2, sentence_pauses + short_pauses)
    return round(min(45.0, max(3.0, seconds)), 1)


def expected_edit_relation(narration_count: int, visual_count: int) -> str:
    if narration_count == 0:
        return "visual_only"
    narration_side = "one" if narration_count == 1 else "many"
    visual_side = "one" if visual_count == 1 else "many"
    return f"{narration_side}_to_{visual_side}"


class ChatCompletionClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: int = 600, max_tokens: int | None = None):
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self.url = base_url + "/chat/completions"
        self.api_key = api_key
        self.app_id, self.app_key = (
            api_key.split(":", 1) if ":" in api_key else ("", "")
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.usage_records: list[dict[str, Any]] = []

    def complete(self, payload: dict[str, Any], *, label: str = "llm") -> str:
        if self.app_id and self.app_key:
            # The model-eval gateway expects APP_ID/APP_KEY headers for
            # app_id:app_key credentials. Sending a Bearer header at the same
            # time makes the gateway reject an otherwise valid credential.
            headers = {
                "APP_ID": self.app_id,
                "APP_KEY": self.app_key,
                "Content-Type": "application/json",
            }
        else:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        if self.max_tokens is not None:
            payload = dict(payload)
            # Use the batch-level budget as an override, not merely a cap.
            # Thinking models spend part of this budget on reasoning, so an
            # agent-local 8k/14k setting can truncate the final JSON even when
            # the caller explicitly requested a larger budget.
            payload["max_tokens"] = self.max_tokens
        if force_json_output() and "response_format" not in payload:
            payload = dict(payload)
            payload["response_format"] = {"type": "json_object"}
        if os.environ.get("VGEN_ENABLE_THINKING", "").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            payload = dict(payload)
            payload.setdefault("thinking", {"type": "enabled"})
            payload.setdefault(
                "reasoning_effort",
                os.environ.get("VGEN_REASONING_EFFORT", "high"),
            )
            payload.setdefault("stream", False)
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        data: dict[str, Any] | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8", errors="replace"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == 3:
                    raise RuntimeError(f"LLM request failed status={exc.code}: {body[:2000]}") from exc
                retry_after = exc.headers.get("Retry-After", "")
                try:
                    delay_seconds = max(float(retry_after), 20.0 * (attempt + 1))
                except ValueError:
                    delay_seconds = 20.0 * (attempt + 1)
                time.sleep(delay_seconds)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == 3:
                    raise RuntimeError(
                        f"LLM request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                time.sleep(20.0 * (attempt + 1))
        if data is None:
            raise RuntimeError("LLM request failed without a response")
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content") or ""
        if (
            not str(content).strip()
            and os.environ.get("VGEN_USE_REASONING_AS_CONTENT", "").strip().lower()
            in {"1", "true", "yes", "on"}
        ):
            reasoning_content = str(message.get("reasoning_content") or "").strip()
            # Some DeepSeek responses put a completed JSON object in
            # reasoning_content when the visible content is empty. Do not use
            # free-form chain-of-thought as an agent answer.
            if reasoning_content.startswith("{"):
                content = reasoning_content
        if not str(content).strip():
            raise RuntimeError(f"LLM returned empty content: {json.dumps(message, ensure_ascii=False)[:1000]}")
        usage = data.get("usage") or {}
        self.usage_records.append(
            {
                "call_index": len(self.usage_records) + 1,
                "label": label,
                "model": self.model,
                "usage": usage,
            }
        )
        return str(content)

    @staticmethod
    def extract_json(content: str) -> dict[str, Any]:
        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object found")
        candidate = text[start:]
        try:
            data, _ = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json

                data = repair_json(candidate, return_objects=True)
            except Exception:
                raise
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data


class OpenAISDKChatCompletionClient:
    """OpenAI-compatible client that preserves the configured base URL exactly."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: int = 1200):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=3,
        )

    def complete(self, payload: dict[str, Any]) -> str:
        response = self._client.chat.completions.create(**payload)
        content = response.choices[0].message.content or ""
        if not str(content).strip():
            raise RuntimeError("LLM returned empty content")
        return str(content)
