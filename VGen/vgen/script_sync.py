from __future__ import annotations

import re
from typing import Any


CLAUSE_SPLIT_RE = re.compile(r"[。！？!?；;]+|\s*\n+\s*")
PAUSE_SPLIT_RE = re.compile(r"[，、：,:]+")
SYNC_PREFIX = "按旁白节奏同步："


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def split_script_beats(text: Any, max_beats: int = 3) -> list[str]:
    value = _clean_text(text)
    if not value:
        return []
    parts = [part.strip(" ，、；;。!！?？") for part in CLAUSE_SPLIT_RE.split(value) if part.strip(" ，、；;。!！?？")]
    if len(parts) <= 1:
        fallback = [part.strip(" ，、；;。!！?？") for part in PAUSE_SPLIT_RE.split(value) if part.strip(" ，、；;。!！?？")]
        if 1 < len(fallback) <= max_beats:
            parts = fallback
    if not parts:
        parts = [value]
    while len(parts) > max_beats:
        parts[-2] = f"{parts[-2]}，{parts[-1]}"
        parts.pop()
    return parts[:max_beats]


def _motion_hint(index: int, total: int) -> str:
    if total <= 1:
        return "镜头轻微推进，保持主体与情绪稳定"
    if index == 0:
        return "先交代人物或核心物象与时代环境，轻微推进起势"
    if index == total - 1:
        return "收束到教学重点或情绪落点，运动减缓并稳定停留"
    return "顺着旁白推进主体动作或空间变化，保持节奏连续"


def _sanitize_existing(items: Any) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    if not isinstance(items, list):
        return cleaned
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        normalized = {
            "time_range": _clean_text(item.get("time_range", "")),
            "audio_cue": _clean_text(item.get("audio_cue", "")),
            "visual_cue": _clean_text(item.get("visual_cue", "")),
            "motion_cue": _clean_text(item.get("motion_cue", "")),
        }
        if any(normalized.values()):
            cleaned.append(normalized)
    return cleaned


def build_default_av_sync_plan(audio_text: Any, duration: int, existing: Any = None) -> list[dict[str, str]]:
    cleaned_existing = _sanitize_existing(existing)
    if cleaned_existing:
        return cleaned_existing
    beats = split_script_beats(audio_text, max_beats=3)
    if not beats:
        return []
    total_seconds = max(1, int(duration or 1))
    item_count = len(beats)
    plans: list[dict[str, str]] = []
    start_second = 0
    for index, beat in enumerate(beats):
        if index == item_count - 1:
            end_second = total_seconds
        else:
            end_second = max(start_second + 1, round(total_seconds * (index + 1) / item_count))
        plans.append(
            {
                "time_range": f"{start_second:02d}-{end_second:02d}s",
                "audio_cue": beat,
                "visual_cue": beat,
                "motion_cue": _motion_hint(index, item_count),
            }
        )
        start_second = end_second
    return plans


def derive_full_script(segments: Any) -> str:
    lines = []
    for segment in segments or []:
        text = _clean_text(getattr(segment, "audio_text", ""))
        if text:
            lines.append(text)
    return "\n".join(lines)


def apply_script_first_sync(plan: Any) -> None:
    segments = list(getattr(plan, "segments", []) or [])
    current_full_script = _clean_text(getattr(plan, "full_script", ""))
    if not current_full_script:
        setattr(plan, "full_script", derive_full_script(segments))
    for segment in segments:
        script_beat = _clean_text(getattr(segment, "script_beat", ""))
        if not script_beat:
            script_beat = _clean_text(getattr(segment, "audio_text", ""))
            setattr(segment, "script_beat", script_beat)
        av_sync_plan = build_default_av_sync_plan(
            script_beat or getattr(segment, "audio_text", ""),
            int(getattr(segment, "duration", 1) or 1),
            existing=getattr(segment, "av_sync_plan", []),
        )
        setattr(segment, "av_sync_plan", av_sync_plan)


def build_sync_suffix(items: Any) -> str:
    cleaned = _sanitize_existing(items)
    if not cleaned:
        return ""
    snippets = []
    for item in cleaned:
        snippets.append(
            f"{item['time_range']} {item['audio_cue']}→{item['visual_cue']}/{item['motion_cue']}".strip()
        )
    return SYNC_PREFIX + "；".join(snippets)


def strip_sync_suffix(prompt: Any) -> str:
    parts = [part.strip() for part in re.split(r"[；;]", _clean_text(prompt)) if part.strip()]
    kept = [part for part in parts if not part.startswith(SYNC_PREFIX)]
    return "；".join(kept)


def upsert_sync_suffix(prompt: Any, items: Any, limit: int | None = None) -> str:
    base = strip_sync_suffix(prompt)
    suffix = build_sync_suffix(items)
    combined = f"{base}；{suffix}" if base and suffix else (suffix or base)
    if limit is not None and len(combined) > int(limit):
        combined = combined[: max(0, int(limit) - 1)].rstrip("；，,。 ") + "…"
    return combined
