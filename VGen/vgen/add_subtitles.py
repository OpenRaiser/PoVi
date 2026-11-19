from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_NAME = "final_poetry_teaching.mp4"
DEFAULT_OUTPUT_NAME = "final_poetry_teaching_subtitled.mp4"
DEFAULT_ASS_NAME = "final_poetry_teaching.ass"
DEFAULT_DIRECTOR_INPUT_NAME = "final_director_graph.mp4"
DEFAULT_DIRECTOR_PLAN_NAME = "execution_plan.json"
DEFAULT_DIRECTOR_REPORT_NAME = "run_report.json"
DEFAULT_PROJECT_FONT_FILE = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
KNOWN_FONT_FAMILY_BY_STEM = {
    "notosanscjksc-regular": "Noto Sans CJK SC",
}

SENTENCE_END = "。！？!?；;"
SOFT_BREAK = "，,、：: "
ASS_SPECIAL_REPLACEMENTS = str.maketrans({"{": "（", "}": "）"})


@dataclass(slots=True)
class SubtitleSource:
    name: str
    text: str
    duration: float


@dataclass(slots=True)
class SubtitleEvent:
    start: float
    end: float
    text: str


def resolve_ffmpeg(explicit_path: str | None = None) -> str:
    if explicit_path:
        ffmpeg = Path(explicit_path).expanduser()
        if ffmpeg.exists():
            return str(ffmpeg)
        raise FileNotFoundError(f"ffmpeg executable does not exist: {ffmpeg}")

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    raise RuntimeError("ffmpeg is not available; install ffmpeg or imageio-ffmpeg.")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ffmpeg_probe_text(ffmpeg: str, path: Path) -> str:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stderr or "") + "\n" + (result.stdout or "")


def parse_duration_from_probe(text: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def wav_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as audio_file:
            frame_rate = audio_file.getframerate()
            if frame_rate > 0:
                return audio_file.getnframes() / float(frame_rate)
    except (EOFError, OSError, wave.Error):
        return None
    return None


def media_duration(ffmpeg: str, path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(path)

    wav_seconds = wav_duration(path)
    if wav_seconds is not None:
        return wav_seconds

    duration = parse_duration_from_probe(ffmpeg_probe_text(ffmpeg, path))
    if duration is None:
        raise RuntimeError(f"Unable to determine media duration: {path}")
    return duration


def detect_video_size(ffmpeg: str, input_video: Path, default_width: int, default_height: int) -> tuple[int, int]:
    text = ffmpeg_probe_text(ffmpeg, input_video)
    for line in text.splitlines():
        if "Video:" not in line:
            continue
        match = re.search(r"(?<![\d.])(\d{3,5})x(\d{3,5})(?![\d.])", line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return default_width, default_height


def find_source_duration(ffmpeg: str, candidates: Iterable[Path], fallback: float | None = None) -> float:
    for path in candidates:
        if not path.exists():
            continue
        try:
            return media_duration(ffmpeg, path)
        except RuntimeError:
            continue
    if fallback is not None and fallback > 0:
        return fallback
    raise RuntimeError("Unable to determine subtitle source duration.")


def load_subtitle_sources(
    output_dir: Path,
    plan_path: Path,
    ffmpeg: str,
    include_intro: bool,
    intro_metadata_path: Path,
) -> list[SubtitleSource]:
    if not plan_path.exists():
        raise FileNotFoundError(f"plan.json does not exist: {plan_path}")

    sources: list[SubtitleSource] = []

    if include_intro and intro_metadata_path.exists():
        metadata = read_json(intro_metadata_path)
        intro_text = str(metadata.get("audio_text") or "").strip()
        fallback_duration = float(metadata.get("title_seconds") or 0) + float(metadata.get("poem_seconds") or 0)
        if float(metadata.get("audio_seconds") or 0) > fallback_duration:
            fallback_duration = float(metadata.get("audio_seconds") or 0)
        intro_duration = find_source_duration(
            ffmpeg,
            [output_dir / "muxed" / "intro.mp4", output_dir / "videos" / "intro_silent.mp4"],
            fallback=fallback_duration,
        )
        sources.append(SubtitleSource(name="intro", text=intro_text, duration=intro_duration))

    plan = read_json(plan_path)
    segments = sorted(plan.get("segments", []), key=lambda item: int(item.get("index", 0)))
    for item in segments:
        index = int(item.get("index", len(sources) + 1))
        text = str(item.get("audio_text") or "").strip()
        fallback_duration = float(item.get("duration") or 0)
        duration = find_source_duration(
            ffmpeg,
            [
                output_dir / "muxed" / f"segment_{index:02d}.mp4",
                output_dir / "audio" / f"segment_{index:02d}.wav",
                output_dir / "videos" / f"segment_{index:02d}.mp4",
            ],
            fallback=fallback_duration,
        )
        sources.append(SubtitleSource(name=f"segment_{index:02d}", text=text, duration=duration))

    if not sources:
        raise RuntimeError("No subtitle sources were found in the output directory.")
    return sources


def load_director_graph_subtitle_sources(
    output_dir: Path,
    plan_path: Path,
    report_path: Path,
    ffmpeg: str,
) -> list[SubtitleSource]:
    if not plan_path.exists():
        raise FileNotFoundError(f"Director execution plan does not exist: {plan_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Director run report does not exist: {report_path}")

    plan = read_json(plan_path)
    planned_by_shot_id = {
        str(item.get("title") or item.get("shot_id") or ""): item
        for item in plan.get("segments", [])
        if isinstance(item, dict)
    }
    report = read_json(report_path)
    sources: list[SubtitleSource] = []
    for position, item in enumerate(report.get("segments", []), start=1):
        if not isinstance(item, dict):
            continue
        shot_id = str(item.get("shot_id") or "").strip()
        kind = str(item.get("kind") or "shot")
        fallback_duration = float(item.get("muxed_seconds") or item.get("audio_seconds") or 0)
        if kind == "poem_card" or shot_id == "poem_card":
            card = report.get("poem_card") if isinstance(report.get("poem_card"), dict) else {}
            text = str(item.get("audio_text") or card.get("audio_text") or "").strip()
            duration = find_source_duration(
                ffmpeg,
                [
                    output_dir / "muxed" / "poem_card.mp4",
                    output_dir / "audio" / "poem_card_padded.wav",
                    output_dir / "audio" / "poem_card.wav",
                ],
                fallback=fallback_duration,
            )
            sources.append(SubtitleSource(name="poem_card", text=text, duration=duration))
            continue

        planned = planned_by_shot_id.get(shot_id, {})
        index = int(item.get("index") or planned.get("index") or position)
        text = str(planned.get("audio_text") or item.get("audio_text") or "").strip()
        duration = find_source_duration(
            ffmpeg,
            [
                output_dir / "muxed" / f"segment_{index:02d}.mp4",
                output_dir / "audio" / f"segment_{index:02d}.wav",
                output_dir / "videos" / f"segment_{index:02d}.mp4",
            ],
            fallback=fallback_duration or float(planned.get("duration") or 0),
        )
        sources.append(SubtitleSource(name=shot_id or f"segment_{index:02d}", text=text, duration=duration))

    if not sources:
        raise RuntimeError("No director-graph subtitle sources were found in the run report.")
    return sources


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()
    return text


def hard_split(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        cut = max_chars
        for index in range(max_chars, max(max_chars // 2, 1), -1):
            if remaining[index - 1] in SOFT_BREAK + SENTENCE_END:
                cut = index
                break
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def split_caption_text(text: str, max_chars: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    phrases: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        current_text = "".join(current).strip()
        if char in SENTENCE_END or (len(current_text) >= max_chars and char in SOFT_BREAK):
            if current_text:
                phrases.append(current_text)
            current = []
    tail = "".join(current).strip()
    if tail:
        phrases.append(tail)

    chunks: list[str] = []
    buffer = ""
    for phrase in phrases:
        candidate = buffer + phrase if not buffer else buffer + phrase
        if not buffer or len(candidate) <= max_chars:
            buffer = candidate
            continue
        chunks.extend(hard_split(buffer, max_chars))
        buffer = phrase
    if buffer:
        chunks.extend(hard_split(buffer, max_chars))

    return [chunk for chunk in chunks if chunk]


def merge_chunks_for_duration(chunks: list[str], duration: float, min_event_seconds: float) -> list[str]:
    if not chunks or min_event_seconds <= 0:
        return chunks
    max_events = max(1, int(math.floor(duration / min_event_seconds)))
    if len(chunks) <= max_events:
        return chunks

    merged = chunks[:]
    while len(merged) > max_events:
        best_index = min(range(len(merged) - 1), key=lambda index: len(merged[index]) + len(merged[index + 1]))
        merged[best_index] = merged[best_index] + merged.pop(best_index + 1)
    return merged


def wrap_caption_text(text: str, line_chars: int, max_lines: int) -> str:
    if line_chars <= 0 or max_lines <= 1 or len(text) <= line_chars:
        return text

    lines: list[str] = []
    remaining = text.strip()
    while remaining and len(lines) < max_lines:
        if len(remaining) <= line_chars:
            lines.append(remaining)
            remaining = ""
            break

        cut = line_chars
        for index in range(line_chars, max(line_chars // 2, 1), -1):
            if remaining[index - 1] in SOFT_BREAK + SENTENCE_END:
                cut = index
                break
        lines.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        lines[-1] = lines[-1] + remaining
    return "\n".join(lines)


def distribute_events(
    sources: Iterable[SubtitleSource],
    max_chars: int,
    line_chars: int,
    max_lines: int,
    padding_seconds: float,
    min_event_seconds: float,
) -> list[SubtitleEvent]:
    events: list[SubtitleEvent] = []
    cursor = 0.0

    for source in sources:
        duration = max(0.0, source.duration)
        chunks = split_caption_text(source.text, max_chars=max_chars)
        chunks = merge_chunks_for_duration(chunks, duration, min_event_seconds=min_event_seconds)
        if duration > 0 and chunks:
            padding = min(max(0.0, padding_seconds), duration * 0.2)
            visible_start = cursor + padding
            visible_duration = max(0.1, duration - padding * 2)
            weights = [max(4, len(chunk)) for chunk in chunks]
            total_weight = float(sum(weights)) or 1.0
            local_cursor = visible_start
            for index, chunk in enumerate(chunks):
                if index == len(chunks) - 1:
                    end = cursor + duration - padding
                else:
                    end = local_cursor + visible_duration * weights[index] / total_weight
                if end > local_cursor:
                    events.append(
                        SubtitleEvent(
                            start=local_cursor,
                            end=end,
                            text=wrap_caption_text(chunk, line_chars=line_chars, max_lines=max_lines),
                        )
                    )
                local_cursor = end
        cursor += duration

    return events


def ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours = centiseconds // 360000
    centiseconds %= 360000
    minutes = centiseconds // 6000
    centiseconds %= 6000
    secs = centiseconds // 100
    centis = centiseconds % 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def ass_escape(text: str) -> str:
    escaped = text.translate(ASS_SPECIAL_REPLACEMENTS)
    escaped = escaped.replace("\\", "\\\\")
    return escaped.replace("\n", r"\N")


def normalize_ass_color(value: str, default: str) -> str:
    value = (value or "").strip()
    if not value:
        return default
    if value.startswith("&H"):
        return value
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        red = value[1:3]
        green = value[3:5]
        blue = value[5:7]
        return f"&H00{blue}{green}{red}"
    return default


def resolve_font_name(font_name: str, font_file: Path | None) -> str:
    if font_name:
        return font_name
    if font_file is None:
        return "sans-serif"
    return KNOWN_FONT_FAMILY_BY_STEM.get(font_file.stem.lower(), font_file.stem)


def validate_subtitle_width_ratio(value: float) -> float:
    if not 0 < value <= 1:
        raise ValueError("--subtitle-width-ratio must be greater than 0 and no more than 1.")
    return value


def estimate_line_chars(width: int, font_size: int, subtitle_width_ratio: float) -> int:
    usable_width = max(1.0, width * subtitle_width_ratio)
    average_cjk_char_width = max(1.0, font_size * 0.95)
    return max(6, int(usable_width // average_cjk_char_width))


def resolve_margin_h(width: int, subtitle_width_ratio: float, explicit_margin_h: int | None) -> int:
    if explicit_margin_h is not None:
        return max(0, explicit_margin_h)
    return max(0, int(round(width * (1.0 - subtitle_width_ratio) / 2.0)))


def write_ass(
    ass_path: Path,
    events: list[SubtitleEvent],
    width: int,
    height: int,
    font_name: str,
    font_size: int,
    primary_color: str,
    outline_color: str,
    back_color: str,
    outline: float,
    shadow: float,
    margin_h: int,
    margin_v: int,
) -> None:
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    primary_color = normalize_ass_color(primary_color, "&H00FFFFFF")
    outline_color = normalize_ass_color(outline_color, "&HAA000000")
    back_color = normalize_ass_color(back_color, "&H66000000")

    content = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "Collisions: Normal",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},{back_color},"
        f"-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for event in events:
        content.append(
            f"Dialogue: 0,{ass_timestamp(event.start)},{ass_timestamp(event.end)},Default,,0,0,0,,{ass_escape(event.text)}"
        )

    ass_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def escape_filter_path(path: str) -> str:
    replacements = {
        "\\": r"\\",
        ":": r"\:",
        "'": r"\'",
        ",": r"\,",
        "[": r"\[",
        "]": r"\]",
        " ": r"\ ",
    }
    for old, new in replacements.items():
        path = path.replace(old, new)
    return path


def burn_subtitles(
    ffmpeg: str,
    input_video: Path,
    output_video: Path,
    ass_path: Path,
    work_dir: Path,
    overwrite: bool,
    crf: int,
    preset: str,
    fonts_dir: Path | None,
) -> None:
    if output_video.exists() and not overwrite:
        raise FileExistsError(f"Output video already exists; pass --overwrite to replace it: {output_video}")

    try:
        ass_filter_path = ass_path.relative_to(work_dir)
    except ValueError:
        ass_filter_path = ass_path

    ass_filter = f"ass={escape_filter_path(str(ass_filter_path))}"
    if fonts_dir is not None:
        try:
            fonts_filter_path = fonts_dir.relative_to(work_dir)
        except ValueError:
            fonts_filter_path = fonts_dir
        ass_filter += f":fontsdir={escape_filter_path(str(fonts_filter_path))}"

    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        ass_filter,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    subprocess.run(command, cwd=str(work_dir), check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add subtitles to an existing poetry teaching video.")
    parser.add_argument("output_dir", type=Path, help="Existing poetry teaching output directory.")
    parser.add_argument(
        "--source-mode",
        choices=["auto", "poetry", "director-graph"],
        default="auto",
        help="Subtitle timeline source. Auto-detects director-graph runs from execution_plan.json.",
    )
    parser.add_argument("--input-video", type=Path, default=None, help="Input final video. Defaults to output_dir/final_poetry_teaching.mp4.")
    parser.add_argument("--output-video", type=Path, default=None, help="Subtitled output video. Defaults to output_dir/final_poetry_teaching_subtitled.mp4.")
    parser.add_argument("--plan", type=Path, default=None, help="Plan JSON path. Defaults to output_dir/plan.json.")
    parser.add_argument("--run-report", type=Path, default=None, help="Director-graph run report containing final segment order and durations.")
    parser.add_argument("--intro-metadata", type=Path, default=None, help="Intro metadata path. Defaults to output_dir/intro/intro_metadata.json.")
    parser.add_argument("--ass-path", type=Path, default=None, help="Generated ASS subtitle path. Defaults to output_dir/subtitles/final_poetry_teaching.ass.")
    parser.add_argument("--ffmpeg", default=None, help="Explicit ffmpeg executable path.")
    parser.add_argument("--no-intro-subtitles", action="store_true", help="Do not add subtitles for intro narration.")
    parser.add_argument("--ass-only", action="store_true", help="Only generate ASS subtitles; do not burn them into the video.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing subtitle/video outputs.")
    parser.add_argument("--width", type=int, default=832, help="Fallback ASS canvas width if video probing fails.")
    parser.add_argument("--height", type=int, default=480, help="Fallback ASS canvas height if video probing fails.")
    parser.add_argument("--font-name", default="", help="ASS subtitle font name. Defaults to the bundled Noto Sans CJK SC font when available.")
    parser.add_argument("--font-file", type=Path, default=None, help="Optional Chinese font file used by ffmpeg/libass, such as a .ttf, .otf or .ttc file. Defaults to assets/fonts/NotoSansCJKsc-Regular.otf when available.")
    parser.add_argument("--fonts-dir", type=Path, default=None, help="Optional font directory passed to the ASS filter. Defaults to --font-file parent if provided.")
    parser.add_argument("--font-size", type=int, default=30, help="ASS subtitle font size.")
    parser.add_argument("--primary-color", default="&H00FFFFFF", help="ASS primary color, or #RRGGBB.")
    parser.add_argument("--outline-color", default="&HAA000000", help="ASS outline color, or #RRGGBB.")
    parser.add_argument("--back-color", default="&H66000000", help="ASS shadow/background color, or #RRGGBB.")
    parser.add_argument("--outline", type=float, default=2.0, help="ASS outline size.")
    parser.add_argument("--shadow", type=float, default=0.8, help="ASS shadow size.")
    parser.add_argument("--subtitle-width-ratio", type=float, default=0.70, help="Maximum subtitle line width as a ratio of video width. Defaults to 0.70.")
    parser.add_argument("--margin-h", type=int, default=None, help="ASS left/right margin. Defaults to centering subtitles inside --subtitle-width-ratio of the video width.")
    parser.add_argument("--margin-v", type=int, default=34, help="Bottom margin for subtitles.")
    parser.add_argument("--max-chars", type=int, default=None, help="Maximum characters per subtitle event before splitting. Defaults to --line-chars multiplied by --max-lines.")
    parser.add_argument("--line-chars", type=int, default=None, help="Preferred characters per subtitle line. Defaults to approximately --subtitle-width-ratio of the video width based on --font-size.")
    parser.add_argument("--max-lines", type=int, default=1, help="Maximum subtitle lines per event. Defaults to 1 for single-line subtitles.")
    parser.add_argument("--padding-seconds", type=float, default=0.12, help="Padding inside each segment timeline.")
    parser.add_argument("--min-event-seconds", type=float, default=1.2, help="Merge subtitle chunks if events would be shorter than this.")
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF used when burning subtitles.")
    parser.add_argument("--preset", default="veryfast", help="x264 preset used when burning subtitles.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    source_mode = args.source_mode
    if source_mode == "auto":
        source_mode = "director-graph" if (output_dir / DEFAULT_DIRECTOR_PLAN_NAME).exists() else "poetry"

    if source_mode == "director-graph":
        report_path = (args.run_report or output_dir / DEFAULT_DIRECTOR_REPORT_NAME).expanduser().resolve()
        input_video = (args.input_video or output_dir / DEFAULT_DIRECTOR_INPUT_NAME).expanduser().resolve()
        output_video = (args.output_video or input_video.with_name(input_video.stem + "_subtitled.mp4")).expanduser().resolve()
        plan_path = (args.plan or output_dir / DEFAULT_DIRECTOR_PLAN_NAME).expanduser().resolve()
        default_ass_path = output_dir / "subtitles" / f"{input_video.stem}.ass"
    else:
        report_path = None
        input_video = (args.input_video or output_dir / DEFAULT_INPUT_NAME).expanduser().resolve()
        output_video = (args.output_video or output_dir / DEFAULT_OUTPUT_NAME).expanduser().resolve()
        plan_path = (args.plan or output_dir / "plan.json").expanduser().resolve()
        default_ass_path = output_dir / "subtitles" / DEFAULT_ASS_NAME
    intro_metadata_path = (args.intro_metadata or output_dir / "intro" / "intro_metadata.json").expanduser().resolve()
    ass_path = (args.ass_path or default_ass_path).expanduser().resolve()

    if not input_video.exists() and not args.ass_only:
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if ass_path.exists() and not args.overwrite:
        raise FileExistsError(f"ASS subtitle file already exists; pass --overwrite to replace it: {ass_path}")

    font_file = args.font_file.expanduser().resolve() if args.font_file is not None else None
    if font_file is None and DEFAULT_PROJECT_FONT_FILE.exists():
        font_file = DEFAULT_PROJECT_FONT_FILE
    if font_file is not None and not font_file.exists():
        raise FileNotFoundError(f"Font file does not exist: {font_file}")
    fonts_dir = args.fonts_dir.expanduser().resolve() if args.fonts_dir is not None else None
    if fonts_dir is None and font_file is not None:
        fonts_dir = font_file.parent
    if fonts_dir is not None and not fonts_dir.exists():
        raise FileNotFoundError(f"Font directory does not exist: {fonts_dir}")
    font_name = resolve_font_name(args.font_name, font_file)

    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    width, height = detect_video_size(ffmpeg, input_video, args.width, args.height) if input_video.exists() else (args.width, args.height)
    subtitle_width_ratio = validate_subtitle_width_ratio(args.subtitle_width_ratio)
    line_chars = args.line_chars if args.line_chars is not None else estimate_line_chars(width, args.font_size, subtitle_width_ratio)
    max_chars = args.max_chars if args.max_chars is not None else line_chars * max(1, args.max_lines)
    margin_h = resolve_margin_h(width, subtitle_width_ratio, args.margin_h)
    if source_mode == "director-graph":
        assert report_path is not None
        sources = load_director_graph_subtitle_sources(
            output_dir=output_dir,
            plan_path=plan_path,
            report_path=report_path,
            ffmpeg=ffmpeg,
        )
    else:
        sources = load_subtitle_sources(
            output_dir=output_dir,
            plan_path=plan_path,
            ffmpeg=ffmpeg,
            include_intro=not args.no_intro_subtitles,
            intro_metadata_path=intro_metadata_path,
        )
    events = distribute_events(
        sources=sources,
        max_chars=max_chars,
        line_chars=line_chars,
        max_lines=args.max_lines,
        padding_seconds=args.padding_seconds,
        min_event_seconds=args.min_event_seconds,
    )
    if not events:
        raise RuntimeError("No subtitle events were generated.")

    write_ass(
        ass_path=ass_path,
        events=events,
        width=width,
        height=height,
        font_name=font_name,
        font_size=args.font_size,
        primary_color=args.primary_color,
        outline_color=args.outline_color,
        back_color=args.back_color,
        outline=args.outline,
        shadow=args.shadow,
        margin_h=margin_h,
        margin_v=args.margin_v,
    )

    if not args.ass_only:
        burn_subtitles(
            ffmpeg=ffmpeg,
            input_video=input_video,
            output_video=output_video,
            ass_path=ass_path,
            work_dir=output_dir,
            overwrite=args.overwrite,
            crf=args.crf,
            preset=args.preset,
            fonts_dir=fonts_dir,
        )

    print(json.dumps({
        "ass_path": str(ass_path),
        "output_video": None if args.ass_only else str(output_video),
        "events": len(events),
        "font_name": font_name,
        "fonts_dir": None if fonts_dir is None else str(fonts_dir),
        "subtitle_width_ratio": subtitle_width_ratio,
        "line_chars": line_chars,
        "max_chars": max_chars,
        "margin_h": margin_h,
        "source_mode": source_mode,
        "run_report": None if report_path is None else str(report_path),
        "sources": [{"name": source.name, "duration": source.duration, "has_text": bool(source.text)} for source in sources],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
