from __future__ import annotations

import html
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


POEM_PHASES = {"poem_scene", "interpretive_turn", "summary"}


@dataclass(slots=True)
class PoemCardSpec:
    title: str
    author: str
    dynasty: str
    lines: list[str]
    audio_text: str
    audio_source: str
    insert_after_shot_id: str
    audio_source_reference: str = ""
    minimum_seconds: float = 8.0


@dataclass(slots=True)
class PoemCardRenderResult:
    html_path: Path
    png_path: Path
    metadata_path: Path


def _title_present(title: str, text: str) -> bool:
    """True if the poem title is named in *text*.

    A verbatim match is preferred, but alias forms are also accepted: a title
    such as "江神子・江城子" should satisfy narration that only names "江城子".
    """
    if title and title in text:
        return True
    for part in re.split(r"[・·\s／/]+", title or ""):
        if len(part) >= 2 and part in text:
            return True
    return False


def build_poem_card_spec(
    graph: dict[str, Any],
    *,
    realized: dict[str, Any] | None = None,
    minimum_seconds: float = 8.0,
) -> PoemCardSpec | None:
    poem = graph.get("poem") or {}
    title = str(poem.get("title") or "").strip()
    author = str(poem.get("author") or "").strip()
    dynasty = str(poem.get("dynasty") or "").strip()
    lines = _poem_lines(poem)
    if not title or not author or not lines:
        return None

    shots = sorted(graph.get("shots", []), key=lambda item: int(item.get("order", 0)))
    insert_after_shot_id = _find_insert_after_shot_id(shots)

    settings = graph.get("poem_card") if isinstance(graph.get("poem_card"), dict) else {}
    explicit_insert_after = str(settings.get("insert_after_shot_id") or "").strip()
    if explicit_insert_after:
        shot_ids = {str(shot.get("shot_id") or "") for shot in shots}
        if shot_ids and explicit_insert_after not in shot_ids:
            raise ValueError(f"poem_card.insert_after_shot_id is not present in shots: {explicit_insert_after}")
        insert_after_shot_id = explicit_insert_after
    generated = (
        realized.get("poem_card_narration")
        if isinstance(realized, dict)
        else None
    )
    if not isinstance(generated, dict):
        raise ValueError(
            "poem_card requires CloseReadingAgent poem_card_narration; "
            "legacy appreciation/knowledge_background narration is disabled"
        )
    audio_text = _clean_narration(generated.get("audio_text"))
    if not audio_text:
        raise ValueError("poem_card_narration.audio_text is empty")
    if not _title_present(title, audio_text) or author not in audio_text:
        raise ValueError(
            "poem_card_narration must name both the poem title and author"
        )
    generated_target_seconds = generated.get("target_seconds")
    if generated_target_seconds not in (None, ""):
        try:
            if float(generated_target_seconds) <= 0:
                raise ValueError("poem_card_narration.target_seconds must be positive")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "poem_card_narration.target_seconds must be a positive number"
            ) from exc

    return PoemCardSpec(
        title=title,
        author=author,
        dynasty=dynasty,
        lines=lines,
        audio_text=audio_text,
        audio_source="close_reading_agent",
        insert_after_shot_id=insert_after_shot_id,
        audio_source_reference="",
        minimum_seconds=max(1.0, float(settings.get("minimum_seconds") or minimum_seconds)),
    )


def _poem_lines(poem: dict[str, Any]) -> list[str]:
    raw_lines = poem.get("lines")
    if isinstance(raw_lines, list):
        ordered = sorted(
            (item for item in raw_lines if isinstance(item, dict)),
            key=lambda item: int(item.get("order", 0)),
        )
        lines = [str(item.get("text") or "").strip() for item in ordered]
        lines = [line for line in lines if line]
        if lines:
            return lines

    text = str(poem.get("text") or "").strip()
    if not text:
        return []
    sentences = [part.strip() for part in re.findall(r"[^。！？]+[。！？]?", text) if part.strip()]
    if len(sentences) <= 4:
        return sentences
    lines: list[str] = []
    for index in range(0, len(sentences), 2):
        lines.append("".join(sentences[index : index + 2]))
    return lines


def _find_insert_after_shot_id(shots: list[dict[str, Any]]) -> str:
    bridge_ids = [str(shot.get("shot_id") or "") for shot in shots if shot.get("phase") == "bridge_to_poem"]
    if bridge_ids:
        return bridge_ids[-1]

    first_poem_index = next(
        (index for index, shot in enumerate(shots) if str(shot.get("phase") or "") in POEM_PHASES),
        None,
    )
    if first_poem_index is None or first_poem_index == 0:
        return ""
    return str(shots[first_poem_index - 1].get("shot_id") or "")


def _clean_context(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip())
    if len(text) > 120:
        text = text[:117].rstrip("，。；：") + "。"
    return text


def _clean_narration(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


class PoemCardRenderer:
    def __init__(self, *, width: int, height: int, font_path: Path):
        self.width = int(width)
        self.height = int(height)
        self.font_path = Path(font_path)

    def render(self, spec: PoemCardSpec, output_dir: Path) -> PoemCardRenderResult:
        if not self.font_path.exists():
            raise FileNotFoundError(f"Poem-card font does not exist: {self.font_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "poem_card.html"
        png_path = output_dir / "poem_card.png"
        metadata_path = output_dir / "poem_card.json"

        svg = self._build_svg(spec)
        document = self._build_html(spec, svg)
        html_path.write_text(document, encoding="utf-8")
        self._render_svg(svg, png_path, output_dir)
        metadata_path.write_text(
            json.dumps(asdict(spec), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return PoemCardRenderResult(
            html_path=html_path,
            png_path=png_path,
            metadata_path=metadata_path,
        )

    def _build_html(self, spec: PoemCardSpec, svg: str) -> str:
        title = html.escape(f"{spec.title} - {spec.author}")
        font_uri = self.font_path.resolve().as_uri()
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    @font-face {{ font-family: 'PoemCardCJK'; src: url('{font_uri}') format('opentype'); }}
    html, body {{ margin: 0; width: 100%; height: 100%; background: #10201f; overflow: hidden; }}
    body {{ display: grid; place-items: center; }}
    svg {{ width: 100vw; height: 100vh; font-family: 'PoemCardCJK', sans-serif; }}
  </style>
</head>
<body>
{svg}
</body>
</html>
"""

    def _build_svg(self, spec: PoemCardSpec) -> str:
        width = self.width
        height = self.height
        title = html.escape(f"《{spec.title}》")
        author = html.escape(f"{spec.dynasty} · {spec.author}" if spec.dynasty else spec.author)
        line_count = max(1, len(spec.lines))
        max_chars = max(len(line) for line in spec.lines)
        poem_font_size = 38 if max_chars <= 18 else 34 if max_chars <= 24 else 30
        poem_start = 294
        poem_end = height - 126
        line_gap = 0 if line_count == 1 else min(82, (poem_end - poem_start) / (line_count - 1))
        poem_nodes = []
        for index, line in enumerate(spec.lines):
            y = poem_start + index * line_gap
            poem_nodes.append(
                f'<text x="{width / 2:.1f}" y="{y:.1f}" text-anchor="middle" '
                f'font-size="{poem_font_size}" fill="#f0eadb">{html.escape(line)}</text>'
            )
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}" font-family="Noto Sans CJK SC, sans-serif">
  <rect width="{width}" height="{height}" fill="#10201f"/>
  <path d="M0 {height * 0.18:.1f} C {width * 0.28:.1f} {height * 0.08:.1f}, {width * 0.68:.1f} {height * 0.28:.1f}, {width} {height * 0.13:.1f}" fill="none" stroke="#31514b" stroke-width="2" opacity="0.55"/>
  <path d="M0 {height * 0.87:.1f} C {width * 0.32:.1f} {height * 0.72:.1f}, {width * 0.72:.1f} {height * 0.96:.1f}, {width} {height * 0.79:.1f}" fill="none" stroke="#31514b" stroke-width="2" opacity="0.55"/>
  <rect x="54" y="48" width="8" height="88" fill="#a8493f"/>
  <text x="92" y="104" font-size="54" fill="#f5f0e5">{title}</text>
  <text x="94" y="151" font-size="26" fill="#b9c6bf">{author}</text>
  <line x1="92" y1="196" x2="{width - 92}" y2="196" stroke="#557069" stroke-width="1"/>
  {''.join(poem_nodes)}
  <line x1="{width - 252}" y1="{height - 72}" x2="{width - 92}" y2="{height - 72}" stroke="#a8493f" stroke-width="4"/>
</svg>"""

    def _render_svg(self, svg: str, output_path: Path, output_dir: Path) -> None:
        fontconfig_path = output_dir / "fontconfig.xml"
        cache_dir = output_dir / ".font-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fontconfig_path.write_text(
            "<?xml version=\"1.0\"?>\n"
            "<!DOCTYPE fontconfig SYSTEM \"fonts.dtd\">\n"
            "<fontconfig>\n"
            f"  <dir>{html.escape(str(self.font_path.parent.resolve()))}</dir>\n"
            f"  <cachedir>{html.escape(str(cache_dir.resolve()))}</cachedir>\n"
            "</fontconfig>\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["FONTCONFIG_FILE"] = str(fontconfig_path)
        env["PATH"] = os.pathsep.join([p for p in os.environ.get("PATH", "").split(os.pathsep) if p] + ["/usr/bin", "/bin"])
        fc_cache_bin = "/usr/bin/fc-cache"
        if not os.path.exists(fc_cache_bin):
            import shutil as _sh
            fc_cache_bin = _sh.which("fc-cache") or "fc-cache"
        subprocess.run([fc_cache_bin, "-f", "-r"], env=env, check=False, capture_output=True)

        previous_fontconfig = os.environ.get("FONTCONFIG_FILE")
        os.environ["FONTCONFIG_FILE"] = str(fontconfig_path)
        try:
            import cairosvg

            cairosvg.svg2png(
                bytestring=svg.encode("utf-8"),
                write_to=str(output_path),
                output_width=self.width,
                output_height=self.height,
            )
        except ImportError as exc:
            raise RuntimeError("HTML poem-card rendering requires cairosvg in the orchestration environment.") from exc
        finally:
            if previous_fontconfig is None:
                os.environ.pop("FONTCONFIG_FILE", None)
            else:
                os.environ["FONTCONFIG_FILE"] = previous_fontconfig
