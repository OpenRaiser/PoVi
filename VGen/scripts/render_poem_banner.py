"""Render the source poem as a banner image for the case-study figure.

The figure's four columns are already full, and adding Chinese to the LaTeX
itself would force XeLaTeX plus a CJK package on whatever template the paper
uses. A rendered banner sidesteps both: it drops in as one \\includegraphics
spanning the text width, and the glyphs travel with the PDF.

Font is a parameter because the machine that builds this may not have the one
you want. Any TTF/OTF/TTC path works; --font-index picks a face out of a
collection.

    python scripts/render_poem_banner.py --out poem_banner.pdf
    python scripts/render_poem_banner.py --font /path/to/weibei.ttf --out ...
"""
from __future__ import annotations

import argparse
import pathlib

from PIL import Image, ImageDraw, ImageFont

# Xin Qiji, Qingyu An (Lantern Festival). Line breaks follow the corpus.
TITLE = "青玉案·元夕"
AUTHOR = "辛弃疾"
LINES = [
    "东风夜放花千树，更吹落、星如雨。",
    "宝马雕车香满路，凤箫声动，玉壶光转，一夜鱼龙舞。",
    "蛾儿雪柳黄金缕，笑语盈盈暗香去。",
    "众里寻他千百度；蓦然回首，那人却在，灯火阑珊处。",
]

DEFAULT_FONT = ""
INK = (28, 28, 32)
MUTED = (120, 120, 128)
PAPER = (255, 255, 255)

# Rendered at this width and scaled by LaTeX to \textwidth. 3x the typical
# printed width keeps the strokes crisp at 300 dpi.
WIDTH = 2400


def load(path: str, size: int, index: int):
    try:
        return ImageFont.truetype(path, size, index=index)
    except Exception:
        return ImageFont.truetype(DEFAULT_FONT, size, index=2)


def render(font_path: str, font_index: int, dest: pathlib.Path,
           columns: bool) -> None:
    body_size = 56 if not columns else 50
    title_size = 62
    pad = 40
    gap = 22

    body = load(font_path, body_size, font_index)
    title = load(font_path, title_size, font_index)
    byline = load(font_path, 34, font_index)

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    # Two lyric lines per row keeps the banner short enough to sit above the
    # frames without competing with them.
    rows = ["　".join(LINES[0:2]), "　".join(LINES[2:4])]
    height = pad * 2 + title_size + gap + len(rows) * (body_size + gap)

    canvas = Image.new("RGB", (WIDTH, height), PAPER)
    draw = ImageDraw.Draw(canvas)

    # Title and author on one centred line.
    head = TITLE
    hw = probe.textlength(head, font=title)
    aw = probe.textlength("　" + AUTHOR, font=byline)
    x = (WIDTH - hw - aw) / 2
    draw.text((x, pad), head, font=title, fill=INK)
    draw.text((x + hw, pad + title_size - 38), "　" + AUTHOR,
              font=byline, fill=MUTED)

    y = pad + title_size + gap
    for row in rows:
        w = probe.textlength(row, font=body)
        draw.text(((WIDTH - w) / 2, y), row, font=body, fill=INK)
        y += body_size + gap

    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest.with_suffix(".png"))
    canvas.save(dest.with_suffix(".pdf"), "PDF", resolution=300.0)
    print(f"{dest.with_suffix('.pdf')}  {WIDTH}x{height}px  "
          f"{WIDTH/300:.2f}x{height/300:.2f} inch @300dpi")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--font", default=DEFAULT_FONT,
                        help="path to a TTF/OTF/TTC with CJK glyphs")
    parser.add_argument("--font-index", type=int, default=2,
                        help="face index inside a .ttc collection")
    parser.add_argument("--columns", action="store_true",
                        help="tighter type, for a narrower banner")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    render(args.font, args.font_index, args.out, args.columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
