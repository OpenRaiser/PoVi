from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


VGEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VGEN_ROOT))

from vgen.poem_card import PoemCardRenderer, build_poem_card_spec


class PoemCardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        graph_path = VGEN_ROOT / "schema" / "examples" / "qingyu_an_yuanxi.director_graph.json"
        cls.graph = json.loads(graph_path.read_text(encoding="utf-8"))

    def test_uses_close_reading_agent_narration(self) -> None:
        realized = {
            "poem_card_narration": {
                        "audio_text": (
                            "《青玉案·元夕》是宋代辛弃疾的词。先看它怎样从元宵夜的繁华转到灯火稀疏处，"
                            "再留意“花千树”把灯火写成繁花，繁花的密集和明亮托出后文寻找与转折的方向；"
                            "“那人”也可以保留字面和寄托两种读法，读者还可结合全词结构理解，不必急着下唯一结论。"
                        )
            }
        }
        spec = build_poem_card_spec(self.graph, realized=realized)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.insert_after_shot_id, "s07")
        self.assertEqual(spec.audio_source, "close_reading_agent")
        self.assertEqual(len(spec.lines), 4)
        self.assertIn("《青玉案·元夕》", spec.audio_text)
        self.assertIn("辛弃疾", spec.audio_text)

    def test_legacy_graph_material_is_not_used_as_narration(self) -> None:
        graph = json.loads(json.dumps(self.graph, ensure_ascii=False))
        with self.assertRaisesRegex(ValueError, "CloseReadingAgent"):
            build_poem_card_spec(graph)

    def test_card_narration_must_name_poem_and_author(self) -> None:
        with self.assertRaisesRegex(ValueError, "poem title and author"):
            build_poem_card_spec(
                self.graph,
                realized={"poem_card_narration": {"audio_text": "只讲灯火和结构，不说身份。"}},
            )

    def test_requires_a_narration_source(self) -> None:
        graph = json.loads(json.dumps(self.graph, ensure_ascii=False))
        with self.assertRaisesRegex(ValueError, "CloseReadingAgent"):
            build_poem_card_spec(graph)

    @unittest.skipUnless(importlib.util.find_spec("cairosvg"), "cairosvg is not installed")
    def test_renders_exact_text_card(self) -> None:
        spec = build_poem_card_spec(
            self.graph,
            realized={
                "poem_card_narration": {
                    "audio_text": (
                        "《青玉案·元夕》是宋代辛弃疾的词。先看它怎样从元宵夜的繁华转到灯火稀疏处，"
                        "再留意“花千树”把灯火写成繁花，繁花的密集和明亮托出后文寻找与转折的方向；"
                        "“那人”也可以保留字面和寄托两种读法。"
                    )
                }
            },
        )
        assert spec is not None
        renderer = PoemCardRenderer(
            width=1280,
            height=720,
            font_path=VGEN_ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = renderer.render(spec, Path(temp_dir))
            self.assertTrue(result.html_path.exists())
            self.assertTrue(result.png_path.exists())
            html_text = result.html_path.read_text(encoding="utf-8")
            for line in spec.lines:
                self.assertIn(line, html_text)


if __name__ == "__main__":
    unittest.main()
