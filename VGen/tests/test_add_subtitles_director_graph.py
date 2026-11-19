from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


VGEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VGEN_ROOT))

from vgen.add_subtitles import SubtitleSource, distribute_events, load_director_graph_subtitle_sources


class DirectorGraphSubtitleSourcesTest(unittest.TestCase):
    def test_visual_only_gap_advances_subtitle_timeline(self) -> None:
        events = distribute_events(
            [
                SubtitleSource(name="g01", text="前句。", duration=2.0),
                SubtitleSource(name="g02", text="", duration=3.0),
                SubtitleSource(name="g03", text="后句。", duration=2.0),
            ],
            max_chars=16,
            line_chars=16,
            max_lines=1,
            padding_seconds=0,
            min_event_seconds=0.5,
        )
        self.assertEqual([event.text for event in events], ["前句。", "后句。"])
        self.assertAlmostEqual(events[1].start, 5.0)

    def test_preserves_report_order_and_poem_card(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            plan_path = output_dir / "execution_plan.json"
            report_path = output_dir / "run_report.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "segments": [
                            {"index": 1, "title": "s01", "audio_text": "背景旁白", "duration": 3},
                            {"index": 2, "title": "s02", "audio_text": "正文旁白", "duration": 4},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            report_path.write_text(
                json.dumps(
                    {
                        "segments": [
                            {"kind": "shot", "index": 1, "shot_id": "s01", "muxed_seconds": 2.8},
                            {
                                "kind": "poem_card",
                                "index": 0,
                                "shot_id": "poem_card",
                                "audio_text": "诗词插页旁白",
                                "muxed_seconds": 5.2,
                            },
                            {"kind": "shot", "index": 2, "shot_id": "s02", "muxed_seconds": 3.9},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            sources = load_director_graph_subtitle_sources(
                output_dir=output_dir,
                plan_path=plan_path,
                report_path=report_path,
                ffmpeg="ffmpeg-not-needed-for-fallbacks",
            )
            self.assertEqual([source.name for source in sources], ["s01", "poem_card", "s02"])
            self.assertEqual([source.text for source in sources], ["背景旁白", "诗词插页旁白", "正文旁白"])
            self.assertEqual([source.duration for source in sources], [2.8, 5.2, 3.9])


if __name__ == "__main__":
    unittest.main()
