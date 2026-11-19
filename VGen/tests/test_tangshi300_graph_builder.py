from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


VGEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VGEN_ROOT))

from vgen.tangshi300_graph_builder import (
    build_locked_graph_fields,
    build_source_packet,
    load_corpus,
    normalize_graph,
    select_records,
    validate_grounding_contract,
)


class PoetryGraphBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus_path = VGEN_ROOT / "data" / "examples" / "qingyu_an_yuanxi.json"
        cls.records, cls.metadata = load_corpus(cls.corpus_path)
        cls.record = cls.records[0]

    def test_loads_and_selects_exact_records(self) -> None:
        self.assertEqual(len(self.records), 1)
        selected = select_records(
            self.records,
            select_all=False,
            indices=[1],
            poem_ids=None,
            titles=None,
            limit=None,
        )
        self.assertEqual([item["title"] for item in selected], ["青玉案·元夕"])

    def test_source_packet_marks_scrape_truncation(self) -> None:
        record = dict(self.record)
        record["sections"] = [{"title": "赏析", "text": "已抓取内容\n展开阅读全文"}]
        packet = build_source_packet(record)
        self.assertTrue(packet["source_completeness"]["contains_truncated_source"])
        self.assertIn("原始抓取内容在此处截断", packet["source_sections"][0]["text"])

    def test_normalization_restores_immutable_corpus_fields(self) -> None:
        packet = build_source_packet(self.record)
        locked_fields = build_locked_graph_fields(self.record, packet, 120.0)
        graph = {
            "schema_version": "wrong",
            "doc_id": "invented",
            "poem": {"title": "错误标题"},
            "evidence_sources": [{"source_id": "invented"}],
            "poem_card": {"appreciation": "模型编写的简析"},
            "metadata": {"note": "preserved"},
        }
        normalized = normalize_graph(
            graph,
            locked_fields,
            record=self.record,
            model="test-model",
            source_hash="abc123",
        )
        self.assertEqual(normalized["poem"]["title"], "青玉案·元夕")
        self.assertEqual(normalized["poem"]["text"], self.record["content"])
        self.assertEqual(normalized["poem_card"], {"minimum_seconds": 8})
        self.assertEqual(normalized["evidence_sources"][0]["source_id"], "e01")
        self.assertEqual(normalized["metadata"]["note"], "preserved")

    def test_benchmark_identity_and_lines_are_preserved(self) -> None:
        record = {
            **self.record,
            "doc_id": "benchmark_case_01",
            "lines": [
                {"line_id": "part_a", "order": 1, "text": "东风夜放花千树。"},
                {"line_id": "part_b", "order": 2, "text": "更吹落、星如雨。"},
            ],
        }
        packet = build_source_packet(record)
        locked_fields = build_locked_graph_fields(record, packet, 180.0)
        self.assertEqual(locked_fields["doc_id"], "benchmark_case_01")
        self.assertEqual(
            [item["line_id"] for item in locked_fields["poem"]["lines"]],
            ["part_a", "part_b"],
        )

    def test_rejects_consensus_without_high_authority_source(self) -> None:
        graph = {
            "evidence_sources": [
                {
                    "source_id": "e01",
                    "authority_level": "medium",
                }
            ],
            "causal_context": {
                "context_nodes": [
                    {
                        "node_id": "c01",
                        "claim_status": "scholarly_consensus",
                        "evidence_ids": ["e01"],
                    }
                ],
                "causal_links": [],
            },
            "interpretation": {"metaphor_nodes": []},
        }
        errors = validate_grounding_contract(graph)
        self.assertTrue(any("scholarly consensus" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
