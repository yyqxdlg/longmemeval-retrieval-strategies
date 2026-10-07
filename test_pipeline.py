from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiment_io import merge_resume_metadata
from generation import (
    build_answer_messages,
    oracle_memories,
    reconstruct_retrieved_memories,
)
from longmemeval_eval import (
    export_hypotheses,
    export_hypotheses_by_condition,
    merge_scores,
    parse_label,
)
from save_runtime_info import disk_usage_snapshot


def sample_record() -> dict:
    return {
        "question_id": "q1",
        "question": "What drink does the user prefer?",
        "answer": "Green tea",
        "question_type": "single-session-user",
        "question_date": "2023/05/30 (Tue) 10:00",
        "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": [
            "2023/05/28 (Sun) 09:00",
            "2023/05/29 (Mon) 09:00",
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I prefer green tea.", "has_answer": True},
                {"role": "assistant", "content": "Noted.", "has_answer": False},
            ],
            [
                {"role": "user", "content": "I visited London."},
                {"role": "assistant", "content": "Sounds fun."},
            ],
        ],
    }


class GenerationInputTests(unittest.TestCase):
    def test_reconstructs_requested_memories_and_hides_labels(self):
        record = sample_record()
        memories = reconstruct_retrieved_memories(
            record,
            "s2::round_0|s1::round_0",
        )
        self.assertEqual([m["id"] for m in memories], ["s2::round_0", "s1::round_0"])

        messages, _ = build_answer_messages(
            question=record["question"],
            question_date=record["question_date"],
            retrieved_memories=memories,
        )
        prompt = json.dumps(messages)
        self.assertNotIn("has_answer", prompt)
        self.assertNotIn("answer_session_ids", prompt)
        self.assertNotIn("is_gold_session", prompt)
        self.assertLess(prompt.index("2023/05/28"), prompt.index("2023/05/29"))

    def test_disk_usage_snapshot_accepts_relative_path(self):
        snapshot = disk_usage_snapshot(Path("."))
        self.assertTrue(snapshot["root"])
        self.assertGreater(snapshot["total_bytes"], 0)
        self.assertGreaterEqual(snapshot["free_bytes"], 0)

    def test_missing_memory_id_fails(self):
        with self.assertRaises(ValueError):
            reconstruct_retrieved_memories(sample_record(), "missing::round_0")

    def test_no_retrieval_prompt_has_no_gold_or_labels(self):
        record = sample_record()
        record["answer"] = "GOLD_ONLY_SENTINEL"
        messages, _ = build_answer_messages(
            question=record["question"],
            question_date=record["question_date"],
            retrieved_memories=[],
        )
        prompt = json.dumps(messages)
        self.assertNotIn("GOLD_ONLY_SENTINEL", prompt)
        self.assertNotIn("has_answer", prompt)
        self.assertNotIn("answer_session_ids", prompt)

    def test_oracle_prompt_contains_only_raw_dialogue_and_dates(self):
        record = sample_record()
        record["answer"] = "GOLD_ONLY_SENTINEL"
        evidence = oracle_memories(record)
        self.assertEqual([item["id"] for item in evidence], ["s1::round_0"])
        messages, _ = build_answer_messages(
            question=record["question"],
            question_date=record["question_date"],
            retrieved_memories=evidence,
        )
        prompt = json.dumps(messages)
        self.assertIn("I prefer green tea.", prompt)
        for forbidden in (
            "GOLD_ONLY_SENTINEL",
            "has_answer",
            "answer_session_ids",
            "is_round_relevant",
            "is_gold_session",
        ):
            self.assertNotIn(forbidden, prompt)


class MetadataTests(unittest.TestCase):
    def test_resume_unions_k_values(self):
        existing = {"model": "x", "k_values": [5], "run_history": [{"run": 1}]}
        current = {"model": "x", "k_values": [10], "run_history": [{"run": 2}]}
        merged = merge_resume_metadata(existing, current, setting_keys=["model"])
        self.assertEqual(merged["k_values"], [5, 10])
        self.assertEqual(len(merged["run_history"]), 2)

    def test_resume_rejects_setting_mismatch(self):
        with self.assertRaises(ValueError):
            merge_resume_metadata(
                {"model": "a", "k_values": [5]},
                {"model": "b", "k_values": [10]},
                setting_keys=["model"],
            )


class OfficialEvaluationIoTests(unittest.TestCase):
    def test_string_false_label_is_not_truthy(self):
        self.assertFalse(parse_label("false"))
        self.assertTrue(parse_label("yes"))

    def test_export_and_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation_csv = root / "generation.csv"
            hypotheses = root / "hypotheses.jsonl"
            judged = root / "judged.jsonl"
            merged_csv = root / "merged.csv"

            with generation_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["question_id", "strategy", "k", "generated_answer"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "question_id": "q1",
                        "strategy": "semantic",
                        "k": 5,
                        "generated_answer": "Green tea",
                    }
                )

            self.assertEqual(export_hypotheses(generation_csv, hypotheses), 1)
            exported = json.loads(hypotheses.read_text(encoding="utf-8"))
            exported["autoeval_label"] = {"model": "judge", "label": True}
            judged.write_text(json.dumps(exported) + "\n", encoding="utf-8")

            matched, unused, missing = merge_scores(
                generation_csv,
                judged,
                merged_csv,
            )
            self.assertEqual((matched, unused, missing), (1, 0, 0))
            with merged_csv.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["answer_correct"], "1")
            self.assertEqual(row["judge_model"], "judge")

    def test_split_export_preserves_condition_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation_csv = root / "generation.csv"
            with generation_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["question_id", "strategy", "k", "generated_answer"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "question_id": "q1",
                        "strategy": "hybrid_rrf",
                        "k": 10,
                        "generated_answer": "answer",
                    }
                )
            counts = export_hypotheses_by_condition(generation_csv, root / "split")
            self.assertEqual(counts, {"hybrid_rrf_k10": 1})
            payload = json.loads(
                (root / "split" / "hybrid_rrf_k10.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                (payload["question_id"], payload["strategy"], payload["k"]),
                ("q1", "hybrid_rrf", 10),
            )


if __name__ == "__main__":
    unittest.main()
