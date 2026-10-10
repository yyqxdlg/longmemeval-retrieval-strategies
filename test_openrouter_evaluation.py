from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parent / "scripts" / "evaluate_qa_openrouter.py"
SPEC = importlib.util.spec_from_file_location("evaluate_qa_openrouter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OpenRouterEvaluationTests(unittest.TestCase):
    @staticmethod
    def temporary_directory() -> tempfile.TemporaryDirectory[str]:
        temp_root = Path(__file__).parent / "tmp" / "tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=temp_root)

    def test_official_prompt_variants(self) -> None:
        for task in (
            "single-session-user",
            "single-session-assistant",
            "multi-session",
            "temporal-reasoning",
            "knowledge-update",
            "single-session-preference",
        ):
            prompt = MODULE.get_anscheck_prompt(task, "Q", "A", "R")
            self.assertIn("Q", prompt)
            self.assertIn("A", prompt)
            self.assertIn("R", prompt)
            self.assertTrue(prompt.endswith("Answer yes or no only."))

    def test_abstention_prompt(self) -> None:
        prompt = MODULE.get_anscheck_prompt(
            "single-session-user", "Q", "A", "R", abstention=True
        )
        self.assertIn("unanswerable question", prompt)
        self.assertIn("Explanation: A", prompt)

    def test_directory_loading_and_duplicate_guard(self) -> None:
        with self.temporary_directory() as raw_dir:
            directory = Path(raw_dir)
            first = {
                "question_id": "q1",
                "hypothesis": "h1",
                "strategy": "semantic",
                "k": 5,
            }
            second = {
                "question_id": "q1",
                "hypothesis": "h2",
                "strategy": "semantic",
                "k": 10,
            }
            (directory / "a.jsonl").write_text(json.dumps(first) + "\n", encoding="utf-8")
            (directory / "b.jsonl").write_text(json.dumps(second) + "\n", encoding="utf-8")
            rows = MODULE.load_hypotheses(directory)
            self.assertEqual(len(rows), 2)

            (directory / "c.jsonl").write_text(json.dumps(first) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.load_hypotheses(directory)

    def test_resume_rejects_model_mismatch(self) -> None:
        row = {
            "question_id": "q1",
            "hypothesis": "h",
            "strategy": "recency",
            "k": 5,
            "autoeval_label": {"model": "another/model", "label": True},
        }
        with self.temporary_directory() as raw_dir:
            path = Path(raw_dir) / "results.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.load_existing_results(path, MODULE.DEFAULT_MODEL)

    def test_request_uses_pinned_protocol_settings(self) -> None:
        response_payload = {
            "model": MODULE.DEFAULT_MODEL,
            "choices": [{"message": {"content": "yes"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        with mock.patch.object(MODULE.urllib.request, "urlopen", return_value=FakeResponse()) as call:
            text, resolved_model, usage = MODULE.request_judgment(
                token="not-a-real-key",
                model=MODULE.DEFAULT_MODEL,
                prompt="judge this",
                provider_order=["openai"],
                allow_provider_fallbacks=False,
                max_retries=0,
                timeout_seconds=3,
            )

        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(text, "yes")
        self.assertEqual(resolved_model, MODULE.DEFAULT_MODEL)
        self.assertEqual(usage["total_tokens"], 11)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 10)
        self.assertEqual(payload["provider"]["order"], ["openai"])
        self.assertFalse(payload["provider"]["allow_fallbacks"])

    def test_cli_default_stays_below_twenty_rpm(self) -> None:
        args = MODULE.build_parser().parse_args(
            [
                "evaluate",
                "--hypotheses",
                "hypotheses",
                "--references",
                "references.json",
                "--output",
                "results.jsonl",
            ]
        )
        self.assertEqual(args.request_delay_seconds, 3.2)


if __name__ == "__main__":
    unittest.main()
