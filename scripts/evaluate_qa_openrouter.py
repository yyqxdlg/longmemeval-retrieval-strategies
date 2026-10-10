"""Run the official LongMemEval answer-correctness judge via OpenRouter.

This is a small, auditable adaptation of LongMemEval's upstream
``src/evaluation/evaluate_qa.py``.  It preserves the task-specific prompts,
``temperature=0``, ``max_tokens=10``, and the official ``"yes" in response``
label rule while using OpenRouter's OpenAI-compatible endpoint.

The API key is read only from ``OPENROUTER_API_KEY``.  Never put a key in this
file, command-line arguments, or a committed .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-2024-08-06"
KEY_FIELDS = ("question_id", "strategy", "k")


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool = False,
) -> str:
    """Return the task-specific prompt from LongMemEval's official evaluator."""
    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\nQuestion: {}\n\nExplanation: "
            "{}\n\nModel Response: {}\n\nDoes the model correctly identify the "
            "question as unanswerable? Answer yes or no only."
        )
    elif task in {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    }:
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
            "{}\n\nIs the model response correct? Answer yes or no only."
        )
    elif task == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no. In addition, do not penalize off-by-one errors for the number of "
            "days.\nIf the question asks for the number of days/weeks/months, "
            "etc., and the model makes off-by-one errors (e.g., predicting 19 days "
            "when the answer is 18), the model's response is still correct. "
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
            "{}\n\nIs the model response correct? Answer yes or no only."
        )
    elif task == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response contains some previous "
            "information along with an updated answer, the response should be "
            "considered as correct as long as the updated answer is the required "
            "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
            "{}\n\nIs the model response correct? Answer yes or no only."
        )
    elif task == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. The "
            "model does not need to reflect all the points in the rubric. The "
            "response is correct as long as it recalls and utilizes the user's "
            "personal information correctly.\n\nQuestion: {}\n\nRubric: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? Answer "
            "yes or no only."
        )
    else:
        raise ValueError(f"Unsupported LongMemEval question type: {task!r}")
    return template.format(question, answer, response)


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(row, dict) for row in parsed):
        raise ValueError(f"Expected a JSON array or JSONL objects in {path}")
    return parsed


def discover_hypothesis_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Hypotheses path does not exist: {path}")
    files = sorted(
        candidate
        for candidate in path.glob("*.jsonl")
        if ".eval-results-" not in candidate.name
    )
    if not files:
        raise ValueError(f"No hypothesis JSONL files found in {path}")
    return files


def load_hypotheses(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for source in discover_hypothesis_files(path):
        for row in read_json_or_jsonl(source):
            missing = [field for field in KEY_FIELDS if field not in row]
            if "hypothesis" not in row:
                missing.append("hypothesis")
            if missing:
                raise ValueError(f"Missing fields {sorted(set(missing))} in {source}")
            key = result_key(row)
            if key in seen:
                raise ValueError(f"Duplicate hypothesis key: {key}")
            seen.add(key)
            rows.append(dict(row))
    return rows


def result_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["question_id"]), str(row["strategy"]), int(row["k"])


def load_existing_results(path: Path, model: str) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, int]] = set()
    for row in read_json_or_jsonl(path):
        key = result_key(row)
        if key in completed:
            raise ValueError(f"Duplicate result key in {path}: {key}")
        label = row.get("autoeval_label")
        if not isinstance(label, dict) or "label" not in label:
            raise ValueError(f"Incomplete result in {path}: {key}")
        prior_model = str(label.get("model", ""))
        if prior_model != model:
            raise ValueError(
                f"Cannot resume with {model}; existing result {key} uses {prior_model}"
            )
        completed.add(key)
    return completed


def api_key() -> str:
    value = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not value:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Set it in the current shell; do not "
            "put the key in source code or commit it."
        )
    return value


def check_key() -> dict[str, Any]:
    request = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/key",
        headers={"Authorization": f"Bearer {api_key()}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter key check failed ({exc.code}): {body}") from exc
    data = payload.get("data", {})
    return {
        "label": data.get("label"),
        "is_free_tier": data.get("is_free_tier"),
        "limit": data.get("limit"),
        "limit_remaining": data.get("limit_remaining"),
        "usage": data.get("usage"),
        "expires_at": data.get("expires_at"),
    }


def request_headers(token: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.getenv("OPENROUTER_APP_NAME", "LongMemEval retrieval strategies").strip()
    if referer:
        headers["HTTP-Referer"] = referer
        headers["X-OpenRouter-Title"] = title
    return headers


def rate_limit_wait_seconds(
    error: urllib.error.HTTPError,
    response_body: str,
) -> float:
    """Return the server-requested 429 wait, capped at 60 seconds."""
    candidates: list[str] = []
    for name in ("Retry-After", "X-RateLimit-Reset"):
        value = error.headers.get(name)
        if value:
            candidates.append(value)
    try:
        payload = json.loads(response_body)
        metadata_headers = payload["error"]["metadata"].get("headers", {})
        for name in ("Retry-After", "X-RateLimit-Reset"):
            value = metadata_headers.get(name)
            if value:
                candidates.append(str(value))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    waits: list[float] = []
    for value in candidates:
        try:
            number = float(value)
        except ValueError:
            continue
        if number > 10_000_000_000:  # Epoch milliseconds.
            number /= 1000
        if number > 1_000_000_000:  # Epoch seconds.
            waits.append(max(0.0, number - time.time()) + 1.0)
        else:  # Retry-After seconds.
            waits.append(max(0.0, number))
    return min(60.0, max(waits, default=0.0))


def iter_pending(
    hypotheses: Iterable[dict[str, Any]],
    completed: set[tuple[str, str, int]],
    limit: int | None,
) -> list[dict[str, Any]]:
    pending = [row for row in hypotheses if result_key(row) not in completed]
    return pending if limit is None else pending[:limit]


def request_judgment(
    *,
    token: str,
    model: str,
    prompt: str,
    provider_order: list[str],
    allow_provider_fallbacks: bool,
    max_retries: int,
    timeout_seconds: float,
) -> tuple[str, str, dict[str, int | None]]:
    request_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "n": 1,
        "temperature": 0,
        "max_tokens": 10,
        "provider": {
            "order": provider_order,
            "allow_fallbacks": allow_provider_fallbacks,
        },
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        server_wait = 0.0
        try:
            request = urllib.request.Request(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                data=json.dumps(request_payload).encode("utf-8"),
                headers=request_headers(token),
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                completion = json.load(response)
            text = str(completion["choices"][0]["message"]["content"] or "").strip()
            if not text:
                raise RuntimeError("Judge returned an empty response")
            usage = completion.get("usage", {})
            usage_data = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
            return text, str(completion.get("model", model)), usage_data
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenRouter HTTP {exc.code}: {body}")
            if exc.code in {400, 401, 402, 403, 404}:
                break
            if exc.code == 429:
                server_wait = rate_limit_wait_seconds(exc, body)
        except Exception as exc:
            last_error = exc
        if attempt >= max_retries:
            break
        delay = min(60.0, max(server_wait, (2**attempt) + random.random()))
        print(
            f"Request failed ({type(last_error).__name__}); retrying in {delay:.1f}s",
            file=sys.stderr,
        )
        time.sleep(delay)
    assert last_error is not None
    raise last_error


def run_evaluation(args: argparse.Namespace) -> int:
    references = read_json_or_jsonl(Path(args.references))
    reference_by_id = {str(row["question_id"]): row for row in references}
    if len(reference_by_id) != len(references):
        raise ValueError("Duplicate question_id in reference file")

    hypotheses = load_hypotheses(Path(args.hypotheses))
    for row in hypotheses:
        if str(row["question_id"]) not in reference_by_id:
            raise ValueError(f"Question missing from references: {row['question_id']}")

    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; pass --resume: {output}")
    completed = load_existing_results(output, args.model) if args.resume else set()
    limit = None if args.limit == 0 else args.limit
    pending = iter_pending(hypotheses, completed, limit)

    print(
        f"Loaded {len(hypotheses)} hypotheses; {len(completed)} already complete; "
        f"{len(pending)} selected for this run."
    )
    if args.dry_run or not pending:
        return 0

    token = api_key()
    output.parent.mkdir(parents=True, exist_ok=True)
    provider_order = [part.strip() for part in args.provider_order.split(",") if part.strip()]
    if not provider_order:
        raise ValueError("--provider-order must contain at least one provider")

    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for index, entry in enumerate(pending, start=1):
            qid = str(entry["question_id"])
            reference = reference_by_id[qid]
            prompt = get_anscheck_prompt(
                str(reference["question_type"]),
                str(reference["question"]),
                str(reference["answer"]),
                str(entry["hypothesis"]),
                abstention="_abs" in qid,
            )
            started = time.perf_counter()
            raw_response, resolved_model, usage = request_judgment(
                token=token,
                model=args.model,
                prompt=prompt,
                provider_order=provider_order,
                allow_provider_fallbacks=args.allow_provider_fallbacks,
                max_retries=args.max_retries,
                timeout_seconds=args.timeout_seconds,
            )
            elapsed = time.perf_counter() - started
            result = dict(entry)
            result["autoeval_label"] = {
                "model": args.model,
                "label": "yes" in raw_response.lower(),
            }
            result["judge_audit"] = {
                "gateway": "openrouter",
                "requested_model": args.model,
                "resolved_model": resolved_model,
                "provider_order": provider_order,
                "allow_provider_fallbacks": args.allow_provider_fallbacks,
                "raw_response": raw_response,
                "latency_seconds": elapsed,
                "usage": usage,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(pending)}] {qid}: {raw_response!r}")
            if args.request_delay_seconds:
                time.sleep(args.request_delay_seconds)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LongMemEval GPT-4o judge through OpenRouter"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "check-key", description="Validate OPENROUTER_API_KEY without model usage"
    )

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--hypotheses", required=True, help="JSONL file or directory")
    evaluate.add_argument("--references", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--model", default=DEFAULT_MODEL)
    evaluate.add_argument(
        "--provider-order",
        default="openai",
        help="Comma-separated OpenRouter provider order (default pins OpenAI)",
    )
    evaluate.add_argument(
        "--allow-provider-fallbacks",
        action="store_true",
        help="Allow providers outside --provider-order (reduces comparability)",
    )
    evaluate.add_argument("--limit", type=int, default=0, help="0 means all pending")
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--max-retries", type=int, default=5)
    evaluate.add_argument("--timeout-seconds", type=float, default=60.0)
    evaluate.add_argument(
        "--request-delay-seconds",
        type=float,
        default=3.2,
        help="Delay after each successful request; 3.2s stays below 20 RPM",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "check-key":
        print(json.dumps(check_key(), ensure_ascii=False, indent=2))
        return 0
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.request_delay_seconds < 0:
        raise ValueError("--request-delay-seconds must be non-negative")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    return run_evaluation(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
