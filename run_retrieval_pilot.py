from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiment_io import (
    merge_resume_metadata,
    read_json,
    runtime_snapshot,
    sha256_file,
    write_json,
)

from memory_adapter import (
    record_to_memories,
    relevant_memory_ids,
    gold_session_ids,
    temporal_diagnostics,
    build_generation_context,
)
from retrieval import (
    MODEL_NAME,
    STELLA_REVISION,
    StellaRetriever,
    retrieve_recency,
    retrieve_semantic_from_scores,
    retrieve_hybrid_from_scores,
    calculate_round_recall_at_k,
    calculate_recall_any_at_k,
    calculate_session_recall_at_k,
    measure_latency,
)


FIELDS = [
    "question_id",
    "task_type",
    "original_question_type",
    "question_date",
    "strategy",
    "k",
    "round_recall_at_k",
    "recall_any_at_k",
    "session_recall_at_k",
    "recall_at_k",
    "latency_ms",
    "token_count",
    "retrieved_context_token_count",
    "num_memories",
    "num_relevant_rounds",
    "num_gold_sessions",
    "input_sorted_by_date",
    "num_future_sessions_excluded",
    "num_future_gold_sessions",
    "num_same_day_later_sessions_retained",
    "num_same_day_later_gold_sessions_retained",
    "mean_memory_tokens_stella",
    "max_memory_tokens_stella",
    "num_memories_over_stella_limit",
    "fraction_memories_over_stella_limit",
    "semantic_score_mean_all",
    "semantic_score_std_all",
    "semantic_score_min_all",
    "semantic_score_max_all",
    "recency_score_mean_all",
    "recency_score_std_all",
    "recency_score_min_all",
    "recency_score_max_all",
    "semantic_minmax_constant",
    "recency_minmax_constant",
    "semantic_scores_all_finite",
    "memory_embeddings_all_finite",
    "gpu_peak_memory_bytes",
    "retrieved_memory_ids",
    "retrieved_session_ids",
]

DEFAULT_STRATEGIES = (
    "recency",
    "semantic",
    "hybrid_raw",
    "hybrid_minmax",
    "hybrid_rrf",
)


RESUME_SETTING_KEYS = (
    "stella_model",
    "stella_revision",
    "device",
    "precision",
    "max_seq_length",
    "batch_size",
    "future_session_policy",
    "generation_tokenizer",
    "generation_tokenizer_revision",
    "trust_generation_tokenizer_code",
    "strategies",
    "selected_question_ids",
)


def load_generation_tokenizer(
    model_name: str | None,
    *,
    revision: str | None,
    trust_remote_code: bool,
):
    if not model_name:
        return None

    from transformers import AutoTokenizer

    print(f"Loading generation tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )


def count_retrieved_context_tokens(retrieved, tokenizer, question_date):
    """Count only retrieved context; full prompt tokens are measured at generation."""
    if tokenizer is None:
        return ""

    text = build_generation_context(
        retrieved,
        question_date=question_date,
        chronological=True,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def run_one_strategy(
    strategy,
    memories,
    score_bundle,
    k,
):
    if strategy == "recency":
        fn = lambda: retrieve_recency(memories, k=k)
    elif strategy == "semantic":
        fn = lambda: retrieve_semantic_from_scores(
            memories,
            score_bundle["cosine"],
            k=k,
        )
    elif strategy in {"hybrid_raw", "hybrid_minmax", "hybrid_rrf"}:
        fn = lambda: retrieve_hybrid_from_scores(
            memories,
            score_bundle,
            strategy=strategy,
            k=k,
        )
    else:
        raise ValueError(strategy)

    latency = measure_latency(fn, warmup=1, repeats=5)
    retrieved = fn()  # untimed call used only to collect selected memories
    return retrieved, latency


def load_completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()

    keys: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                keys.add(
                    (
                        str(row["question_id"]),
                        str(row["strategy"]),
                        int(row["k"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5],
        help=(
            "One or more retrieval depths, e.g. --k 5 or --k 5 10. "
            "Rows keep k explicitly so analyses can be separated."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--question-id",
        nargs="+",
        default=None,
        help="Optional exact question IDs for smoke or sensitivity runs.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=list(DEFAULT_STRATEGIES),
        help="Retrieval conditions to run; defaults to all confirmatory strategies.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help=(
            "Primary Stella setting. 512 is kept as the default because the "
            "model card recommends 512; longer values should be sensitivity "
            "analyses, not silent replacements."
        ),
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 instead of fp16 on CUDA.",
    )
    parser.add_argument(
        "--include-future",
        action="store_true",
        help=(
            "Include sessions on calendar dates strictly later than the "
            "question date. Same-day sessions are retained by default even "
            "when their HH:MM is later than question_date."
        ),
    )
    parser.add_argument(
        "--allow-future-gold",
        action="store_true",
        help=(
            "Allow an item whose gold evidence is on a strictly later calendar "
            "date. Default behavior is to stop because that item would become "
            "unanswerable after future-day filtering."
        ),
    )
    parser.add_argument(
        "--generation-tokenizer",
        default=None,
        help=(
            "Tokenizer used to count retrieved context tokens. It should be "
            "the SAME tokenizer as the final answer-generation LLM. Example: "
            "meta-llama/Llama-3.1-8B-Instruct"
        ),
    )
    parser.add_argument(
        "--generation-tokenizer-revision",
        default=None,
        help="Optional immutable Hugging Face revision for the generation tokenizer.",
    )
    parser.add_argument(
        "--trust-generation-tokenizer-code",
        action="store_true",
        help="Allow custom tokenizer code. Not needed for Llama 3.1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append to an existing CSV and skip already completed "
            "(question_id, strategy, k) rows. Use only with the same settings."
        ),
    )
    args = parser.parse_args()

    k_values = sorted(set(args.k))
    strategies = tuple(dict.fromkeys(args.strategies))
    if any(k <= 0 for k in k_values):
        raise SystemExit("All k values must be positive integers.")

    pilot_path = Path(args.pilot_file)
    with pilot_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if args.question_id:
        requested_ids = list(dict.fromkeys(str(x) for x in args.question_id))
        by_id = {str(record.get("question_id")): record for record in records}
        missing_ids = [qid for qid in requested_ids if qid not in by_id]
        if missing_ids:
            raise SystemExit(f"Unknown --question-id values: {missing_ids}")
        records = [by_id[qid] for qid in requested_ids]
    else:
        requested_ids = None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = out_path.with_suffix(".metadata.json")

    existing_keys = load_completed_keys(out_path) if args.resume else set()
    if args.resume and existing_keys:
        print(f"Resume mode: {len(existing_keys)} completed rows found.")

    mode = "a" if args.resume and out_path.exists() else "w"
    need_header = mode == "w" or out_path.stat().st_size == 0

    retriever = StellaRetriever(
        device=args.device,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        use_fp16=not args.fp32,
    )
    tokenizer = load_generation_tokenizer(
        args.generation_tokenizer,
        revision=args.generation_tokenizer_revision,
        trust_remote_code=args.trust_generation_tokenizer_code,
    )

    if tokenizer is None:
        print(
            "WARNING: --generation-tokenizer was not supplied. token_count "
            "will be blank. Do not use the token-count part of RQ3 until the "
            "final generation tokenizer is fixed."
        )

    metadata = {
        "stella_model": MODEL_NAME,
        "stella_revision": STELLA_REVISION,
        "device": retriever.device,
        "precision": "fp16" if retriever.use_fp16 else "fp32",
        "max_seq_length": retriever.max_seq_length,
        "batch_size": retriever.batch_size,
        "k_values": k_values,
        "strategies": list(strategies),
        "selected_question_ids": requested_ids,
        "future_session_policy": (
            "include_all"
            if args.include_future
            else "exclude_strictly_later_calendar_dates; retain_same_day"
        ),
        "generation_tokenizer": args.generation_tokenizer,
        "generation_tokenizer_revision": args.generation_tokenizer_revision,
        "trust_generation_tokenizer_code": args.trust_generation_tokenizer_code,
        "input_sha256": sha256_file(pilot_path),
        "latency": "1 warm-up + 5 measured runs; median reported",
        "latency_scope": (
            "ranking from shared precomputed document/query embeddings; "
            "Stella encoding is excluded"
        ),
        "checkpointing": "CSV flushed after every result row",
        "run_history": [runtime_snapshot()],
    }
    if args.resume:
        if not metadata_path.exists():
            raise SystemExit(
                "--resume requires the existing metadata JSON; use a new output "
                "file rather than mixing undocumented settings."
            )
        try:
            metadata = merge_resume_metadata(
                read_json(metadata_path),
                metadata,
                setting_keys=RESUME_SETTING_KEYS,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    # Write metadata BEFORE the expensive run so a partial CSV still has its
    # experimental settings if the process is interrupted.
    write_json(metadata_path, metadata)

    total = len(records)
    rows_written = 0

    with out_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if need_header:
            writer.writeheader()
            f.flush()

        for idx, record in enumerate(records, start=1):
            qid = str(record.get("question_id"))
            task = record.get("_pilot_task")
            qtype = record.get("question_type")
            question = record.get("question", "")
            question_date = record.get("question_date")

            expected_keys = {
                (qid, strategy, k)
                for k in k_values
                for strategy in strategies
            }
            if expected_keys.issubset(existing_keys):
                print(f"[{idx}/{total}] {qid} already complete; skipping")
                continue

            tdiag = temporal_diagnostics(record)
            print(
                f"[{idx}/{total}] {qid} task={task} type={qtype} "
                f"sorted_by_date={tdiag['sorted_by_date']} "
                f"future_days={tdiag['num_future_sessions']} "
                f"same_day_later_retained={tdiag['num_same_day_later_sessions']}"
            )

            if (
                not args.include_future
                and tdiag["num_future_gold_sessions"] > 0
                and not args.allow_future_gold
            ):
                raise RuntimeError(
                    f"Question {qid} has gold evidence on a calendar date "
                    f"after the question date: "
                    f"{tdiag['future_gold_session_ids']}. Earlier rows are "
                    "already checkpointed in the CSV. Review this item or use "
                    "--allow-future-gold only if documented."
                )

            memories = record_to_memories(
                record,
                exclude_future=not args.include_future,
            )
            if not memories:
                print("  WARNING: no memories after preprocessing; skipping")
                continue

            gold_round_ids = relevant_memory_ids(memories)
            gold_sessions = gold_session_ids(record)

            if not gold_round_ids:
                print(
                    "  WARNING: no relevant round remains after preprocessing. "
                    "Session-level recall may still be available."
                )

            # One-time document embedding preprocessing is OUTSIDE retrieval latency.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            memory_embeddings = retriever.encode_memories(memories)
            memory_embeddings_finite = bool(np.isfinite(memory_embeddings).all())
            if not memory_embeddings_finite:
                raise FloatingPointError(
                    f"Question {qid} produced non-finite Stella embeddings"
                )
            # The query is encoded exactly once, then every strategy and k reuses
            # the same semantic and recency components.
            score_bundle = retriever.score_query(
                question,
                memories,
                memory_embeddings,
            )
            gpu_peak_memory_bytes = (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else ""
            )

            length_diag = retriever.token_length_diagnostics(memories)
            score_diag = retriever.hybrid_component_diagnostics(
                question,
                memories,
                memory_embeddings,
                score_bundle=score_bundle,
            )

            for k in k_values:
                for strategy in strategies:
                    key = (qid, strategy, k)
                    if key in existing_keys:
                        continue

                    retrieved, latency = run_one_strategy(
                        strategy,
                        memories,
                        score_bundle,
                        k,
                    )

                    round_recall = calculate_round_recall_at_k(
                        retrieved,
                        gold_round_ids,
                    )
                    recall_any = calculate_recall_any_at_k(
                        retrieved,
                        gold_round_ids,
                    )
                    session_recall = calculate_session_recall_at_k(
                        retrieved,
                        gold_sessions,
                    )

                    retrieved_context_tokens = count_retrieved_context_tokens(
                        retrieved,
                        tokenizer,
                        question_date,
                    )

                    row = {
                        "question_id": qid,
                        "task_type": task,
                        "original_question_type": qtype,
                        "question_date": question_date,
                        "strategy": strategy,
                        "k": k,
                        "round_recall_at_k": (
                            "" if round_recall is None else round_recall
                        ),
                        "recall_any_at_k": (
                            "" if recall_any is None else recall_any
                        ),
                        "session_recall_at_k": (
                            "" if session_recall is None else session_recall
                        ),
                        # Backward-compatible column name for older analysis.
                        "recall_at_k": (
                            "" if round_recall is None else round_recall
                        ),
                        "latency_ms": latency["median_ms"],
                        # Backward-compatible alias. This is retrieved context
                        # only, not the complete chat-template prompt.
                        "token_count": retrieved_context_tokens,
                        "retrieved_context_token_count": retrieved_context_tokens,
                        "num_memories": len(memories),
                        "num_relevant_rounds": len(gold_round_ids),
                        "num_gold_sessions": len(gold_sessions),
                        "input_sorted_by_date": tdiag["sorted_by_date"],
                        "num_future_sessions_excluded": (
                            0
                            if args.include_future
                            else tdiag["num_future_sessions"]
                        ),
                        "num_future_gold_sessions": tdiag[
                            "num_future_gold_sessions"
                        ],
                        "num_same_day_later_sessions_retained": tdiag[
                            "num_same_day_later_sessions"
                        ],
                        "num_same_day_later_gold_sessions_retained": tdiag[
                            "num_same_day_later_gold_sessions"
                        ],
                        **length_diag,
                        **score_diag,
                        "semantic_minmax_constant": score_bundle[
                            "semantic_minmax_constant"
                        ],
                        "recency_minmax_constant": score_bundle[
                            "recency_minmax_constant"
                        ],
                        "semantic_scores_all_finite": True,
                        "memory_embeddings_all_finite": memory_embeddings_finite,
                        "gpu_peak_memory_bytes": gpu_peak_memory_bytes,
                        "retrieved_memory_ids": "|".join(
                            str(x["id"]) for x in retrieved
                        ),
                        "retrieved_session_ids": "|".join(
                            str(x["session_id"]) for x in retrieved
                        ),
                    }

                    writer.writerow(row)
                    # Critical for long runs: preserve completed work even if a
                    # later question raises or the process is interrupted.
                    f.flush()
                    existing_keys.add(key)
                    rows_written += 1

    final_keys = load_completed_keys(out_path)
    metadata["completed_at_utc"] = runtime_snapshot()["captured_at_utc"]
    metadata["expected_rows"] = len(records) * len(strategies) * len(k_values)
    metadata["actual_rows"] = len(final_keys)
    metadata["complete"] = metadata["actual_rows"] == metadata["expected_rows"]
    metadata["rows_written_this_invocation"] = rows_written
    write_json(metadata_path, metadata)

    print(f"\nSaved/checkpointed: {out_path}")
    print(f"New rows written: {rows_written}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
