from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_io import sha256_file  # noqa: E402
from memory_adapter import record_to_memories  # noqa: E402
from retrieval import (  # noqa: E402
    MODEL_NAME,
    STELLA_REVISION,
    StellaRetriever,
    retrieve_semantic_from_scores,
)


CSV_FIELDS = [
    "question_id",
    "task_type",
    "N_rounds",
    "kind",
    "repeat",
    "seconds",
]


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""
    return (completed.stdout or completed.stderr).strip()


def active_gpu_compute_processes() -> list[str]:
    overview = run_text(["nvidia-smi"])
    if "No running processes found" in overview:
        return []
    raw = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw or "No running processes found" in raw:
        return []
    own_pid = str(os.getpid())
    return [line for line in raw.splitlines() if line.strip() and not line.startswith(own_pid + ",")]


def timed_cuda_call(function: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    finished = time.perf_counter()
    return result, finished - started


def load_selection(selection_path: Path, confirmatory_path: Path) -> list[dict[str, str]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    confirmatory = json.loads(confirmatory_path.read_text(encoding="utf-8"))
    source_questions = confirmatory["questions"]
    source_ids = [str(item["question_id"]) for item in source_questions]
    expected_ids = random.Random(42).sample(source_ids, 10)
    selected_questions = selection["questions"]
    selected_ids = [str(item["question_id"]) for item in selected_questions]
    if selected_ids != expected_ids:
        raise ValueError(
            "Selection file does not equal random.Random(42).sample(ids, 10): "
            f"expected {expected_ids}, found {selected_ids}"
        )
    return selected_questions


def load_saved_top5(csv_path: Path, selected_ids: list[str]) -> dict[str, list[str]]:
    selected = set(selected_ids)
    saved: dict[str, list[str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            qid = str(row["question_id"])
            if qid not in selected or row["strategy"] != "semantic" or int(row["k"]) != 5:
                continue
            saved[qid] = [value for value in row["retrieved_memory_ids"].split("|") if value]
    missing = [qid for qid in selected_ids if qid not in saved]
    if missing:
        raise ValueError(f"Saved semantic Top-5 rows are missing for: {missing}")
    return saved


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Stella document and query encoding latency with main-run code paths."
    )
    parser.add_argument(
        "--pilot-file",
        default="outputs/confirmatory_200/pilot_questions.json",
    )
    parser.add_argument(
        "--confirmatory-ids",
        default="config/confirmatory_200_ids.json",
    )
    parser.add_argument(
        "--selection-file",
        default="config/stella_encoding_timing_10_ids.json",
    )
    parser.add_argument(
        "--retrieval-results",
        default="outputs/confirmatory_200/retrieval_stella512_fp32_k5_k10.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/embedding_timing",
    )
    parser.add_argument("--expected-gpu-substring", default="RTX 5070")
    parser.add_argument("--allow-active-gpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this measurement.")

    gpu_name = torch.cuda.get_device_name(0)
    if args.expected_gpu_substring and args.expected_gpu_substring not in gpu_name:
        raise SystemExit(
            f"Expected GPU containing {args.expected_gpu_substring!r}, found {gpu_name!r}."
        )

    active_processes = active_gpu_compute_processes()
    if active_processes and not args.allow_active_gpu:
        raise SystemExit(
            "Other GPU compute processes are active; refusing to benchmark:\n"
            + "\n".join(active_processes)
        )

    pilot_path = repo_path(args.pilot_file)
    confirmatory_path = repo_path(args.confirmatory_ids)
    selection_path = repo_path(args.selection_file)
    retrieval_path = repo_path(args.retrieval_results)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements_path = output_dir / "stella512_fp32_encoding_measurements.csv"
    summary_path = output_dir / "stella512_fp32_encoding_summary.md"
    metadata_path = output_dir / "stella512_fp32_encoding_metadata.json"

    selected_questions = load_selection(selection_path, confirmatory_path)
    selected_ids = [str(item["question_id"]) for item in selected_questions]
    task_by_id = {
        str(item["question_id"]): str(item["task_type"])
        for item in selected_questions
    }
    saved_top5 = load_saved_top5(retrieval_path, selected_ids)

    records = json.loads(pilot_path.read_text(encoding="utf-8"))
    by_id = {str(record["question_id"]): record for record in records}
    missing_records = [qid for qid in selected_ids if qid not in by_id]
    if missing_records:
        raise SystemExit(f"Selected records are missing from pilot file: {missing_records}")

    prepared: list[dict[str, Any]] = []
    for qid in selected_ids:
        record = by_id[qid]
        memories = record_to_memories(record, exclude_future=True)
        if not memories:
            raise SystemExit(f"No memories remain after preprocessing for {qid}")
        prepared.append(
            {
                "question_id": qid,
                "task_type": task_by_id[qid],
                "question": str(record.get("question", "")),
                "memories": memories,
            }
        )

    print(
        f"Loading {MODEL_NAME}@{STELLA_REVISION} on {gpu_name} "
        "(fp32, batch_size=1, max_seq_length=512)",
        flush=True,
    )
    retriever = StellaRetriever(
        model_name=MODEL_NAME,
        revision=STELLA_REVISION,
        device="cuda",
        max_seq_length=512,
        batch_size=1,
        use_fp16=False,
    )

    warm_record = prepared[0]
    _, warm_doc_seconds = timed_cuda_call(
        lambda: retriever.encode_memories(warm_record["memories"])
    )
    warm_query_seconds: list[float] = []
    for _ in range(5):
        _, seconds = timed_cuda_call(
            lambda: retriever.encode_query(warm_record["question"])
        )
        warm_query_seconds.append(seconds)
    print(
        f"Warm-up discarded: doc={warm_doc_seconds:.6f}s; "
        f"queries={[round(value * 1000, 3) for value in warm_query_seconds]} ms",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    vectors: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    with measurements_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()

        for index, item in enumerate(prepared, start=1):
            qid = item["question_id"]
            n_rounds = len(item["memories"])
            print(
                f"[{index}/10] {qid} task={item['task_type']} N_rounds={n_rounds}",
                flush=True,
            )

            for repeat in range(1, 4):
                embeddings, seconds = timed_cuda_call(
                    lambda item=item: retriever.encode_memories(item["memories"])
                )
                vectors[qid]["documents"] = np.asarray(embeddings)
                row = {
                    "question_id": qid,
                    "task_type": item["task_type"],
                    "N_rounds": n_rounds,
                    "kind": "doc",
                    "repeat": repeat,
                    "seconds": f"{seconds:.9f}",
                }
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                print(f"  doc repeat {repeat}/3: {seconds:.6f}s", flush=True)

            for repeat in range(1, 6):
                embedding, seconds = timed_cuda_call(
                    lambda item=item: retriever.encode_query(item["question"])
                )
                vectors[qid]["query"] = np.asarray(embedding)
                row = {
                    "question_id": qid,
                    "task_type": item["task_type"],
                    "N_rounds": n_rounds,
                    "kind": "query",
                    "repeat": repeat,
                    "seconds": f"{seconds:.9f}",
                }
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                print(f"  query repeat {repeat}/5: {seconds * 1000:.3f}ms", flush=True)

    consistency: list[dict[str, Any]] = []
    for item in prepared:
        qid = item["question_id"]
        scores = vectors[qid]["documents"] @ vectors[qid]["query"]
        retrieved = retrieve_semantic_from_scores(item["memories"], scores, k=5)
        measured_ids = [str(memory["id"]) for memory in retrieved]
        expected_ids = saved_top5[qid]
        consistency.append(
            {
                "question_id": qid,
                "exact_order_match": measured_ids == expected_ids,
                "same_set": set(measured_ids) == set(expected_ids),
                "overlap": len(set(measured_ids) & set(expected_ids)),
                "measured_ids": measured_ids,
                "saved_ids": expected_ids,
            }
        )

    boundary_swaps = [
        item for item in consistency
        if not item["same_set"] and item["overlap"] == 4
    ]
    large_mismatches = [
        item for item in consistency
        if not item["same_set"] and item["overlap"] < 4
    ]

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    n_by_id: dict[str, int] = {}
    for row in rows:
        grouped[(str(row["question_id"]), str(row["kind"]))].append(float(row["seconds"]))
        n_by_id[str(row["question_id"])] = int(row["N_rounds"])

    per_question: list[dict[str, Any]] = []
    for qid in selected_ids:
        doc_seconds = grouped[(qid, "doc")]
        query_seconds = grouped[(qid, "query")]
        doc_median = float(statistics.median(doc_seconds))
        query_median = float(statistics.median(query_seconds))
        per_question.append(
            {
                "question_id": qid,
                "task_type": task_by_id[qid],
                "N_rounds": n_by_id[qid],
                "doc_median_seconds": doc_median,
                "doc_ms_per_round": doc_median * 1000.0 / n_by_id[qid],
                "query_median_ms": query_median * 1000.0,
            }
        )

    doc_summary = summarize([item["doc_median_seconds"] for item in per_question])
    n_summary = summarize([float(item["N_rounds"]) for item in per_question])
    per_round_summary = summarize([item["doc_ms_per_round"] for item in per_question])
    query_summary = summarize([item["query_median_ms"] for item in per_question])

    anomalies: list[str] = []
    for (qid, kind), values in grouped.items():
        center = statistics.median(values)
        if center > 0 and values[0] > center * 1.5:
            anomalies.append(
                f"{qid} {kind}: first repeat {values[0]:.6f}s was "
                f"{values[0] / center:.2f}x its median"
            )
        if center > 0 and (max(values) - min(values)) / center > 0.25:
            anomalies.append(
                f"{qid} {kind}: repeat range was "
                f"{(max(values) - min(values)) / center:.1%} of the median"
            )
    anomalies = list(dict.fromkeys(anomalies))
    if boundary_swaps:
        anomalies.append(
            f"Semantic Top-5 had a one-item boundary swap for "
            f"{len(boundary_swaps)}/10 questions; all retained 4/5 overlap"
        )

    gpu_line = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()[0]
    hardware_software_line = (
        f"GPU/driver/VRAM: {gpu_line}; CUDA (PyTorch): {torch.version.cuda}; "
        f"cuDNN: {torch.backends.cudnn.version()}; Python: {platform.python_version()}; "
        f"PyTorch: {torch.__version__}; Transformers: {package_version('transformers')}; "
        f"Sentence Transformers: {package_version('sentence-transformers')}"
    )

    git_commit = run_text(["git", "rev-parse", "HEAD"])
    git_status = run_text(["git", "status", "--short"])
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "revision": STELLA_REVISION,
        "precision": "fp32",
        "batch_size": 1,
        "max_seq_length": 512,
        "query_prompt_name": "s2p_query",
        "document_repeats": 3,
        "query_repeats": 5,
        "warmup": {"document_histories": 1, "queries": 5},
        "timing_clock": "time.perf_counter",
        "cuda_sync_before_start_and_stop": True,
        "selected_question_ids": selected_ids,
        "input_sha256": sha256_file(pilot_path),
        "selection_sha256": sha256_file(selection_path),
        "retrieval_results_sha256": sha256_file(retrieval_path),
        "measurements_sha256": sha256_file(measurements_path),
        "hardware_software": hardware_software_line,
        "git_commit_at_run": git_commit,
        "git_dirty_at_run": bool(git_status),
        "warmup_seconds": {
            "document": warm_doc_seconds,
            "queries": warm_query_seconds,
        },
        "summary": {
            "document_seconds_per_history": doc_summary,
            "round_count": n_summary,
            "document_ms_per_round": per_round_summary,
            "query_ms": query_summary,
        },
        "consistency": consistency,
        "consistency_interpretation": {
            "exact_order_matches": sum(item["exact_order_match"] for item in consistency),
            "same_set_matches": sum(item["same_set"] for item in consistency),
            "total_top5_overlap": sum(item["overlap"] for item in consistency),
            "boundary_swap_questions": [item["question_id"] for item in boundary_swaps],
            "large_mismatch_questions": [item["question_id"] for item in large_mismatches],
        },
        "anomalies": anomalies,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stella 512 fp32 encoding-time measurement",
        "",
        "The aggregate statistics use each question's median across repeats,",
        "then report the median/minimum/maximum across the fixed 10 questions.",
        "Model loading, JSON reading, timestamp parsing, future-session filtering,",
        "similarity calculation, and ranking are outside the timed regions.",
        "",
        "## Results",
        "",
        f"- Document encoding total (s/history): median {doc_summary['median']:.6f}, min {doc_summary['min']:.6f}, max {doc_summary['max']:.6f}",
        f"- History size N (rounds): median {n_summary['median']:.1f}, range {n_summary['min']:.0f}-{n_summary['max']:.0f}",
        f"- Document encoding normalized (ms/round): median {per_round_summary['median']:.3f}, min {per_round_summary['min']:.3f}, max {per_round_summary['max']:.3f}",
        f"- Query encoding (ms/query): median {query_summary['median']:.3f}, min {query_summary['min']:.3f}, max {query_summary['max']:.3f}",
        "",
        "## Semantic Top-5 consistency",
        "",
        f"- Exact ordered matches: {sum(item['exact_order_match'] for item in consistency)}/10",
        f"- Same Top-5 sets: {sum(item['same_set'] for item in consistency)}/10",
        f"- Retrieved-ID overlap: {sum(item['overlap'] for item in consistency)}/50",
        "",
        "## Hardware and software",
        "",
        hardware_software_line,
        "",
        "## Anomalies",
        "",
    ]
    if anomalies:
        lines.extend(f"- {item}" for item in anomalies)
    else:
        lines.append("- None detected after the prescribed warm-up.")
    lines.extend(
        [
            "",
            "## Per-question medians",
            "",
            "| question_id | task | N rounds | doc s/history | doc ms/round | query ms |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in per_question:
        lines.append(
            f"| {item['question_id']} | {item['task_type']} | {item['N_rounds']} | "
            f"{item['doc_median_seconds']:.6f} | {item['doc_ms_per_round']:.3f} | "
            f"{item['query_median_ms']:.3f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {measurements_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)
    print(
        f"Semantic Top-5 exact order: {sum(item['exact_order_match'] for item in consistency)}/10; "
        f"same set: {sum(item['same_set'] for item in consistency)}/10",
        flush=True,
    )
    if boundary_swaps:
        print(
            "WARNING: one-item semantic Top-5 boundary swaps were observed for: "
            + ", ".join(item["question_id"] for item in boundary_swaps),
            flush=True,
        )
    if large_mismatches:
        raise SystemExit(
            "Large semantic Top-5 mismatch detected; inspect metadata consistency details before use."
        )


if __name__ == "__main__":
    main()
