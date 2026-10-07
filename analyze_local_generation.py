from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def render_gold(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def completion_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, k), group in df.groupby(["strategy", "k"], dropna=False):
        answer = group["generated_answer"].fillna("").astype(str).str.strip()
        row: dict[str, Any] = {
            "strategy": strategy,
            "k": int(k),
            "expected_rows": len(group),
            "completed_rows": int(answer.ne("").sum()),
            "empty_answers": int(answer.eq("").sum()),
            "completion_rate": float(answer.ne("").mean()),
        }
        for column in (
            "prompt_token_count",
            "completion_token_count",
            "generation_latency_ms",
            "total_latency_ms",
        ):
            values = pd.to_numeric(group.get(column), errors="coerce")
            row[f"mean_{column}"] = float(values.mean())
            row[f"median_{column}"] = float(values.median())
        row["exploratory_normalized_exact_match_rate"] = float(
            pd.to_numeric(
                group["exploratory_normalized_exact_match"], errors="coerce"
            ).mean()
        )
        row["exploratory_answer_contains_gold_rate"] = float(
            pd.to_numeric(
                group["exploratory_answer_contains_gold"], errors="coerce"
            ).mean()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "strategy"])


def choose_manual_audit(df: pd.DataFrame, size: int) -> pd.DataFrame:
    task_order = ["IE", "MR", "KU", "TR"]
    strategy_order = [
        "recency",
        "semantic",
        "hybrid_raw",
        "hybrid_minmax",
        "hybrid_rrf",
        "no_retrieval",
        "oracle",
    ]
    chosen: list[pd.Series] = []
    used: set[tuple[str, str, int]] = set()

    for task_index, task in enumerate(task_order):
        for strategy_index, strategy in enumerate(strategy_order):
            candidates = df.loc[
                (df["task_type"] == task) & (df["strategy"] == strategy)
            ].copy()
            if candidates.empty:
                continue
            if strategy not in {"no_retrieval", "oracle"}:
                target_k = 5 if (task_index + strategy_index) % 2 == 0 else 10
                at_k = candidates.loc[candidates["k"] == target_k]
                if not at_k.empty:
                    candidates = at_k
                target_recall = (task_index + strategy_index) % 2
                recall = pd.to_numeric(
                    candidates.get("recall_any_at_k"), errors="coerce"
                )
                matched = candidates.loc[recall == target_recall]
                if not matched.empty:
                    candidates = matched
            candidates["_audit_sort_tokens"] = pd.to_numeric(
                candidates.get("prompt_token_count"), errors="coerce"
            ).fillna(-1)
            candidates = candidates.sort_values(
                ["_audit_sort_tokens", "question_id"], ascending=[False, True]
            )
            for _, candidate in candidates.iterrows():
                key = (
                    str(candidate["question_id"]),
                    str(candidate["strategy"]),
                    int(candidate["k"]),
                )
                if key not in used:
                    chosen.append(candidate)
                    used.add(key)
                    break

    if len(chosen) < size:
        remainder = df.copy()
        remainder["_audit_sort_tokens"] = pd.to_numeric(
            remainder.get("prompt_token_count"), errors="coerce"
        ).fillna(-1)
        remainder = remainder.sort_values(
            ["_audit_sort_tokens", "question_id"], ascending=[False, True]
        )
        for _, candidate in remainder.iterrows():
            key = (
                str(candidate["question_id"]),
                str(candidate["strategy"]),
                int(candidate["k"]),
            )
            if key in used:
                continue
            chosen.append(candidate)
            used.add(key)
            if len(chosen) >= size:
                break

    audit = pd.DataFrame(chosen[:size])
    audit["manual_correctness"] = ""
    audit["manual_notes"] = ""
    columns = [
        "question_id",
        "task_type",
        "question",
        "gold_answer",
        "generated_answer",
        "strategy",
        "k",
        "recall_any_at_k",
        "retrieved_memory_ids",
        "context_sha256",
        "prompt_token_count",
        "manual_correctness",
        "manual_notes",
    ]
    return audit.reindex(columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze unjudged local generation without claiming official accuracy."
    )
    parser.add_argument("--generation", required=True)
    parser.add_argument("--pilot-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manual-audit-output", required=True)
    parser.add_argument("--audit-size", type=int, default=28)
    args = parser.parse_args()
    if not 24 <= args.audit_size <= 30:
        raise SystemExit("--audit-size must be between 24 and 30")

    generation_path = Path(args.generation)
    records = json.loads(Path(args.pilot_file).read_text(encoding="utf-8"))
    by_id = {str(record["question_id"]): record for record in records}
    df = pd.read_csv(generation_path, keep_default_na=False)
    required = {"question_id", "strategy", "k", "generated_answer"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing generation columns: {sorted(missing)}")
    if df.duplicated(["question_id", "strategy", "k"]).any():
        raise SystemExit("Duplicate (question_id, strategy, k) generation rows found")
    expected_conditions = {
        (strategy, k)
        for strategy in (
            "recency",
            "semantic",
            "hybrid_raw",
            "hybrid_minmax",
            "hybrid_rrf",
        )
        for k in (5, 10)
    } | {("no_retrieval", 0), ("oracle", 0)}
    present_conditions = {
        (str(row.strategy), int(row.k))
        for row in df[["strategy", "k"]].drop_duplicates().itertuples(index=False)
    }
    if present_conditions != expected_conditions:
        raise SystemExit(
            "Generation condition mismatch: "
            f"missing={sorted(expected_conditions - present_conditions)}, "
            f"unexpected={sorted(present_conditions - expected_conditions)}"
        )
    expected_per_condition = int(df["question_id"].astype(str).nunique())
    condition_counts = df.groupby(["strategy", "k"]).size()
    bad_counts = condition_counts.loc[condition_counts != expected_per_condition]
    if not bad_counts.empty:
        raise SystemExit(f"Incomplete condition counts:\n{bad_counts}")
    if "answer_correct" in df and df["answer_correct"].astype(str).str.strip().ne("").any():
        raise SystemExit(
            "answer_correct is not blank; use a separately labeled scored analysis."
        )
    for fixed_field in (
        "answer_model_revision",
        "quantization",
        "generation_torch_dtype",
        "max_new_tokens",
        "generation_seed",
    ):
        if fixed_field in df and df[fixed_field].astype(str).nunique(dropna=False) != 1:
            raise SystemExit(f"Mixed generation setting in {fixed_field}")
    if "context_sha256" in df and df["context_sha256"].astype(str).str.strip().eq("").any():
        raise SystemExit("Missing context_sha256 values")

    df["question"] = df["question_id"].map(
        lambda qid: str(by_id[str(qid)].get("question", ""))
    )
    df["gold_answer"] = df["question_id"].map(
        lambda qid: render_gold(by_id[str(qid)].get("answer", ""))
    )
    if "task_type" not in df.columns:
        df["task_type"] = df["question_id"].map(
            lambda qid: str(by_id[str(qid)].get("_pilot_task", ""))
        )
    answer_norm = df["generated_answer"].map(normalize_text)
    gold_norm = df["gold_answer"].map(normalize_text)
    nonempty = answer_norm.ne("") & gold_norm.ne("")
    df["exploratory_normalized_exact_match"] = (
        (answer_norm == gold_norm) & nonempty
    ).astype(int)
    df["exploratory_answer_contains_gold"] = pd.Series(
        [bool(g and g in a) for a, g in zip(answer_norm, gold_norm)],
        index=df.index,
    ).astype(int)
    df["generated_answer_char_count"] = df["generated_answer"].astype(str).str.len()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "generation_row_diagnostics.csv", index=False)
    completion_summary(df).to_csv(
        out_dir / "generation_completion_summary.csv", index=False
    )
    (out_dir / "HEURISTIC_NOTICE.txt").write_text(
        "Normalized exact match and answer-contains-gold are exploratory heuristics.\n"
        "They are not GPT-4o judgments and must not be reported as official answer accuracy.\n",
        encoding="utf-8",
    )
    (out_dir / "generation_validation.json").write_text(
        json.dumps(
            {
                "question_count": expected_per_condition,
                "condition_count": len(expected_conditions),
                "expected_rows": expected_per_condition * len(expected_conditions),
                "actual_rows": len(df),
                "unique_keys": True,
                "answer_correct_all_blank": True,
                "context_hashes_present": True,
                "condition_counts": {
                    f"{strategy}_k{int(k)}": int(count)
                    for (strategy, k), count in condition_counts.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    audit_path = Path(args.manual_audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    choose_manual_audit(df, args.audit_size).to_csv(
        audit_path, index=False, quoting=csv.QUOTE_MINIMAL
    )
    print(f"Local generation analysis: {out_dir}")
    print(f"Manual audit rows: {args.audit_size} -> {audit_path}")


if __name__ == "__main__":
    main()
