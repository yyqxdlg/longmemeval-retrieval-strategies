from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiment_io import read_json, sha256_file, write_json


ALL_STRATEGIES = [
    "recency",
    "semantic",
    "hybrid_raw",
    "hybrid_minmax",
    "hybrid_rrf",
]
MAIN_STRATEGIES = ["recency", "semantic", "hybrid_raw"]
HYBRID_STRATEGIES = ["hybrid_raw", "hybrid_minmax", "hybrid_rrf"]


def write_subset(
    df: pd.DataFrame,
    path: Path,
    *,
    source_path: Path,
    parent_metadata: dict,
    strategies: list[str],
    k: int,
) -> None:
    subset = df.loc[(df["k"] == k) & df["strategy"].isin(strategies)].copy()
    subset = subset.sort_values(["question_id", "strategy"])
    path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(path, index=False)
    metadata = dict(parent_metadata)
    metadata.update(
        {
            "derived_from": str(source_path),
            "derived_from_sha256": sha256_file(source_path),
            "strategies": strategies,
            "k_values": [k],
            "actual_rows": len(subset),
            "question_count": int(subset["question_id"].nunique()),
        }
    )
    write_json(path.with_suffix(".metadata.json"), metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and split the 200-question confirmatory retrieval run."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--main-output-dir", required=True)
    parser.add_argument("--hybrid-output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    required = {"question_id", "strategy", "k"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if df.duplicated(["question_id", "strategy", "k"]).any():
        raise SystemExit("Duplicate (question_id, strategy, k) rows found")

    df["k"] = pd.to_numeric(df["k"], errors="raise").astype(int)
    question_count = int(df["question_id"].nunique())
    expected_rows = question_count * len(ALL_STRATEGIES) * 2
    if set(df["strategy"]) != set(ALL_STRATEGIES):
        raise SystemExit(f"Unexpected strategy set: {sorted(set(df['strategy']))}")
    if set(df["k"]) != {5, 10}:
        raise SystemExit(f"Unexpected k set: {sorted(set(df['k']))}")
    if len(df) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} rows, found {len(df)}")
    for field in ("semantic_scores_all_finite", "memory_embeddings_all_finite"):
        if field in df.columns and not df[field].astype(str).str.lower().eq("true").all():
            raise SystemExit(f"Non-finite diagnostic found in {field}")

    metadata_path = input_path.with_suffix(".metadata.json")
    parent_metadata = read_json(metadata_path) if metadata_path.exists() else {}
    main_dir = Path(args.main_output_dir)
    hybrid_dir = Path(args.hybrid_output_dir)
    for k in (5, 10):
        write_subset(
            df,
            main_dir / f"retrieval_main_k{k}.csv",
            source_path=input_path,
            parent_metadata=parent_metadata,
            strategies=MAIN_STRATEGIES,
            k=k,
        )
        write_subset(
            df,
            hybrid_dir / f"retrieval_hybrid_sensitivity_k{k}.csv",
            source_path=input_path,
            parent_metadata=parent_metadata,
            strategies=HYBRID_STRATEGIES,
            k=k,
        )
    print(
        f"Validated {len(df)} rows for {question_count} questions; "
        "wrote main and hybrid-sensitivity subsets."
    )


if __name__ == "__main__":
    main()
