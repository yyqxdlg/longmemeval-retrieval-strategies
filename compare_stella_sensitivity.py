from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = [
    "round_recall_at_k",
    "recall_any_at_k",
    "session_recall_at_k",
    "latency_ms",
]


def summarize(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    rows = []
    for (max_seq_length, k, strategy), group in df.groupby(
        ["max_seq_length", "k", "strategy"]
    ):
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "subset": subset_name,
                    "max_seq_length": int(max_seq_length),
                    "k": int(k),
                    "strategy": strategy,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k, strategy), group in df.groupby(["k", "strategy"]):
        for metric in METRICS:
            pivot = group.pivot_table(
                index="question_id",
                columns="max_seq_length",
                values=metric,
                aggfunc="first",
            )
            if 512 not in pivot or 1024 not in pivot:
                continue
            pair = pivot[[512, 1024]].apply(pd.to_numeric, errors="coerce").dropna()
            delta = pair[1024] - pair[512]
            if len(pair) == 0:
                stat = p = np.nan
            elif np.allclose(delta, 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(
                    pair[1024], pair[512], zero_method="wilcox", method="auto"
                )
            rows.append(
                {
                    "k": int(k),
                    "strategy": strategy,
                    "metric": metric,
                    "n_pairs": len(pair),
                    "mean_512": float(pair[512].mean()),
                    "mean_1024": float(pair[1024].mean()),
                    "mean_difference_1024_minus_512": float(delta.mean()),
                    "wilcoxon_statistic": stat,
                    "p_raw_exploratory": p,
                }
            )
    return pd.DataFrame(rows)


def plot_recall(summary: pd.DataFrame, output_dir: Path) -> None:
    local = summary.loc[
        (summary["subset"] == "all")
        & (summary["metric"] == "round_recall_at_k")
    ]
    for k in sorted(local["k"].unique()):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        selected = local.loc[local["k"] == k]
        for strategy, group in selected.groupby("strategy"):
            group = group.sort_values("max_seq_length")
            ax.plot(
                group["max_seq_length"],
                group["mean"],
                marker="o",
                label=strategy,
            )
        ax.set_xticks([512, 1024])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Stella max sequence length")
        ax.set_ylabel(f"Mean round recall@{int(k)}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"stella_512_1024_round_recall_k{int(k)}.png", dpi=300)
        fig.savefig(output_dir / f"stella_512_1024_round_recall_k{int(k)}.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired Stella 512 and 1024 retrieval sensitivity runs."
    )
    parser.add_argument("--input-512", required=True)
    parser.add_argument("--input-1024", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    frames = []
    for length, path in ((512, args.input_512), (1024, args.input_1024)):
        frame = pd.read_csv(path)
        frame["max_seq_length"] = length
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if df.duplicated(["question_id", "strategy", "k", "max_seq_length"]).any():
        raise SystemExit("Duplicate sensitivity keys found")
    counts = df.groupby("max_seq_length")["question_id"].nunique().to_dict()
    if counts.get(512) != counts.get(1024):
        raise SystemExit(f"Question counts differ: {counts}")
    for field in ("semantic_scores_all_finite", "memory_embeddings_all_finite"):
        if field in df and not df[field].astype(str).str.lower().eq("true").all():
            raise SystemExit(f"Non-finite diagnostic found in {field}")

    source_512 = frames[0]
    long_ids = set(
        source_512.loc[
            pd.to_numeric(
                source_512["num_memories_over_stella_limit"], errors="coerce"
            )
            > 0,
            "question_id",
        ].astype(str)
    )
    df["question_id"] = df["question_id"].astype(str)
    all_summary = summarize(df, "all")
    long_summary = summarize(df.loc[df["question_id"].isin(long_ids)], "over_512_rounds")
    summary = pd.concat([all_summary, long_summary], ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "stella_512_1024_summary.csv", index=False)
    paired_tests(df).to_csv(output_dir / "stella_512_1024_paired_tests.csv", index=False)
    pd.DataFrame(
        [
            {
                "questions_per_condition": counts.get(512),
                "questions_with_any_round_over_512_tokens": len(long_ids),
                "all_embeddings_and_scores_finite": True,
            }
        ]
    ).to_csv(output_dir / "stella_512_1024_diagnostics.csv", index=False)
    plot_recall(summary, output_dir)
    print(
        f"Compared {counts.get(512)} paired questions; "
        f"{len(long_ids)} had at least one round over 512 Stella tokens."
    )


if __name__ == "__main__":
    main()
