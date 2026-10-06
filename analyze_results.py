from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


STRATEGIES = ["recency", "semantic", "hybrid"]
TASK_ORDER = ["IE", "MR", "KU", "TR"]

# Okabe-Ito colorblind-safe palette. Markers/hatches also encode strategy so
# figures remain distinguishable in grayscale printing.
PALETTE = {
    "recency": "#0072B2",   # blue
    "semantic": "#D55E00",  # vermillion
    "hybrid": "#009E73",    # bluish green
}
MARKERS = {"recency": "o", "semantic": "s", "hybrid": "^"}
HATCHES = {"recency": "//", "semantic": "..", "hybrid": "xx"}

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 42


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values in the original order."""
    if not p_values:
        return []

    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    adjusted = np.full(len(p), np.nan, dtype=float)

    if not finite.any():
        return adjusted.tolist()

    finite_idx = np.where(finite)[0]
    finite_p = p[finite]
    order_local = np.argsort(finite_p)
    m = len(finite_p)
    running = 0.0

    adjusted_sorted = np.zeros(m, dtype=float)
    for rank, local_idx in enumerate(order_local):
        value = (m - rank) * finite_p[local_idx]
        running = max(running, value)
        adjusted_sorted[rank] = min(running, 1.0)

    for rank, local_idx in enumerate(order_local):
        adjusted[finite_idx[local_idx]] = adjusted_sorted[rank]

    return adjusted.tolist()


def bootstrap_ci(
    values: Iterable[float],
    *,
    statistic: str = "mean",
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = BOOTSTRAP_REPS,
) -> tuple[float, float, float]:
    """Estimate a mean/median and its non-parametric 95% bootstrap CI."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan

    fn = np.mean if statistic == "mean" else np.median
    observed = float(fn(x))
    if len(x) == 1:
        return observed, observed, observed

    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True)
    boot_stats = fn(samples, axis=1)
    lo, hi = np.quantile(boot_stats, [0.025, 0.975])
    return observed, float(lo), float(hi)


def paired_bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = BOOTSTRAP_REPS,
) -> tuple[float, float, float]:
    """Observed mean(a-b) and a paired bootstrap 95% CI."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan, np.nan, np.nan

    observed = float(diff.mean())
    if len(diff) == 1:
        return observed, observed, observed

    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(n_boot, len(diff)), replace=True)
    boot = samples.mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, float(lo), float(hi)


def coerce_binary(series: pd.Series) -> pd.Series:
    """Convert common binary encodings to numeric 0/1, leaving unknowns NaN."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    numeric = pd.to_numeric(series, errors="coerce")
    unresolved = numeric.isna() & series.notna()
    if unresolved.any():
        mapping = {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "correct": 1.0,
            "incorrect": 0.0,
        }
        mapped = series.astype(str).str.strip().str.lower().map(mapping)
        numeric.loc[unresolved] = mapped.loc[unresolved]
    numeric.loc[~numeric.isin([0, 1]) & numeric.notna()] = np.nan
    return numeric.astype(float)


def paired_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot_table(
        index="question_id",
        columns="strategy",
        values=metric,
        aggfunc="first",
    )


def paired_wilcoxon_table(
    df: pd.DataFrame,
    metric: str,
    *,
    scope: str = "overall",
    task_type: str | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    raw_p: list[float] = []
    pivot = paired_pivot(df, metric)

    for a, b in itertools.combinations(STRATEGIES, 2):
        if a not in pivot.columns or b not in pivot.columns:
            x = y = np.array([], dtype=float)
        else:
            pair = pivot[[a, b]].dropna()
            x = pair[a].to_numpy(dtype=float)
            y = pair[b].to_numpy(dtype=float)

        diff = x - y if len(x) else np.array([], dtype=float)

        if len(x) == 0:
            stat, p = np.nan, np.nan
            mean_diff, ci_lo, ci_hi = np.nan, np.nan, np.nan
        elif np.allclose(diff, 0):
            stat, p = 0.0, 1.0
            mean_diff, ci_lo, ci_hi = 0.0, 0.0, 0.0
        else:
            stat, p = wilcoxon(
                x,
                y,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            mean_diff, ci_lo, ci_hi = paired_bootstrap_mean_diff(x, y)

        rows.append(
            {
                "scope": scope,
                "task_type": task_type or "ALL",
                "metric": metric,
                "strategy_a": a,
                "strategy_b": b,
                "n_pairs": len(x),
                "mean_difference_a_minus_b": mean_diff,
                "bootstrap95_lo": ci_lo,
                "bootstrap95_hi": ci_hi,
                "wilcoxon_statistic": stat,
                "p_raw": p,
            }
        )
        raw_p.append(p)

    adjusted = holm_adjust(raw_p)
    for row, p_adj in zip(rows, adjusted):
        row["p_holm"] = p_adj
    return pd.DataFrame(rows)


def exact_mcnemar_table(
    df: pd.DataFrame,
    metric: str,
    *,
    scope: str = "overall",
    task_type: str | None = None,
) -> pd.DataFrame:
    """Exact paired McNemar test via a two-sided binomial test on discordance."""
    local = df.copy()
    local[metric] = coerce_binary(local[metric])
    pivot = paired_pivot(local, metric)

    rows: list[dict] = []
    raw_p: list[float] = []

    for a, b in itertools.combinations(STRATEGIES, 2):
        if a not in pivot.columns or b not in pivot.columns:
            x = y = np.array([], dtype=float)
        else:
            pair = pivot[[a, b]].dropna()
            x = pair[a].to_numpy(dtype=int)
            y = pair[b].to_numpy(dtype=int)

        if len(x) == 0:
            n10 = n01 = n_discordant = 0
            p = np.nan
            mean_diff, ci_lo, ci_hi = np.nan, np.nan, np.nan
        else:
            # a=1,b=0 and a=0,b=1 are the discordant cells.
            n10 = int(np.sum((x == 1) & (y == 0)))
            n01 = int(np.sum((x == 0) & (y == 1)))
            n_discordant = n10 + n01
            p = 1.0 if n_discordant == 0 else float(
                binomtest(n10, n_discordant, p=0.5, alternative="two-sided").pvalue
            )
            mean_diff, ci_lo, ci_hi = paired_bootstrap_mean_diff(x, y)

        rows.append(
            {
                "scope": scope,
                "task_type": task_type or "ALL",
                "metric": metric,
                "strategy_a": a,
                "strategy_b": b,
                "n_pairs": len(x),
                "a1_b0": n10,
                "a0_b1": n01,
                "n_discordant": n_discordant,
                "paired_proportion_difference_a_minus_b": mean_diff,
                "bootstrap95_lo": ci_lo,
                "bootstrap95_hi": ci_hi,
                "p_raw_exact_mcnemar": p,
            }
        )
        raw_p.append(p)

    adjusted = holm_adjust(raw_p)
    for row, p_adj in zip(rows, adjusted):
        row["p_holm"] = p_adj
    return pd.DataFrame(rows)


def ci_summary(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    by_task: bool,
) -> pd.DataFrame:
    group_cols = ["task_type", "strategy"] if by_task else ["strategy"]
    rows: list[dict] = []

    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))

        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                continue

            mean, mean_lo, mean_hi = bootstrap_ci(values, statistic="mean")
            med, med_lo, med_hi = bootstrap_ci(values, statistic="median")
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "mean_ci95_lo": mean_lo,
                    "mean_ci95_hi": mean_hi,
                    "median": med,
                    "median_ci95_lo": med_lo,
                    "median_ci95_hi": med_hi,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )

    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_box(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_dir: Path,
) -> None:
    data = []
    labels = []
    for strategy in STRATEGIES:
        values = pd.to_numeric(
            df.loc[df["strategy"] == strategy, metric], errors="coerce"
        ).dropna().to_numpy()
        data.append(values)
        labels.append(strategy)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    box = ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
    )
    for patch, strategy in zip(box["boxes"], STRATEGIES):
        patch.set_facecolor(PALETTE[strategy])
        patch.set_alpha(0.45)
        patch.set_hatch(HATCHES[strategy])
        patch.set_edgecolor("black")

    # Raw observations make the small pilot sample visible.
    for i, strategy in enumerate(STRATEGIES, start=1):
        values = pd.to_numeric(
            df.loc[df["strategy"] == strategy, metric], errors="coerce"
        ).dropna().to_numpy()
        offsets = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else [0]
        ax.scatter(
            np.asarray(offsets) + i,
            values,
            marker=MARKERS[strategy],
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            s=28,
            zorder=3,
        )

    ax.set_xlabel("Retrieval strategy")
    ax.set_ylabel(ylabel)
    if metric == "latency_ms":
        ax.set_yscale("log")
        ax.set_ylabel("Retrieval latency (ms, log scale)")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir, f"boxplot_{metric}")


def plot_overall_ci(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(STRATEGIES))

    for i, strategy in enumerate(STRATEGIES):
        values = pd.to_numeric(
            df.loc[df["strategy"] == strategy, metric], errors="coerce"
        ).dropna().to_numpy()
        mean, lo, hi = bootstrap_ci(values, statistic="mean")
        if np.isnan(mean):
            continue
        ax.errorbar(
            i,
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt=MARKERS[strategy],
            markersize=8,
            color=PALETTE[strategy],
            markeredgecolor="black",
            capsize=4,
            linewidth=1.6,
            label=strategy,
        )
        offsets = np.linspace(-0.05, 0.05, len(values)) if len(values) > 1 else [0]
        ax.scatter(
            np.asarray(offsets) + i,
            values,
            s=18,
            marker=MARKERS[strategy],
            color=PALETTE[strategy],
            alpha=0.35,
        )

    ax.set_xticks(xs, STRATEGIES)
    ax.set_xlabel("Retrieval strategy")
    ax.set_ylabel(ylabel)
    if "recall" in metric or metric == "answer_correct":
        ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir, f"ci_{metric}_overall")


def plot_task_ci(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_dir: Path,
) -> None:
    tasks = [t for t in TASK_ORDER if t in set(df["task_type"].dropna())]
    if not tasks:
        tasks = sorted(df["task_type"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    x = np.arange(len(tasks), dtype=float)
    offsets = {"recency": -0.22, "semantic": 0.0, "hybrid": 0.22}

    for strategy in STRATEGIES:
        means, lows, highs = [], [], []
        for task in tasks:
            values = pd.to_numeric(
                df.loc[
                    (df["strategy"] == strategy) & (df["task_type"] == task),
                    metric,
                ],
                errors="coerce",
            ).dropna().to_numpy()
            mean, lo, hi = bootstrap_ci(values, statistic="mean")
            means.append(mean)
            lows.append(lo)
            highs.append(hi)

            # Show individual questions next to the estimate.
            px = x[tasks.index(task)] + offsets[strategy]
            raw_offsets = (
                np.linspace(-0.035, 0.035, len(values)) if len(values) > 1 else [0]
            )
            ax.scatter(
                px + np.asarray(raw_offsets),
                values,
                marker=MARKERS[strategy],
                color=PALETTE[strategy],
                alpha=0.30,
                s=18,
            )

        means_arr = np.asarray(means, dtype=float)
        lows_arr = np.asarray(lows, dtype=float)
        highs_arr = np.asarray(highs, dtype=float)
        valid = np.isfinite(means_arr)
        ax.errorbar(
            x[valid] + offsets[strategy],
            means_arr[valid],
            yerr=[
                means_arr[valid] - lows_arr[valid],
                highs_arr[valid] - means_arr[valid],
            ],
            fmt=MARKERS[strategy],
            markersize=7,
            color=PALETTE[strategy],
            markeredgecolor="black",
            capsize=3,
            linewidth=1.4,
            label=strategy,
        )

    ax.set_xticks(x, tasks)
    ax.set_xlabel("Task type")
    ax.set_ylabel(ylabel)
    if "recall" in metric or metric == "answer_correct":
        ax.set_ylim(-0.05, 1.05)
    ax.legend(title="Strategy")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir, f"task_ci_{metric}")


def plot_tradeoff(
    df: pd.DataFrame,
    *,
    performance_metric: str,
    cost_metric: str,
    performance_label: str,
    cost_label: str,
    out_dir: Path,
) -> None:
    if performance_metric not in df.columns or cost_metric not in df.columns:
        return
    if not pd.to_numeric(df[cost_metric], errors="coerce").notna().any():
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for strategy in STRATEGIES:
        sub = df.loc[df["strategy"] == strategy]
        perf = pd.to_numeric(sub[performance_metric], errors="coerce").dropna().to_numpy()
        cost = pd.to_numeric(sub[cost_metric], errors="coerce").dropna().to_numpy()
        if len(perf) == 0 or len(cost) == 0:
            continue

        y, ylo, yhi = bootstrap_ci(perf, statistic="mean")
        # Latency/token counts are commonly skewed; use median cost.
        x, xlo, xhi = bootstrap_ci(cost, statistic="median")

        ax.errorbar(
            x,
            y,
            xerr=[[x - xlo], [xhi - x]],
            yerr=[[y - ylo], [yhi - y]],
            fmt=MARKERS[strategy],
            markersize=9,
            color=PALETTE[strategy],
            markeredgecolor="black",
            capsize=4,
            linewidth=1.5,
            label=strategy,
        )
        ax.annotate(
            strategy,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(cost_label)
    ax.set_ylabel(performance_label)
    if "recall" in performance_metric or performance_metric == "answer_correct":
        ax.set_ylim(-0.05, 1.05)
    if cost_metric == "latency_ms":
        ax.set_xscale("log")
    ax.grid(alpha=0.25)
    ax.legend(title="Strategy")
    save_figure(
        fig,
        out_dir,
        f"tradeoff_{performance_metric}_vs_{cost_metric}",
    )


def plot_hybrid_scale_diagnostic(df: pd.DataFrame, out_dir: Path) -> None:
    required = {"semantic_score_std_all", "recency_score_std_all"}
    if not required.issubset(df.columns):
        return

    q = (
        df.drop_duplicates("question_id")
        [["question_id", "semantic_score_std_all", "recency_score_std_all"]]
        .dropna()
    )
    if q.empty:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(
        q["recency_score_std_all"],
        q["semantic_score_std_all"],
        marker="o",
        facecolors="none",
        edgecolors="black",
    )
    lim_max = max(
        q["recency_score_std_all"].max(),
        q["semantic_score_std_all"].max(),
        0.01,
    )
    ax.plot([0, lim_max], [0, lim_max], linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("Recency-score SD across all memories")
    ax.set_ylabel("Semantic-score SD across all memories")
    ax.set_title("Hybrid component scale diagnostic")
    ax.grid(alpha=0.25)
    save_figure(fig, out_dir, "hybrid_component_scale")


def check_completeness(df: pd.DataFrame) -> pd.DataFrame:
    expected = set(STRATEGIES)
    rows = []
    for (qid, k), group in df.groupby(["question_id", "k"]):
        present = set(group["strategy"].dropna())
        rows.append(
            {
                "question_id": qid,
                "k": int(k),
                "n_rows": len(group),
                "strategies_present": "|".join(sorted(present)),
                "missing_strategies": "|".join(sorted(expected - present)),
                "complete": present == expected,
            }
        )
    return pd.DataFrame(rows)


def analyze_one_k(df: pd.DataFrame, selected_k: int, base_out_dir: Path) -> None:
    local = df.loc[df["k"] == selected_k].copy()
    out_dir = base_out_dir / f"k{selected_k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Coerce known metrics. This also makes blank token_count fields become NaN.
    for col in [
        "round_recall_at_k",
        "recall_any_at_k",
        "session_recall_at_k",
        "latency_ms",
        "token_count",
        "retrieved_context_token_count",
        "prompt_token_count",
        "completion_token_count",
        "generation_latency_ms",
        "total_latency_ms",
        "answer_correct",
    ]:
        if col in local.columns:
            local[col] = (
                coerce_binary(local[col])
                if col in {"recall_any_at_k", "answer_correct"}
                else pd.to_numeric(local[col], errors="coerce")
            )

    print(f"\n=== Analyzing k={selected_k} ===")
    print("Shape:", local.shape)
    print("Strategy counts:\n", local["strategy"].value_counts())

    pd.DataFrame(
        {
            "column": local.columns,
            "missing_count": [int(local[c].isna().sum()) for c in local.columns],
            "missing_fraction": [float(local[c].isna().mean()) for c in local.columns],
        }
    ).to_csv(out_dir / "missing_values.csv", index=False)

    numeric_metrics = [
        c
        for c in [
            "round_recall_at_k",
            "recall_any_at_k",
            "session_recall_at_k",
            "latency_ms",
            "token_count",
            "retrieved_context_token_count",
            "prompt_token_count",
            "completion_token_count",
            "generation_latency_ms",
            "total_latency_ms",
            "answer_correct",
        ]
        if c in local.columns and local[c].notna().any()
    ]

    ci_summary(local, numeric_metrics, by_task=False).to_csv(
        out_dir / "descriptive_ci_by_strategy.csv", index=False
    )
    ci_summary(local, numeric_metrics, by_task=True).to_csv(
        out_dir / "descriptive_ci_by_task_strategy.csv", index=False
    )

    # Paired continuous/bounded metrics: exploratory Wilcoxon + Holm.
    wilcoxon_metrics = [
        c
        for c in [
            "round_recall_at_k",
            "session_recall_at_k",
            "latency_ms",
            "token_count",
            "retrieved_context_token_count",
            "prompt_token_count",
            "completion_token_count",
            "generation_latency_ms",
            "total_latency_ms",
        ]
        if c in numeric_metrics
    ]
    overall_w = [paired_wilcoxon_table(local, m) for m in wilcoxon_metrics]
    if overall_w:
        pd.concat(overall_w, ignore_index=True).to_csv(
            out_dir / "paired_wilcoxon_holm_overall.csv", index=False
        )

    task_w = []
    for task in [t for t in TASK_ORDER if t in set(local["task_type"].dropna())]:
        task_df = local.loc[local["task_type"] == task]
        for metric in wilcoxon_metrics:
            task_w.append(
                paired_wilcoxon_table(
                    task_df,
                    metric,
                    scope="task",
                    task_type=task,
                )
            )
    if task_w:
        pd.concat(task_w, ignore_index=True).to_csv(
            out_dir / "paired_wilcoxon_holm_by_task.csv", index=False
        )

    # Binary paired outcomes: exact McNemar + Holm.
    binary_metrics = [
        c for c in ["recall_any_at_k", "answer_correct"] if c in numeric_metrics
    ]
    overall_m = [exact_mcnemar_table(local, m) for m in binary_metrics]
    if overall_m:
        pd.concat(overall_m, ignore_index=True).to_csv(
            out_dir / "paired_mcnemar_holm_overall.csv", index=False
        )

    task_m = []
    for task in [t for t in TASK_ORDER if t in set(local["task_type"].dropna())]:
        task_df = local.loc[local["task_type"] == task]
        for metric in binary_metrics:
            task_m.append(
                exact_mcnemar_table(
                    task_df,
                    metric,
                    scope="task",
                    task_type=task,
                )
            )
    if task_m:
        pd.concat(task_m, ignore_index=True).to_csv(
            out_dir / "paired_mcnemar_holm_by_task.csv", index=False
        )

    # Preprocessing diagnostics: one row/question is sufficient within a k.
    qdiag = local.drop_duplicates("question_id")
    diag_cols = [
        "question_id",
        "task_type",
        "input_sorted_by_date",
        "num_future_sessions_excluded",
        "num_future_gold_sessions",
        "num_same_day_later_sessions_retained",
        "num_same_day_later_gold_sessions_retained",
        "fraction_memories_over_stella_limit",
        "semantic_score_std_all",
        "recency_score_std_all",
    ]
    qdiag[[c for c in diag_cols if c in qdiag.columns]].to_csv(
        out_dir / "preprocessing_diagnostics.csv", index=False
    )

    # Core figures.
    if "round_recall_at_k" in numeric_metrics:
        plot_metric_box(
            local,
            "round_recall_at_k",
            f"Round-level Recall@{selected_k}",
            out_dir,
        )
        plot_overall_ci(
            local,
            "round_recall_at_k",
            f"Mean round-level Recall@{selected_k} (95% bootstrap CI)",
            out_dir,
        )
        plot_task_ci(
            local,
            "round_recall_at_k",
            f"Mean round-level Recall@{selected_k} (95% bootstrap CI)",
            out_dir,
        )

    if "recall_any_at_k" in numeric_metrics:
        plot_overall_ci(
            local,
            "recall_any_at_k",
            f"Recall-any@{selected_k} rate (95% bootstrap CI)",
            out_dir,
        )
        plot_task_ci(
            local,
            "recall_any_at_k",
            f"Recall-any@{selected_k} rate (95% bootstrap CI)",
            out_dir,
        )

    if "session_recall_at_k" in numeric_metrics:
        plot_overall_ci(
            local,
            "session_recall_at_k",
            f"Mean session-level Recall@{selected_k} (95% bootstrap CI)",
            out_dir,
        )

    if "latency_ms" in numeric_metrics:
        plot_metric_box(local, "latency_ms", "Retrieval latency (ms)", out_dir)

    if "answer_correct" in numeric_metrics:
        plot_overall_ci(
            local,
            "answer_correct",
            "Answer accuracy (95% bootstrap CI)",
            out_dir,
        )
        plot_task_ci(
            local,
            "answer_correct",
            "Answer accuracy (95% bootstrap CI)",
            out_dir,
        )

    # RQ3 retrieval-only trade-offs now; answer-accuracy trade-offs are added
    # automatically later when answer_correct exists.
    performance_metrics = []
    if "round_recall_at_k" in numeric_metrics:
        performance_metrics.append(
            ("round_recall_at_k", f"Mean round Recall@{selected_k}")
        )
    if "answer_correct" in numeric_metrics:
        performance_metrics.append(("answer_correct", "Answer accuracy"))

    for perf_metric, perf_label in performance_metrics:
        latency_metric = (
            "total_latency_ms"
            if perf_metric == "answer_correct" and "total_latency_ms" in numeric_metrics
            else "latency_ms"
        )
        latency_label = (
            "Median retrieval + generation latency (ms, log scale)"
            if latency_metric == "total_latency_ms"
            else "Median retrieval latency (ms, log scale)"
        )
        if latency_metric in numeric_metrics:
            plot_tradeoff(
                local,
                performance_metric=perf_metric,
                cost_metric=latency_metric,
                performance_label=perf_label,
                cost_label=latency_label,
                out_dir=out_dir,
            )

        token_metric = (
            "prompt_token_count"
            if perf_metric == "answer_correct" and "prompt_token_count" in numeric_metrics
            else "token_count"
        )
        token_label = (
            "Median complete prompt token count"
            if token_metric == "prompt_token_count"
            else "Median retrieved context token count"
        )
        if token_metric in numeric_metrics:
            plot_tradeoff(
                local,
                performance_metric=perf_metric,
                cost_metric=token_metric,
                performance_label=perf_label,
                cost_label=token_label,
                out_dir=out_dir,
            )

    plot_hybrid_scale_diagnostic(local, out_dir)

    (out_dir / "analysis_selection.txt").write_text(
        "\n".join(
            [
                f"k={selected_k}",
                f"n_rows={len(local)}",
                f"n_questions={local['question_id'].nunique()}",
                "task-level tests are exploratory for the 20-question pilot",
                "Holm correction is applied within each metric/scope family of three strategy pairs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def cross_k_summary(df: pd.DataFrame, out_dir: Path) -> None:
    ks = sorted(int(x) for x in df["k"].dropna().unique())
    if len(ks) < 2:
        return

    cross_dir = out_dir / "cross_k"
    cross_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        c
        for c in [
            "round_recall_at_k",
            "recall_any_at_k",
            "session_recall_at_k",
            "latency_ms",
            "token_count",
            "retrieved_context_token_count",
            "prompt_token_count",
            "completion_token_count",
            "generation_latency_ms",
            "total_latency_ms",
            "answer_correct",
        ]
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]

    # Long-format comparison table, easy to use in the report or pivot later.
    rows = []
    for k in ks:
        for strategy in STRATEGIES:
            sub = df.loc[(df["k"] == k) & (df["strategy"] == strategy)]
            for metric in metrics:
                vals = (
                    coerce_binary(sub[metric])
                    if metric in {"recall_any_at_k", "answer_correct"}
                    else pd.to_numeric(sub[metric], errors="coerce")
                ).dropna().to_numpy()
                if len(vals) == 0:
                    continue
                mean, mean_lo, mean_hi = bootstrap_ci(vals, statistic="mean")
                med, med_lo, med_hi = bootstrap_ci(vals, statistic="median")
                rows.append(
                    {
                        "k": k,
                        "strategy": strategy,
                        "metric": metric,
                        "n": len(vals),
                        "mean": mean,
                        "mean_ci95_lo": mean_lo,
                        "mean_ci95_hi": mean_hi,
                        "median": med,
                        "median_ci95_lo": med_lo,
                        "median_ci95_hi": med_hi,
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(cross_dir / "topk_comparison_table.csv", index=False)

    # Wide report-friendly table for the first two k values (typically 5/10).
    if len(ks) >= 2 and not summary.empty:
        k1, k2 = ks[0], ks[1]
        wide_rows = []
        for strategy in STRATEGIES:
            for metric in metrics:
                a = summary.loc[
                    (summary["k"] == k1)
                    & (summary["strategy"] == strategy)
                    & (summary["metric"] == metric)
                ]
                b = summary.loc[
                    (summary["k"] == k2)
                    & (summary["strategy"] == strategy)
                    & (summary["metric"] == metric)
                ]
                if a.empty or b.empty:
                    continue
                ar, br = a.iloc[0], b.iloc[0]
                wide_rows.append(
                    {
                        "strategy": strategy,
                        "metric": metric,
                        f"k{k1}_n": ar["n"],
                        f"k{k1}_mean": ar["mean"],
                        f"k{k1}_mean_ci95_lo": ar["mean_ci95_lo"],
                        f"k{k1}_mean_ci95_hi": ar["mean_ci95_hi"],
                        f"k{k1}_median": ar["median"],
                        f"k{k2}_n": br["n"],
                        f"k{k2}_mean": br["mean"],
                        f"k{k2}_mean_ci95_lo": br["mean_ci95_lo"],
                        f"k{k2}_mean_ci95_hi": br["mean_ci95_hi"],
                        f"k{k2}_median": br["median"],
                        f"mean_difference_k{k2}_minus_k{k1}": br["mean"] - ar["mean"],
                    }
                )
        pd.DataFrame(wide_rows).to_csv(
            cross_dir / f"top{k1}_vs_top{k2}_table.csv", index=False
        )

    # Paired k-vs-k tests within each strategy. This is separate from the
    # between-strategy tests above.
    if len(ks) == 2:
        k1, k2 = ks
        test_rows = []
        for metric in metrics:
            metric_rows = []
            raw_p = []
            for strategy in STRATEGIES:
                sub = df.loc[df["strategy"] == strategy]
                pivot = sub.pivot_table(
                    index="question_id",
                    columns="k",
                    values=metric,
                    aggfunc="first",
                )
                if k1 not in pivot.columns or k2 not in pivot.columns:
                    continue
                pair = pivot[[k1, k2]].dropna()
                if metric in {"recall_any_at_k", "answer_correct"}:
                    x = coerce_binary(pair[k1]).to_numpy()
                    y = coerce_binary(pair[k2]).to_numpy()
                else:
                    x = pd.to_numeric(pair[k1], errors="coerce").to_numpy()
                    y = pd.to_numeric(pair[k2], errors="coerce").to_numpy()
                valid = np.isfinite(x) & np.isfinite(y)
                x, y = x[valid], y[valid]

                row = {
                    "metric": metric,
                    "strategy": strategy,
                    "k_a": k1,
                    "k_b": k2,
                    "n_pairs": len(x),
                }

                if metric in {"recall_any_at_k", "answer_correct"}:
                    x = x.astype(int)
                    y = y.astype(int)
                    n10 = int(np.sum((x == 1) & (y == 0)))
                    n01 = int(np.sum((x == 0) & (y == 1)))
                    disc = n10 + n01
                    p = 1.0 if disc == 0 else float(
                        binomtest(n10, disc, p=0.5, alternative="two-sided").pvalue
                    )
                    row.update(
                        {
                            "test": "exact_mcnemar",
                            "k_a1_k_b0": n10,
                            "k_a0_k_b1": n01,
                            "statistic": np.nan,
                            "p_raw": p,
                        }
                    )
                else:
                    diff = x - y
                    if len(x) == 0:
                        stat, p = np.nan, np.nan
                    elif np.allclose(diff, 0):
                        stat, p = 0.0, 1.0
                    else:
                        stat, p = wilcoxon(x, y, zero_method="wilcox", method="auto")
                    row.update(
                        {
                            "test": "wilcoxon",
                            "statistic": stat,
                            "p_raw": p,
                        }
                    )

                mean_diff, lo, hi = paired_bootstrap_mean_diff(y, x)
                row.update(
                    {
                        f"mean_difference_k{k2}_minus_k{k1}": mean_diff,
                        "bootstrap95_lo": lo,
                        "bootstrap95_hi": hi,
                    }
                )
                metric_rows.append(row)
                raw_p.append(p)

            adjusted = holm_adjust(raw_p)
            for row, p_adj in zip(metric_rows, adjusted):
                row["p_holm_across_strategies"] = p_adj
            test_rows.extend(metric_rows)

        pd.DataFrame(test_rows).to_csv(
            cross_dir / "paired_topk_tests.csv", index=False
        )


def load_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        frame = pd.read_csv(p)
        frame["_source_file"] = p
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    required = {"question_id", "strategy", "k"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df.dropna(subset=["k"]).copy()
    df["k"] = df["k"].astype(int)

    key_cols = ["question_id", "strategy", "k"]
    dup = df.duplicated(key_cols, keep=False)
    if dup.any():
        print(
            f"WARNING: found {int(dup.sum())} duplicate rows by {key_cols}; "
            "keeping the last occurrence."
        )
        df = df.drop_duplicates(key_cols, keep="last")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EDA, paired statistical tests, confidence intervals, figures, "
            "RQ3 trade-off plots, and optional cross-k comparison for "
            "LongMemEval retrieval results."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help=(
            "One or more retrieval-results CSV files. Multiple files are "
            "concatenated, which allows separate top-5/top-10 runs."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Analyze only one k. If omitted, every k value is analyzed in its "
            "own folder and a cross-k comparison is created when possible."
        ),
    )
    args = parser.parse_args()

    df = load_inputs(args.input)
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    completeness = check_completeness(df)
    completeness.to_csv(base_out / "completeness_check.csv", index=False)
    incomplete = completeness.loc[~completeness["complete"]]
    if not incomplete.empty:
        print(
            f"WARNING: {len(incomplete)} question/k cells are incomplete. "
            "See completeness_check.csv before interpreting paired tests."
        )

    available_k = sorted(int(x) for x in df["k"].dropna().unique())
    if args.k is not None:
        if args.k not in available_k:
            raise SystemExit(
                f"Requested k={args.k}, but available values are {available_k}."
            )
        ks_to_analyze = [args.k]
    else:
        ks_to_analyze = available_k

    for k in ks_to_analyze:
        analyze_one_k(df, k, base_out)

    if args.k is None and len(available_k) > 1:
        cross_k_summary(df, base_out)

    (base_out / "analysis_notes.txt").write_text(
        "\n".join(
            [
                "Inferential tests on the 20-question pilot are exploratory/procedure checks.",
                "Round/session recall and continuous cost metrics use paired Wilcoxon tests with Holm correction.",
                "recall_any_at_k uses exact McNemar tests because it is paired binary data.",
                "If answer_correct is later added, the same exact McNemar analysis is applied automatically.",
                "Task-level tests have very small n in the pilot and should not be used for strong claims.",
                "Figures use the Okabe-Ito colorblind-safe palette plus distinct markers/hatches.",
                "Error bars are non-parametric bootstrap 95% confidence intervals across questions.",
                "Retrieval trade-offs use retrieval latency/context tokens; answer trade-offs prefer total latency/full prompt tokens.",
                "Top-k conditions are analyzed separately; cross-k output is created only from explicitly present k values.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nAnalysis written to: {base_out}")
    print(
        "Reminder: with only 20 pilot questions, treat p-values and task-level "
        "intervals as exploratory rather than final evidence."
    )


if __name__ == "__main__":
    main()
