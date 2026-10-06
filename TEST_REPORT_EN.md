# Preliminary Quantitative Analysis of LongMemEval-S Retrieval Strategies

**II2201 Quantitative Methods exercise | Group preliminary report | 2 October 2026**

**Status.** This is an exploratory retrieval-only pilot. It does not measure whether a language model ultimately answers the questions correctly. The individual Canvas quiz and later peer feedback are separate parts of the course exercise.

## 1. Dataset and research question

We used the classic cleaned LongMemEval-S dataset, specifically `longmemeval_s_cleaned.json` from the data archive supplied to the group. The file is 277,383,467 bytes and has SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. The original benchmark contains 500 questions. After removing abstention items, the available pool in our processing was 470 questions: 150 information extraction (IE), 121 multi-session reasoning (MR), 72 knowledge update (KU), and 127 temporal reasoning (TR). We sampled five questions per group with seed 42, giving 20 questions. The same 20 questions were evaluated under every retrieval strategy, so the 60 CSV rows represent **20 paired experimental units**, not 60 independent questions.

Our preliminary question is: how do recency, semantic, and fixed-weight hybrid retrieval differ in evidence recall and retrieval latency at top-5? The unit retrieved is one dialogue round. Each question supplies a timestamped conversation history and gold evidence labels.

## 2. Variables and expected properties

| Role | Variable | Definition and expected property |
|---|---|---|
| Independent | Retrieval strategy | Categorical, with recency, semantic, and hybrid conditions applied to each question. |
| Grouping factor | Task type | IE, MR, KU, or TR; five pilot questions in each group. |
| Fixed condition | Retrieval depth | `k=5` for all results in this report. |
| Dependent | Round Recall@5 | Fraction of gold dialogue rounds among the five retrieved rounds; bounded in [0, 1]. |
| Dependent | Recall-any@5 | Binary indicator that at least one gold dialogue round was retrieved. |
| Dependent | Session Recall@5 | Fraction of gold sessions represented by the retrieved rounds; bounded in [0, 1]. |
| Dependent | Retrieval latency | Positive time in milliseconds, expected to vary substantially by strategy and potentially be skewed. |

The repository outline also identifies retrieved token count and final answer correctness as useful outcomes. They are **not available** in this pilot: no answer-generation tokenizer was specified, and answer generation was not run. These variable properties are design assumptions, not post-hoc findings. We did not treat an intuitive hope that hybrid retrieval would balance quality and speed as a preregistered hypothesis.

## 3. Preprocessing and exploratory analysis

The pilot preparation mapped original question types to IE/MR/KU/TR, excluded abstention IDs, sorted candidate IDs before seeded sampling, and kept the same questions for all three strategies. It parsed timestamps instead of relying on JSON list order. One dialogue round became one memory item. Sessions on calendar dates strictly later than the question were excluded, while later times on the same date were retained, following the project's documented handling of potentially randomized within-day times. Stella embeddings used the pinned revision `7817065102fd9e1b031fe874e910c01f40b2f001`, with a 512-token input limit. The hybrid score was fixed at `0.5 * (cosine + 1) / 2 + 0.5 * normalized_recency`.

Exploratory checks found exactly 20 questions and 60 complete strategy rows (five questions per task-strategy cell). Round recall, Recall-any, session recall, and latency had no missing entries. All 60 token-count entries were missing because no generation tokenizer was supplied. Ten of the 20 input histories were not chronologically sorted. Four questions contained at least one same-day session timestamped after the question; one contained a gold session with such a timestamp. Across 4,984 memory rounds, 1,813 (36.4%) exceeded the 512-token Stella limit. The final fp32 run had finite semantic score diagnostics for every question.

## 4. Statistical method

Because every strategy was applied to the same questions, comparisons were paired by question ID. We summarized each strategy with means, medians, and non-parametric bootstrap 95% confidence intervals (10,000 resamples, seed 42). For round and session recall and latency, the analysis used pairwise Wilcoxon signed-rank tests and paired bootstrap intervals for mean differences. For binary Recall-any, it used exact McNemar tests. Holm correction covered the three strategy pairs **within each metric and analysis scope**; it did not correct over every metric and task group jointly. Task-level results have only five paired questions and are treated descriptively.

No regression model was fit to this 20-question pilot. Strategy is an experimentally varied categorical condition and the immediate goal is a paired comparison, so these tests answer the preliminary question directly. Principal component analysis would not establish a strategy effect and would be difficult to interpret with this small sample and predefined outcomes. A larger study could fit a hierarchical or mixed-effects model with question and task effects after specifying the estimand and sample-size target.

## 5. Results

| Strategy | Mean round Recall@5 (95% CI) | Recall-any@5 | Mean session Recall@5 | Mean latency, ms |
|---|---:|---:|---:|---:|
| Recency | 0.000 (0.000-0.000) | 0.00 | 0.000 | 0.022 |
| Semantic | 0.483 (0.300-0.675) | 0.65 | 0.703 | 34.475 |
| Hybrid | 0.050 (0.000-0.125) | 0.10 | 0.085 | 34.848 |

Semantic exceeded hybrid by 0.433 in paired mean round Recall@5 (bootstrap 95% CI 0.267-0.608; Holm-adjusted Wilcoxon `p=0.00339`). Its Recall-any rate exceeded hybrid by 0.55 (Holm-adjusted exact McNemar `p=0.00195`). In this sample, recency retrieved no gold round. The semantic mean round recall by task was 0.600 for IE, 0.333 for MR, 0.600 for KU, and 0.400 for TR; these five-question task estimates are imprecise.

Figure 1 (`outputs/analysis/k5/ci_round_recall_at_k_overall.png`) shows overall mean round recall, individual-question points, and bootstrap intervals. Figure 2 (`task_ci_round_recall_at_k.png`) shows the four task groups. Figure 3 (`tradeoff_round_recall_at_k_vs_latency_ms.png`) plots recall against median measured retrieval latency on a logarithmic time axis. Figure 4 (`hybrid_component_scale.png`) compares the within-question standard deviations of the two hybrid score components. Vector PDF versions are available in the same analysis directory. The charts use different marker shapes in addition to the Okabe-Ito color palette.

## 6. Discussion

The observed semantic advantage is larger than we would have expected from a hybrid strategy intended to combine semantic relevance and recency. The most useful data for understanding this discrepancy were the per-question relevance labels, retrieved round IDs, and component-score diagnostics. Those fields let us distinguish a genuine evidence hit from merely retrieving the right session and reveal that the hybrid components do not contribute on comparable scales. Across questions, the recency-score standard deviation averaged 0.290, while the semantic-score standard deviation averaged 0.033: roughly an 8.8-fold difference despite equal nominal weights. The hybrid therefore behaves much more like a recency ranker in this pilot. This is a diagnosis, not a reason to change the primary formula after seeing the outcome; normalized-score or weight variations should be labeled as sensitivity analyses.

The results are limited in both accuracy and generality. Twenty paired questions, with only five in each task category, cannot establish reliable task-specific patterns. The overall semantic round-recall interval is also wide (0.300-0.675). More questions from the same cleaned dataset would narrow sampling uncertainty; a prespecified minimum important effect and power analysis, informed by pilot variance and discordant-pair counts, should determine the final sample size. Different data would also help: answer-generation accuracy, token use with the final model tokenizer, and end-to-end time including memory embedding would test whether higher retrieval recall produces a useful system-level gain. Over one-third of memory rounds exceed the embedding input limit, and one question retains a gold session timestamped later on the same day. Removing that question leaves semantic mean round recall at 0.456, but the time rule and truncation both warrant dedicated sensitivity checks. This pilot therefore supports an exploratory finding for these sampled LongMemEval-S questions, not a general claim about all tasks, datasets, or complete assistant performance.

## 7. Reproducibility and limitations

The valid run used Python 3.10.20, PyTorch 2.11.0+cu128, Sentence Transformers 3.0.1, Transformers 4.42.3, an RTX 5070 Ti Laptop GPU, fp32 precision, batch size 1, and the pinned Stella revision above. The machine's initial fp16 run produced non-finite semantic scores and was discarded; a small sanity example alone did not reveal that failure. The repository's recommendation of Python 3.11 and PyTorch 2.3.1 was not used on this GPU. The historical `outputs/pilot/environment.txt`, `gpu_info.txt`, and `python_version.txt` files were captured on a separate RTX 4070 preparation machine and are not the provenance record for this valid run. Measured latency includes retrieval after memory embeddings are prepared, but excludes one-time embedding preprocessing. Answer correctness and retrieved token costs remain unmeasured.

Run commands (from the repository root):

```powershell
python prepare_pilot_data.py --data-root data/longmemeval-cleaned --per-task 5 --seed 42 --output-dir outputs/pilot
python run_retrieval_pilot.py --pilot-file outputs/pilot/pilot_questions.json --output outputs/pilot/retrieval_results_v4_k5.csv --k 5 --fp32 --batch-size 1
python analyze_results.py --input outputs/pilot/retrieval_results_v4_k5.csv --output-dir outputs/analysis --k 5
```

## References

1. Wu, D. et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*. Official benchmark repository: https://github.com/xiaowu0162/LongMemEval
2. Cleaned LongMemEval dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
3. Group implementation and analysis scripts: https://github.com/yyqxdlg/longmemeval-retrieval-strategies (commit `8f94fc4aff288a98b0f6dcbe609c9fdf1d0cb959`).
