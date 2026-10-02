# LongMemEval Retrieval Pilot — Fixed Version v4

This version keeps the time-handling and retrieval fixes introduced in v3, while substantially expanding the quantitative analysis: paired tests, task-level analysis, 95% confidence intervals, RQ3 trade-off analysis, top-k comparison, and colorblind-friendly visualizations.

## What Changed in This Version

### 1. Recency no longer depends on list position

`memory_adapter.py` now parses timestamps such as:

```text
2023/05/30 (Tue) 11:46
```

and uses the actual timestamp to calculate recency. Therefore, Recency and Hybrid retrieval remain correct even when sessions in the JSON file are not ordered chronologically.

Availability filtering is now performed at the **calendar-day level**: a session is excluded only when `session_date.date() > question_date.date()`. A session that occurs later in HH:MM on the same day as the question is retained. This is necessary because LongMemEval may assign times within the same day in a way that would make minute-level comparison incorrectly classify valid same-day sessions as future information.

The full timestamp is still used for Recency ranking. Only the future-session filter uses the calendar day. The program also records `same_day_later` diagnostics so this behavior can be documented in the report. If gold evidence is found on a genuinely later calendar day, the program still stops and asks for manual inspection.

### 2. Recall is reported using three definitions

The retrieval output now includes:

- `round_recall_at_k`: proportion of relevant dialogue rounds retrieved;
- `recall_any_at_k`: whether at least one relevant round is retrieved in the top-k (0/1);
- `session_recall_at_k`: proportion of `answer_session_ids` covered by the retrieved memories.

This makes it possible to distinguish the granularity of turn-level `has_answer=True` annotations from session-level `answer_session_ids`.

### 3. Stella still uses 512 tokens as the main setting

The main experiment keeps:

```text
max_seq_length = 512
```

rather than directly changing the input length to 1024 or 2048. This keeps the main condition aligned with the selected Stella setup instead of introducing a new retrieval condition after inspecting the pilot.

The output CSV therefore adds diagnostics for truncation risk:

- `mean_memory_tokens_stella`
- `max_memory_tokens_stella`
- `num_memories_over_stella_limit`
- `fraction_memories_over_stella_limit`

These values can be used to quantify how often memory items exceed the 512-token input limit. A longer sequence length can later be evaluated separately as a sensitivity analysis.

### 4. The primary Hybrid formula is unchanged, but scale diagnostics are added

The main experiment still follows the Research Plan:

```text
S_semantic = (cosine + 1) / 2
S_recency = 1 - rank / (N - 1)
S_hybrid = 0.5 * S_semantic + 0.5 * S_recency
```

The output CSV now also records the mean, standard deviation, minimum, and maximum of semantic and recency scores across all memories. These diagnostics make it possible to check whether the recency score varies much more strongly than the semantic score.

Do not change the primary Hybrid formula after inspecting pilot results. If an alternative normalization such as min-max normalization is evaluated, report it separately as a sensitivity analysis.

### 5. GPU settings and reproducibility

- Stella revision is pinned to: `7817065102fd9e1b031fe874e910c01f40b2f001`.
- CUDA uses fp16 by default to reduce GPU memory usage.
- Default batch size is 4.
- Direct dependencies are version-pinned in `requirements.txt`.
- The complete runtime environment should still be saved with `pip freeze`, including transitive dependencies such as `tokenizers` and `huggingface_hub`.
- `.gitignore` excludes `.idea/`, `__pycache__/`, and `.venv/`.

### 6. Generation context includes dates

`build_generation_context()` sorts retrieved memories chronologically before passing them to the answer-generation model and formats them as:

```text
[Question date: ...]
[Session date: ...]
User: ...
Assistant: ...
```

Therefore, Recency, Semantic, and Hybrid all use the same context-ordering rule at generation time.

---

# Installation

A working Python 3.11 + CUDA environment can be reused.

For PyTorch on an RTX 4070 / CUDA 12.1 environment:

```powershell
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies with:

```powershell
pip install -r requirements.txt
```

---

# 1. Validate the Selected 20 Pilot Questions

The already checked pilot sample does not need to be resampled. Continue using:

```text
outputs/pilot/pilot_questions.json
```

Run:

```powershell
python validate_pilot.py --pilot-file outputs/pilot/pilot_questions.json
```

The script reports:

- histories that are not ordered by timestamp;
- questions containing sessions on a **later calendar day** than the question date (excluded by default);
- whether any gold evidence appears on a later calendar day;
- questions containing sessions that are later in HH:MM but still on the same day (retained and reported only as diagnostics).

---

# 2. Run Retrieval Again

Because the Recency / Hybrid time logic has changed, previous `retrieval_results.csv` files should not be used for the final analysis. Rerun retrieval with:

```powershell
python run_retrieval_pilot.py --pilot-file outputs/pilot/pilot_questions.json --output outputs/pilot/retrieval_results_fixed.csv --k 5
```

Default settings:

```text
Stella: NovaSearch/stella_en_1.5B_v5
revision: 7817065102fd9e1b031fe874e910c01f40b2f001
precision: fp16 on CUDA
max_seq_length: 512
batch_size: 4
future sessions: only sessions on strictly later calendar dates are excluded; same-day sessions are retained
latency: 1 warm-up + 5 measured runs, median reported
```

If an RTX 4070 runs out of GPU memory:

```powershell
python run_retrieval_pilot.py --pilot-file outputs/pilot/pilot_questions.json --output outputs/pilot/retrieval_results_fixed.csv --k 5 --batch-size 2
```

If necessary, reduce further to:

```powershell
--batch-size 1
```

All formal comparisons must use the same fixed batch size.

### Top-5 + Top-10

`--k` now accepts multiple values:

```powershell
python run_retrieval_pilot.py --pilot-file outputs/pilot/pilot_questions.json --output outputs/pilot/retrieval_results_fixed.csv --k 5 10
```

The CSV stores `k` explicitly. The analysis script does not mix top-5 and top-10 observations.

### Checkpointing and resume for long runs

Each completed result row is written and flushed to the CSV immediately. If the process fails at question 19, results from the first 18 questions are preserved.

If a run is interrupted, resume it with the same experimental settings:

```powershell
python run_retrieval_pilot.py --pilot-file outputs/pilot/pilot_questions.json --output outputs/pilot/retrieval_results_fixed.csv --k 5 --resume
```

The script skips existing `(question_id, strategy, k)` combinations.

---

# 3. Token Count

If the final answer-generation tokenizer is not provided, `token_count` remains empty and the script prints a warning. This is intentional: token cost should be measured with the tokenizer of the final generation LLM, not with the Stella tokenizer.

If the final generation model is Llama 3.1 8B Instruct and Hugging Face access has been configured:

```powershell
python run_retrieval_pilot.py --pilot-file outputs/pilot/pilot_questions.json --output outputs/pilot/retrieval_results_fixed.csv --k 5 --generation-tokenizer meta-llama/Llama-3.1-8B-Instruct
```

---

# 4. EDA, Statistical Analysis, and Figures

### Analyze top-5 only

```powershell
python analyze_results.py --input outputs/pilot/retrieval_results_fixed.csv --output-dir outputs/analysis --k 5
```

Results are written to:

```text
outputs/analysis/k5/
```

### Analyze a CSV containing both top-5 and top-10

Omit `--k`:

```powershell
python analyze_results.py --input outputs/pilot/retrieval_results_fixed.csv --output-dir outputs/analysis
```

The script will:

1. analyze `k=5` and `k=10` separately rather than mixing them;
2. create `outputs/analysis/cross_k/` automatically;
3. generate top-5 vs top-10 comparison tables and paired cross-k tests.

If top-5 and top-10 are stored in separate CSV files, both can be supplied:

```powershell
python analyze_results.py --input outputs/pilot/top5.csv outputs/pilot/top10.csv --output-dir outputs/analysis
```

### Main outputs produced for each k

```text
outputs/analysis/k5/
├── missing_values.csv
├── descriptive_ci_by_strategy.csv
├── descriptive_ci_by_task_strategy.csv
├── paired_wilcoxon_holm_overall.csv
├── paired_wilcoxon_holm_by_task.csv
├── paired_mcnemar_holm_overall.csv
├── paired_mcnemar_holm_by_task.csv
├── preprocessing_diagnostics.csv
├── ci_round_recall_at_k_overall.pdf/png
├── task_ci_round_recall_at_k.pdf/png
├── ci_recall_any_at_k_overall.pdf/png
├── task_ci_recall_any_at_k.pdf/png
├── boxplot_round_recall_at_k.pdf/png
├── boxplot_latency_ms.pdf/png
├── tradeoff_round_recall_at_k_vs_latency_ms.pdf/png
├── tradeoff_round_recall_at_k_vs_token_count.pdf/png   # when token_count is available
├── hybrid_component_scale.pdf/png
└── analysis_selection.txt
```

The analysis uses:

- paired Wilcoxon tests with Holm correction for `round_recall_at_k`, `session_recall_at_k`, latency, and token count;
- **exact McNemar tests with Holm correction** for the paired binary outcome `recall_any_at_k`;
- both overall comparisons and **task-level comparisons for IE / MR / KU / TR**;
- **95% non-parametric bootstrap confidence intervals** for plotted performance estimates;
- individual-question points in task-level figures so the small-sample structure is visible rather than hidden behind bar heights;
- **Recall vs latency** trade-off plots for RQ3, plus **Recall vs token count** when token-count data are available;
- automatic accuracy analysis if a future CSV contains `answer_correct`, including exact McNemar tests, task-level accuracy summaries, and Accuracy–Latency / Accuracy–Token trade-off figures;
- an **Okabe–Ito colorblind-friendly palette**, together with different markers and hatching so the figures remain distinguishable in grayscale.

### Top-5 vs Top-10 outputs

If the input contains more than one k value, the script additionally creates:

```text
outputs/analysis/cross_k/
├── topk_comparison_table.csv
├── top5_vs_top10_table.csv
└── paired_topk_tests.csv
```

`paired_topk_tests.csv` compares k=5 and k=10 within the same retrieval strategy:

- continuous or bounded outcomes: paired Wilcoxon test;
- binary outcomes such as recall-any: exact McNemar test;
- Holm correction is applied across the three strategy-specific cross-k tests for each metric.

**Pilot interpretation note:** each task currently contains only five questions. Task-level p-values should therefore be treated as an analysis-pipeline check or exploratory evidence, not as strong inferential conclusions. The main inferential interpretation should be performed on the larger experiment with approximately 50 questions per task.

---

# 5. IE Sampling

The manually checked 20-question pilot does not need to be resampled.

For the larger experiment, IE should preferably be stratified across its three subtypes:

```text
single-session-user
single-session-assistant
single-session-preference
```

`prepare_pilot_data.py` supports `--stratify-ie`:

```powershell
python prepare_pilot_data.py --data-root data/longmemeval-cleaned --per-task 50 --seed 42 --stratify-ie --output-dir outputs/full_sample
```

---

# 6. Save the Complete Runtime Environment

`requirements.txt` pins the direct dependencies used by this project. After the final experimental environment is working, save the complete environment with:

```powershell
python -m pip freeze > outputs/pilot/environment.txt
python --version > outputs/pilot/python_version.txt
nvidia-smi > outputs/pilot/gpu_info.txt
```

This records transitive dependencies such as `tokenizers` and `huggingface_hub` without guessing their versions. Python 3.11 is recommended for the current environment.

---

# 7. Report

See:

```text
REPORT_OUTLINE.md
```

The report should explicitly state:

- the dataset file used: `longmemeval_s_cleaned.json`;
- one dialogue round = one memory item;
- Recency is calculated from parsed `haystack_dates` timestamps;
- future-session policy: only sessions whose calendar date is strictly later than the question date are excluded; later times on the same day are retained;
- Stella's 512-token input limit and the observed truncation proportion;
- definitions of round-level, any-hit, and session-level Recall;
- Hybrid 0.5/0.5 weighting is the pre-registered primary condition; score-scale imbalance is treated as a diagnostic or limitation;
- the 20-question pilot has limited statistical power and should not be used for strong conclusions.
