# LongMemEval Retrieval and Answer-Generation Pipeline — v5

This repository contains the pilot implementation for the project **How Memory Retrieval Strategies Affect LLM Agent Performance**.

The experiment compares three memory retrieval strategies on the classic cleaned LongMemEval benchmark:

- **Recency retrieval**
- **Semantic retrieval**
- **Hybrid retrieval**

The pilot uses four task groups:

- **IE** — Information Extraction
- **MR** — Multi-session Reasoning
- **KU** — Knowledge Update
- **TR** — Temporal Reasoning

The current pilot contains **20 questions: 5 per task type**.

---

## Main Experimental Design

The primary independent variable is the **retrieval strategy**:

1. Recency
2. Semantic
3. Hybrid

The main retrieval metrics are:

- `round_recall_at_k`
- `recall_any_at_k`
- `session_recall_at_k`
- `latency_ms`

The primary retrieval depth is:

```text
k = 5
```

Top-10 can be evaluated separately as an additional comparison.

---

# Installation

Python **3.11** is recommended.

The working environment uses:

```text
Python: 3.11
PyTorch: 2.3.1 + CUDA 12.1
Transformers: 4.42.3
Sentence Transformers: 3.0.1
Embedding model: NovaSearch/stella_en_1.5B_v5
Stella revision: 7817065102fd9e1b031fe874e910c01f40b2f001
```

For Windows with CUDA 12.1, install PyTorch first:

```powershell
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

Keep the checked retrieval environment separate from answer generation. Llama
3.1 requires Transformers 4.43.2 or newer, while the completed retrieval run
used Transformers 4.42.3. On the generation machine, install a matching CUDA
PyTorch build and then:

```powershell
pip install -r requirements-generation.txt
```

Accept the Meta Llama 3.1 license on Hugging Face and authenticate before the
first model download. A 4-bit run is the default in `run_generation.py`; use the
same quantization and model revision for every experimental condition.

`requirements-lock.txt` records the complete working environment, including transitive dependencies.

---

# Dataset

This project uses the **classic cleaned LongMemEval dataset**, not LongMemEval-V2.

Clone the dataset into the local `data/` directory:

```powershell
git clone https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned data/longmemeval-cleaned
```

The pilot uses:

```text
data/longmemeval-cleaned/longmemeval_s_cleaned.json
```

The repository does not track the dataset because `data/` is excluded by `.gitignore`.

Expected local structure:

```text
data/
└── longmemeval-cleaned/
    ├── longmemeval_m_cleaned.json
    ├── longmemeval_oracle.json
    ├── longmemeval_s_cleaned.json
    └── README.md
```

---

# Data Processing

## 1. Task Mapping

`prepare_pilot_data.py` maps the original LongMemEval question types into four task groups:

```text
single-session-user        -> IE
single-session-assistant   -> IE
single-session-preference  -> IE
multi-session              -> MR
knowledge-update           -> KU
temporal-reasoning         -> TR
```

Abstention questions whose IDs end with:

```text
_abs
```

are excluded from the pilot.

---

## 2. Pilot Sampling

The current pilot uses:

```text
5 IE
5 MR
5 KU
5 TR
```

for a total of:

```text
20 questions
```

The fixed random seed is:

```text
42
```

Generate the pilot with:

```powershell
python prepare_pilot_data.py `
  --data-root data/longmemeval-cleaned `
  --per-task 5 `
  --seed 42 `
  --output-dir outputs/pilot
```

For the current checked pilot, do **not** use `--stratify-ie`.

The script produces:

```text
outputs/pilot/
├── pilot_questions.json
├── pilot_ids.json
└── pilot_manifest.csv
```

`pilot_questions.json` contains the full selected LongMemEval records.

`pilot_ids.json` records the selected question IDs and task types.

`pilot_manifest.csv` provides a compact summary of the selected questions.

The checked pilot IDs are also stored in:

```text
config/pilot_ids.json
```

This makes the pilot sampling reproducible. Selected pilot and analysis outputs
are currently committed; the dataset and local caches remain ignored.

---

## 3. Verify the Pilot IDs

After generating the pilot, verify that the generated IDs match the checked pilot:

```powershell
python -c "import json; a=json.load(open('outputs/pilot/pilot_ids.json', encoding='utf-8')); b=json.load(open('config/pilot_ids.json', encoding='utf-8')); x=[q['question_id'] for q in a['questions']]; y=[q['question_id'] for q in b['questions']]; print('Pilot IDs match:', x==y)"
```

Expected output:

```text
Pilot IDs match: True
```

---

## 4. Validate the Pilot

Run:

```powershell
python validate_pilot.py --pilot-file outputs/pilot/pilot_questions.json
```

For the checked 20-question pilot, the validation result is:

```text
Task counts: {'IE': 5, 'MR': 5, 'KU': 5, 'TR': 5}
Unsorted histories: 10/20
Questions with sessions on a STRICTLY LATER CALENDAR DAY: 0/20
Questions with GOLD sessions on a strictly later calendar day: 0/20
Questions with same-day sessions later by HH:MM (retained): 4/20
Questions with same-day GOLD sessions later by HH:MM (retained): 1/20
```

The unsorted histories are expected. The retrieval implementation does not rely on the JSON list order for recency.

---

# Memory Construction

One **dialogue round** is treated as one memory item.

The memory adapter extracts memory items from the LongMemEval session histories and preserves:

- question ID
- session ID
- round index
- session timestamp
- dialogue text
- relevance labels

Relevant rounds are identified using turn-level `has_answer=True` labels when available.

Session-level `answer_session_ids` are retained separately for session-level recall.

---

# Time Handling

Some LongMemEval histories are not stored in chronological order.

Therefore, Recency and Hybrid retrieval use parsed timestamps rather than list positions.

Timestamps such as:

```text
2023/05/30 (Tue) 11:46
```

are parsed directly.

The preprocessing policy is:

- full timestamps are used for Recency ranking;
- sessions on a **strictly later calendar date** than the question are excluded;
- sessions later in HH:MM on the same calendar day are retained;
- same-day-later cases are recorded as diagnostics.

For the current pilot:

```text
strictly later calendar-day questions: 0/20
strictly later calendar-day gold questions: 0/20
same-day-later questions: 4/20
same-day-later gold questions: 1/20
```

This calendar-day filtering rule is a benchmark-specific preprocessing decision and should be documented in the report.

---

# Retrieval Strategies

## Recency

Recency retrieval ranks memories using the actual parsed timestamps.

The normalized recency score is:

```text
S_recency = 1 - rank / (N - 1)
```

where rank 0 is the newest memory.

---

## Semantic

Semantic retrieval uses:

```text
NovaSearch/stella_en_1.5B_v5
```

Pinned revision:

```text
7817065102fd9e1b031fe874e910c01f40b2f001
```

The query is encoded using Stella's:

```text
s2p_query
```

Memory embeddings are normalized and reused.

Semantic similarity is cosine similarity.

---

## Hybrid

The Hybrid strategy uses the fixed formula from the Research Plan:

```text
S_semantic = (cosine + 1) / 2
S_recency = 1 - rank / (N - 1)
S_hybrid = 0.5 * S_semantic + 0.5 * S_recency
```

The primary 0.5 / 0.5 weighting is kept fixed.

The implementation also records semantic-score and recency-score statistics so that scale imbalance can be inspected without changing the primary formula.

---

# Stella Input Length

The main experiment uses:

```text
max_seq_length = 512
```

The output records:

```text
mean_memory_tokens_stella
max_memory_tokens_stella
num_memories_over_stella_limit
fraction_memories_over_stella_limit
```

These diagnostics quantify possible truncation.

A longer input length should be treated as a separate sensitivity analysis.

---

# Sanity Tests

Run the retrieval sanity test:

```powershell
python test_retrieval.py
```

A successful run should print:

```text
RECENCY
SEMANTIC
HYBRID
```

Run the time-handling test:

```powershell
python test_time_handling.py
```

The warning:

```text
Torch was not compiled with flash attention
```

is not an error. It may affect speed but does not invalidate the retrieval result.

---

# Run Top-5 Retrieval

Run:

```powershell
python run_retrieval_pilot.py `
  --pilot-file outputs/pilot/pilot_questions.json `
  --output outputs/pilot/retrieval_results_v4_k5.csv `
  --k 5
```

Default settings:

```text
Embedding model: NovaSearch/stella_en_1.5B_v5
Revision: 7817065102fd9e1b031fe874e910c01f40b2f001
Precision: fp16 on CUDA
Max sequence length: 512
Batch size: 4
Latency: 1 warm-up + 5 measured runs; median reported
Future-session policy: exclude strictly later calendar dates; retain same-day sessions
```

The expected output size is:

```text
20 questions × 3 strategies = 60 rows
```

If GPU memory is insufficient, use:

```text
--batch-size 2
```

or:

```text
--batch-size 1
```

Use the same batch size across compared conditions.

---

# Checkpointing and Resume

Each completed result row is written and flushed immediately.

If the experiment is interrupted, resume with:

```powershell
python run_retrieval_pilot.py `
  --pilot-file outputs/pilot/pilot_questions.json `
  --output outputs/pilot/retrieval_results_v4_k5.csv `
  --k 5 `
  --resume
```

`--resume` skips already completed:

```text
(question_id, strategy, k)
```

combinations.

---

# Check Retrieval Output

After Top-5 finishes:

```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs/pilot/retrieval_results_v4_k5.csv'); print(df.shape); print(df['strategy'].value_counts()); print(df.groupby(['task_type','strategy']).size())"
```

Expected total:

```text
60 rows
```

Expected strategy counts:

```text
recency     20
semantic    20
hybrid      20
```

Each task type should have 5 observations for each strategy.

---

# Retrieval Metrics

The output contains three recall definitions.

## Round Recall@k

```text
round_recall_at_k
```

The proportion of relevant dialogue rounds retrieved.

## Recall-any@k

```text
recall_any_at_k
```

Binary retrieval-success variable:

```text
1 = at least one relevant round was retrieved
0 = no relevant round was retrieved
```

## Session Recall@k

```text
session_recall_at_k
```

The proportion of benchmark `answer_session_ids` covered by retrieved memories.

---

# Retrieval Latency

Retrieval latency is measured using:

```text
1 warm-up run
5 measured runs
median latency reported
```

One-time memory embedding preprocessing is excluded from retrieval latency.

CUDA synchronization is used when measuring GPU retrieval operations.

---

# Token Count

If no generation tokenizer is supplied, `token_count` remains empty.

This is intentional because token cost should be measured using the tokenizer of the final answer-generation model, not the Stella embedding tokenizer.

The retrieval runner can optionally count only the retrieved context with the
final model tokenizer. Use a separate output file rather than overwriting the
checked result:

```powershell
python run_retrieval_pilot.py `
  --pilot-file outputs/pilot/pilot_questions.json `
  --output outputs/pilot/retrieval_results_v4_k5_with_context_tokens.csv `
  --k 5 `
  --generation-tokenizer meta-llama/Llama-3.1-8B-Instruct
```

This value excludes the system prompt, question, chat template, and generated
answer. `run_generation.py` records both `retrieved_context_token_count` and the
complete `prompt_token_count`; use the latter for end-to-end cost analysis.

---

# Quantitative Analysis

Analyze Top-5 with:

```powershell
python analyze_results.py `
  --input outputs/pilot/retrieval_results_v4_k5.csv `
  --output-dir outputs/analysis `
  --k 5
```

Results are written to:

```text
outputs/analysis/k5/
```

Main output tables include:

```text
missing_values.csv
descriptive_ci_by_strategy.csv
descriptive_ci_by_task_strategy.csv
paired_wilcoxon_holm_overall.csv
paired_wilcoxon_holm_by_task.csv
paired_mcnemar_holm_overall.csv
paired_mcnemar_holm_by_task.csv
preprocessing_diagnostics.csv
analysis_selection.txt
```

Main figures include:

```text
ci_round_recall_at_k_overall.pdf/png
task_ci_round_recall_at_k.pdf/png
ci_recall_any_at_k_overall.pdf/png
task_ci_recall_any_at_k.pdf/png
boxplot_round_recall_at_k.pdf/png
boxplot_latency_ms.pdf/png
tradeoff_round_recall_at_k_vs_latency_ms.pdf/png
hybrid_component_scale.pdf/png
```

If token counts are available:

```text
tradeoff_round_recall_at_k_vs_token_count.pdf/png
```

---

# Statistical Tests

The analysis includes:

- descriptive statistics;
- 95% non-parametric bootstrap confidence intervals;
- paired Wilcoxon tests with Holm correction for:
  - `round_recall_at_k`
  - `session_recall_at_k`
  - latency
  - token count, when available;
- exact McNemar tests with Holm correction for:
  - `recall_any_at_k`;
- overall comparisons;
- task-level comparisons for:
  - IE
  - MR
  - KU
  - TR;
- Recall vs latency trade-off plots;
- Recall vs token-count trade-off plots when token counts are available;
- Okabe-Ito colorblind-friendly plotting together with different markers and hatching.

The pilot contains only five questions per task type, so task-level p-values should be interpreted as exploratory results rather than strong inferential conclusions.

---

# Top-5 and Top-10

Run Top-10 into a separate raw-results file. This preserves the checked Top-5
artifact and prevents different environments or tokenizer settings from being
silently mixed:

```powershell
python run_retrieval_pilot.py `
  --pilot-file outputs/pilot/pilot_questions.json `
  --output outputs/pilot/retrieval_results_v4_k10.csv `
  --k 10 `
  --fp32 `
  --batch-size 1
```

With both Top-5 and Top-10 result files, run:

```powershell
python analyze_results.py `
  --input outputs/pilot/retrieval_results_v4_k5.csv outputs/pilot/retrieval_results_v4_k10.csv `
  --output-dir outputs/analysis
```

Do not specify `--k`.

The analysis then creates:

```text
outputs/analysis/
├── k5/
├── k10/
└── cross_k/
```

Cross-k outputs include:

```text
topk_comparison_table.csv
top5_vs_top10_table.csv
paired_topk_tests.csv
```

---

# Larger Sample

For the larger experiment, IE can be stratified across:

```text
single-session-user
single-session-assistant
single-session-preference
```

Example:

```powershell
python prepare_pilot_data.py `
  --data-root data/longmemeval-cleaned `
  --per-task 50 `
  --seed 42 `
  --stratify-ie `
  --output-dir outputs/full_sample
```

---

# Answer Generation

The completed retrieval-host run and the authorized-host generation handoff
are summarized in `LOCAL_RUN_SUMMARY.md` and `TEAMMATE_HANDOFF.md`. The
retrieval host was denied access to the official gated Llama repository, so it
does not claim local generation results.

## Confirmatory 200-question local run

The confirmatory run is separate from the checked 20-question pilot. Its fixed
IDs are stored in `config/confirmatory_200_ids.json`. Generate the full local
records from the ignored dataset with:

```powershell
python prepare_pilot_data.py `
  --data-root data/longmemeval-cleaned `
  --per-task 50 `
  --seed 42 `
  --stratify-ie `
  --output-dir outputs/confirmatory_200
```

Run all five retrieval conditions and both retrieval depths in one pass so the
Stella document and query embeddings are shared:

```powershell
python run_retrieval_pilot.py `
  --pilot-file outputs/confirmatory_200/pilot_questions.json `
  --output outputs/confirmatory_200/retrieval_stella512_fp32_k5_k10.csv `
  --k 5 10 `
  --fp32 `
  --batch-size 1 `
  --max-seq-length 512
```

The default conditions are `recency`, `semantic`, `hybrid_raw`,
`hybrid_minmax`, and `hybrid_rrf`. The original primary hybrid formula is
unchanged; the latter two are sensitivity conditions. Use `--resume` with the
same output and settings after interruption. Use `--question-id ...` for a
small smoke or sensitivity subset.

After validation, split the primary and hybrid-sensitivity tables without
recomputing embeddings:

```powershell
python organize_confirmatory_retrieval.py `
  --input outputs/confirmatory_200/retrieval_stella512_fp32_k5_k10.csv `
  --main-output-dir outputs/confirmatory_200 `
  --hybrid-output-dir outputs/sensitivity_hybrid
```

Stella 1024 is a separate retrieval-only sensitivity run. Compare its complete
CSV to the 512 CSV with `compare_stella_sensitivity.py`; do not pool it into the
main analysis.

Generate the retrieval conditions plus the two k=0 baselines locally with one
frozen Llama revision:

```powershell
python run_generation.py `
  --pilot-file outputs/confirmatory_200/pilot_questions.json `
  --retrieval-results outputs/confirmatory_200/retrieval_stella512_fp32_k5_k10.csv `
  --output outputs/generation_local/generation_all_conditions.csv `
  --model-name meta-llama/Llama-3.1-8B-Instruct `
  --model-revision <immutable-commit> `
  --quantization 4bit `
  --max-new-tokens 128 `
  --seed 42 `
  --include-baselines
```

The script refuses CPU/disk model offload unless explicitly overridden. It
flushes every row and validates model, revision, quantization, prompt, and
decoding settings before resume. Export one pending official-evaluation JSONL
per condition with:

```powershell
python longmemeval_eval.py export `
  --input outputs/generation_local/generation_all_conditions.csv `
  --output outputs/hypotheses_pending `
  --split-by-condition
```

For local-only completion, token/latency, exploratory string diagnostics, and a
28-row manual audit template, use `analyze_local_generation.py`. Its string
metrics are explicitly exploratory and are not official answer accuracy.

The repository separates retrieval from answer generation so raw retrieval
results remain immutable:

```text
retrieved Top-k memories
        ↓
answer-generation LLM
        ↓
generated answer
        ↓
answer correctness evaluation
```

The answer model is:

```text
Llama 3.1 8B Instruct
```

Install the generation dependencies in a separate environment, then generate
answers for each retrieval file. The default is deterministic 4-bit inference
with `do_sample=False`, seed 42, and at most 128 new tokens:

```powershell
python run_generation.py `
  --pilot-file outputs/pilot/pilot_questions.json `
  --retrieval-results outputs/pilot/retrieval_results_v4_k5.csv `
  --output outputs/generation/generation_results_k5.csv `
  --model-name meta-llama/Llama-3.1-8B-Instruct `
  --model-revision main `
  --quantization 4bit
```

Before the full run, append `--limit 1` as a smoke test. If that row is valid,
rerun the same command without `--limit` and add `--resume`; the completed row
will be skipped after the metadata settings are checked.

The run records the resolved Hugging Face commit. For subsequent confirmatory
runs, pass that immutable commit instead of `main`. The generated prompt is
constructed only from the question, question date, and retrieved memory text;
gold answers, relevance flags, answer-session IDs, and task labels are not
accepted by the prompt-building function.

For Top-10, repeat the command with the Top-10 retrieval CSV and a separate
generation output. Do not change the prompt, model, revision, quantization, or
decoding settings between strategies or k values.

Generation outputs add:

```text
generated_answer
retrieved_context_token_count
prompt_token_count
completion_token_count
generation_latency_ms
total_latency_ms
answer_model_revision
context_sha256
```

## Official Answer Correctness Evaluation

Export generated answers in the official LongMemEval hypothesis format:

```powershell
python longmemeval_eval.py export `
  --input outputs/generation/generation_results_k5.csv `
  --output outputs/evaluation/hypotheses_k5.jsonl
```

Run LongMemEval's official evaluator from a checked-out copy of the benchmark.
The evaluator requires `OPENAI_API_KEY` when `gpt-4o` is used as the judge:

```powershell
python path/to/LongMemEval/src/evaluation/evaluate_qa.py `
  gpt-4o `
  outputs/evaluation/hypotheses_k5.jsonl `
  outputs/pilot/pilot_questions.json
```

If a direct OpenAI API key is unavailable, the repository also includes an
OpenRouter adapter that preserves the official LongMemEval judge prompts and
pins the same `gpt-4o-2024-08-06` snapshot:

```powershell
# Set this only in the local process; never commit the key.
$env:OPENROUTER_API_KEY = "<set-locally>"

# Export, run one paid smoke-test judgment, and create a partial scored CSV.
.\scripts\run_openrouter_evaluation.ps1 -Limit 1

# Resume the same log and finish all remaining judgments.
.\scripts\run_openrouter_evaluation.ps1 -Limit 0 -Resume
```

The default route is restricted to OpenRouter's OpenAI provider. Report the
result as "LongMemEval official judge protocol using GPT-4o-2024-08-06 via
OpenRouter." See `scripts/OPENROUTER_EVALUATION.md` for secure key entry, a
no-cost dry run, and the exact output paths.

Merge the official labels back into a new, scored CSV:

```powershell
python longmemeval_eval.py merge `
  --generation-results outputs/generation/generation_results_k5.csv `
  --evaluation-log outputs/evaluation/hypotheses_k5.jsonl.eval-results-gpt-4o `
  --output outputs/generation/generation_results_k5_scored.csv
```

Then analyze scored Top-5 and Top-10 files together:

```powershell
python analyze_results.py `
  --input outputs/generation/generation_results_k5_scored.csv outputs/generation/generation_results_k10_scored.csv `
  --output-dir outputs/analysis_end_to_end
```

The analysis then includes:

- answer accuracy;
- exact McNemar tests for paired answer correctness;
- Accuracy vs total retrieval-plus-generation latency;
- Accuracy vs complete prompt token count.

The retrieval-only McNemar analysis for `recall_any_at_k` remains distinct from
the final McNemar analysis of `answer_correct`.

---

# Save Runtime Information

Save provenance into a run-specific directory rather than overwriting another
machine's metadata:

```powershell
python save_runtime_info.py --output-dir outputs/runtime/retrieval_YYYYMMDD
python save_runtime_info.py --output-dir outputs/runtime/generation_YYYYMMDD
```

The files are written as UTF-8 and include a structured `runtime.json`. The
older files under `outputs/pilot/` describe the RTX 4070 preparation machine;
the preliminary report documents the separate RTX 5070 Ti run that produced
the checked fp32 results. Do not treat either machine's files as provenance for
a new run.

---

# Repository Structure

```text
longmemeval-retrieval-strategies/
│
├── README.md
├── requirements.txt
├── requirements-generation.txt
├── requirements-lock.txt
├── .gitignore
│
├── config/
│   └── pilot_ids.json
│
├── prepare_pilot_data.py
├── inspect_dataset.py
├── validate_pilot.py
├── memory_adapter.py
├── retrieval.py
├── run_retrieval_pilot.py
├── generation.py
├── run_generation.py
├── longmemeval_eval.py
├── experiment_io.py
├── save_runtime_info.py
├── analyze_results.py
├── test_retrieval.py
├── test_time_handling.py
├── test_pipeline.py
│
├── REPORT_OUTLINE.md
├── CHANGELOG_v4.md
├── CHANGELOG_v5.md
│
├── data/       # local only, ignored by Git
└── outputs/    # checked artifacts plus new run outputs
```
