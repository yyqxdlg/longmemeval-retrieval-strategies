# Official Answer Evaluation Status

## Current status

Local answer generation and answer-correctness evaluation are complete for all
2,400 question-condition rows. The evaluation follows the LongMemEval official
judge protocol using the fixed `gpt-4o-2024-08-06` snapshot via OpenRouter, with
the OpenAI provider pinned.

The auditable judge log is stored at
`outputs/evaluation_openrouter/gpt4o_2024_08_06_all_conditions.jsonl`. The fully
merged result is stored at
`outputs/generation_local/generation_all_conditions_scored_openrouter.csv`.
Every row has a non-empty `answer_correct` label.

`longmemeval_eval.py export --split-by-condition` has created these 12
unscored JSONLs under `outputs/hypotheses_pending`, each with 200 hypotheses:

- `recency_k5.jsonl`, `recency_k10.jsonl`
- `semantic_k5.jsonl`, `semantic_k10.jsonl`
- `hybrid_raw_k5.jsonl`, `hybrid_raw_k10.jsonl`
- `hybrid_minmax_k5.jsonl`, `hybrid_minmax_k10.jsonl`
- `hybrid_rrf_k5.jsonl`, `hybrid_rrf_k10.jsonl`
- `no_retrieval_k0.jsonl`
- `oracle_k0.jsonl`

## Reproducing the official evaluation

The unscored exports are preserved so the evaluation can be independently
reproduced. With a separately supplied paid API key, the upstream benchmark
evaluator can be run for every JSONL. Example:

```powershell
python path/to/LongMemEval/src/evaluation/evaluate_qa.py `
  gpt-4o `
  outputs/hypotheses_pending/semantic_k5.jsonl `
  outputs/confirmatory_200/pilot_questions.json
```

Merge each resulting evaluation log into a new scored CSV; never overwrite the
unscored generation file:

```powershell
python longmemeval_eval.py merge `
  --generation-results outputs/generation_local/generation_all_conditions.csv `
  --evaluation-log <condition-evaluation-log> `
  --output outputs/generation_local/generation_scored_partial.csv `
  --allow-partial
```

Any repeat evaluation must verify all 2,400 condition keys before computing
accuracy, answer-correctness McNemar tests, or `answer_correct × recall_any`
tables. Blank `answer_correct` always means unknown, not incorrect.

## OpenRouter route

To reproduce the completed OpenRouter run, use
`scripts/run_openrouter_evaluation.ps1`. It calls the same fixed
`gpt-4o-2024-08-06` model snapshot through OpenRouter, preserves the upstream
LongMemEval prompts and decoding settings, pins the OpenAI provider by default,
and writes an auditable, resumable JSONL log.

The key must be supplied as `OPENROUTER_API_KEY` outside the repository. Start
with `-DryRun -Limit 0`, then `-Limit 1`, then finish with
`-Limit 0 -Resume`. Full
instructions are in `scripts/OPENROUTER_EVALUATION.md`. Report the gateway
explicitly; do not describe the run as a direct OpenAI API call.
