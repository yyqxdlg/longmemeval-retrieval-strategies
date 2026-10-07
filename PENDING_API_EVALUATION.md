# Pending Official Answer Evaluation

## Current status

No paid model API has been called, no official judge labels exist, and no
official answer accuracy is reported. The retrieval host could not create local
answers because its request for the official gated Llama repository was
rejected. An authorized teammate must first complete `TEAMMATE_HANDOFF.md`.

After the 2,400-row generation run, `longmemeval_eval.py export
--split-by-condition` should create these 12 unscored JSONLs, each with 200
hypotheses:

- `recency_k5.jsonl`, `recency_k10.jsonl`
- `semantic_k5.jsonl`, `semantic_k10.jsonl`
- `hybrid_raw_k5.jsonl`, `hybrid_raw_k10.jsonl`
- `hybrid_minmax_k5.jsonl`, `hybrid_minmax_k10.jsonl`
- `hybrid_rrf_k5.jsonl`, `hybrid_rrf_k10.jsonl`
- `no_retrieval_k0.jsonl`
- `oracle_k0.jsonl`

## Later official evaluation

Only after the team intentionally enables the paid judge and supplies its API
key outside the repository, run the benchmark evaluator separately for every
JSONL. Example:

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

The final merge must verify all 2,400 condition keys before computing official
accuracy, answer-correctness McNemar tests, or `answer_correct × recall_any`
tables. Until then, blank `answer_correct` means unknown, not incorrect.
