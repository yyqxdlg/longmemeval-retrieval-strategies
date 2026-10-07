# Authorized-Host Generation Handoff

The retrieval host has finished all 200-question 512 and 1024 retrieval work.
Do not recompute retrieval. This handoff starts from the committed fixed IDs
and `outputs/confirmatory_200/retrieval_stella512_fp32_k5_k10.csv`.

## 1. Reconstruct the ignored full records

Place the classic cleaned LongMemEval data under
`data/longmemeval-cleaned/`, then run:

```powershell
python prepare_pilot_data.py `
  --data-root data/longmemeval-cleaned `
  --per-task 50 `
  --seed 42 `
  --stratify-ie `
  --output-dir outputs/confirmatory_200
```

The reconstructed `pilot_questions.json` must have SHA256
`586a14e73f598885be4480094a98aabd263b1378e5beee062c4304df84ba7b19`
and its ordered `(question_id, task_type, question_type)` entries must match
`config/confirmatory_200_ids.json`. Stop if either check differs.

## 2. Download and freeze the official model

Use an account that has approved access to the official gated repository. Do
not paste a token into chat or a command-line argument.

```powershell
$env:HF_HOME = 'D:\AIProject\model'
hf auth login
hf auth whoami
$snapshot = hf download meta-llama/Llama-3.1-8B-Instruct --revision main
$revision = Split-Path $snapshot -Leaf
$revision
```

The printed snapshot directory name is the immutable commit revision. Use it
for every formal generation condition. Do not use an unofficial mirror and do
not copy model weights into the repository.

## 3. Smoke test and resume

Use the separate generation environment and keep the same settings for the
smoke and formal run:

```powershell
python run_generation.py `
  --pilot-file outputs/confirmatory_200/pilot_questions.json `
  --retrieval-results outputs/confirmatory_200/retrieval_stella512_fp32_k5_k10.csv `
  --output outputs/generation_local/generation_all_conditions.csv `
  --model-name meta-llama/Llama-3.1-8B-Instruct `
  --model-revision $revision `
  --quantization 4bit `
  --max-new-tokens 128 `
  --seed 42 `
  --include-baselines `
  --limit 1
```

Verify that the warm-up and row stay on GPU, the answer is non-empty, the
resolved revision equals `$revision`, and no CPU/disk offload occurs. Then run:

```powershell
python run_generation.py `
  --pilot-file outputs/confirmatory_200/pilot_questions.json `
  --retrieval-results outputs/confirmatory_200/retrieval_stella512_fp32_k5_k10.csv `
  --output outputs/generation_local/generation_all_conditions.csv `
  --model-name meta-llama/Llama-3.1-8B-Instruct `
  --model-revision $revision `
  --quantization 4bit `
  --max-new-tokens 128 `
  --seed 42 `
  --include-baselines `
  --resume
```

Expected output: 2,400 unique rows: five retrieval strategies times two k
values times 200 questions, plus 200 no-retrieval and 200 oracle rows. Run the
same command with `--resume` once more; it must add zero rows. Keep
`answer_correct` blank.

## 4. Local analysis and pending hypotheses

```powershell
python analyze_local_generation.py `
  --generation outputs/generation_local/generation_all_conditions.csv `
  --pilot-file outputs/confirmatory_200/pilot_questions.json `
  --output-dir outputs/analysis_generation `
  --manual-audit-output outputs/manual_audit/manual_audit_28.csv `
  --audit-size 28

python longmemeval_eval.py export `
  --input outputs/generation_local/generation_all_conditions.csv `
  --output outputs/hypotheses_pending `
  --split-by-condition
```

The local analysis may report completion, tokens, latency, empty-answer counts,
and explicitly labeled exploratory string heuristics. It must not claim
official answer accuracy. The manual audit CSV and raw dataset remain ignored.

## 5. Return artifacts

Commit the generation CSV and reasonably sized analysis/hypothesis files only
after scanning for secrets. Never commit the model cache, token, raw dataset,
or API key. Update `LOCAL_RUN_SUMMARY.md` and `PENDING_API_EVALUATION.md` with
the immutable Llama revision, actual counts, hashes, and generation status.
