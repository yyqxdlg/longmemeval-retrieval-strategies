# Local Confirmatory Run Summary

## Outcome

The 200-question retrieval experiment is complete on the fixed balanced sample.
Both Stella 512 (main) and Stella 1024 (retrieval-only sensitivity) completed
all five strategies at Top-5 and Top-10. The official Llama repository denied
this host account's access request, so model download and local answer
generation were not run here. No paid inference or judge API was called.

## Completed artifacts

| Artifact | Status |
| --- | --- |
| Fixed 200 IDs (`config/confirmatory_200_ids.json`) | Complete: IE/MR/KU/TR = 50 each; IE = 17/17/16 |
| Stella 512 combined retrieval CSV | Complete: 2,000 rows, 200 questions, 2,000 unique condition keys |
| Main recency/semantic/hybrid_raw splits | Complete for k=5 and k=10 |
| Hybrid min-max/RRF sensitivity splits | Complete for k=5 and k=10 |
| Stella 1024 combined retrieval CSV | Complete: 2,000 rows, 200 questions, 2,000 unique condition keys |
| Retrieval tables, paired tests, and figures | Complete for 512, 1024, and paired 512/1024 comparison |
| Resume validation | Complete: zero new rows for both finished combined runs |
| Local Llama generation | Blocked: official gated-repository request rejected |
| Official answer judging | Not run; official accuracy is unavailable |

The ignored full question records remain local as
`outputs/confirmatory_200/pilot_questions.json`; the dataset is not committed.
The committed fixed IDs allow an authorized teammate with the same cleaned
dataset to reconstruct and verify the exact sample.

## Retrieval configuration

- Stella: `NovaSearch/stella_en_1.5B_v5`
- Immutable revision: `7817065102fd9e1b031fe874e910c01f40b2f001`
- Precision / batch: fp32 / 1
- Main maximum sequence length: 512
- Sensitivity maximum sequence length: 1024
- Strategies: recency, semantic, hybrid_raw, hybrid_minmax, hybrid_rrf
- Retrieval depths: 5 and 10
- Input SHA256: `586a14e73f598885be4480094a98aabd263b1378e5beee062c4304df84ba7b19`
- Future-session policy: exclude strictly later calendar dates; retain same-day sessions

## Retrieval results

`recall_any_at_k` on the paired 200-question sample:

| Strategy | Stella 512 k=5 | Stella 512 k=10 | Stella 1024 k=5 | Stella 1024 k=10 |
| --- | ---: | ---: | ---: | ---: |
| recency | 0.015 | 0.100 | 0.015 | 0.100 |
| semantic | 0.835 | 0.925 | 0.825 | 0.920 |
| hybrid_raw | 0.275 | 0.310 | 0.260 | 0.310 |
| hybrid_minmax | 0.630 | 0.720 | 0.620 | 0.730 |
| hybrid_rrf | 0.460 | 0.640 | 0.450 | 0.640 |

These are retrieval metrics, not answer accuracy. All 200 questions had at
least one round exceeding 512 Stella tokens, and all recorded embeddings and
scores were finite in both runs.

Integrity hashes:

- 512 combined CSV SHA256: `b3531f30d75f9f7f72170b4a8e660d2965e08bab569b850fc96544bab9d5b42c`
- 1024 combined CSV SHA256: `77e7a04b23c645e3ce654048193b52f33b0fef30c6eef791adc3cf93ca7789b2`

## Main versus sensitivity results

Stella 512 is the confirmatory main condition. `hybrid_minmax`, `hybrid_rrf`,
and Stella 1024 are sensitivity analyses and must not replace or be pooled with
the pre-specified 512 main conditions. The checked 20-question pilot and small
smoke runs are also separate from all 200-question estimates above.

## Runtime

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12,227 MiB
- NVIDIA driver: 610.47
- Retrieval environment: Python 3.11.17, PyTorch 2.7.1+cu128,
  Transformers 4.42.3, Sentence Transformers 3.0.1
- Prepared generation environment: Python 3.11.17, PyTorch 2.7.1+cu128,
  Transformers 4.57.6, Accelerate 1.15.0, bitsandbytes 0.50.2
- Runtime metadata: `outputs/runtime/`

## Unfinished work

An authorized teammate should follow `TEAMMATE_HANDOFF.md` to download the
official model, freeze the resolved immutable revision, run the 2,400-row local
generation matrix, export 12 hypothesis JSONLs, run local-only diagnostics, and
create the manual audit file. `answer_correct` must remain blank until an
official judge or human review supplies labels. See `PENDING_API_EVALUATION.md`.
