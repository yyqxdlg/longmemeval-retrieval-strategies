# End-to-End Answer Evaluation Summary

## Evaluation status

- 200 questions, balanced across IE, MR, KU, and TR (50 each).
- 12 conditions and 2,400 judged answers; no missing judge labels.
- Judge protocol: LongMemEval official prompt and decision rule using the fixed
  `gpt-4o-2024-08-06` snapshot via OpenRouter, with the OpenAI provider pinned.
- Accuracy confidence intervals are non-parametric bootstrap 95% intervals over
  questions. Pairwise binary comparisons use exact McNemar tests.

## Main result

| Condition | Correct / 200 | Accuracy | Bootstrap 95% CI |
|---|---:|---:|---:|
| no retrieval | 11 | 5.5% | 2.5–9.0% |
| recency@5 | 16 | 8.0% | 4.5–12.0% |
| recency@10 | 24 | 12.0% | 7.5–16.5% |
| hybrid raw@5 | 40 | 20.0% | 14.5–26.0% |
| hybrid raw@10 | 44 | 22.0% | 16.5–28.0% |
| hybrid RRF@5 | 46 | 23.0% | 17.5–29.0% |
| hybrid RRF@10 | 63 | 31.5% | 25.0–38.0% |
| hybrid min-max@5 | 71 | 35.5% | 29.0–42.0% |
| hybrid min-max@10 | 83 | 41.5% | 35.0–48.5% |
| semantic@5 | 95 | 47.5% | 40.5–54.5% |
| **semantic@10** | **114** | **57.0%** | **50.0–64.0%** |
| oracle | 144 | 72.0% | 65.5–78.0% |

Semantic@10 is the strongest non-oracle condition. It improves over no
retrieval by 51.5 percentage points and remains 15.0 points below oracle. Both
paired differences remain significant after Holm correction across all 66
condition pairs (adjusted p < 0.001 and p = 0.0055, respectively).

Within k=10, semantic is significantly better than every other retrieval
strategy. Its advantage over the second-best hybrid min-max condition is 15.5
points (exact McNemar, within-k Holm-adjusted p = 0.00019).

## Effect of increasing k

| Strategy | Accuracy@5 | Accuracy@10 | Difference | Cross-k Holm p |
|---|---:|---:|---:|---:|
| recency | 8.0% | 12.0% | +4.0 pp | 0.3032 |
| semantic | 47.5% | 57.0% | +9.5 pp | 0.0439 |
| hybrid raw | 20.0% | 22.0% | +2.0 pp | 0.5716 |
| hybrid min-max | 35.5% | 41.5% | +6.0 pp | 0.1283 |
| hybrid RRF | 23.0% | 31.5% | +8.5 pp | 0.0299 |

The cross-k p-values above use the planned family of five within-strategy
comparisons. Under this correction, the gains for semantic and hybrid RRF are
statistically significant; the other three are not.

## Task-type results

| Condition | IE | MR | KU | TR |
|---|---:|---:|---:|---:|
| semantic@10 | 68% | 34% | 74% | 52% |
| hybrid min-max@10 | 56% | 22% | 64% | 24% |
| oracle | 84% | 58% | 74% | 72% |

Semantic@10 reaches oracle accuracy on KU, but the remaining gaps on IE, MR,
and TR are 16, 24, and 20 points. The 50-question task cells should be treated
as secondary analyses rather than independent confirmatory tests.

## Retrieval success and answer accuracy

- Semantic recall-any rises from 83.5% at k=5 to 92.5% at k=10.
- For semantic@10, answer accuracy is 61.1% when at least one gold memory is
  retrieved (113/185), versus 6.7% when none is retrieved (1/15).
- Hybrid min-max@10 reaches 53.5% when recall-any is true, but its recall-any
  rate is only 72.0%; this explains much of its lower overall accuracy.
- Even with oracle context, accuracy is 72.0%, so retrieval is not the only
  error source. Local answer generation and/or judge-sensitive reasoning still
  impose a substantial ceiling.

## Interpretation and caveat

The end-to-end evidence supports semantic retrieval over recency and the tested
hybrid formulations. The raw weighted hybrid is especially weak, showing that
combining incomparable score scales without normalization is not reliable.
Increasing k helps semantic and RRF, but also roughly doubles the median prompt
size, so k=10 trades context cost for accuracy.

The recorded `latency_ms` values cover ranking only and exclude Stella document
and query encoding. They must not be reported as full semantic/hybrid retrieval
latency. The separate encoding benchmark is required for the complete RQ3
efficiency comparison.
