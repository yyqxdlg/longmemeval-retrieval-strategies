# Confirmatory Local-Run Method Changes

This file records prospective changes made before the confirmatory results were
inspected. The original checked 20-question Top-5 pilot remains unchanged.

## Fixed confirmatory sample

- 200 non-abstention questions, seed 42.
- 50 questions for each of IE, MR, KU, and TR.
- IE is stratified across the three original single-session subtypes (17/17/16).
- The existing future-session policy is retained: exclude sessions on strictly
  later calendar dates and retain same-day sessions regardless of HH:MM.
- Every retrieval and generation condition uses the same fixed question IDs.

## Retrieval conditions

The primary conditions remain recency, semantic, and the pre-registered raw
hybrid formula. The raw hybrid output name is now `hybrid_raw` to distinguish it
from two sensitivity conditions:

```text
semantic_scaled = (cosine + 1) / 2
recency_score = 1 - rank / (N - 1)
hybrid_raw = 0.5 * semantic_scaled + 0.5 * recency_score
```

`hybrid_minmax` normalizes semantic cosine and recency scores independently
within each question and combines them with equal weights. If a component is
constant, every normalized value for that component is set to 0.5 and the event
is recorded in the output.

`hybrid_rrf` uses one-based, deterministic ordinal ranks:

```text
1 / (60 + semantic_rank) + 1 / (60 + recency_rank)
```

Ties are broken by stable memory identity. Document embeddings and the query
embedding are computed once per question and reused by all strategies and both
retrieval depths. No post-hoc 0.9/0.1 weighting is introduced.

Per-strategy retrieval latency measures deterministic ranking from these shared
precomputed scores (one warm-up plus five measured runs, median reported).
Stella document/query encoding is treated as shared preprocessing and excluded
from per-strategy latency rather than charged repeatedly to every condition.

The main Stella condition is the pinned revision
`7817065102fd9e1b031fe874e910c01f40b2f001`, fp32, batch size 1, and maximum
sequence length 512. Sequence length 1024 is a retrieval-only sensitivity
condition and does not replace the 512 main analysis.

## Local answer generation

All answer conditions use the same local Llama 3.1 8B Instruct immutable
revision, deterministic decoding (`do_sample=False`), seed 42, NF4 4-bit
quantization, `device_map=auto`, and a fixed system prompt. CPU/disk model
offload is rejected by default.

Two baselines are added once per question at `k=0`:

- `no_retrieval`: no dialogue memory is supplied.
- `oracle`: raw dialogue rounds carrying `has_answer=True` are supplied; when
  turn-level labels are unavailable, raw rounds from `answer_session_ids` are
  used instead.

Prompt construction accepts only the question, question date, raw dialogue
text, and system prompt. Gold answers, task labels, relevance flags, answer
session IDs, and other evaluation metadata are not accepted by that boundary.

## Evaluation status

No paid model API is used in this local run. `answer_correct` remains blank
until a later official judge run. Normalized exact match and string containment
are emitted only as explicitly labeled exploratory heuristics and are not
official LongMemEval answer accuracy.
