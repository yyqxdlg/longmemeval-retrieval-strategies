# Stella 512 fp32 encoding-time measurement

The aggregate statistics use each question's median across repeats,
then report the median/minimum/maximum across the fixed 10 questions.
Model loading, JSON reading, timestamp parsing, future-session filtering,
similarity calculation, and ranking are outside the timed regions.

## Results

- Document encoding total (s/history): median 28.738382, min 27.821085, max 29.618723
- History size N (rounds): median 243.5, range 218-277
- Document encoding normalized (ms/round): median 119.625, min 106.326, max 127.620
- Query encoding (ms/query): median 37.062, min 33.178, max 39.891

## Semantic Top-5 consistency

- Exact ordered matches: 8/10
- Same Top-5 sets: 8/10
- Retrieved-ID overlap: 48/50
- The two mismatches each replace only the fifth-ranked boundary item. A focused
  repeatability check under the same settings found non-bitwise-identical CUDA
  embeddings (maximum absolute difference: 0.032560 for document vectors and
  0.000823 for query vectors), so this is recorded as encoding nondeterminism
  rather than a settings mismatch.

## Hardware and software

GPU/driver/VRAM: NVIDIA GeForce RTX 5070 Ti Laptop GPU, 610.47, 12227; CUDA (PyTorch): 12.8; cuDNN: 90701; Python: 3.11.17; PyTorch: 2.7.1+cu128; Transformers: 4.42.3; Sentence Transformers: 3.0.1

## Anomalies

- No timing outlier was detected after the prescribed warm-up.
- Semantic Top-5 had a one-item boundary swap for 2/10 questions
  (`gpt4_4edbafa2`, `a1cc6108`); all retained 4/5 overlap.

## Per-question medians

| question_id | task | N rounds | doc s/history | doc ms/round | query ms |
| --- | --- | ---: | ---: | ---: | ---: |
| gpt4_4edbafa2 | TR | 231 | 28.423575 | 123.046 | 36.293 |
| 7161e7e2 | IE | 277 | 29.452239 | 106.326 | 38.486 |
| f8c5f88b | IE | 246 | 29.593802 | 120.300 | 33.178 |
| 8c18457d | TR | 218 | 27.821085 | 127.620 | 38.761 |
| 1192316e | MR | 234 | 28.842049 | 123.257 | 37.167 |
| 60036106 | MR | 244 | 28.634715 | 117.355 | 38.731 |
| a1cc6108 | MR | 249 | 29.145941 | 117.052 | 33.860 |
| b6025781 | IE | 231 | 28.439327 | 123.114 | 36.956 |
| a3838d2b | TR | 243 | 28.304801 | 116.481 | 36.893 |
| ceb54acb | IE | 249 | 29.618723 | 118.951 | 39.891 |
