# Limitations of the Local Confirmatory Run

- Official answer correctness is unavailable until the exported hypotheses are
  evaluated with the benchmark's official judge. Blank `answer_correct` values
  are unknown, not incorrect.
- Any normalized exact-match or answer-containment figures are exploratory
  string heuristics. They are not substitutes for semantic judging and must not
  be presented as official accuracy.
- Local answer generation was not run on the retrieval host because Meta
  rejected that Hugging Face account's gated-repository access request. The
  checked generation pipeline is configured for NF4 4-bit on a 12 GB laptop
  GPU, but no Llama weights or generated answers are claimed in this run.
- The generation results must be produced on an authorized host from one
  immutable official-model revision. Quantized NF4 results may differ from
  unquantized Llama 3.1 8B inference.
- Windows WDDM, background display workloads, and laptop power/thermal behavior
  can affect latency. Latency is valid for this recorded machine and runtime,
  not a universal hardware benchmark.
- The benchmark-specific future-session policy retains same-calendar-day
  sessions even when their HH:MM timestamp is later than the question time.
- Stella 512 is the main condition. Stella 1024 is a retrieval-only sensitivity
  analysis and should not be pooled with the primary condition.
- The confirmatory sample is balanced by task group rather than sampled in the
  benchmark's natural task proportions.
- Oracle context is an upper-reference generation condition, not a deployable
  retrieval strategy. It relies on benchmark evidence annotations to select raw
  dialogue text, while keeping those annotations themselves out of the prompt.
- No GPT-4o judge, official answer accuracy, answer-correctness McNemar test, or
  `answer_correct × recall_any` cross-tab is reported in the no-paid-API stage.
