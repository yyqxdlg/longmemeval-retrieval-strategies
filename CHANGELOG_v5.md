# v5 End-to-End Pipeline Update

## Added

- deterministic Llama 3.1 8B Instruct answer generation from retrieved memories;
- separate generation outputs so checked retrieval artifacts remain immutable;
- complete prompt, completion, retrieval-context, generation-latency, and total-latency measurements;
- context hashing, resolved model revision, quantization, and runtime provenance;
- official LongMemEval hypothesis export and judge-label merge;
- end-to-end accuracy/cost analysis support;
- UTF-8 runtime capture and dependency separation;
- unit tests for prompt isolation, memory reconstruction, resume metadata, and evaluation I/O.

## Changed

- Top-10 is documented as a separate raw result file;
- resume mode validates fixed settings and preserves the union of k values;
- generation-tokenizer remote code is disabled unless explicitly requested;
- answer-accuracy trade-offs use total latency and complete prompt tokens.

## Preserved

- the checked Top-5 retrieval CSV and its primary retrieval conclusions;
- the preregistered fixed-weight hybrid formula;
- the distinction between exploratory retrieval success and final answer correctness.
