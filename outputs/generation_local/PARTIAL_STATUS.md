# Generation checkpoint status

Updated: 2026-10-09. Primary generation is complete: 2400 unique nonempty answers, with 200 questions across 12 conditions. All four task types have 600 rows each. Answer correctness is not yet scored.

Primary settings: meta-llama/Llama-3.1-8B-Instruct, revision 0e9e39f249a16976918f6564b8830bc894c89659, NF4 4bit, seed 42, max_new_tokens 128. Attention uses cuDNN prefill and math single-token decode.

The earlier 1419-row partial checkpoint was resumed and completed. The primary result has been preserved without replacing answers with longer outputs.

Separate post-hoc length sensitivity analysis: all 236 primary answers with completion_token_count equal to 128 were regenerated with max_new_tokens 256 and otherwise matching generation settings. Results are in ../generation_length256/generation_limit236_256.csv. All 236 are complete as saved rows; 63 again reached the output limit. These are separate sensitivity results, not scored answer accuracy.

Validation and visible-text ending reviews are in ../analysis_generation/length_limit_checks/. See their limitations before interpreting answer quality.