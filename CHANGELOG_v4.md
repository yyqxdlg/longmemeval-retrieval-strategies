# v4 Analysis Update

This update keeps the v3 retrieval / time-handling fixes and expands `analyze_results.py`.

## Added

- exact McNemar + Holm for paired `recall_any_at_k`
- automatic exact McNemar for `answer_correct` when that column is later added
- overall and task-level paired statistical tests
- bootstrap 95% CIs in summary CSVs and figures
- individual-question points on small-sample figures
- explicit Okabe-Ito colorblind-safe strategy palette plus distinct markers/hatches
- Recall-vs-latency RQ3 trade-off plot
- Recall-vs-token trade-off plot when token counts exist
- Accuracy-vs-latency/token plots when `answer_correct` exists
- automatic separate analysis for every k value when `--k` is omitted
- cross-k `topk_comparison_table.csv`
- report-friendly `top5_vs_top10_table.csv` when 5 and 10 are present
- paired k-vs-k Wilcoxon / exact McNemar tests
- support for multiple input CSV files with `--input file1.csv file2.csv`
- completeness check for all `(question_id, k)` strategy triplets

## Interpretation

The 20-question pilot is still a small exploratory dataset. In particular, task-level tests have about five paired questions per task and should be treated as analysis-pipeline checks rather than strong inferential evidence.
