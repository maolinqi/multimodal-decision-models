# Completed ScienceQA paired evaluation

Five models each evaluated the same 100 fixed image-bearing ScienceQA test questions. The model JSON files contain all 500 per-question native and decision predictions, candidate distributions, correctness, agreement, logit differences, token counts, and synchronized forward timings.

`summary.json` is recalculated from these records by `scripts/summarize_scienceqa.py`. The script checks completed status, source revisions, the fixed suite hash, exact question IDs and gold labels, visual tokens, and unchanged parameter versions before calculating accuracy and latency.

[English results](../../docs/scienceqa-results.md) · [中文结果](../../docs/scienceqa-results_zh.md) · [Protocol](../../docs/scienceqa.md) · [Fixed manifest](../../benchmarks/scienceqa-test-100-manifest.json)
