# Completed ScienceQA evaluation aggregates

`completed-summaries.json` stores metrics captured from completed evaluation outputs:

- Four models, each evaluated on the same 100 image-bearing test questions.
- Native and decision accuracy, agreement, logit differences, and parameter-version checks read from the evaluator's completed `SUMMARY` output.
- Three models' median, P95, and mean forward timings read from completed per-question reports, rounded to 0.1 ms.

The `evidence_kind`, `accuracy_source`, and `latency_source` fields identify this record as captured aggregate metrics. Dataset revisions, the fixed manifest hash, hardware, precision, and timing scope are included.

[English protocol](../../docs/scienceqa.md) · [中文方法](../../docs/scienceqa_zh.md) · [Fixed manifest](../../benchmarks/scienceqa-test-100-manifest.json)
