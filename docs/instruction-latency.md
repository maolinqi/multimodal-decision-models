# Natural generation vs. training-free decisions

| Backbone | Dataset | Questions | Repetitions | Median output tokens | Native median (ms) | Decision median (ms) | Speedup | Paired 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma-3n-E4B-it | ScienceQA | 100 | 3 | 62.5 | 4880.7 | 152.3 | **32.06×** | 21.48–41.43 |

| Backbone | Decision correct | Native parsed correct | Unparsed native answers | Truncated native calls | Identical first-step controls |
|---|---:|---:|---:|---:|---:|
| Gemma-3n-E4B-it | 228/300 | 162/300 | 117/300 | 0/300 | 100/100 |

Both paths receive the same image, question, options and prompt. Native answer extraction uses a fixed formatting parser. Timings retain all calls, including short and capped answers.

[Per-question measurements](../evidence/instruction-latency/) · [Fixed ScienceQA sample](../benchmarks/scienceqa-test-100-manifest.json)
