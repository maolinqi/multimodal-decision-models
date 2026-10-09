# ScienceQA paired evaluation

100 fixed image-bearing questions from the official test split, seed 42. Four completed model evaluations use the same subset, yielding 400 native/decision pairs. Each pair shares weights, prepared inputs, prompts, candidates, precision, and attention implementation.

| Model | Additional training | Native candidate accuracy | Decision accuracy | Agreement |
|---|---:|---:|---:|---:|
| Gemma-3n-E2B-it | 0 steps | 80% | 80% | 100% |
| Gemma-3n-E4B-it | 0 steps | 84% | 84% | 100% |
| MiniCPM-V-4.5 | 0 steps | 98% | 98% | 100% |
| InternVL3.5-8B | 0 steps | 93% | 93% | 100% |

400/400 candidate decisions agree. Maximum full-vocabulary logit and candidate-probability differences are 0; parameter objects and version counters remain unchanged. Accuracy applies identical candidate softmax and argmax to native generation first-step logits and decision-forward logits.

The statistics are captured from completed evaluator SUMMARY outputs. Published evidence is explicitly marked as aggregate metrics.

[Protocol](scienceqa.md) · [Fixed manifest](../benchmarks/scienceqa-test-100-manifest.json) · [Completed aggregates](../evidence/scienceqa/completed-summaries.json)
