# ScienceQA paired evaluation

100 fixed image-bearing questions from the official test split, seed 42. Each backbone uses the same subset and paired prepared inputs. Accuracy is scored from candidate-label logits on the official generation first step and the decision forward path using the same softmax and argmax.

| Model | Native candidate accuracy | Decision accuracy | Difference (pp) | Agreement | Max logit difference |
|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 89.0% | 89.0% | +0.0 | 100.0% | 0 |
| Qwen3.5-2B-Base | 82.0% | 82.0% | +0.0 | 100.0% | 0 |
| Gemma-3n-E2B-it | 80.0% | 80.0% | +0.0 | 100.0% | 0 |
| Gemma-3n-E4B-it | 84.0% | 84.0% | +0.0 | 100.0% | 0 |
| MiniCPM-V-4.5 | 98.0% | 98.0% | +0.0 | 100.0% | 0 |
| InternVL3.5-8B | 93.0% | 93.0% | +0.0 | 100.0% | 0 |
| InternVL3.5-14B | 92.0% | 92.0% | +0.0 | 100.0% | 0 |

[Method](scienceqa.md) · [Manifest](../benchmarks/scienceqa-test-100-manifest.json) · [Raw records](../evidence/scienceqa/)
