# Measured decision-forward latency

ScienceQA fixed image-test subset, 100 questions per model. NVIDIA A800-SXM4-80GB, BF16, PyTorch 2.8.0, Transformers 4.57.1, shared GPU. Models are loaded before synchronized measurement. Timing covers visual encoding, fusion and the language-model forward from prepared tensors to returned logits. Each input contains one image and one candidate-choice question. P95 uses nearest rank.

| Model | Median forward | P95 forward |
|---|---:|---:|
| Gemma-3n-E2B-it | 213.5 ms | 253.1 ms |
| Gemma-3n-E4B-it | 222.8 ms | 275.4 ms |
| MiniCPM-V-4.5 | 148.9 ms | 256.1 ms |

Values were read from completed per-question reports and rounded to 0.1 ms. This table measures the model decision-forward stage.

[Completed aggregates](../evidence/scienceqa/completed-summaries.json) · [Protocol](scienceqa.md)
