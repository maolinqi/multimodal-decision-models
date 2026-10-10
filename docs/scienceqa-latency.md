# Measured decision-forward latency

ScienceQA fixed image-test subset, 100 questions per model. NVIDIA A800-SXM4-80GB, BF16, PyTorch 2.8.0, Transformers 4.57.1 for the original five adapters and 5.19.0 for Qwen3.5 and Gemma 4, shared GPU. Models are loaded before measurement. The synchronized timer covers visual encoding, fusion, and the language-model forward from prepared tensors to returned logits. Each request contains one image and one choice question. Median is the sample median; P95 uses the nearest-rank method.

| Model | Questions | Median forward (ms) | P95 forward (ms) |
|---|---:|---:|---:|
| Gemma-4-26B-A4B-it | 100 | 264.3 | 333.0 |
| Qwen3.5-2B-Base | 100 | 90.4 | 121.4 |
| Gemma-3n-E2B-it | 100 | 213.5 | 253.1 |
| Gemma-3n-E4B-it | 100 | 222.8 | 275.4 |
| MiniCPM-V-4.5 | 100 | 148.9 | 256.1 |
| InternVL3.5-8B | 100 | 291.1 | 384.7 |
| InternVL3.5-14B | 100 | 263.8 | 440.3 |

[Raw per-question timings](../evidence/scienceqa/) · [Protocol](scienceqa.md)
