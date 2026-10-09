# Validation evidence

2026-10-09，Linux / NVIDIA A800 / BF16。CPU 协议与安全边界测试：11 passed。图像探针为简单合成圆形，不是现实任务 benchmark。

| Model | Evidence | Position probes | Official first-step candidate logits |
|---|---|---|---|
| Gemma 3n E2B | earlier deployed adapter | 1/2 | max absolute difference 0 |
| Gemma 3n E4B | earlier deployed adapter | 2/2 (left probe tied at 0.5/0.5) | max absolute difference 0 |
| MiniCPM-V-4.5 | current package, real GPU | 2/2 | max absolute difference 0 |
| InternVL3.5-8B | current package, real GPU | 2/2 | max absolute difference 0 |
| InternVL3.5-14B | current package, real GPU | 2/2 | max absolute difference 0 |

See JSON records in [`evidence/`](../evidence/). Gemma records retain their earlier test inputs and `evidence_origin`; their probe images differ from the new script, so do not treat these rows as a matched-model benchmark. E4B chose `six` for the earlier `2+3` probe; E2B chose `five`.

Interface verification covers positive visual-token counts, normalized candidate probabilities, `choice`, `noul`, `score`, four-axis proposals, and native first-step parity. It does not establish calibrated probabilities, trained policies, navigation improvement, low-latency closed-loop control, real-image generalization or physical flight success.
