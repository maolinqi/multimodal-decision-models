# Validation evidence

2026-10-09–10，Linux / NVIDIA A800 / BF16。CPU 协议与安全边界测试：11 passed。图像探针为简单合成圆形，不是现实任务 benchmark。

| Model | Evidence | Position probes | Official first-step candidate logits |
|---|---|---|---|
| Gemma 4 26B A4B | isolated 5.19.0 runtime, real GPU/API | 2/2 | max absolute difference 0 |
| Qwen3.5-2B-Base | isolated 5.19.0 runtime, real GPU/API | 2/2 | max absolute difference 0 |
| Gemma 3n E2B | earlier deployed adapter | 1/2 | max absolute difference 0 |
| Gemma 3n E4B | earlier deployed adapter | 2/2 (left probe tied at 0.5/0.5) | max absolute difference 0 |
| MiniCPM-V-4.5 | current package, real GPU | 2/2 | max absolute difference 0 |
| InternVL3.5-8B | current package, real GPU | 2/2 | max absolute difference 0 |
| InternVL3.5-14B | current package, real GPU | 2/2 | max absolute difference 0 |

See JSON records in [`evidence/`](../evidence/). Gemma records retain their earlier test inputs and `evidence_origin`; their probe images differ from the new script, so do not treat these rows as a matched-model benchmark. E4B chose `six` for the earlier `2+3` probe; E2B chose `five`.

Interface verification covers positive visual-token counts, normalized candidate probabilities, `choice`, `noul`, `score`, four-axis proposals, and native first-step parity. It does not establish calibrated probabilities, trained policies, navigation improvement, low-latency closed-loop control, real-image generalization or physical flight success.

## 扩展基座行为对照

六个模型各 20 项固定文字与合成图像探针的原生第一步配对结果已公开，见 [基座行为保持对比](base-retention-results.md)。它验证相同已预处理输入下的候选决策与完整词表 logits 一致性。

Qwen3.5 的独立 API 完成健康检查、默认模型选择、两次图像请求与卸载；新旧运行环境的 11 项 CPU 协议测试均通过。运行与复现方法见 [Qwen3.5 独立环境](qwen35-runtime.md)。
