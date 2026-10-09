# 决策前向延迟实测

ScienceQA 固定图像测试子集，每模型 100 题。NVIDIA A800-SXM4-80GB、BF16、PyTorch 2.8.0、Transformers 4.57.1，共享 GPU 实测。模型加载后使用 GPU 同步计时，范围为已预处理张量输入到 logits 返回，包含视觉编码、融合与语言模型前向。每次输入一张图像与一个候选选择问题，P95 使用最近秩法。

| 模型 | 前向中位数 | 前向 P95 |
|---|---:|---:|
| Gemma-3n-E2B-it | 213.5 ms | 253.1 ms |
| Gemma-3n-E4B-it | 222.8 ms | 275.4 ms |
| MiniCPM-V-4.5 | 148.9 ms | 256.1 ms |

数据来自已完成逐题报告的统计读取，保留一位小数；本表衡量模型决策前向阶段。

[完成汇总](../evidence/scienceqa/completed-summaries.json) · [评测方法](scienceqa_zh.md)
