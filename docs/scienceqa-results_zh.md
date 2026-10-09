# ScienceQA 基座配对评测结果

ScienceQA 官方 test 图像子集固定 100 题，seed 42。四个已完成模型评测使用同一组题目，累计 400 组原生路径与决策路径配对。每组共享权重、已预处理输入、提示、候选集、精度与注意力实现。

| 模型 | 额外训练 | 原生候选正确率 | 决策适配正确率 | 决策一致率 |
|---|---:|---:|---:|---:|
| Gemma-3n-E2B-it | 0 步 | 80% | 80% | 100% |
| Gemma-3n-E4B-it | 0 步 | 84% | 84% | 100% |
| MiniCPM-V-4.5 | 0 步 | 98% | 98% | 100% |
| InternVL3.5-8B | 0 步 | 93% | 93% | 100% |

400/400 候选决策一致；完整词表 logits 和候选概率最大差均为 0，参数对象与版本计数保持不变。正确率通过对原生生成第一步与决策前向的候选 logits 使用相同 softmax 和 argmax 计算。

本页统计来源为评测程序已完成执行的 SUMMARY 输出，公开记录按来源标记为汇总指标。

[评测方法](scienceqa_zh.md) · [固定样本清单](../benchmarks/scienceqa-test-100-manifest.json) · [完成汇总](../evidence/scienceqa/completed-summaries.json)
