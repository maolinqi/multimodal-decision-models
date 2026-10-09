# ScienceQA 基座配对评测方法

[English](scienceqa.md)

评测采用 **ScienceQA 官方 test 划分中的 100 道带图像题目**。先对题目 ID 按数值排序，再用固定 seed 42 抽样；五个基座使用完全相同的题目。准确率为这组固定子集上的候选选择正确率。

原生对照调用官方生成实现的第一步，读取原始候选标签 logits；决策适配直接读取最后位置的 logits。每组配对共享权重、已预处理输入、提示、选项、精度和注意力实现，使用相同候选 softmax 与 argmax。该口径验证项目预处理配置下的原生候选决策保持情况。

模型输入包含题目、已有提示、图像与选项。答案、lecture 和 solution 用于核验与评分。公开清单保存题目 ID、标签、来源版本及输入哈希；原始题目与图像保存在本地 `data/` 目录。

## 复现

```bash
.venv/bin/pip install -e '.[benchmark]'
.venv/bin/python scripts/prepare_scienceqa.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compare_base_retention.py gemma-e2b \
  --suite data/scienceqa/suite.json --out evidence/scienceqa/gemma-e2b.json
```

分别对五个 model_id 运行对照。`scripts/summarize_scienceqa.py` 按固定清单核查完整报告，从逐题预测重算正确率并生成表格。报告保存错误预测、决策一致性、完整词表 logits 差异与参数版本检查。

## 数据来源与许可

- [ScienceQA 作者仓库](https://github.com/lupantech/ScienceQA)，固定版本 `2cbf8318e07b9ece895bb2ae605e71e38d623264`。
- [作者链接的 Hugging Face 数据集](https://huggingface.co/datasets/derek-thomas/ScienceQA)，固定版本 `f18b0a70359ebfb41f658fd564208d0355b013f4`。
- 按作者说明，数据集采用 **CC-BY-NC-SA-4.0**；与项目自有代码的 Apache-2.0 许可分别适用。

Lu 等（2022），*Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*，NeurIPS。

## 延迟计时

每条记录保存 GPU 同步后的 `decision_forward_ms`，范围为已预处理张量输入到 logits 返回，包含视觉编码、融合和语言模型前向。模型加载后计时，按每模型全部 100 题统计中位数与最近秩 P95。硬件为 NVIDIA A800-SXM4-80GB、BF16，共享 GPU 实测。
