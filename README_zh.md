<div align="center">

# System One
### 全模态 System One 决策模型

**无需额外训练，让多模态基座直接做决策。**

基于 Gemma 3n、MiniCPM-V 与 InternVL 的开源多模态决策实现

**免额外训练实测：ScienceQA 固定 100 道图像题，MiniCPM-V-4.5 候选正确率 98%，决策前向中位数 148.9 ms。**

[English](README.md) · **简体中文**

[![Code License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml/badge.svg)](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml)
[![Training](https://img.shields.io/badge/Additional_Training-0_steps-2563eb)](#免额外训练的决策改造)
[![Adapters](https://img.shields.io/badge/Native_Adapters-6-2563eb)](#模型基座)
[![Paired Decisions](https://img.shields.io/badge/ScienceQA_Pairs-600%2F600_agree-2563eb)](docs/scienceqa-results_zh.md)

[项目介绍](#项目介绍) · [免训练改造](#免额外训练的决策改造) · [输入输出](#多模态输入结构化输出) · [模型基座](#模型基座) · [免训练准确率](#免额外训练的实测准确率) · [实测延迟](#低延迟决策具体用了多久) · [快速开始](#快速开始) · [开源范围](#开源范围与许可)

</div>

---

## 项目介绍

**System One 是一个面向多模态理解与低延迟决策的开源模型项目，将视觉语言模型的理解能力转化为可直接调用的结构化决策。** 输入任务文字、图像或视频帧，以及待判断的问题，模型返回候选选择、真假判断或等级评分，并给出对应的概率分布。文字描述任务和候选，图像提供对象、位置与空间关系，按时间排列的视频帧提供前后变化；这些信息共同进入模型的原生多模态通路。

我们基于 **Qwen3.5-2B-Base、Gemma-3n-E4B / E2B、MiniCPM-V-4.5 和 InternVL3.5-8B / 14B** 完成六个模型的决策适配，**直接使用已有预训练权重，无需额外训练或微调即可运行**。改造保留基座的原生多模态编码、视觉融合和语言头，从候选标签 logits 直接读出决策。一次选择、判断或评分使用一次语言模型前向，生成 **0 个新文本 token**，直接返回决策分布，减少逐 token 解码环节带来的等待。

项目提供统一的观测协议、决策 API 和网页决策台，让六个模型使用相同的输入输出接口。运行代码、模型适配、评测脚本和配对验证记录一并开源，支持在统一页面切换模型、查看候选分布并复现基座对照结果。

## 免额外训练的决策改造

我们的改造把基座已有的多模态理解能力接到统一决策接口：保留原生视觉编码与融合，将选项映射到单 token 标签，再通过已有语言头读取候选 logits，归一化为候选分布。

| 改造特点 | 本版本实现 |
|---|---|
| 额外训练步骤 | **0** |
| 新增模型参数 | **0**，复用基座原生语言头 |
| 基座权重更新 | **0**，直接加载官方权重 |
| 单个选择、判断或评分 | **1 次语言模型前向** |
| 决策路径生成文本 | **0 个新 token** |
| 输出 | 稳定选项 ID、候选概率与输入记录 |

候选分布由基座当前输入下的标签 logits 计算：

$$p_i = \frac{\exp(z_{t_i})}{\sum_{j=1}^{K}\exp(z_{t_j})}$$

其中 $z_{t_i}$ 为第 $i$ 个候选标签的 logit，$K$ 为候选数量。

这种方式将图像和视频理解转化为程序可调用的选择、判断与评分；六个模型通过同一套协议接入同一个决策台。

## 多模态输入，结构化输出

- **图文联合判断：** 图像提供视觉内容，文字给出任务、上下文与候选选项，共同参与决策。
- **多帧时序信息：** 支持多张 RGB 图像与最多 8 帧带时间戳的视频抽帧，保留相机标识与时间顺序。
- **程序可调用的结果：** 将判断读出为稳定 ID 与候选分布，统一用于选择、真假判断和有序评分。

| 决策类型 | 解决的问题 | 返回结果 |
|---|---|---|
| `choice` | 从 2–26 个已定义选项中选择 | 稳定选项 ID 与候选概率 |
| `noul` | 判断一个陈述是否成立 | true/false 与对应概率 |
| `score` | 在有序等级中评分 | 等级分布与期望分数 |

候选概率表示当前选项之间的相对偏好。

## 从同一套协议，走向六个基座

模型之间的差异，首先出现在视觉信息如何进入语言模型。Gemma、MiniCPM 与 InternVL 各有自己的预处理、视觉编码和融合实现。我们的工作是为它们逐个建立原生适配，保留这些通路，再用同一套协议连接到决策读出和网页。

| 我们实现的部分 | 具体内容 |
|---|---|
| **原生模型适配** | Gemma processor / forward；MiniCPM 图像切片、视觉重采样和 embedding 融合；InternVL 动态切片与视觉 token 融合 |
| **统一观测协议** | 按相机和时间戳组织 RGB 帧，验证输入字段与数值 |
| **统一决策读出** | 候选标签使用单 token，直接读取基座已有语言头的 logits |
| **统一决策台与 API** | 同一页面切换六个模型，可接入已有 Qwen 服务，按需加载并管理单模型驻留 |
| **可复现验证** | 固定输入、原生对照、协议测试、实际 GPU 记录和失败样本一并公开 |

## 模型基座

| model_id | 官方基座 |
|---|---|
| `qwen35-2b` | [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) |
| `gemma-e4b` | [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) |
| `gemma-e2b` | [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) |
| `minicpm-v45` | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) |
| `internvl35-8b` | [OpenGVLab/InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) |
| `internvl35-14b` | [OpenGVLab/InternVL3_5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) |

Qwen3.5 使用[项目内独立运行环境](docs/qwen35-runtime.md)，保留原五个适配器的环境。`QWEN_DECISION_URL` 仍可接入已有的 Qwen3-VL-4B 服务。

本版本的决策适配直接使用官方基座权重与原生语言头。

## 免额外训练的实测准确率

**MiniCPM-V-4.5 在 ScienceQA 固定图像测试子集上的候选选择正确率为 98%，额外训练 0 步。** Gemma E2B / E4B 与 InternVL 8B 分别达到 80%、84% 和 93%；InternVL 14B 为 92%；Qwen3.5-2B-Base 为 82%。

测试采用 **ScienceQA 官方 test 划分中的 100 道带图像题目，seed 42**，所有表中模型使用同一份固定清单。模型输入包含题目、已有提示、图像与选项；原生对照和决策适配使用相同权重、已预处理输入、提示、候选集、精度及注意力实现。原生对照读取官方生成第一步的候选 logits，采用相同候选 softmax 与 argmax 评分。

| 模型 | 额外训练 | 原生候选正确率 | 决策适配正确率 | 决策一致率 |
|---|---:|---:|---:|---:|
| Qwen3.5-2B-Base | 0 步 | 82% | 82% | 100% |
| Gemma-3n-E2B-it | 0 步 | 80% | 80% | 100% |
| Gemma-3n-E4B-it | 0 步 | 84% | 84% | 100% |
| MiniCPM-V-4.5 | 0 步 | 98% | 98% | 100% |
| InternVL3.5-8B | 0 步 | 93% | 93% | 100% |
| InternVL3.5-14B | 0 步 | 92% | 92% | 100% |

**600 组配对全部一致，完整词表 logits 和候选概率最大差均为 0。** 这验证了已测输入与配置下，免额外训练的决策适配保持了基座的原生候选决策行为。

[测试集与方法](docs/scienceqa_zh.md) · [固定样本清单](benchmarks/scienceqa-test-100-manifest.json) · [完整结果](docs/scienceqa-results_zh.md) · [逐题记录](evidence/scienceqa/)

## 低延迟决策：具体用了多久？

单个选择、判断或评分直接读出一次前向的候选 logits，生成 0 个新文本 token。输入经过多模态编码与融合后即可返回结构化分布，减少逐 token 解码与自由文本解析环节。

以下为同一 ScienceQA 固定 100 题图像子集的实测，每次输入一张图像与一个选择问题。**模型已加载，计时范围为预处理完成后的视觉编码、融合与语言模型前向，直到 logits 返回。** 使用 NVIDIA A800-SXM4-80GB、BF16、GPU 同步计时，共享 GPU；统计全部 100 题的中位数与 P95。

| 模型 | 前向中位数 | 前向 P95 |
|---|---:|---:|
| Qwen3.5-2B-Base | 90.4 ms | 121.4 ms |
| Gemma-3n-E2B-it | 213.5 ms | 253.1 ms |
| Gemma-3n-E4B-it | 222.8 ms | 275.4 ms |
| MiniCPM-V-4.5 | 148.9 ms | 256.1 ms |
| InternVL3.5-8B | 291.1 ms | 384.7 ms |
| InternVL3.5-14B | 263.8 ms | 440.3 ms |

原五个适配器使用 Transformers 4.57.1，Qwen3.5 使用独立的 5.19.0 环境；跨模型前向时间仅作描述，不属于改造前后的配对加速比较。

[延迟统计与口径](docs/scienceqa-latency_zh.md) · [完整结果汇总](evidence/scienceqa/summary.json)

此外，六个基座完成了颜色、计数、OCR、空间关系、双图和时序等固定探针的配对验证，见 [原生通路验证](docs/base-retention-results.md)。

## 统一决策台

你可以在统一决策台中上传图像或视频片段，填写问题与候选选项，切换模型，查看和导出决策分布。视频通过带时间戳的 RGB 抽帧输入，最多 8 帧。

在程序中，可以通过统一 API 接入候选选择、真假判断与等级评分。视觉候选选择、事件判断和有序程度评估使用同一协议。

## 快速开始

Linux、Python 3.12、NVIDIA GPU；已测试 PyTorch 2.8.0、Transformers 4.57.1 和 BF16。以下以 MiniCPM-V-4.5 为例：

```bash
git clone https://github.com/maolinqi/multimodal-decision-models.git
cd multimodal-decision-models
./setup.sh
.venv/bin/python scripts/download_models.py minicpm-v45
CUDA_VISIBLE_DEVICES=0 ./start_all.sh
```

浏览器打开 [本地决策台](http://127.0.0.1:8456)，选择已下载的模型。视频需要系统已有 `ffmpeg` 和 `ffprobe`。停止服务使用 `./stop_all.sh`；启动脚本使用已有项目环境。

```bash
curl -H 'Content-Type: application/json' \
  --data-binary @examples/choice.json http://127.0.0.1:8457/decide
```

<details>
<summary>下载其他模型、已有模型目录与显存要求</summary>

Gemma 模型需要在 Hugging Face 接受上游条款并登录：

```bash
.venv/bin/hf auth login
.venv/bin/python scripts/download_models.py gemma-e2b gemma-e4b internvl35-8b internvl35-14b
```

已下载权重可通过 `MODEL_ROOT=/path/to/models` 指向共享模型目录；下载和启动使用同一个变量。单后端按需加载，一个时刻驻留一个模型。显存预留门槛为 E2B 16 GiB、E4B 20 GiB、MiniCPM / InternVL 8B 22 GiB、InternVL 14B 34 GiB；长输入可能需要更多。

总输入上限为 8192 tokens，超限返回错误。当前使用有限切片预算；安装后的网页服务依赖仓库中的 `web/`，建议保留 checkout 和 editable 安装。服务默认绑定本机。

</details>

## 复现验证

```bash
.venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compare_base_retention.py minicpm-v45 \
  --out evidence/retention/minicpm-v45.json
.venv/bin/python scripts/summarize_retention.py
```

CPU 协议测试 **11 项通过**。最后一条命令核查并汇总仓库中的六份完整配对记录。单模型接口验证另见 `scripts/validate_model.py`；已有 Gemma 及新增三个模型的结果见 [验证记录](docs/validation.md)。

ScienceQA 固定图像测试子集的复现命令与数据来源见 [公开测试集评测方法](docs/scienceqa_zh.md)；六份逐题报告可用 `.venv/bin/python scripts/summarize_scienceqa.py` 重新核查并生成准确率与延迟表。

## 开源范围与许可

**运行代码已公开开源。** 你可以下载、运行、检查实现并复现对比：

```text
src/multimodal_decision/  原生适配、协议、输入检查、API 与网页网关
web/                     统一决策台
scripts/                 下载、接口验证、基座配对比较与汇总
benchmarks/              固定输入探针
evidence/                实际验证记录，含错误样本
tests/                   协议与输入检查测试
docs/                    架构、对照方法与验证结果
setup.sh / start_all.sh / stop_all.sh
```

自有接口代码采用 **Apache-2.0**。官方模型权重与运行时加载的自定义代码遵循各自上游条款，详见 [模型许可说明](docs/model-licenses.md)。模型权重通过上游渠道下载。

欢迎使用固定输入报告问题，或提交模型适配与验证改进，见 [贡献说明](CONTRIBUTING.md)。


### 冻结基座准确率对照

仅用官方基座权重，无额外训练。作者训练后的分数作为报告参考，不重跑作者模型。Qwen/Gemma 作者分数对照采用单独记录的输入协议；Qwen3.5 已接入控制台，Gemma 4 的输入适配仍在推进。

|基座|测试内容|本次正确/题数|本次准确率|作者训练模型报告|
|---|---|---:|---:|---:|
|Qwen3.5-2B-Base|RAVEN|178/300|59.33%|Decider 80%|
|Qwen3.5-2B-Base|Visual7W|271/300|90.33%|Decider 89%|
|Gemma-4-26B-A4B-it|128 道公开预览 + 8 张示例卡|90/136|66.18%|Rune v3 75.7%，280 图像令牌|

736 题均完成，无推理异常。作者逐题样本及最终渲染未完全核验，因此不能据此证明非劣效。Visual7W 的数值接近；RAVEN 和 Gemma 仍有较大差距。公开全部适配失败、标记诊断及 197 道错题。[完整结果及限制](docs/frozen-backbone-accuracy.md) · [逐题准确率证据](evidence/frozen-backbone/accuracy/)

**已完成的单字母对照：**两边输入完全相同，并共同要求只回答一个字母。Qwen 600 题、每题每条路径三次，原模型生成中位耗时 118.3 ms，决策评分 84.1 ms，减少 28.9%。原模型实际只生成字母和结束符两个 token，没有触及上限；两边 600 道题的答案全部一致。[协议和原始耗时](docs/frozen-backbone-accuracy.md#completed-minimal-answer-latency-control)。自然生成的完整对照及 Gemma 单字母对照仍在运行/排队，完成后补充。
