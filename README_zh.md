<div align="center">

# System One
### 全模态 System One 决策模型

**看见环境，读懂任务，给出明确选择。**

基于 Gemma 3n、MiniCPM-V 与 InternVL 的开源多模态决策实现

[English](README.md) · **简体中文**

[![Code License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml/badge.svg)](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml)
[![Adapters](https://img.shields.io/badge/Native_Adapters-5-2563eb)](#模型基座)
[![Paired Probes](https://img.shields.io/badge/Paired_Probes-100%2F100_agree-2563eb)](docs/base-retention-results.md)

[项目介绍](#项目介绍) · [决策通路](#一次前向直接读出决策) · [模型基座](#模型基座) · [基座对比](#基座对比) · [快速开始](#快速开始) · [开源范围](#开源范围与许可)

</div>

---

## 项目介绍

**System One 是一个开源多模态决策模型项目，将视觉语言模型的理解能力转化为可直接调用的结构化决策。** 输入任务文字、图像或视频帧，以及待判断的问题，模型返回候选选择、真假判断或等级评分，并给出对应的概率分布。

我们基于 **Gemma-3n-E4B / E2B、MiniCPM-V-4.5 和 InternVL3.5-8B / 14B** 完成五个模型的决策适配。改造保留基座的原生多模态编码、视觉融合和语言头，从候选标签 logits 直接读出决策。一次选择、判断或评分使用一次语言模型前向，生成 **0 个新文本 token**。

项目提供统一的观测协议、决策 API 和网页决策台，让五个模型使用相同的输入输出接口。运行代码、模型适配、评测脚本和配对验证记录一并开源，支持在统一页面切换模型、查看候选分布并复现基座对照结果。

## 一次前向，直接读出决策

本项目用 **System One** 表示从当前观测直接到结构化选择的通路。对于一次候选决策，模型完成一次语言模型前向，读取最后位置的候选标签 logits，然后计算候选集合内的概率。

**生产决策路径生成 0 个新文本 token，直接返回结构化候选分布。**

| 决策类型 | 解决的问题 | 返回结果 |
|---|---|---|
| `choice` | 从 2–26 个已定义选项中选择 | 稳定选项 ID 与候选概率 |
| `noul` | 判断一个陈述是否成立 | true/false 与对应概率 |
| `score` | 在有序等级中评分 | 等级分布与期望分数 |

候选概率表示当前选项之间的相对偏好。

## 从同一套协议，走向五个基座

模型之间的差异，首先出现在视觉信息如何进入语言模型。Gemma、MiniCPM 与 InternVL 各有自己的预处理、视觉编码和融合实现。我们的工作是为它们逐个建立原生适配，保留这些通路，再用同一套协议连接到决策读出和网页。

| 我们实现的部分 | 具体内容 |
|---|---|
| **原生模型适配** | Gemma processor / forward；MiniCPM 图像切片、视觉重采样和 embedding 融合；InternVL 动态切片与视觉 token 融合 |
| **统一观测协议** | 按相机和时间戳组织 RGB 帧，验证输入字段与数值 |
| **统一决策读出** | 候选标签使用单 token，直接读取基座已有语言头的 logits |
| **统一决策台与 API** | 同一页面切换五个模型，可接入已有 Qwen 服务，按需加载并管理单模型驻留 |
| **可复现验证** | 固定输入、原生对照、协议测试、实际 GPU 记录和失败样本一并公开 |

## 模型基座

| model_id | 官方基座 |
|---|---|
| `gemma-e4b` | [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) |
| `gemma-e2b` | [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) |
| `minicpm-v45` | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) |
| `internvl35-8b` | [OpenGVLab/InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) |
| `internvl35-14b` | [OpenGVLab/InternVL3_5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) |

可通过 `QWEN_DECISION_URL` 接入已有的兼容 Qwen 服务。

本版本的决策适配直接使用官方基座权重与原生语言头。

## 基座对比

我们通过配对测试核查视觉融合与决策读出是否保持基座的原生计算。

为回答这些问题，我们让每个基座完成同一组 **20 个固定探针**：文字计算与逻辑、颜色、计数、OCR、位置关系、双图判断和时序变化。每个输入分别经过官方原生生成第一步和本项目决策前向，控制权重、已预处理输入、提示、候选集、精度和注意力实现。

| 模型 | 原生对照正确数 | 决策改造正确数 | 决策一致率 | 完整词表 logits 最大差 |
|---|---:|---:|---:|---:|
| Gemma-3n-E2B-it | 16/20 | 16/20 | 100% | 0 |
| Gemma-3n-E4B-it | 16/20 | 16/20 | 100% | 0 |
| MiniCPM-V-4.5 | 20/20 | 20/20 | 100% | 0 |
| InternVL3.5-8B | 19/20 | 19/20 | 100% | 0 |
| InternVL3.5-14B | 20/20 | 20/20 | 100% | 0 |

**100 组配对中，候选决策全部一致；完整词表 logits 和候选概率最大差均为 0。** 运行期间参数对象和版本计数未变化。基座答错的样本也保留在报告中，正确数由同一批固定输入计算。

**在已测输入与配置下，决策改造保持了基座的原生第一步计算及候选决策行为。** 评测口径为同一模型、同一配置下的候选决策对照，输入与计算方法见下方文档。

[查看对照方法](docs/base-retention.md) · [查看完整结果](docs/base-retention-results.md) · [查看固定输入](benchmarks/base_retention_v1.json) · [查看原始记录](evidence/retention/)

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

CPU 协议测试 **11 项通过**。最后一条命令核查并汇总仓库中的五份完整配对记录。单模型接口验证另见 `scripts/validate_model.py`；已有 Gemma 及新增三个模型的结果见 [验证记录](docs/validation.md)。

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
