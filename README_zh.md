<div align="center">

# System One
### 全模态 System One 决策模型

**无需额外训练，让多模态基座直接做决策。**

基于 Qwen3.5、Gemma 3n/4、MiniCPM-V 与 InternVL 的开源多模态决策实现

[English](README.md) · **简体中文**

[![Code License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml/badge.svg)](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml)
[![Training](https://img.shields.io/badge/Additional_Training-0_steps-2563eb)](#免额外训练的决策改造)
[![Adapters](https://img.shields.io/badge/Native_Adapters-7-2563eb)](#模型基座)
[![Paired Decisions](https://img.shields.io/badge/ScienceQA_Pairs-700%2F700_agree-2563eb)](docs/scienceqa-results_zh.md)

[项目介绍](#项目介绍) · [免训练改造](#免额外训练的决策改造) · [输入输出](#多模态输入结构化输出) · [模型基座](#模型基座) · [免训练准确率](#免额外训练的实测准确率) · [实测延迟](#低延迟决策具体用了多久) · [快速开始](#快速开始) · [开源范围](#开源范围与许可)

</div>

---

## 项目介绍

**System One 将多模态基座改造成结构化决策接口，无需额外训练或微调。** 输入文字、图像或视频帧及候选选项，直接返回选择、真假判断或等级评分与候选概率。项目支持七个基座，提供统一 API、网页决策台和评测记录。

## 免额外训练的实测准确率

| 基座 | 评测集（n） | 训练模型 | [训练后报告](docs/frozen-backbone-accuracy.md) | 本方法（免训练） | 差值（pp） |
|---|---|---|---:|---:|---:|
| Qwen3.5-2B-Base | Visual7W (300) | Decider-2B-Vision | 89.00% | **90.33%** | +1.33 |
| Gemma-4-26B-A4B-it | Rune 公开重建集 (136) | Rune v3 | 75.70% | **68.38%** | -7.32 |

### ScienceQA（固定 100 题）

| 基座 | 额外训练 | 原生候选准确率 | 本方法准确率 | 选择一致 |
|---|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 0 | 89% | 89% | 100/100 |
| Qwen3.5-2B-Base | 0 | 82% | 82% | 100/100 |
| Gemma-3n-E2B-it | 0 | 80% | 80% | 100/100 |
| Gemma-3n-E4B-it | 0 | 84% | 84% | 100/100 |
| MiniCPM-V-4.5 | 0 | 98% | 98% | 100/100 |
| InternVL3.5-8B | 0 | 93% | 93% | 100/100 |
| InternVL3.5-14B | 0 | 92% | 92% | 100/100 |

[准确率记录](docs/frozen-backbone-accuracy.md) · [ScienceQA 协议与结果](docs/scienceqa_zh.md)

## 低延迟决策：具体用了多久？

| 基座 | 回答方式 | 评测集（n） | 原生 token 中位 | 原生中位（ms） | 本方法决策延迟（ms，中位数） | 加速比 |
|---|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 自然生成 | Rune (136) | 240 | 12512.1[†](docs/identical-input-latency.md#what-was-timed) | **248.2** | **50.40×** |
| Gemma-4-26B-A4B-it | 仅回答字母 | Rune (136) | 2 | 308.1[†](docs/frozen-backbone-accuracy.md#completed-minimal-answer-latency-control) | **244.6** | **1.26×** |
| Gemma-3n-E4B-it | 自然生成 | ScienceQA (100) | 62.5 | 4880.7 | **152.3** | **32.06×** |

[逐题结果](docs/identical-input-latency.md) · [ScienceQA 延迟对照](docs/instruction-latency.md)

<details>
<summary>ScienceQA 决策前向耗时</summary>

| 基座 | 前向中位数（ms） | 前向 P95（ms） |
|---|---:|---:|
| Gemma-4-26B-A4B-it | 264.3 | 333.0 |
| Qwen3.5-2B-Base | 90.4 | 121.4 |
| Gemma-3n-E2B-it | 213.5 | 253.1 |
| Gemma-3n-E4B-it | 222.8 | 275.4 |
| MiniCPM-V-4.5 | 148.9 | 256.1 |
| InternVL3.5-8B | 291.1 | 384.7 |
| InternVL3.5-14B | 263.8 | 440.3 |

[完整结果](docs/scienceqa-latency_zh.md)

</details>

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

## 从同一套协议，走向七个基座

模型之间的差异，首先出现在视觉信息如何进入语言模型。Qwen、Gemma、MiniCPM 与 InternVL 各有自己的预处理、视觉编码和融合实现。我们的工作是为它们逐个建立原生适配，保留这些通路，再用同一套协议连接到决策读出和网页。

| 我们实现的部分 | 具体内容 |
|---|---|
| **原生模型适配** | Qwen 原生图像 token 与稳定 Base 答案边界；Gemma processor / forward；MiniCPM 图像切片、视觉重采样和 embedding 融合；InternVL 动态切片与视觉 token 融合 |
| **统一观测协议** | 按相机和时间戳组织 RGB 帧，验证输入字段与数值 |
| **统一决策读出** | 候选标签使用单 token，直接读取基座已有语言头的 logits |
| **统一决策台与 API** | 同一页面切换七个模型，可接入已有 Qwen 服务，按需加载并管理单模型驻留 |
| **可复现验证** | 固定输入、原生对照、协议测试、实际 GPU 记录和失败样本一并公开 |

## 模型基座

| model_id | 官方基座 |
|---|---|
| `qwen35-2b` | [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) |
| `gemma4-a4b` | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| `gemma-e4b` | [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) |
| `gemma-e2b` | [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) |
| `minicpm-v45` | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) |
| `internvl35-8b` | [OpenGVLab/InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) |
| `internvl35-14b` | [OpenGVLab/InternVL3_5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) |

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

两个新增基座使用各自的项目内环境，官方权重下载固定到已验证版本：

```bash
./scripts/setup_qwen35.sh
./scripts/setup_gemma4.sh
.venv/bin/python scripts/download_models.py qwen35-2b gemma4-a4b
CUDA_VISIBLE_DEVICES=GPU-YOUR_AVAILABLE_GPU_UUID ./start_all.sh
```

Qwen3.5 与 Gemma 4 服务分别使用 8460、8461 端口，加载前要求至少 8、62 GiB 空闲显存；各后端独立管理驻留模型。[Qwen3.5 环境说明](docs/qwen35-runtime.md)与 [Gemma 4 环境说明](docs/gemma4-runtime.md)提供完整的路由、卸载与验证命令。

## 复现验证

```bash
.venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compare_base_retention.py minicpm-v45 \
  --out evidence/retention/minicpm-v45.json
.venv/bin/python scripts/summarize_retention.py
```

CPU 协议测试 **11 项通过**。最后一条命令核查并汇总仓库中的七份完整配对记录。单模型接口验证另见 `scripts/validate_model.py`；已有 Gemma 及新增三个模型的结果见 [验证记录](docs/validation.md)。

ScienceQA 固定图像测试子集的复现命令与数据来源见 [公开测试集评测方法](docs/scienceqa_zh.md)；七份逐题报告可用 `.venv/bin/python scripts/summarize_scienceqa.py` 重新核查并生成准确率与延迟表。

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
