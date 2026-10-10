<div align="center">

# 天择多模态决策大模型框架
### TianZe-MJev: A Multimodal Jev-style Decision Model

**看见现场，读懂任务，直接作出选择。**

[English](README.md) · **简体中文**

[![Code License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/jiangfeibo/TianZe-MJev/actions/workflows/tests.yml/badge.svg)](https://github.com/jiangfeibo/TianZe-MJev/actions/workflows/tests.yml)
[![Backbones](https://img.shields.io/badge/Backbones-7-2563eb)](#安装使用)
[![Training](https://img.shields.io/badge/Additional_Training-0_steps-2563eb)](#优势)

[介绍](#介绍) · [优势](#优势) · [实验结果](#实验结果) · [界面展示](#界面展示) · [安装使用](#安装使用) · [作者信息](#作者信息) · [开源许可](#开源许可)

</div>

## 介绍

### 项目背景：为什么决策一定要先生成文字？

想象一架无人机正在高速飞行，前方突然出现障碍物。此时系统需要立即判断：向左、向右，还是悬停？

对于这样的任务，我们真正需要的不是模型生成一段关于障碍物位置、飞行方向和避障策略的文字分析，而是**根据当前多模态观测，迅速选择最合适的动作**。

当前大语言模型主要采用自回归（Autoregressive）生成机制，通过逐 Token 预测生成后续内容。当模型通过生成文字、分析步骤或推理链来完成决策时，可以将这种处理路径理解为一种受 System 2 启发的审慎推理方式。

这种方法适合复杂推理、开放式问答及需要解释的任务，但对于实时控制、游戏决策、GUI 操作以及其他有限候选决策任务，传统生成式路径并不总是最优选择。

需要说明的是，System 1 / System 2 原本是认知心理学中的处理方式划分，并不与自回归、非自回归模型严格等价。这里主要比较的是**通过连续文本生成完成决策的路径**与**直接读取候选决策分布的路径**。

### 一、现有自回归决策方法的四个主要局限

**1. 串行 Token 生成带来较高决策延迟**

传统自回归生成需要逐步预测下一个 Token，后续生成依赖已经生成的内容。

即使任务最终只需要一个动作标签，模型仍可能生成大量中间文字，造成额外的解码开销。

对于无人机避障、机器人控制和实时交互等任务，这种等待可能限制系统的响应能力。

**2. 自然语言输出与结构化决策需求不匹配**

传统生成式模型通常输出自然语言，而实际决策系统需要的是明确的动作编号、类别标签或控制指令。

将生成文本转换为机器可执行动作，可能需要额外的格式约束、文本解析或异常处理。

此外，自然语言回答可能包含解释、无效格式或不在候选集合内的内容，增加决策接口的复杂度。

**3. 难以直接获得完整的候选决策分布**

普通文本生成接口通常返回一个生成结果，而不是所有候选动作的完整概率分布。

例如，当无人机需要在“左转、右转、悬停”之间选择时，系统除了最终动作，还可能希望知道不同候选动作之间的相对偏好。

仅依赖最终生成文本，不便于直接获得这一信息。若需要候选分布，通常还需要额外的 logits 读取或评分机制。

**4. 通用生成路径与重复性决策任务之间存在效率不匹配**

在大量重复的有限选项任务中，系统往往需要连续作出简单、明确的选择，而不需要每次都重新生成完整回答。

如果进一步构建专用决策模型，还可能引入额外的监督微调、强化学习、数据收集和模型维护成本。

因此，值得研究一个关键问题：**能否不重新训练模型，直接将已有多模态基座的理解能力转化为低延迟的结构化决策能力？**

### 二、TianZe-MJev：面向多模态任务的 System 1 式直接决策框架

针对上述问题，我们提出 **TianZe-MJev（天择多模态决策大模型框架）**。

该框架受到 System 1 快速决策机制的启发，在保留已有多模态模型感知与理解能力的基础上，将传统的连续文本生成输出转换为直接候选决策。

模型接收任务描述、环境状态、候选选项以及图像或视频帧，通过基座模型原生的多模态编码和融合机制理解当前观测，再从候选标签对应的 logits 直接计算决策分布。

框架不需要生成完整自然语言回答，即可输出选项 ID、概率分布及相应的结构化结果。

### 三、我们的方法具有四个核心优势

**1. 直接决策：显著减少自回归解码延迟**

针对传统生成式方法的串行解码问题，我们使用单次语言模型前向计算读取候选标签 logits，无需逐 Token 生成最终回答。

- 单次选择、判断或评分只需要一次语言模型前向。
- 决策路径生成 0 个新文本 Token。
- 保留必要的输入编码和多模态特征计算，省去后续连续文本解码。

在现有自然生成对照实验中，Gemma 4 的响应延迟从 **12,512.1 ms 降低至 248.2 ms**，实现约 **50.4 倍加速，延迟降低 98.0%**。

**2. 结构化输出：让模型决策直接对接应用程序**

针对自然语言生成结果难以直接执行的问题，我们设计统一的结构化决策接口。

目前支持三类决策任务：

- `choice`：从 2–26 个候选选项中直接选择。
- `noul`：对给定陈述进行二元真假判断。
- `score`：输出有序等级分布及期望评分。

系统直接返回选项 ID 与对应分布，减少对自由文本解析的依赖，方便接入机器人、游戏智能体与其他自动化系统。

**3. 免额外训练：直接复用已有多模态大模型能力**

针对构建专用决策模型可能产生的训练成本，我们采用冻结基座的决策适配方式。

- 额外决策训练步骤：**0**
- 新增模型参数：**0**
- 基座模型权重更新：**0**

框架目前支持 Qwen3.5、Gemma 3n / 4、MiniCPM-V 和 InternVL 等系列共七个官方基座模型。

这使得已有模型能够在不额外进行决策微调的情况下接入统一决策框架，降低实验和部署门槛。

**4. 多模态融合与候选分布：兼顾环境理解与决策可观察性**

框架保留基座模型原生的视觉编码与多模态融合能力，支持结合任务文字、状态信息、RGB 图像及带时间顺序的视频抽帧进行判断。

与只返回单一文本答案的调用方式不同，TianZe-MJev 可以输出全部候选选项的相对概率分布，并支持通过网页查看、比较和导出决策结果。

这为后续开展决策置信度分析、时序状态理解和智能体控制研究提供了统一接口。

需要注意，当前候选概率表示模型在给定选项之间的相对偏好，尚不能直接等同于经过校准的决策正确率或安全置信度。

### 四、实验验证：我们的改进有多大？

我们在 NVIDIA A800-SXM4-80GB GPU 上进行了自然文本生成与直接决策的延迟对照实验。两种方法使用相同官方基座和相同任务输入，对比正常自回归生成回答与直接读取候选决策分布的耗时。

**实验表明，在测试的有限候选任务中，跳过连续文本生成可以显著缩短决策响应时间。**

在准确率方面，冻结的 Qwen3.5-2B-Base 在 Visual7W 的 300 道测试题上取得 **90.33%** 的准确率，与 Decider-2B-Vision 作者报告的 89.00% 接近，数值高出 1.33 个百分点。然而，RAVEN 与 Rune 重建集的结果仍低于相应训练模型报告值，说明免训练适配的效果存在任务差异，不能据此断言整体优于专门训练的决策模型。

此外，较大的延迟收益可能伴随答案质量差异。例如在 Gemma 4 的中性提示对照中，首轮直接决策答对 30/136 题，自然生成答案按保守解析规则答对 95/136 题；而在双方都要求单字母回答的控制实验中，延迟下降为 20.6%。

因此，我们的核心贡献并不是证明直接决策在所有任务上都比自回归生成更准确，而是：

**在无需额外训练、不修改基座权重的条件下，为现有多模态模型提供一条可复现、可比较的低延迟结构化决策路径，并明确分析速度与决策质量之间的权衡。**

### 项目愿景

我们希望让多模态大模型不仅能够理解图像、描述场景和回答问题，还能够在具体任务中快速作出可执行的选择。

**从“理解环境并生成答案”，走向“理解环境并直接决策”。**

## 优势

- **免额外训练。** 直接使用官方基座权重，决策适配额外训练步骤、权重更新与新增模型参数均为 0。
- **缩短决策输出路径。** 单个选择、判断或评分使用一次语言模型前向，决策路径生成 0 个新文本 token；实测耗时与质量见下方实验。
- **七个基座，同一接口。** Qwen3.5、Gemma 3n / 4、MiniCPM-V 和 InternVL 接入统一 API 与网页，保留各自原生视觉处理。
- **输入与结果可追溯。** 网页展示候选分布并导出输入记录与结果；仓库保留评测方法、逐题输出和错误样本。

## 实验结果

### 实验一：免训练决策的准确率

**实验目标。** 衡量官方冻结基座经决策适配后的任务准确率，并与公开训练模型的报告值建立参考。

**比较的方法。** 本方法加载官方权重、不加决策训练；参考方法为 Decider-2B-Vision 和 Rune v3 的作者报告。本轮未运行这两个训练模型，样本版本与最终图像呈现未完全配对，因此差值只作数值参考。

**测试的基准。** Visual7W 和 RAVEN 各 300 题；Rune 的公开重建集为 128 道预览题加 8 张示例卡，共 136 题，使用 280 图像 token。重建集含 GUI、科学图表、几何、金融表格等任务；本方法的 FinQA 输入还包含原始结构化表格单元格文字。

**实验结果。**

| 基座 | 测试基准（题数） | 对比模型 | 作者报告准确率 | 本方法实测准确率 | 差值（百分点） |
|---|---|---|---:|---:|---:|
| Qwen3.5-2B-Base | Visual7W (300) | [Decider-2B-Vision](https://huggingface.co/Mapika/decider-2b-vision) | 89.00% | **90.33% (271/300)** | +1.33 |
| Qwen3.5-2B-Base | RAVEN (300) | [Decider-2B-Vision](https://huggingface.co/Mapika/decider-2b-vision) | 80.00% | **59.33% (178/300)** | −20.67 |
| Gemma-4-26B-A4B-it | Rune 公开重建集 (136) | [Rune v3](https://huggingface.co/surogate/rune-26b-a4b-GGUF) | 75.70% | **68.38% (93/136)** | −7.32 |

**实验结果解析。** Visual7W 的免训练结果接近作者参考值，而 RAVEN 的差距较大；Gemma 4 的重建集结果也低于 Rune 报告值。免训练适配在部分任务上已经可用，训练的收益与任务类型有关。这些不同输入条件下的参考结果不能证明整体优于训练模型，也不能证明严格非劣效。[评测协议、逐题记录与错误账本](docs/frozen-backbone-accuracy.md)

**对比模型做了什么？**

[**Decider-2B-Vision（Mapika）**](https://github.com/Mapika/decider/blob/e50e549b47e2da69223734fee4efa1ddd4528e93/MODEL_CARD_VISION.md)将 v5 文本决策权重移入 Qwen3.5-2B 视觉语言模型，在答案位置读出候选概率。视觉阶段使用约 8 万样本（约 5 万带图）训练一轮，数据包括脚本策略标注的游戏画面、DAgger 采集、Cauldron 选择题与文本回放，随后在 Breakout / Pong 上进行基于像素的 PPO。

[**Rune v3（Invergent / Surogate）**](https://huggingface.co/surogate/rune-26b-a4b-GGUF)基于 Gemma 4 26B-A4B-it，保留视觉塔，按决策协议返回 choice / noul / score。作者确认使用 [Surogate 训练引擎](https://invergent.ai/blog/rune/)进行了决策训练；公开模型卡与发布文章没有提供完整数据配方，也没有明确该版本采用全参数还是 LoRA、是否使用 RL。这里按已公开范围介绍，避免把训练引擎支持的方法当作 Rune 的实际训练方法。

### 实验二：直接决策减少了多少等待？

**实验目标。** 比较同一个官方基座对同一输入自然生成回答与直接返回候选分布的响应时间。

**比较的方法。** 两条路径收到相同的图像、问题、状态、候选与中性提示；自然生成使用贪心解码至 EOS，直接决策进行一次前向。每题各测三次，再先取题内中位数。Gemma 4 两侧均关闭 thinking；生成上限为 2048 token。

**测试的基准。** Rune 公开重建集 136 题和固定 ScienceQA 图像测试子集 100 题。使用 NVIDIA A800-SXM4-80GB、BF16；计时包含输入处理、传输、模型计算与结果读出，排除模型加载、网络及排队时间。

**实验结果。**

| 基座模型 | 测试集（题数） | 原模型输出长度（token，中位数） | 改造前延迟（ms，中位数） | 改造后延迟（ms，中位数） |
|---|---|---:|---:|---:|
| Gemma-4-26B-A4B-it | Rune (136) | 240 | 12512.1 | **248.2** |
| Gemma-3n-E4B-it | ScienceQA (100) | 62.5 | 4880.7 | **152.3** |

**实验结果解析。** 省去连续文本解码可以显著减少这类候选任务的等待，但答案质量必须同时看。Gemma 4 在本实验的自然提示下，首轮直接决策为 **30/136**，生成回答按保守解析规则为 **95/136**；Gemma 3n 的三轮记录分别为 **228/300** 与 **162/300**，其中生成回答有 **117/300** 未被解析。后者不能直接解读为生成模型能力较差。这些分数也不能与实验一使用明确选项标签提示的准确率混用。

Gemma 4 的 408 次生成调用中有 3 次触及上限，表中保留其观测耗时。另一个双方都要求单字母答案的控制实验测得 **308.1 ms → 244.6 ms**，说明输出已经很短时，延迟差距明显缩小。[Gemma 4 逐题输出与质量分析](docs/identical-input-latency.md) · [Gemma 3n 逐题记录](docs/instruction-latency.md)

## 界面展示

网页支持模型切换、文字状态、图片与视频抽帧、候选分布和 JSON 导出。下面的挑战都可以点击网页中的示例按钮载入；示例只提供输入，答案由当前模型实际计算。

### 公开国考图形推理

从实际测试中精选三个答对的公开国考题作为展示：**InternVL3.5-14B 的旋转题、多元素题，以及 MiniCPM-V-4.5 的复合规律题**。保留原题图片，输入只含问题、题图和 A–D 候选，不提供答案或解析。[题目与答案出处](https://gs.huatu.com/2024/0808/1771187.html)

![旋转题：InternVL3.5-14B 选择 A](docs/images/civil-internvl-2023-73.png)

![多元素题：InternVL3.5-14B 选择 A](docs/images/civil-internvl-2022-74.png)

![复合规律题：MiniCPM-V-4.5 选择 D](docs/images/civil-2019-73.png)

[精选正确示例与复现说明](docs/console-demos.md#公开国考图形推理) · [独立保存的完整测试记录](docs/civil-reasoning-results.md)

### 红球去哪了？

四帧小实验：红球进入左侧杯子，杯子遮住它，随后两个杯子交换位置。最后一帧看不到球，需要结合前面的画面回答。它让“看一张图”和“看一个事件”产生区别，也便于尝试删帧或乱序的影响。

![红球追踪的实际网页与模型返回](docs/images/red-ball.png)

### 网页里的文字陷阱

一张虚构展览预约页面里，真正的预约按钮旁放着一句“请选择取消”的干扰文字。模型需要根据用户的目标选按钮。这是一个可控的视觉任务与文字干扰示例，适合更换目标后观察分布变化。

![网页按钮挑战的实际网页与模型返回](docs/images/web-trap.png)

### 太空温室值班员

仪表板给出温度、湿度与土壤含水率，文字状态定义触发阈值。你来选下一步处理：浇水、通风、加热或保持。修改阈值而保留相同图片，可以检验模型是否同时使用视觉读数和文字条件。

![图文联合判断的实际网页与模型返回](docs/images/space-greenhouse.png)

公开考题与合成场景均保留实际接口结果，供交互体验；准确率结论以实验部分为准。[示例输入与复现说明](docs/console-demos.md)

## 安装使用

### 环境与七个模型

使用 **Linux x86_64、NVIDIA GPU、可用的 CUDA 驱动、Python 3.10–3.12（用于引导）和 Git**。初始化脚本在项目目录安装 Python 3.12 与三个独立环境。PyTorch 使用 CUDA 12.8 构建，请先用 `nvidia-smi` 确认驱动支持。完整七模型单卡依次使用建议 **80 GB 显存**；较小显卡可选下表中符合门槛的模型。门槛是加载前所需空闲显存，长输入可能增加需求。

| 网页模型 / model_id | 官方权重 | 环境 / 服务端口 | 最低空闲显存 |
|---|---|---|---:|
| `qwen35-2b` | [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | `.venv-qwen35` / 8460 | 8 GiB |
| `gemma4-a4b` | [Gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | `.venv-gemma4` / 8461 | 62 GiB |
| `gemma-e4b` | [Gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) | `.venv` / 8457 | 20 GiB |
| `gemma-e2b` | [Gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) | `.venv` / 8457 | 16 GiB |
| `minicpm-v45` | [MiniCPM-V-4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5) | `.venv` / 8457 | 22 GiB |
| `internvl35-8b` | [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | `.venv` / 8457 | 22 GiB |
| `internvl35-14b` | [InternVL3.5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) | `.venv` / 8457 | 34 GiB |

基础五模型使用 Transformers 4.57.1，Qwen3.5 与 Gemma 4 各用 Transformers 5.19.0。三个环境都在项目目录；启动只使用已装环境。下载全部 BF16 权重建议预留 **160 GB 以上**磁盘空间，另留环境与缓存空间。Gemma 模型需先在上述官方页面接受许可并获得访问权，再登录。

### 完整七模型部署

```bash
git clone https://github.com/jiangfeibo/TianZe-MJev.git
cd TianZe-MJev
./scripts/setup_all.sh
.venv/bin/hf auth login
export MODEL_ROOT="$PWD/models"
.venv/bin/python scripts/download_models.py qwen35-2b gemma4-a4b gemma-e2b gemma-e4b minicpm-v45 internvl35-8b internvl35-14b
CUDA_VISIBLE_DEVICES=0 ./start_all.sh
```

打开 [本地决策台](http://127.0.0.1:8456)，点击示例载入输入和对应模型；也可选择其他已下载模型，再点击“计算概率并选择”。首次提交加载权重，所以会比后续推理慢。网页切换后端时会卸载本项目其他后端的驻留模型，便于七个模型依次共用一张显卡。直接访问各后端 API 时，由调用者通过各自 `/unload` 管理驻留。

远程服务器使用本机终端建立访问通道：

```bash
ssh -L 8456:127.0.0.1:8456 USER@SERVER
```

将 `USER@SERVER` 替换为你的服务器登录地址；非默认 SSH 端口追加 `-p PORT`。随后在本机浏览器打开同一个本地决策台地址。

### 只部署一个模型

例如 MiniCPM-V-4.5，无需初始化另外两个环境：

```bash
./setup.sh
export MODEL_ROOT="$PWD/models"
.venv/bin/python scripts/download_models.py minicpm-v45
CUDA_VISIBLE_DEVICES=0 ./start_all.sh
```

Qwen3.5 或 Gemma 4 单模型部署需在 `./setup.sh` 后分别执行 `./scripts/setup_qwen35.sh` 或 `./scripts/setup_gemma4.sh`，再下载相应 `model_id`。若没有安装 uv，先执行完整初始化脚本，或者使用已有 uv；基础环境的备用安装方式要求本机已有合适版本的 Python。

### 调用 API、检查与停止

下面这条命令与网页走同一网关，可路由七个模型。先下载所选模型，将 `model_id` 改为上表对应值：

```bash
curl --fail-with-body http://127.0.0.1:8456/api/decide \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"minicpm-v45","question":"2+3等于多少？","state_text":"","options":[{"id":"four","text":"4"},{"id":"five","text":"5"},{"id":"six","text":"6"}],"images":[]}'
curl --fail http://127.0.0.1:8456/api/models
./stop_all.sh
```

返回结果包含 `answer`、`distribution`、模型名称与耗时。网页当前提供候选选择；`noul` / `score` 使用模型对应的服务端口 `/decide`，见 [API 示例](examples/)和 [Qwen3.5](docs/qwen35-runtime.md) / [Gemma 4](docs/gemma4-runtime.md)说明。`/api/models` 的 `model_ready` 表示下载完成且后端可达；`loaded` 表示模型已驻留。

视频需系统已有 `ffmpeg` 和 `ffprobe`，也可使用下方项目内安装方法：

```bash
mkdir -p .tools/bin
curl -fL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C .tools bin/micromamba
.tools/bin/micromamba create -y -p "$PWD/.video-env" -c conda-forge ffmpeg
export PATH="$PWD/.video-env/bin:$PATH"
./stop_all.sh
CUDA_VISIBLE_DEVICES=0 ./start_all.sh
```

已有权重时，下载与启动前统一设置 `MODEL_ROOT=/你的模型目录`，目录名按下载脚本约定。启动会检查服务是否就绪；失败查看 `logs/`，常见原因是端口占用、依赖初始化失败、许可未授权或可用显存不足。服务默认绑定本机。

下载齐七个模型并启动后，可运行 `.venv/bin/python scripts/verify_installation.py`，逐个验证图片请求、候选分布和切换后的模型驻留；记录保存到 `run/installation-check.json`。这是安装功能检查，实际验证环境与记录见 [七模型部署验证](docs/installation-verification.md)。

## 作者信息

**作者：** 毛林祺，[maolinqi@hunnu.edu.cn](mailto:maolinqi@hunnu.edu.cn)，湖南师范大学，计算机技术专业，在读研究生。

**指导老师：** 江沸菠，[jiangfb@hunnu.edu.cn](mailto:jiangfb@hunnu.edu.cn)，湖南师范大学，信息科学与工程学院，副教授。

项目由 [maolinqi](https://github.com/maolinqi) 维护。问题反馈、示例输入和复现记录可提交到 [GitHub Issues](https://github.com/jiangfeibo/TianZe-MJev/issues)；贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。基座模型由 Qwen、Google、OpenBMB 与 OpenGVLab 发布；Decider 和 Rune 的结果归各自作者。

## 开源许可

本项目接口代码采用 **[Apache-2.0](LICENSE)**，包括原生适配器、观测与决策协议、API、网页、生命周期脚本和评测代码。模型权重从上游下载，遵循各自许可；详见 [模型许可说明](docs/model-licenses.md)。仓库的 `evidence/` 保留实际验证记录，`docs/` 提供评测协议与结果。
