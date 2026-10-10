<div align="center">

# 天择-MJev: 一种基于Jev范式的多模态决策模型框架
### TianZe-MJev: A Multimodal Jev-style Decision Model

**看见现场，读懂任务，直接作出选择。**

[English](README_en.md) · **简体中文**

[![Code License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/jiangfeibo/TianZe-MJev/actions/workflows/tests.yml/badge.svg)](https://github.com/jiangfeibo/TianZe-MJev/actions/workflows/tests.yml)
[![Backbones](https://img.shields.io/badge/Backbones-7-2563eb)](#安装使用)
[![Training](https://img.shields.io/badge/Additional_Training-0_steps-2563eb)](#介绍)

[介绍](#介绍) · [实验结果](#实验结果) · [界面展示](#界面展示) · [安装使用](#安装使用) · [作者信息](#作者信息) · [开源许可](#开源许可)

</div>

## 介绍

### 一、项目背景：从“生成答案”走向“直接决策”

近年来，以大语言模型（LLM）和多模态大语言模型（MLLM）为代表的人工智能技术快速发展，使机器逐步具备了理解自然语言、感知视觉环境和处理复杂任务的能力。然而，**理解世界并不等于能够高效地作出决策**。随着人工智能从对话交互走向机器人控制、无人机自主导航、游戏智能体和智能终端等实际应用，模型不仅需要回答“发生了什么”和“为什么”，还需要进一步判断“现在应该做什么”。

目前，主流大模型主要采用自回归（Autoregressive）生成机制，通过逐 Token 预测形成文本输出。当面对复杂问题时，模型可以通过生成中间分析步骤完成推理；但对于大量具有明确候选选项的决策任务，系统最终可能只需要一个简单的选择结果。

例如，当无人机检测到前方障碍物时，系统需要从“向左避障、向右避障、悬停”等候选动作中选择合适的操作，而不一定需要先生成一段完整的自然语言分析。同样，在机器人动作选择、游戏操作和GUI交互等任务中，模型需要的是明确、快速、结构化的决策结果。

**这引出了一个值得研究的问题：能否充分利用现有多模态大模型的感知与理解能力，绕过不必要的文本生成过程，直接完成智能决策？**

### 二、认知启发：System 1与System 2双系统机制

认知心理学中的双加工理论（Dual-Process Theory）为理解这一问题提供了重要启发。Daniel Kahneman在《思考，快与慢》（Thinking, Fast and Slow）中，通过System 1和System 2描述人类思维中的两类典型认知加工方式。

#### System 1：快速直觉系统（Fast and Intuitive Thinking）

System 1主要表现为快速、自动、低意识参与的判断过程。人们通常不需要进行复杂的显式推理，就能够根据已有经验和当前感知信息形成初步判断。例如，识别熟悉的物体、判断明显的危险，或者在熟悉场景中迅速选择行动。

其核心特点是：**无需展开冗长的显式分析，即可根据当前信息快速形成判断。**

#### System 2：审慎分析系统（Slow and Deliberative Thinking）

System 2主要表现为有意识、受控制且需要较多认知资源的分析过程。当面对复杂数学问题、多步骤逻辑推理或需要权衡多个约束条件的任务时，人们通常需要进行更深入的分析，并逐步形成结论。

其核心特点是：**通过有意识的分析、推理和比较，解决需要深度思考的复杂问题。**

需要强调的是，System 1和System 2是认知加工方式的概念划分，并不严格对应人工智能中的非自回归和自回归模型。本文借鉴的是两种不同的决策路径：一种通过显式生成中间内容辅助决策，另一种直接根据模型内部表征对候选结果进行判断。

### 三、研究动机：为什么大模型需要System 1式决策能力？

当前多模态大模型已经具备较强的环境理解能力，但其常见交互形式仍以文本生成为中心。对于需要明确选择结果的任务，这种方式存在三个值得关注的问题。

**首先，连续文本生成可能引入不必要的决策延迟。** 自回归模型需要逐步生成输出Token，当模型生成较长的分析内容时，决策结果必须等待生成过程完成。这种串行解码机制可能难以满足低时延交互和高频决策的需求。

**其次，自然语言生成与实际决策接口之间存在形式上的差异。** 传统生成模型主要输出自由文本，而智能体控制系统通常需要动作类别、候选编号或结构化控制指令。将生成内容转换为可执行决策，可能需要额外的格式约束和解析处理。

**最后，传统文本生成接口不一定直接提供完整的候选决策分布。** 对于多个候选动作，系统不仅需要确定最终选择，还可能需要比较各个候选结果的相对偏好，以支持后续的决策分析和系统控制。

因此，对于有限候选集合中的选择任务，值得探索一种区别于连续文本生成的处理方式：**利用多模态模型已经学习到的知识与表征能力，直接对候选结果进行评分和选择。**

这种设计并不意味着System 1能够取代System 2。对于需要复杂规划和深度推理的任务，审慎分析仍然具有重要价值。研究重点在于：对于能够直接判断的任务，是否可以采用更简洁的决策路径，减少不必要的生成计算。

### 四、TianZe-MJev：受System 1启发的多模态直接决策框架

基于上述研究背景，我们提出 **TianZe-MJev（天择-MJev: 一种基于Jev范式的多模态决策模型框架）**，一种受System 1快速判断机制启发、面向有限候选任务的多模态直接决策框架。

TianZe-MJev的核心思想是：**保留现有多模态大模型的感知与理解能力，将传统的“理解—文本生成—结果提取”路径，转换为“理解—候选评分—直接选择”的决策路径。**

具体而言，框架接收自然语言任务指令、环境状态、候选选项以及图像或视频帧等多模态输入。首先，利用已有多模态模型的视觉编码、语言理解和跨模态融合能力提取任务相关信息；随后，通过一次语言模型前向计算，读取候选标签对应的输出Logits，计算候选决策的相对概率分布，并选择相应的决策结果。

这一过程无需逐Token生成完整的自然语言回答，也不依赖额外训练专用决策网络。

与传统生成式决策路径相比，TianZe-MJev具有以下四个核心特点：

1. **直接决策（Direct Decision）**：通过单次语言模型前向计算获得候选结果分布，避免连续文本生成带来的额外解码开销。

2. **免训练适配（Training-Free Adaptation）**：直接复用已有多模态模型的预训练权重，无需额外决策微调，不更新基座参数，也不引入新的可训练参数。

3. **多模态感知（Multimodal Perception）**：保留基座模型原生的多模态理解能力，支持融合文本、环境状态、图像和视频帧等输入信息完成任务判断。

4. **结构化选择（Structured Selection）**：直接输出候选选项及其相对概率分布，支持选择（Choice）、二元判断（Noul）和评分（Score）等任务形式，便于对接智能体及自动化系统。

目前，TianZe-MJev已支持Qwen3.5、Gemma 3n/4、MiniCPM-V和InternVL等系列的七个官方多模态基座模型，并提供统一的决策接口与可视化交互环境。

## 实验结果

### 实验一：免训练决策的准确率

**实验目标。** 衡量官方冻结基座经决策适配后的任务准确率，并与公开训练模型的报告值建立参考。

**比较的方法。** 本方法加载官方权重、不加决策训练；参考方法为 Decider-2B-Vision 和 Rune v3 的作者报告。本轮未运行这两个训练模型，样本版本与最终图像呈现未完全配对，因此差值只作数值参考。

**测试的基准。** Visual7W 300 题；Rune 的公开重建集为 128 道预览题加 8 张示例卡，共 136 题，使用 280 图像 token。重建集含 GUI、科学图表、几何、金融表格等任务；本方法的 FinQA 输入还包含原始结构化表格单元格文字。

**实验结果。**

| 基座 | 测试基准（题数） | 对比模型 | 作者报告准确率 | 本方法实测准确率 | 差值（百分点） |
|---|---|---|---:|---:|---:|
| Qwen3.5-2B-Base | Visual7W (300) | [Decider-2B-Vision](https://huggingface.co/Mapika/decider-2b-vision) | 89.00% | **90.33% (271/300)** | +1.33 |
| Gemma-4-26B-A4B-it | Rune 公开重建集 (136) | [Rune v3](https://huggingface.co/surogate/rune-26b-a4b-GGUF) | 75.70% | **68.38% (93/136)** | −7.32 |

**实验结果解析。** Visual7W 的免训练结果接近作者参考值；Gemma 4 的重建集结果也低于 Rune 报告值。免训练适配在部分任务上已经可用，训练的收益与任务类型有关。这些不同输入条件下的参考结果不能证明整体优于训练模型，也不能证明严格非劣效。[评测协议、逐题记录与错误账本](docs/frozen-backbone-accuracy.md)

**对比模型做了什么？**

[**Decider-2B-Vision（Mapika）**](https://github.com/Mapika/decider/blob/e50e549b47e2da69223734fee4efa1ddd4528e93/MODEL_CARD_VISION.md)将 v5 文本决策权重移入 Qwen3.5-2B 视觉语言模型，在答案位置读出候选概率。视觉阶段使用约 8 万样本（约 5 万带图）训练一轮，数据包括脚本策略标注的游戏画面、DAgger 采集、Cauldron 选择题与文本回放，随后在 Breakout / Pong 上进行基于像素的 PPO。

[**Rune v3（Invergent / Surogate）**](https://huggingface.co/surogate/rune-26b-a4b-GGUF)基于 Gemma 4 26B-A4B-it，保留视觉塔，按决策协议返回 choice / noul / score。作者确认使用 [Surogate 训练引擎](https://invergent.ai/blog/rune/)进行了决策训练；公开模型卡与发布文章没有提供完整数据配方，也没有明确该版本采用全参数还是 LoRA、是否使用 RL。这里按已公开范围介绍，避免把训练引擎支持的方法当作 Rune 的实际训练方法。

### 实验二：直接决策减少了多少等待？

**实验目标。** 比较同一个官方基座对同一输入自然生成回答与直接返回候选分布的响应时间。

**比较的方法。** 两条路径收到相同的图像、问题、状态、候选与中性提示；自然生成使用贪心解码至 EOS，直接决策进行一次前向。每题各测三次，再先取题内中位数。Gemma 4 两侧均关闭 thinking；生成上限为 2048 token。

**测试的基准。** Rune 公开重建集 136 题和固定 ScienceQA 图像测试子集 100 题。使用 NVIDIA A800-SXM4-80GB、BF16；计时包含输入处理、传输、模型计算与结果读出，排除模型加载、网络及排队时间。

**实验结果。**

| 基座模型 | 测试集（题数） | 原模型输出长度（token，中位数） | 改造前延迟（ms，中位数） | 改造后延迟（ms，中位数） | 加速比例 |
|---|---|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | Rune (136) | 240 | 12512.1 | **248.2** | **50.4×** |
| Gemma-3n-E4B-it | ScienceQA (100) | 62.5 | 4880.7 | **152.3** | **32.0×** |

加速比例 = 改造前延迟 ÷ 改造后延迟（按表中中位数计算）。

其余五个基座已改用“先分析、比较选项并说明理由，再给出选择”的提示，重新安排双卡实验。自然生成与直接决策使用相同提示，但直接决策仅读取首步候选 logits；新组与上表的中性提示结果分开报告。旧补测已废弃，新结果尚未完成。[新实验与暂停规则](docs/latency-reasoning-20261010.md)

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
