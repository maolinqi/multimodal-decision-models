<div align="center">

# 天择多模态决策大模型框架
### TianZe-MJev: A Multimodal Jev-style Decision Model

**看见现场，读懂任务，直接作出选择。**

[English](README.md) · **简体中文**

[![Code License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml/badge.svg)](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml)
[![Backbones](https://img.shields.io/badge/Backbones-7-2563eb)](#安装使用)
[![Training](https://img.shields.io/badge/Additional_Training-0_steps-2563eb)](#优势)

[介绍](#介绍) · [优势](#优势) · [实验结果](#实验结果) · [界面展示](#界面展示) · [安装使用](#安装使用) · [作者信息](#作者信息) · [开源许可](#开源许可)

</div>

## 介绍

你在玩一个小游戏。红球从左边滚来，下一刻就要撞上挡板。你需要决定向左还是向右，却不需要先写一段关于球速、方向和游戏规则的说明。

很多软件也在等这样的一个选择：屏幕上哪个按钮对应当前目标？连续画面里的物体去了哪里？这条观测是否符合给定条件？系统需要的是能进入下一步的答案。

认知心理学用 **System 1** 和 **System 2** 描述两类处理方式：前者快速、自主地产生反应，后者借助注意与工作记忆进行审慎推理。[Evans 与 Stanovich（2013）](https://journals.sagepub.com/doi/10.1177/1745691612460685)讨论了这一区分。在规律稳定、线索有效、又有充分练习和反馈的环境中，快速直觉可以形成可靠的技能；遇到陌生问题、含糊证据和多步推理，审慎分析仍然重要。[Kahneman 与 Klein（2009）](https://bear.warrington.ufl.edu/brenner/mar7588/Papers/kahneman-klein-2009.pdf)

把这个问题带到大模型里：如果候选答案已经给定，为什么每次都要等它把答案写成一段话？自回归模型逐个预测后续 token，前面的文字生成完，后面的文字才继续。在大量重复的有限选项判断中，应用最终可能只保留一个标签，却付出了整段输出的解码时间。解释有用途时值得生成；只需要一个选择时，可以把输出路径缩短。

**天择从这个需求出发：让已有的多模态理解能力直接进入决策接口。** 输入问题、状态和候选选项，配上图像或连续画面，返回稳定选项 ID 与候选分布。System 1 在这里是快速决策通路的设计启发，心理学分类与模型推理方式并非一一对应；这条通路适合开展低延迟候选决策实验，效果由具体任务的验证决定。

文字告诉模型“要判断什么”，图像提供“眼前是什么”；多张 RGB 图像和最多 **8 帧带时间戳的视频抽帧**保留相机标识与先后顺序。你可以用 `choice` 在 **2–26 个候选**中选择，用 `noul` 判断陈述的真假，或用 `score` 返回有序等级分布与期望分数。结果能直接交给程序，也能在网页上比较。候选概率表达的是给定选项内的相对偏好。

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
git clone https://github.com/maolinqi/multimodal-decision-models.git
cd multimodal-decision-models
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

项目由 [maolinqi](https://github.com/maolinqi) 维护。问题反馈、示例输入和复现记录可提交到 [GitHub Issues](https://github.com/maolinqi/multimodal-decision-models/issues)；贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。基座模型由 Qwen、Google、OpenBMB 与 OpenGVLab 发布；Decider 和 Rune 的结果归各自作者。

## 开源许可

本项目接口代码采用 **[Apache-2.0](LICENSE)**，包括原生适配器、观测与决策协议、API、网页、生命周期脚本和评测代码。模型权重从上游下载，遵循各自许可；详见 [模型许可说明](docs/model-licenses.md)。仓库的 `evidence/` 保留实际验证记录，`docs/` 提供评测协议与结果。
