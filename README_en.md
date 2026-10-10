<div align="center">

# TianZe-MJev: A Multimodal Jev-style Decision Model
### 天择-MJev: 一种基于Jev范式的多模态决策模型框架

**See the scene. Understand the task. Make a choice.**

**English** · [简体中文](README.md)

[![License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/jiangfeibo/TianZe-MJev/actions/workflows/tests.yml/badge.svg)](https://github.com/jiangfeibo/TianZe-MJev/actions/workflows/tests.yml)
[![Backbones](https://img.shields.io/badge/Backbones-7-2563eb)](#installation-and-use)

[Introduction](#introduction) · [Experiments](#experiments) · [Web-console-gallery](#web-console-gallery) · [Installation](#installation-and-use) · [Authors](#authors) · [License](#license)

</div>

## Introduction

### 1. Background: from generating answers to making decisions directly

Recent advances in artificial intelligence, represented by large language models (LLMs) and multimodal large language models (MLLMs), have enabled machines to understand natural language, perceive visual environments and handle complex tasks. However, **understanding the world does not necessarily mean making decisions efficiently**. As AI moves from conversational interaction into applications such as robot control, autonomous drone navigation, game agents and intelligent devices, models need to answer not only “What happened?” and “Why?” but also “What should we do now?”

Most mainstream large models currently use autoregressive generation, predicting text output one token at a time. For complex questions, models can reason by generating intermediate analytical steps. Yet many decision tasks with explicit candidate options ultimately require only a simple choice.

For example, when a drone detects an obstacle ahead, the system needs to choose an appropriate action from candidates such as avoiding it to the left, avoiding it to the right or hovering, without necessarily generating a complete natural-language analysis first. Likewise, robot action selection, game operations and GUI interaction require explicit, fast and structured decisions.

**This raises a research question: can we fully use the perception and understanding capabilities of existing multimodal models to make intelligent decisions directly, bypassing unnecessary text generation?**

### 2. Cognitive inspiration: the dual-system mechanisms of System 1 and System 2

Dual-process theory in cognitive psychology offers useful inspiration for understanding this question. In *Thinking, Fast and Slow*, Daniel Kahneman describes two typical modes of human cognitive processing through System 1 and System 2.

#### System 1: fast and intuitive thinking

System 1 primarily involves fast, automatic judgments with limited conscious involvement. People can often form initial judgments from prior experience and current perceptual information without complex explicit reasoning. Examples include recognizing familiar objects, identifying obvious dangers and quickly choosing an action in familiar situations.

Its defining characteristic is: **forming judgments quickly from current information without lengthy explicit analysis.**

#### System 2: slow and deliberative thinking

System 2 primarily involves conscious, controlled analysis that requires more cognitive resources. Complex mathematical problems, multi-step logical reasoning and tasks involving tradeoffs among multiple constraints typically require deeper analysis to reach a conclusion.

Its defining characteristic is: **solving complex problems that require deep thought through conscious analysis, reasoning and comparison.**

System 1 and System 2 are conceptual categories of cognitive processing, rather than strict equivalents of non-autoregressive and autoregressive AI models. Here, we draw inspiration from two different decision paths: one generates intermediate content explicitly to support decisions, while the other evaluates candidate outcomes directly from the model's internal representations.

### 3. Research motivation: why do large models need System 1-style decision capabilities?

Current multimodal models have strong capabilities for understanding their environments, but their common forms of interaction remain centered on text generation. For tasks requiring explicit choices, this approach raises three issues.

**First, continuous text generation may introduce unnecessary decision latency.** Autoregressive models generate output tokens step by step. When a model produces a lengthy analysis, the decision must wait for generation to finish. This serial decoding mechanism may struggle to meet the needs of low-latency interaction and frequent decisions.

**Second, natural-language generation differs in form from practical decision interfaces.** Conventional generative models mainly produce free text, while agent control systems typically require action categories, candidate IDs or structured control commands. Converting generated content into executable decisions may require additional format constraints and parsing.

**Finally, conventional text-generation interfaces do not necessarily provide complete candidate decision distributions directly.** With multiple candidate actions, a system may need both a final choice and a comparison of relative preferences among candidates to support subsequent decision analysis and control.

For selection tasks within finite candidate sets, it is therefore worth exploring an alternative to continuous text generation: **using the knowledge and representations already learned by multimodal models to score and select candidate outcomes directly.**

This design does not imply that System 1 can replace System 2. Deliberate analysis remains valuable for tasks requiring complex planning and deep reasoning. The research focus is whether tasks that permit direct judgments can use a simpler decision path and reduce unnecessary generation computation.

### 4. TianZe-MJev: a multimodal direct decision framework inspired by System 1

Building on this research background, we propose **TianZe-MJev (天择-MJev: 一种基于Jev范式的多模态决策模型框架)**, a multimodal direct decision framework inspired by System 1's fast judgment mechanism and designed for finite-candidate tasks.

The core idea of TianZe-MJev is to **retain the perception and understanding capabilities of existing multimodal models while transforming the conventional “understanding → text generation → result extraction” path into “understanding → candidate scoring → direct selection.”**

The framework receives multimodal inputs such as natural-language task instructions, environment state, candidate options, images and video frames. It first extracts task-relevant information through the existing model's visual encoding, language understanding and cross-modal fusion capabilities. It then reads the output logits corresponding to candidate labels in a single language-model forward pass, computes their relative probability distribution and selects the corresponding decision.

This process does not require generating a complete natural-language answer token by token or training an additional dedicated decision network.

Compared with conventional generative decision paths, TianZe-MJev has four core characteristics:

1. **Direct Decision:** obtain the candidate distribution in a single language-model forward pass, avoiding the additional decoding overhead of continuous text generation.

2. **Training-Free Adaptation:** directly reuse the pretrained weights of existing multimodal models, without additional decision fine-tuning, backbone parameter updates or new trainable parameters.

3. **Multimodal Perception:** retain the backbone's native multimodal understanding capabilities and combine text, environment state, images and video frames to make task judgments.

4. **Structured Selection:** directly return candidate options and their relative probability distributions, supporting Choice, binary judgment (Noul) and Score tasks for integration with agents and automated systems.

TianZe-MJev currently supports seven official multimodal backbones across the Qwen3.5, Gemma 3n/4, MiniCPM-V and InternVL families, with a unified decision interface and visual interaction environment.

## Experiments

### 1. Accuracy without additional training

**Goal:** measure frozen-backbone decision accuracy and contextualize it with trained-model reports.

**Methods compared:** our official frozen backbones versus authors' reported Decider-2B-Vision and Rune v3 results. Trained reference weights were not run here; sample revisions and final image presentations are not fully paired.

**Benchmarks:** Visual7W and RAVEN, 300 questions each; Rune's 128 public preview items plus eight cards, 136 total, with a 280-image-token budget. Our reconstructed GUI presentation and structured FinQA table-cell text differ from the author inputs.

**Results:**

| Backbone | Benchmark (n) | Reference model | Author report | Ours, measured | Difference (pp) |
|---|---|---|---:|---:|---:|
| Qwen3.5-2B-Base | Visual7W (300) | [Decider-2B-Vision](https://huggingface.co/Mapika/decider-2b-vision) | 89.00% | **90.33% (271/300)** | +1.33 |
| Qwen3.5-2B-Base | RAVEN (300) | [Decider-2B-Vision](https://huggingface.co/Mapika/decider-2b-vision) | 80.00% | **59.33% (178/300)** | −20.67 |
| Gemma-4-26B-A4B-it | Rune public reconstruction (136) | [Rune v3](https://huggingface.co/surogate/rune-26b-a4b-GGUF) | 75.70% | **68.38% (93/136)** | −7.32 |

**Interpretation:** Visual7W is numerically close to the reference, while RAVEN and the Rune reconstruction show substantial gaps. Training benefits depend on the task. These references establish neither overall superiority nor paired non-inferiority. [Protocol, raw results and error ledger](docs/frozen-backbone-accuracy.md).

**Reference models and their training:**

[**Decider-2B-Vision**](https://github.com/Mapika/decider/blob/e50e549b47e2da69223734fee4efa1ddd4528e93/MODEL_CARD_VISION.md) transplants v5 text weights into Qwen3.5-2B's vision-language model and reads option probabilities at an answer slot. One epoch uses 80k examples (50k with images): scripted game policies, DAgger frames, Cauldron multiple-choice tasks and text replay. Pixel-based PPO on Breakout/Pong follows.

[**Rune v3**](https://huggingface.co/surogate/rune-26b-a4b-GGUF) adapts Gemma 4 26B-A4B-it, retaining vision and a choice/noul/score protocol. Its author confirms training with [Surogate](https://invergent.ai/blog/rune/). The reviewed primary sources do not disclose a full data recipe, whether v3 uses full fine-tuning or LoRA, or whether it uses RL; engine capabilities are not evidence of Rune's actual recipe.

### 2. Response latency

**Goal:** compare natural text responses with direct finite-choice decisions from the same official backbone.

**Methods compared:** identical images, state, question, candidates and neutral prompts; greedy native generation to EOS versus one decision forward. Three repetitions per question, summarized first by within-question median. Gemma 4 disables thinking on both paths and uses a 2,048-token emergency cap.

**Benchmarks:** 136 Rune public reconstruction questions and 100 fixed ScienceQA image-test questions; A800-SXM4-80GB, BF16. Timing includes input preparation, transfer, computation and readout; excludes loading, network and queues.

**Results:**

| Backbone | Test set (n) | Native output tokens, median | Before (ms, median) | After (ms, median) |
|---|---|---:|---:|---:|
| Gemma-4-26B-A4B-it | Rune (136) | 240 | 12512.1 | **248.2** |
| Gemma-3n-E4B-it | ScienceQA (100) | 62.5 | 4880.7 | **152.3** |

**Interpretation:** avoiding sequential decoding reduces waiting, but quality must be assessed separately. Under this neutral prompt, Gemma 4 scores **30/136** for direct decisions versus **95/136** for conservatively parsed generation in the first repetition. Gemma 3n scores **228/300** versus **162/300** across three repetitions, with **117/300** generated answers unparsed; this is not evidence that generation has lower intrinsic accuracy. These are different prompts from experiment 1.

Gemma 4 hits the cap on 3/408 generation calls; their observed times remain included. A separate shared single-letter-prompt control measures **308.1 → 244.6 ms**, showing much less benefit when generation is already short. [Gemma 4 quality and timing](docs/identical-input-latency.md), [Gemma 3n records](docs/instruction-latency.md).

## Web console gallery

The console switches models, accepts text, images and sampled video, plots candidate distributions and exports JSON. The example buttons load inputs only; results come from the selected model.

**Public civil-service figure reasoning: selected correct examples.** These three actual requests matched the published answer key: rotation and multiple-element items with **InternVL3.5-14B**, and a compound-rule item with **MiniCPM-V-4.5**. Inputs contain the original image, question and A–D candidates without answers or explanations. [Source and answer key](https://gs.huatu.com/2024/0808/1771187.html).

![Rotation: InternVL3.5-14B selects A](docs/images/civil-internvl-2023-73.png)

![Multiple elements: InternVL3.5-14B selects A](docs/images/civil-internvl-2022-74.png)

![Compound rule: MiniCPM-V-4.5 selects D](docs/images/civil-2019-73.png)

[Selected screenshots and reproduction](docs/console-demos.md#公开国考图形推理) · [Separate complete test record](docs/civil-reasoning-results.md)

**Where did the red ball go?** Four frames show a ball entering a cup, becoming hidden, and the cups exchanging places. Try dropping or reordering frames.

![Actual red-ball console response](docs/images/red-ball.png)

**A text trap on a web page.** A fictional exhibition booking page contains distracting instructions next to the real booking button. Choose based on the user's goal.

![Actual web-button console response](docs/images/web-trap.png)

**Space greenhouse operator.** Combine pictured instrument readings with textual thresholds to choose watering, ventilation, heating or holding steady. Change the threshold with the same image.

![Actual greenhouse console response](docs/images/space-greenhouse.png)

Public exam items and synthetic scenes retain real API responses, not benchmark accuracy claims. [Inputs, records and reproduction](docs/console-demos.md).

## Installation and use

Use Linux x86_64, an NVIDIA GPU and driver supporting the PyTorch CUDA 12.8 build, Git, and Python 3.10–3.12 for bootstrapping. Setup installs Python 3.12 and three project-local runtimes. For all seven models sequentially on one GPU, an **80 GB card** is recommended. Allow **160 GB or more for weights**, plus runtime/cache space. Accept the upstream Gemma terms and obtain access before downloading.

| model_id | Official weights | Runtime / port | Free GPU memory admission |
|---|---|---|---:|
| `qwen35-2b` | [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | `.venv-qwen35` / 8460 | 8 GiB |
| `gemma4-a4b` | [Gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | `.venv-gemma4` / 8461 | 62 GiB |
| `gemma-e4b` | [Gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) | `.venv` / 8457 | 20 GiB |
| `gemma-e2b` | [Gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) | `.venv` / 8457 | 16 GiB |
| `minicpm-v45` | [MiniCPM-V-4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5) | `.venv` / 8457 | 22 GiB |
| `internvl35-8b` | [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | `.venv` / 8457 | 22 GiB |
| `internvl35-14b` | [InternVL3.5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) | `.venv` / 8457 | 34 GiB |

These are free-memory admission thresholds, not peak-memory guarantees. The five original backbones use Transformers 4.57.1; Qwen3.5/Gemma 4 each use 5.19.0 in separate environments.

```bash
git clone https://github.com/jiangfeibo/TianZe-MJev.git
cd TianZe-MJev
./scripts/setup_all.sh
.venv/bin/hf auth login
export MODEL_ROOT="$PWD/models"
.venv/bin/python scripts/download_models.py qwen35-2b gemma4-a4b gemma-e2b gemma-e4b minicpm-v45 internvl35-8b internvl35-14b
CUDA_VISIBLE_DEVICES=0 ./start_all.sh
```

Open [the local console](http://127.0.0.1:8456), load an example and its model, optionally choose another downloaded model, then submit. First use includes weight loading. Console requests unload other project-managed backends before switching, allowing sequential use of one GPU. Direct backend API callers must manage residency with `/unload` themselves.

For a remote server, forward the console to your computer:

```bash
ssh -L 8456:127.0.0.1:8456 USER@SERVER
```

Replace the login address; append `-p PORT` for a non-default SSH port.

For just MiniCPM-V-4.5:

```bash
./setup.sh
export MODEL_ROOT="$PWD/models"
.venv/bin/python scripts/download_models.py minicpm-v45
CUDA_VISIBLE_DEVICES=0 ./start_all.sh
```

For Qwen3.5 or Gemma 4, also run the corresponding `scripts/setup_qwen35.sh` or `scripts/setup_gemma4.sh` after base setup, then download that model ID. Those scripts require uv; `setup_all.sh` bootstraps it locally when missing.

Call the shared seven-model gateway, inspect readiness, and stop:

```bash
curl --fail-with-body http://127.0.0.1:8456/api/decide \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"minicpm-v45","question":"What is 2+3?","state_text":"","options":[{"id":"four","text":"4"},{"id":"five","text":"5"},{"id":"six","text":"6"}],"images":[]}'
curl --fail http://127.0.0.1:8456/api/models
./stop_all.sh
```

Responses include `answer`, `distribution`, model and timing. The console currently offers choice; use the corresponding backend `/decide` for noul/score. See [examples](examples/), [Qwen3.5](docs/qwen35-runtime.md) and [Gemma 4](docs/gemma4-runtime.md). `model_ready` means reachable backend and downloaded weights; `loaded` means resident weights.

Video requires `ffmpeg` and `ffprobe`. A project-local installation is available:

```bash
mkdir -p .tools/bin
curl -fL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C .tools bin/micromamba
.tools/bin/micromamba create -y -p "$PWD/.video-env" -c conda-forge ffmpeg
export PATH="$PWD/.video-env/bin:$PATH"
./stop_all.sh
CUDA_VISIBLE_DEVICES=0 ./start_all.sh
```

Set `MODEL_ROOT` consistently for downloads/startup to reuse existing weights with the downloader's directory names. Startup checks service readiness; inspect `logs/` for port conflicts or initialization failures. Services bind to localhost. Authorization and free GPU memory must be available for the selected model.

After downloading all seven models and starting the services, run `.venv/bin/python scripts/verify_installation.py` to check image requests, candidate distributions and single-model residency. Results go to `run/installation-check.json`. This checks installation functionality; see [the verified environment and records](docs/installation-verification.md).

## Authors

**Author:** Linqi Mao (毛林祺), [maolinqi@hunnu.edu.cn](mailto:maolinqi@hunnu.edu.cn), Hunan Normal University, postgraduate student in Computer Technology.

**Advisor:** Feibo Jiang (江沸菠), [jiangfb@hunnu.edu.cn](mailto:jiangfb@hunnu.edu.cn), Hunan Normal University, College of Information Science and Engineering, Associate Professor.

Maintained by [maolinqi](https://github.com/maolinqi). Share reproducible inputs and issues through [GitHub Issues](https://github.com/jiangfeibo/TianZe-MJev/issues); see [Contributing](CONTRIBUTING.md). Backbones are published by Qwen, Google, OpenBMB and OpenGVLab. Decider/Rune reference results belong to their authors.

## License

Our adapters, protocols, API, console, lifecycle scripts and evaluation code use **[Apache-2.0](LICENSE)**. Upstream weights and custom code retain their respective licenses; see [Model licenses](docs/model-licenses.md). `evidence/` contains recorded validation and `docs/` contains protocols and results.
