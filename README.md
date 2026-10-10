<div align="center">

# TianZe-MJev: A Multimodal Jev-style Decision Model
### 天择多模态决策大模型框架

**See the scene. Understand the task. Make a choice.**

**English** · [简体中文](README_zh.md)

[![License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml/badge.svg)](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml)
[![Backbones](https://img.shields.io/badge/Backbones-7-2563eb)](#installation-and-use)

[Introduction](#introduction) · [Advantages](#advantages) · [Experiments](#experiments) · [Web-console-gallery](#web-console-gallery) · [Installation](#installation-and-use) · [Authors](#authors) · [License](#license)

</div>

## Introduction

### Background: why must a decision start with generating text?

Imagine a drone flying at high speed when an obstacle suddenly appears ahead. The system must immediately decide: turn left, turn right, or hover?

For this task, what we need is not a paragraph analyzing the obstacle's position, flight direction and avoidance strategy, but **a fast choice of the most suitable action based on the current multimodal observations**.

Today's large language models mainly use autoregressive generation, predicting subsequent content one token at a time. When a model makes decisions by generating text, analytical steps or a chain of thought, this processing path can be understood as a form of deliberate reasoning inspired by System 2.

This approach suits complex reasoning, open-ended questions and tasks requiring explanations. For real-time control, game decisions, GUI operations and other finite-candidate decision tasks, however, the conventional generative path is not always the best choice.

System 1 and System 2 originally describe modes of processing in cognitive psychology; they are not strictly equivalent to autoregressive and non-autoregressive models. Here, we mainly compare **making decisions through continuous text generation** with **directly reading candidate decision distributions**.

### 1. Four main limitations of existing autoregressive decision methods

**1. Serial token generation increases decision latency**

Conventional autoregressive generation predicts the next token step by step, with subsequent generation depending on the content already generated.

Even when the final task only requires an action label, a model may generate considerable intermediate text, adding decoding overhead.

For drone obstacle avoidance, robot control and real-time interaction, this waiting time can limit the system's responsiveness.

**2. Natural-language output does not match structured decision requirements**

Conventional generative models typically produce natural language, while decision systems need explicit action IDs, category labels or control commands.

Converting generated text into executable actions may require additional format constraints, text parsing or exception handling.

Natural-language answers may also contain explanations, invalid formats or content outside the candidate set, increasing the complexity of the decision interface.

**3. Complete candidate decision distributions are difficult to obtain directly**

Ordinary text-generation interfaces typically return a generated result rather than a complete probability distribution over all candidate actions.

For example, when a drone chooses between turning left, turning right and hovering, the system may want both the final action and the relative preference among the candidates.

The final generated text alone does not make this information readily available. Obtaining a candidate distribution usually requires an additional mechanism for reading logits or scoring candidates.

**4. General-purpose generation can be inefficient for repetitive decision tasks**

Many repetitive finite-choice tasks require a continuous series of simple, explicit decisions, without generating a complete answer each time.

Building a dedicated decision model may also introduce additional costs for supervised fine-tuning, reinforcement learning, data collection and model maintenance.

This motivates a key research question: **Can we turn the understanding capabilities of existing multimodal backbones directly into structured decisions with low latency, without retraining the models?**

### 2. TianZe-MJev: a System 1-style direct decision framework for multimodal tasks

To address these questions, we propose **TianZe-MJev (天择多模态决策大模型框架)**.

Inspired by System 1's fast decision mechanism, the framework retains the perception and understanding capabilities of existing multimodal models while replacing continuous text generation with direct candidate decisions.

The model receives a task description, environment state, candidate options and images or video frames. It understands the observations through the backbone's native multimodal encoding and fusion mechanisms, then directly computes the decision distribution from the logits corresponding to candidate labels.

The framework returns option IDs, probability distributions and associated structured results without generating a complete natural-language answer.

### 3. Four core advantages of our approach

**1. Direct decisions: substantially reducing autoregressive decoding latency**

To address serial decoding, we read candidate-label logits from a single language-model forward pass, without generating the final answer token by token.

- A single choice, judgment or score requires one language-model forward pass.
- The decision path generates zero new text tokens.
- Necessary input encoding and multimodal feature computation remain, while subsequent continuous text decoding is omitted.

In the existing natural-generation comparison, Gemma 4's response latency decreased from **12,512.1 ms to 248.2 ms**, corresponding to approximately **50.4× speedup and a 98.0% latency reduction**.

**2. Structured output: connecting model decisions directly to applications**

To address the difficulty of executing natural-language results directly, we provide a unified structured decision interface.

Three decision tasks are currently supported:

- `choice`: select directly from 2–26 candidate options.
- `noul`: make a binary true/false judgment about a statement.
- `score`: return an ordinal distribution and expected score.

The system directly returns option IDs and their distributions, reducing dependence on free-text parsing and facilitating integration with robots, game agents and other automated systems.

**3. No additional training: reusing existing multimodal model capabilities directly**

To avoid the training costs of building a dedicated decision model, we adapt frozen backbones for decision tasks.

- Additional decision-training steps: **0**
- Added model parameters: **0**
- Backbone weight updates: **0**

The framework currently supports seven official backbones across the Qwen3.5, Gemma 3n/4, MiniCPM-V and InternVL families.

Existing models can therefore use the unified decision framework without additional decision fine-tuning, lowering the barrier to experimentation and deployment.

**4. Multimodal fusion and candidate distributions: understanding the environment and observing decisions**

The framework retains the backbone's native visual encoding and multimodal fusion capabilities. It supports joint judgments using task text, state information, RGB images and video frames sampled in temporal order.

Unlike calls that return only a single text answer, TianZe-MJev returns relative probability distributions over all candidate options. The web console supports viewing, comparing and exporting decision results.

This provides a unified interface for further research into decision confidence, temporal state understanding and agent control.

The current candidate probabilities express relative preference among the supplied options. They should not be directly equated with calibrated decision accuracy or safety confidence.

### 4. Experimental validation: how much improvement do we observe?

We compared the latency of natural text generation and direct decisions on an NVIDIA A800-SXM4-80GB GPU. Both methods used the same official backbones and task inputs, comparing normal autoregressive answer generation with directly reading candidate decision distributions.

**The experiments show that skipping continuous text generation can substantially reduce decision response time on the tested finite-candidate tasks.**

For accuracy, frozen Qwen3.5-2B-Base achieved **90.33%** on 300 Visual7W questions, close to the Decider-2B-Vision authors' reported 89.00% and numerically higher by 1.33 percentage points. Results on RAVEN and the reconstructed Rune set remain below the corresponding trained-model reports, however. The effectiveness of adaptation without training varies by task, so these results do not establish overall superiority over dedicated trained decision models.

Large latency gains may also accompany differences in answer quality. For example, in the Gemma 4 comparison using neutral prompts, the first direct-decision run answered 30/136 questions correctly, while natural-generation answers scored 95/136 under conservative parsing. In the control experiment requiring a single-letter answer from both methods, latency decreased by 20.6%.

Our core contribution is therefore not to prove that direct decisions are more accurate than autoregressive generation on every task, but to provide:

**A reproducible and comparable path to structured decisions with low latency for existing multimodal models, without additional training or backbone weight changes, together with an explicit analysis of the tradeoff between speed and decision quality.**

### Vision

We want multimodal models to do more than understand images, describe scenes and answer questions: they should also make fast, executable choices for specific tasks.

**From “understand the environment and generate an answer” to “understand the environment and decide directly.”**

## Advantages

- **No additional decision training:** official frozen weights, zero added parameters and zero weight updates.
- **Short decision output path:** one language-model forward per choice, judgment or score; zero generated text tokens.
- **Seven backbones, one interface:** Qwen3.5, Gemma 3n/4, MiniCPM-V and InternVL retain their native visual processing.
- **Traceable inputs and results:** inspect distributions, export JSON, and read per-question evaluation records including failures.

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
git clone https://github.com/maolinqi/multimodal-decision-models.git
cd multimodal-decision-models
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

Maintained by [maolinqi](https://github.com/maolinqi). Share reproducible inputs and issues through [GitHub Issues](https://github.com/maolinqi/multimodal-decision-models/issues); see [Contributing](CONTRIBUTING.md). Backbones are published by Qwen, Google, OpenBMB and OpenGVLab. Decider/Rune reference results belong to their authors.

## License

Our adapters, protocols, API, console, lifecycle scripts and evaluation code use **[Apache-2.0](LICENSE)**. Upstream weights and custom code retain their respective licenses; see [Model licenses](docs/model-licenses.md). `evidence/` contains recorded validation and `docs/` contains protocols and results.
