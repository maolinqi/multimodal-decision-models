<div align="center">

# System One
### Omni-modal System One Decision Models

**Turn multimodal backbones into decision interfaces, with no additional training.**

Open-source multimodal decision interfaces built on Qwen3.5, Gemma 3n/4, MiniCPM-V, and InternVL

**English** · [简体中文](README_zh.md)

[![Code License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml/badge.svg)](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml)
[![Training](https://img.shields.io/badge/Additional_Training-0_steps-2563eb)](#training-free-decision-adaptation)
[![Adapters](https://img.shields.io/badge/Native_Adapters-7-2563eb)](#supported-backbones)
[![Paired Decisions](https://img.shields.io/badge/ScienceQA_Pairs-700%2F700_agree-2563eb)](docs/scienceqa-results.md)

[Overview](#overview) · [Training-free method](#training-free-decision-adaptation) · [Inputs and outputs](#multimodal-inputs-structured-outputs) · [Models](#supported-backbones) · [Training-free accuracy](#measured-accuracy-with-no-additional-training) · [Latency](#measured-decision-latency) · [Quick start](#quick-start) · [Open source](#open-source-and-licenses)

</div>

---

## Overview

**System One turns multimodal backbones into structured decision interfaces, with no additional training or fine-tuning.** Given text, images or video frames, and candidate options, it returns a choice, binary judgment, or ordinal score with candidate probabilities. Seven backbones share one API and web console, with evaluation records included.

## Measured accuracy with no additional training

| Backbone | Benchmark (questions) | Trained model | [Reported accuracy](docs/frozen-backbone-accuracy.md) | Ours (0 training) | Delta (pp) |
|---|---|---|---:|---:|---:|
| Qwen3.5-2B-Base | Visual7W (300) | Decider-2B-Vision | 89.00% | **90.33%** | +1.33 |
| Gemma-4-26B-A4B-it | Rune public reconstruction (136) | Rune v3 | 75.70% | **68.38%** | -7.32 |

### ScienceQA (fixed 100 questions)

| Backbone | Extra training | Native candidate accuracy | Ours | Agreement |
|---|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 0 | 89% | 89% | 100/100 |
| Qwen3.5-2B-Base | 0 | 82% | 82% | 100/100 |
| Gemma-3n-E2B-it | 0 | 80% | 80% | 100/100 |
| Gemma-3n-E4B-it | 0 | 84% | 84% | 100/100 |
| MiniCPM-V-4.5 | 0 | 98% | 98% | 100/100 |
| InternVL3.5-8B | 0 | 93% | 93% | 100/100 |
| InternVL3.5-14B | 0 | 92% | 92% | 100/100 |

[Accuracy evidence](docs/frozen-backbone-accuracy.md) · [ScienceQA protocol and results](docs/scienceqa.md)

## Measured decision latency

| Base model | Original response mode | Test set (questions) | Original output length (tokens, median) | Before conversion (ms, median) | After conversion (ms, median) | Latency saved (ms) | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | Natural generation | Rune (136) | 240 | 12512.1[†](docs/identical-input-latency.md#what-was-timed) | **248.2** | 12263.9 | **50.40×** |
| Gemma-3n-E4B-it | Natural generation | ScienceQA (100) | 62.5 | 4880.7 | **152.3** | 4728.5 | **32.06×** |

[Per-question results](docs/identical-input-latency.md) · [ScienceQA response comparison](docs/instruction-latency.md)

<details>
<summary>ScienceQA decision-forward timings</summary>

| Backbone | Median forward (ms) | P95 forward (ms) |
|---|---:|---:|
| Gemma-4-26B-A4B-it | 264.3 | 333.0 |
| Qwen3.5-2B-Base | 90.4 | 121.4 |
| Gemma-3n-E2B-it | 213.5 | 253.1 |
| Gemma-3n-E4B-it | 222.8 | 275.4 |
| MiniCPM-V-4.5 | 148.9 | 256.1 |
| InternVL3.5-8B | 291.1 | 384.7 |
| InternVL3.5-14B | 263.8 | 440.3 |

[Full results](docs/scienceqa-latency.md)

</details>

## Training-free decision adaptation

The adaptation connects existing multimodal understanding to a shared decision interface: preserve native visual encoding and fusion, map options to single-token labels, read candidate logits through the existing language head, and normalize them into a candidate distribution.

| Property | Implemented in this release |
|---|---|
| Additional training steps | **0** |
| Added model parameters | **0**; reuse the native language head |
| Backbone weight updates | **0**; load official weights directly |
| One choice, judgment, or score | **1 language-model forward pass** |
| Generated text in the decision path | **0 new tokens** |
| Output | Stable option IDs, candidate probabilities, and input provenance |

The candidate distribution is computed from the backbone’s label logits for the current input:

$$p_i = \frac{\exp(z_{t_i})}{\sum_{j=1}^{K}\exp(z_{t_j})}$$

Here, $z_{t_i}$ is the logit for candidate $i$’s label token and $K$ is the number of candidates.

## Multimodal inputs, structured outputs

- **Joint visual and textual judgment:** images provide visual content; text provides the task, context, and candidate options. Both enter the decision pipeline.
- **Multi-frame temporal context:** support multiple RGB images and up to eight timestamped video frames, retaining camera IDs and temporal order.
- **Software-ready results:** read judgments into stable IDs and candidate distributions for selection, binary judgment, and ordinal scoring.

| Decision type | Question | Output |
|---|---|---|
| `choice` | Which of 2–26 defined options? | Stable option ID and candidate probabilities |
| `noul` | Does this statement hold? | true/false and probabilities |
| `score` | Where on an ordered scale? | Level distribution and expected score |

Candidate probabilities express relative preference within the supplied options.

## What we implemented

Qwen, Gemma, MiniCPM, and InternVL encode and fuse visual inputs differently. We built a native adapter for each family, preserved its input pipeline, and connected it to the same decision protocol and console.

| Component | Implementation |
|---|---|
| **Native model adapters** | Qwen native image tokens and stable Base answer boundary; Gemma processor / forward; MiniCPM image slicing, visual resampling, and embedding fusion; InternVL dynamic tiling and visual-token fusion |
| **Shared observation protocol** | RGB frames organized by camera and timestamp, with field and numeric validation |
| **Shared decision readout** | Single-token candidate labels read through the backbone’s existing language head |
| **Unified console and API** | Switch among seven models on one page; connect a compatible Qwen service; load models on demand with one resident model per backend |
| **Reproducible validation** | Fixed inputs, native-path comparisons, protocol tests, GPU records, and incorrect predictions published together |

## Supported backbones

| model_id | Official backbone |
|---|---|
| `qwen35-2b` | [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) |
| `gemma4-a4b` | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| `gemma-e4b` | [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) |
| `gemma-e2b` | [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) |
| `minicpm-v45` | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) |
| `internvl35-8b` | [OpenGVLab/InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) |
| `internvl35-14b` | [OpenGVLab/InternVL3_5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) |


## Unified decision console

Upload an image or video clip, enter a question and candidate options, switch models, and inspect or export the decision distribution. Video input is sampled into up to eight timestamped RGB frames.

Choice, binary judgment, and ordinal scoring are available through the shared API. Visual selection, event judgments, and ordered-level assessment use the same protocol.

## Quick start

Linux, Python 3.12, and an NVIDIA GPU. Tested with PyTorch 2.8.0, Transformers 4.57.1, and BF16. MiniCPM-V-4.5 example:

```bash
git clone https://github.com/maolinqi/multimodal-decision-models.git
cd multimodal-decision-models
./setup.sh
.venv/bin/python scripts/download_models.py minicpm-v45
CUDA_VISIBLE_DEVICES=0 ./start_all.sh
```

Open the [local console](http://127.0.0.1:8456) and select a downloaded model. Video processing uses system `ffmpeg` and `ffprobe`. Stop services with `./stop_all.sh`. Startup uses the existing project environment.

```bash
curl -H 'Content-Type: application/json' \
  --data-binary @examples/choice.json http://127.0.0.1:8457/decide
```

<details>
<summary>Other models, existing weights, and GPU memory requirements</summary>

For Gemma, accept the upstream terms on Hugging Face and log in:

```bash
.venv/bin/hf auth login
.venv/bin/python scripts/download_models.py gemma-e2b gemma-e4b internvl35-8b internvl35-14b
```

Set `MODEL_ROOT=/path/to/models` to reuse downloaded weights. Downloads and startup use the same variable. Each backend loads on demand and keeps one model resident at a time. Free-memory admission thresholds are 16 GiB for E2B, 20 GiB for E4B, 22 GiB for MiniCPM / InternVL 8B, and 34 GiB for InternVL 14B. Longer inputs can require more memory.

The input budget is 8,192 tokens with bounded image tiling. The console uses the checkout’s `web/` directory; retain the checkout and editable installation. Services bind to localhost by default.

</details>

For the two newly validated backbones, initialize their separate runtimes and download the pinned official weights:

```bash
./scripts/setup_qwen35.sh
./scripts/setup_gemma4.sh
.venv/bin/python scripts/download_models.py qwen35-2b gemma4-a4b
CUDA_VISIBLE_DEVICES=GPU-YOUR_AVAILABLE_GPU_UUID ./start_all.sh
```

The Qwen3.5 and Gemma 4 services use ports 8460 and 8461; free-memory admission is 8 GiB and 62 GiB respectively. Each backend manages its own resident model. See [Qwen3.5](docs/qwen35-runtime.md) and [Gemma 4](docs/gemma4-runtime.md) for API routing, unload and verification commands.

## Reproduce validation

```bash
.venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compare_base_retention.py minicpm-v45 \
  --out evidence/retention/minicpm-v45.json
.venv/bin/python scripts/summarize_retention.py
```

**11 CPU protocol tests passed.** The final command checks and aggregates the seven completed paired reports. Per-model interface checks are implemented in `scripts/validate_model.py`; recorded results are in [Validation](docs/validation.md).

Reproduction commands and dataset provenance for the fixed ScienceQA image-test subset are in [Evaluation protocol](docs/scienceqa.md). Recheck the seven per-question reports and regenerate accuracy and latency tables with `.venv/bin/python scripts/summarize_scienceqa.py`.

## Open source and licenses

**The runnable implementation is open source.** Download it, run it, inspect the adapters, and reproduce the comparisons:

```text
src/multimodal_decision/  Native adapters, protocols, checks, API, and web gateway
web/                     Unified decision console
scripts/                 Downloads, interface validation, paired evaluation, and summaries
benchmarks/              Fixed-input probes
evidence/                Recorded validation, including incorrect predictions
tests/                   Protocol and input-validation tests
docs/                    Architecture, evaluation methods, and results
setup.sh / start_all.sh / stop_all.sh
```

Our interface code is licensed under **Apache-2.0**. Official weights and upstream custom code follow their respective licenses; see [Model licenses](docs/model-licenses.md). Weights are downloaded from the upstream repositories.

Contributions with reproducible inputs, model adapters, and validation improvements are welcome. See [Contributing](CONTRIBUTING.md).
