<div align="center">

# System One
### Omni-modal System One Decision Models

**See the environment. Understand the task. Make an explicit choice.**

Open-source multimodal decision interfaces built on Gemma 3n, MiniCPM-V, and InternVL

**English** · [简体中文](README_zh.md)

[![Code License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml/badge.svg)](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml)
[![Adapters](https://img.shields.io/badge/Native_Adapters-5-2563eb)](#supported-backbones)
[![Paired Probes](https://img.shields.io/badge/Paired_Probes-100%2F100_agree-2563eb)](docs/base-retention-results.md)

[Overview](#overview) · [Method](#direct-decision-readout) · [Models](#supported-backbones) · [Evaluation](#base-model-comparison) · [Quick start](#quick-start) · [Open source](#open-source-and-licenses)

</div>

---

## Overview

A red target appears on the left of an image. The system asks: “Which side is the target on?”

The answer needs to enter the next step of a program: a stable option ID, a distribution over alternatives, and an inspectable record of the observation and inference.

**System One connects multimodal understanding to this decision interface.**

Building on our Qwen3-VL decision-console work, we implemented a shared observation and decision protocol for Gemma-3n-E4B / E2B, MiniCPM-V-4.5, and InternVL3.5-8B / 14B. Text, images, and timestamped video frames pass through each backbone’s native multimodal pipeline and become choices, binary judgments, or ordinal scores.

> **v0.1 supports:** text, RGB images, timestamped video frames, and five native model adapters.

## Direct decision readout

Here, **System One** describes the path from the current observation to a structured decision. A candidate decision uses one language-model forward pass: read the candidate-label logits at the final position, then normalize over the candidate set.

**The production decision path generates 0 new text tokens and returns a structured candidate distribution.**

| Decision type | Question | Output |
|---|---|---|
| `choice` | Which of 2–26 defined options? | Stable option ID and candidate probabilities |
| `noul` | Does this statement hold? | true/false and probabilities |
| `score` | Where on an ordered scale? | Level distribution and expected score |

Candidate probabilities express relative preference within the supplied options.

## What we implemented

Gemma, MiniCPM, and InternVL encode and fuse visual inputs differently. We built a native adapter for each family, preserved its input pipeline, and connected it to the same decision protocol and console.

| Component | Implementation |
|---|---|
| **Native model adapters** | Gemma processor / forward; MiniCPM image slicing, visual resampling, and embedding fusion; InternVL dynamic tiling and visual-token fusion |
| **Shared observation protocol** | RGB frames organized by camera and timestamp, with field and numeric validation |
| **Shared decision readout** | Single-token candidate labels read through the backbone’s existing language head |
| **Unified console and API** | Switch among five models on one page; connect a compatible Qwen service; load models on demand with one resident model per backend |
| **Reproducible validation** | Fixed inputs, native-path comparisons, protocol tests, GPU records, and incorrect predictions published together |

## Supported backbones

| model_id | Official backbone |
|---|---|
| `gemma-e4b` | [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) |
| `gemma-e2b` | [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) |
| `minicpm-v45` | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) |
| `internvl35-8b` | [OpenGVLab/InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) |
| `internvl35-14b` | [OpenGVLab/InternVL3_5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) |

These decision adapters use the official backbone weights and native language heads. An existing compatible Qwen service can be connected through `QWEN_DECISION_URL`.

## Base-model comparison

We used paired tests to check whether visual fusion and decision readout preserve the backbone’s native computation.

Each backbone ran the same **20 fixed probes** covering text arithmetic and logic, colors, counting, OCR, spatial relations, two-image judgments, and temporal changes. Every input was evaluated through the official generation first step and our decision forward path, with checkpoint, prepared input, prompt, candidate set, precision, and attention implementation held constant.

| Model | Native correct | Decision correct | Decision agreement | Max full-vocabulary logit difference |
|---|---:|---:|---:|---:|
| Gemma-3n-E2B-it | 16/20 | 16/20 | 100% | 0 |
| Gemma-3n-E4B-it | 16/20 | 16/20 | 100% | 0 |
| MiniCPM-V-4.5 | 20/20 | 20/20 | 100% | 0 |
| InternVL3.5-8B | 19/20 | 19/20 | 100% | 0 |
| InternVL3.5-14B | 20/20 | 20/20 | 100% | 0 |

**All 100 paired decisions agreed. Maximum differences in full-vocabulary logits and candidate probabilities were both 0.** Parameter objects and version counters remained unchanged during evaluation. Incorrect native predictions are included in the reports.

**On these tested inputs and settings, the adapters preserved native first-step computation and candidate decisions.** The evaluation compares each model against its own native path under identical conditions; the linked protocol specifies inputs and scoring.

[Protocol](docs/base-retention.md) · [Results](docs/base-retention-results.md) · [Fixed inputs](benchmarks/base_retention_v1.json) · [Raw records](evidence/retention/)

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

Set `MODEL_ROOT=/path/to/models` to reuse downloaded weights. Downloads and startup use the same variable. One backend loads on demand and keeps one model resident at a time. Free-memory admission thresholds are 16 GiB for E2B, 20 GiB for E4B, 22 GiB for MiniCPM / InternVL 8B, and 34 GiB for InternVL 14B. Longer inputs can require more memory.

The input budget is 8,192 tokens with bounded image tiling. The console uses the checkout’s `web/` directory; retain the checkout and editable installation. Services bind to localhost by default.

</details>

## Reproduce validation

```bash
.venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compare_base_retention.py minicpm-v45 \
  --out evidence/retention/minicpm-v45.json
.venv/bin/python scripts/summarize_retention.py
```

**11 CPU protocol tests passed.** The final command checks and aggregates the five completed paired reports. Per-model interface checks are implemented in `scripts/validate_model.py`; recorded results are in [Validation](docs/validation.md).

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
