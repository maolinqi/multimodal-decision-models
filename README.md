<div align="center">

# System One
### Omni-modal System One Decision Models

**Turn multimodal backbones into decision interfaces, with no additional training.**

Open-source multimodal decision interfaces built on Gemma 3n, MiniCPM-V, and InternVL

**With no additional training: MiniCPM-V-4.5 achieves 98% candidate accuracy and 148.9 ms median decision-forward latency on 100 fixed ScienceQA image-test questions.**

**English** · [简体中文](README_zh.md)

[![Code License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml/badge.svg)](https://github.com/maolinqi/multimodal-decision-models/actions/workflows/tests.yml)
[![Training](https://img.shields.io/badge/Additional_Training-0_steps-2563eb)](#training-free-decision-adaptation)
[![Adapters](https://img.shields.io/badge/Native_Adapters-5-2563eb)](#supported-backbones)
[![Paired Decisions](https://img.shields.io/badge/ScienceQA_Pairs-500%2F500_agree-2563eb)](docs/scienceqa-results.md)

[Overview](#overview) · [Training-free method](#training-free-decision-adaptation) · [Inputs and outputs](#multimodal-inputs-structured-outputs) · [Models](#supported-backbones) · [Training-free accuracy](#measured-accuracy-with-no-additional-training) · [Latency](#measured-decision-latency) · [Quick start](#quick-start) · [Open source](#open-source-and-licenses)

</div>

---

## Overview

**System One is an open-source model project for multimodal understanding and low-latency decisions, turning vision-language understanding into structured decisions callable from software.** Given task text, images or video frames, and a question, it returns a candidate choice, binary judgment, or ordinal score with a corresponding probability distribution. Text defines the task and candidates, images provide objects and spatial relationships, and timestamped video frames provide changes over time. These inputs are combined through the backbone’s native multimodal pipeline.

We implemented five decision adapters on **Gemma-3n-E4B / E2B, MiniCPM-V-4.5, and InternVL3.5-8B / 14B**, **using their existing pretrained weights directly, with no additional training or fine-tuning**. Each adapter preserves the backbone's native multimodal encoding, visual fusion, and language head, and reads decisions directly from candidate-label logits. A choice, judgment, or score uses one language-model forward pass and generates **0 new text tokens**, returning the decision distribution directly and reducing the wait associated with token-by-token decoding.

The project provides a shared observation protocol, decision API, and web console, giving all five models the same input and output interface. Runnable code, native adapters, evaluation scripts, and paired validation records are open source. Switch models on one page, inspect candidate distributions, and reproduce comparisons against the native backbone paths.

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

This turns image and video understanding into choices, judgments, and scores callable from software. All five models connect to one console through the same protocol.

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

## Measured accuracy with no additional training

**MiniCPM-V-4.5 achieves 98% candidate-choice accuracy on the fixed ScienceQA image-test subset, with 0 additional training steps.** Gemma E2B / E4B and InternVL 8B achieve 80%, 84%, and 93%, respectively; InternVL 14B achieves 92%.

The evaluation uses **100 image-bearing questions from the official ScienceQA test split, seed 42**, with identical IDs for every model in the table. Inputs contain the question, available hint, image, and options. Native and decision paths share checkpoint, prepared input, prompt, candidate set, precision, and attention implementation. The baseline reads candidate logits from the official generation first step and uses the same candidate softmax and argmax for scoring.

| Model | Additional training | Native candidate accuracy | Decision accuracy | Agreement |
|---|---:|---:|---:|---:|
| Gemma-3n-E2B-it | 0 steps | 80% | 80% | 100% |
| Gemma-3n-E4B-it | 0 steps | 84% | 84% | 100% |
| MiniCPM-V-4.5 | 0 steps | 98% | 98% | 100% |
| InternVL3.5-8B | 0 steps | 93% | 93% | 100% |
| InternVL3.5-14B | 0 steps | 92% | 92% | 100% |

**All 500 paired decisions agree; maximum full-vocabulary logit and candidate-probability differences are 0.** On these tested inputs and settings, training-free adaptation preserves the backbone's native candidate decisions.

[Dataset and protocol](docs/scienceqa.md) · [Fixed manifest](benchmarks/scienceqa-test-100-manifest.json) · [Full results](docs/scienceqa-results.md) · [Per-question records](evidence/scienceqa/)

## Measured decision latency

A choice, judgment, or score reads candidate logits from one forward pass and generates 0 new text tokens. Multimodal encoding and fusion feed directly into a structured distribution, reducing token-by-token decoding and free-text parsing stages.

Measurements use the same fixed ScienceQA subset, 100 image questions per model, with one image and one choice question per input. **Models are loaded before timing; the timer covers visual encoding, fusion, and the language-model forward from prepared tensors to returned logits.** Hardware: NVIDIA A800-SXM4-80GB, BF16, synchronized timing, shared GPU. Median and nearest-rank P95 use all 100 questions.

| Model | Median forward | P95 forward |
|---|---:|---:|
| Gemma-3n-E2B-it | 213.5 ms | 253.1 ms |
| Gemma-3n-E4B-it | 222.8 ms | 275.4 ms |
| MiniCPM-V-4.5 | 148.9 ms | 256.1 ms |
| InternVL3.5-8B | 291.1 ms | 384.7 ms |
| InternVL3.5-14B | 263.8 ms | 440.3 ms |

[Timing details](docs/scienceqa-latency.md) · [Full results](docs/scienceqa-results.md) · [Per-question records](evidence/scienceqa/)

All five backbones also completed paired fixed probes covering color, counting, OCR, spatial relationships, two-image judgments, and temporal changes. See [Native-path validation](docs/base-retention-results.md).


### Frozen-backbone accuracy experiment

Official weights only, no additional training; trained authors' scores are reported references, not reruns. These Qwen/Gemma experimental runners are separate from the five console adapters.

| Backbone | Evaluation | Our correct / questions | Our accuracy | Author trained-model report |
|---|---|---:|---:|---:|
| Qwen3.5-2B-Base | RAVEN | 178/300 | 59.33% | Decider 80% |
| Qwen3.5-2B-Base | Visual7W | 271/300 | 90.33% | Decider 89% |
| Gemma-4-26B-A4B-it | 128 public preview + 8 cards | 90/136 | 66.18% | Rune v3 75.7%, 280 image tokens |

Zero inference exceptions. Exact author samples/rendering are not independently matched; these figures do not establish non-inferiority. All adaptation failures, marker diagnostics and 197 wrong cases are retained. [Detailed results and limitations](docs/frozen-backbone-accuracy.md) · [Raw accuracy evidence](evidence/frozen-backbone/accuracy/)

**Completed one-letter control:** identical initial information and a shared one-letter instruction on 600 Qwen questions, three repetitions each: native generation 118.3 ms versus decision scoring 84.1 ms, reducing latency by 28.9%. Native generation actually emitted two tokens, with no cap hits; both paths agree on all 600 answers. [Protocol and raw timings](docs/frozen-backbone-accuracy.md#completed-minimal-answer-latency-control). Full natural-output timing and Gemma's short-answer control remain pending.

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

Reproduction commands and dataset provenance for the fixed ScienceQA image-test subset are in [Evaluation protocol](docs/scienceqa.md). Recheck the five per-question reports and regenerate accuracy and latency tables with `.venv/bin/python scripts/summarize_scienceqa.py`.

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
