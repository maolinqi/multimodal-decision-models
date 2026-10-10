# Qwen3.5-2B-Base decision adapter

`qwen35-2b` is the sixth registered adapter. It uses the official **Base** checkpoint at revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, without training, fine-tuning, extra parameters or an Instruct substitution. Native image tokens, visual fusion and the language head are preserved. The answer prefix ends in a newline; every request checks that each candidate letter remains one token in context.

## Verified results

- Fixed ScienceQA image-test subset: **82/100** for both native first-step candidate scoring and decision scoring; **100/100** candidate answers agree, full-vocabulary and candidate logit differences are zero, parameter versions unchanged.
- Shared 20-probe suite: **16/20** for both paths, **20/20** agreement, including single-image, multi-image and temporal inputs. Incorrect native answers remain in the report.
- Isolated API: both synthetic left/right image requests correct, distributions normalized, visual tokens present, default model selection and explicit unload verified. `choice`, `noul`, `score` and proposal-only `velocity4` interface probes completed.
- Separate Decider-reference experiment: Visual7W **271/300 (90.33%)** versus reported 89%; RAVEN **178/300 (59.33%)** versus reported 80%. The RAVEN gap remains **20.67 percentage points**. These are reported references under separately recorded input protocols, not a paired trained-model rerun.

The ScienceQA baseline is native **candidate scoring**, not arbitrary free-text generation. These tests establish the measured inference-path agreement, not universal capability preservation, calibrated probabilities or physical control success.

[ScienceQA evidence](../evidence/scienceqa/qwen35-2b.json) · [20 paired probes](../evidence/retention/qwen35-2b.json) · [API check](../evidence/qwen35-api-validation.json) · [Typed interface check](../evidence/qwen35-interface.json) · [Decider/Rune reference experiment](frozen-backbone-accuracy.md)

## Isolated runtime

The original five adapters retain `.venv` with Transformers 4.57.1. Qwen3.5 uses project-local `.venv-qwen35` with Transformers 5.19.0; do not upgrade the existing environment or install its pinned project dependencies into the new runtime.

```bash
./setup.sh                       # only for a fresh checkout without the existing .venv
./scripts/setup_qwen35.sh         # initializes the separate runtime
.venv/bin/python scripts/download_models.py qwen35-2b
export MODEL_ROOT="$PWD/models"
export CUDA_VISIBLE_DEVICES=GPU-YOUR_AVAILABLE_GPU_UUID
./start_all.sh
```

The console exposes `qwen35-2b`. Its isolated backend listens on loopback port **8460**, while the existing backend and gateway retain 8457/8456. Set `QWEN35_DECISION_URL` to route an independently deployed Qwen3.5 backend. This is distinct from the pre-existing optional `QWEN_DECISION_URL` integration for Qwen3-VL-4B. Model loading occurs on the first request; `stop_all.sh` stops the new project-owned service as well as the original services. The existing project image-root restrictions and memory admission checks apply.

For standalone validation:

```bash
export PYTHONPATH="$PWD/src"
.venv-qwen35/bin/python scripts/validate_model.py qwen35-2b --out evidence/qwen35-interface.json
.venv-qwen35/bin/python scripts/compare_base_retention.py qwen35-2b --out evidence/retention/qwen35-2b.json
.venv-qwen35/bin/python scripts/validate_qwen35_api.py
```

The Qwen runtime uses PyTorch fallback implementations for optional causal-convolution/gated-delta-rule kernels in the recorded deployment. Performance is specific to this runtime and shared A800 hardware. The paired response-latency experiment separately records full preprocessing and actual generation lengths; ScienceQA forward timing alone is not a generation-speedup comparison.

To release only this Qwen3.5 backend before loading another large model, send `POST http://127.0.0.1:8460/unload` with an empty JSON object. Unload refuses while the backend is busy; it does not stop other project jobs.
