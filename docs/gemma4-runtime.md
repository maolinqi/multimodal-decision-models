# Gemma 4 26B A4B decision adapter

`gemma4-a4b` uses official `google/gemma-4-26B-A4B-it` revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, BF16, native image processing and language head, without additional training or adapter parameters. The native instruction template uses `enable_thinking=False`; decision readout uses one forward and zero generated text tokens. Candidate token boundaries are checked per request. Image budget is 280 soft tokens, matching the recorded Rune-reference experiment.

## Verified scope

- Fixed ScienceQA image-test subset: **89/100** on both paths, **100/100** agreement, zero full-vocabulary/candidate/probability differences, unchanged parameter versions.
- Fixed 20-probe suite: **20/20** correct on both paths, including multi-image and temporal probes, **20/20** agreement and zero full-vocabulary differences.
- Real isolated API: two synthetic left/right images correct; default model selection, normalized distribution, visual tokens and unload verified.
- Typed interface: choice, binary judgment, score and proposal-only four-axis outputs passed; two image probes match native first-step candidate logits exactly.
- Separate normalized Rune-reference benchmark: **93/136 (68.38%)**, versus reported Rune v3 **75.7%**, a **7.32-point gap**. No trained Rune weights were run. All 136 same-input native first-step controls match the benchmark scorer's full-vocabulary/candidate logits exactly.

The reference benchmark uses 128 public preview questions plus eight published cards, not the author's private 1,196-question set. GUI panel mappings and 455 original structured table-cell strings are shared observations in the latest presentation. They reduce reading ambiguity, but differ from the author's input presentation and do not establish paired non-inferiority. This is a finite-choice decision interface; it does not establish calibrated probabilities, universal capability retention or physical control success.

[API evidence](../evidence/gemma4-api-validation.json) · [Typed interface](../evidence/gemma4-interface.json) · [Rune-reference accuracy](frozen-backbone-accuracy.md) · [First-step controls and short-answer timings](../evidence/frozen-backbone/latency-short-answer/gemma/)

## Isolated runtime and lifecycle

The original five adapters retain their existing Transformers 4.57.1 runtime. Gemma 4 uses project-local `.venv-gemma4` with Transformers 5.19.0; Qwen3.5 retains its separate `.venv-qwen35`. Startup uses existing environments and does not install dependencies.

```bash
./scripts/setup_gemma4.sh
.venv/bin/python scripts/download_models.py gemma4-a4b
export MODEL_ROOT="$PWD/models"
export CUDA_VISIBLE_DEVICES=GPU-YOUR_AVAILABLE_GPU_UUID
./start_all.sh
```

The Gemma 4 backend listens on loopback **8461**, routed by model ID `gemma4-a4b`. `GEMMA4_DECISION_URL` can override its backend URL. Existing backend/gateway ports remain 8457/8456. Model load occurs on first request and requires at least 62 GiB free GPU memory. `stop_all.sh` stops only project-owned services. Image-root restrictions and observation validation are inherited from the shared protocol.

```bash
export PYTHONPATH="$PWD/src"
.venv-gemma4/bin/python scripts/validate_gemma4_api.py
.venv-gemma4/bin/python scripts/validate_model.py gemma4-a4b --out evidence/gemma4-interface.json
.venv-gemma4/bin/python scripts/compare_base_retention.py gemma4-a4b --out evidence/retention/gemma4-a4b.json
.venv-gemma4/bin/python scripts/compare_base_retention.py gemma4-a4b --suite data/scienceqa/suite.json --out evidence/scienceqa/gemma4-a4b.json
```

The expanded [20-probe](../evidence/retention/gemma4-a4b.json) and [fixed ScienceQA](../evidence/scienceqa/gemma4-a4b.json) reports are complete and verified.

To release only this Gemma 4 backend before loading another large model, send `POST http://127.0.0.1:8461/unload` with an empty JSON object. Unload refuses while the backend is busy; it does not stop other project jobs.
