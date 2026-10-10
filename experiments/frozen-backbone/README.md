# Frozen-backbone accuracy and identical-input latency experiments

These experimental runners evaluate Qwen3.5-2B-Base and Gemma-4-26B-A4B-it. Their benchmark input protocols are recorded separately from the console interfaces. Original official weights are frozen, with no extra training or adapter parameters.

Use this directory's isolated environment: Linux x86-64, Python 3.12, `uv`, and a sufficiently free NVIDIA GPU. The recorded environment is in `requirements.lock.txt`; `setup.sh` creates only this directory's `.venv` and installs the recorded dependency versions. Install Python 3.12 and `uv` before invoking it.

```bash
cd experiments/frozen-backbone
./setup.sh
export BENCH_MODEL_ROOT=/path/to/official-models
# Accept upstream model access terms before downloading Gemma.
.venv/bin/python download_models.py
.venv/bin/python verify_weights.py
.venv/bin/python build_qwen_projected.py
.venv/bin/python audit_qwen_tokens.py
.venv/bin/python build_gemma_examples.py
.venv/bin/python build_gemma_reconstruction.py
.venv/bin/python build_gemma_clarity_v3.py
.venv/bin/python build_gemma_tables_v4.py
.venv/bin/python build_gemma_gallery_v5.py
.venv/bin/python build_gemma_normalized_v6.py
# Use an idle GPU UUID, obtained from nvidia-smi. Gemma admission requires 62 GiB free.
export CUDA_VISIBLE_DEVICES=GPU-REPLACE_WITH_YOUR_IDLE_GPU_UUID
.venv/bin/python evaluate_qwen_v2.py --model qwen --records data/qwen/records.jsonl --run-name qwen-v2-600
.venv/bin/python evaluate.py --model gemma --records data/gemma-normalized-v6/records.jsonl --run-name gemma-normalized-v6-136
.venv/bin/python benchmark_natural_latency.py --model qwen --records data/qwen/records.jsonl --repeats 3
.venv/bin/python benchmark_natural_latency.py --model gemma --records data/gemma-normalized-v6/records.jsonl --run-name gemma-v6 --repeats 3
```

Every timed native answer-only generation, decision scoring call, and one-token raw-logit control receives the **same image, state, question, options, prompt and prepared-input format**. No side receives extra information or a request for additional reasoning. The final natural-output experiment removes the single-letter instruction from BOTH paths. Native generation uses greedy decoding and normal EOS, with a 2,048-token emergency cap; cap hits and truncations are retained. The earlier benchmark_identical_latency.py is a separate short-answer control and is not the natural-output experiment. Decision scoring returns a candidate distribution without generating text. Same-prompt hashes are verified per question. Model load, network and service queue time are excluded; normal input preparation, synchronized computation and answer readout are included. One-token retention controls time preprocessing and native computation; their extra validation forward is outside timing.

The scripts preserve every repetition and failure. Question-level paired comparisons use three repetition medians and paired bootstrap intervals. All native generation outputs, actual token counts, invalid answers and truncations are retained. Weight parameter versions must remain unchanged. The first-step control checks candidate/raw-logit retention; it is not a free-text accuracy equivalence claim.

Source datasets, published metadata and model revisions are pinned in `reference/`. Builders use hf-mirror.com for dataset I/O in the recorded server environment. Download transport can be changed separately from revision and integrity checks. Source images are reconstructed, not redistributed. GUI distractors/crops and rendered tables have not been matched to the trained authors' final pixels. Author figures are reported references, not rerun models or paired non-inferiority evidence.

The benchmark and evaluator source is preserved; portability changes use `BENCH_MODEL_ROOT` rather than the original server's model directory. Model weights and image assets are intentionally absent from Git. See `NOTICE.md`, the repository license and upstream model/dataset terms.

The current Gemma v6 presentation adds an explicit A–E candidate-panel mapping and 455 original structured table-cell strings to shared state. Both inference paths receive this same state and the gallery/table images. Cell text is obtained from structured dataset values, not an automatically validated screenshot OCR pipeline. All original questions/options/labels remain; this presentation differs from the author reference. Historical coordinate and shuffled-letter audits are retained, including failed gates.

To verify published completed groups without rerunning models, run `python scripts/audit_paired_latency.py` from the repository root. It checks every raw call, paired prompt/token counts, repetitions, termination counts, first-step differences and recomputed medians/reduction. Question repetitions remain distinct from independent question count; confidence intervals are retained from the original source summaries.
