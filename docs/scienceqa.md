# ScienceQA paired evaluation protocol

[中文](scienceqa_zh.md)

The evaluation uses **100 image-bearing questions from the official ScienceQA test split**, sampled once with seed 42 from numerically sorted problem IDs. All seven backbones use these same IDs. Accuracy is the percentage of correct candidate choices on this fixed subset.

The baseline runs the official generation implementation for one step and reads its raw candidate-label logits. The decision adapter directly reads the final-position logits. Each pair shares the checkpoint, prepared input, prompt, options, precision, and attention implementation. Candidate softmax and argmax are identical in both paths. This measures native candidate-decision retention under the project's preprocessing settings.

Only the question, available hint, image, and options are supplied as model input. Answer labels, lectures, and solutions are used for auditing and scoring. The published manifest contains IDs, labels, source revisions, and input hashes; source questions and images remain in the local `data/` directory.

## Reproduction

```bash
.venv/bin/pip install -e '.[benchmark]'
.venv/bin/python scripts/prepare_scienceqa.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compare_base_retention.py gemma-e2b \
  --suite data/scienceqa/suite.json --out evidence/scienceqa/gemma-e2b.json
```

Run the comparison once for each of the seven registered model IDs. `scripts/summarize_scienceqa.py` checks completed reports against the frozen manifest, recalculates accuracy from per-question predictions, and renders the table. Per-question incorrect predictions, agreement, full-vocabulary logit differences, and parameter-version checks are retained.

## Sources and licensing

- [ScienceQA authors' repository](https://github.com/lupantech/ScienceQA), revision `2cbf8318e07b9ece895bb2ae605e71e38d623264`.
- [Hugging Face dataset linked by the authors](https://huggingface.co/datasets/derek-thomas/ScienceQA), revision `f18b0a70359ebfb41f658fd564208d0355b013f4`.
- Dataset license: **CC-BY-NC-SA-4.0**, according to the authors. The dataset's terms apply separately from this project's Apache-2.0 code license.

Lu et al. (2022), *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*, NeurIPS.

## Timing

Each record includes synchronized `decision_forward_ms`, covering visual encoding, fusion, and the language-model forward from prepared tensors to returned logits. Models are loaded before measurement. Median and nearest-rank P95 are calculated from all 100 questions per model. Hardware: NVIDIA A800-SXM4-80GB, BF16.

Qwen3.5 uses the separate Transformers 5.19.0 runtime: `PYTHONPATH=src .venv-qwen35/bin/python scripts/compare_base_retention.py qwen35-2b --suite data/scienceqa/suite.json --out evidence/scienceqa/qwen35-2b.json`. Existing adapters keep their Transformers 4.57.1 environment.

Gemma 4 uses `.venv-gemma4` with Transformers 5.19.0: `PYTHONPATH=src .venv-gemma4/bin/python scripts/compare_base_retention.py gemma4-a4b --suite data/scienceqa/suite.json --out evidence/scienceqa/gemma4-a4b.json`. It uses 280 image soft tokens; see [runtime details](gemma4-runtime.md).
