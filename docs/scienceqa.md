# ScienceQA paired evaluation protocol

[中文](scienceqa_zh.md)

The evaluation uses **100 image-bearing questions from the official ScienceQA test split**, sampled once with seed 42 from numerically sorted problem IDs. All five backbones use these same IDs. Accuracy is the percentage of correct candidate choices on this fixed subset.

The baseline runs the official generation implementation for one step and reads its raw candidate-label logits. The decision adapter directly reads the final-position logits. Each pair shares the checkpoint, prepared input, prompt, options, precision, and attention implementation. Candidate softmax and argmax are identical in both paths. This measures native candidate-decision retention under the project's preprocessing settings.

Only the question, available hint, image, and options are supplied as model input. Answer labels, lectures, and solutions are used for auditing and scoring. The published manifest contains IDs, labels, source revisions, and input hashes; source questions and images remain in the local `data/` directory.

## Reproduction

```bash
.venv/bin/pip install -e '.[benchmark]'
.venv/bin/python scripts/prepare_scienceqa.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compare_base_retention.py gemma-e2b \
  --suite data/scienceqa/suite.json --out evidence/scienceqa/gemma-e2b.json
```

Run the comparison once for each of the five registered model IDs. `scripts/summarize_scienceqa.py` checks completed reports against the frozen manifest, recalculates accuracy from per-question predictions, and renders the table. Per-question incorrect predictions, agreement, full-vocabulary logit differences, and parameter-version checks are retained.

## Sources and licensing

- [ScienceQA authors' repository](https://github.com/lupantech/ScienceQA), revision `2cbf8318e07b9ece895bb2ae605e71e38d623264`.
- [Hugging Face dataset linked by the authors](https://huggingface.co/datasets/derek-thomas/ScienceQA), revision `f18b0a70359ebfb41f658fd564208d0355b013f4`.
- Dataset license: **CC-BY-NC-SA-4.0**, according to the authors. The dataset's terms apply separately from this project's Apache-2.0 code license.

Lu et al. (2022), *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*, NeurIPS.
