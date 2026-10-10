# Identical-input multiple-choice generation versus decision latency

Measured on the frozen official Gemma 4 backbone, with no additional training. **Both sides receive exactly the same image, state, question, candidates and prompt.** Every question is measured three times per path, in randomized paired order. Both paths use the same neutral task prompt, without an added one-letter, brevity or explanation instruction. Source question text is retained, including source questions that themselves ask for a letter. Gemma uses the same native instruction template with `enable_thinking=False` on both paths. This experiment does not compare against Gemma’s thinking-enabled mode. Native generation uses normal EOS with a disclosed emergency cap; capped calls are retained.

| Backbone | Questions | Native observed-response median | Decision median | Difference | Speedup, paired 95% CI | Latency reduction |
|---|---:|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 136 | 12512.1 ms | 248.2 ms | 12263.9 ms | 50.40× [47.28, 56.37] | 98.0% |

Medians in this table are computed after collapsing each question's three repetitions to its per-path median. Confidence intervals resample paired questions (2,000 bootstrap resamples), rather than treating repetitions as independent questions. The raw timing records retain every repetition; nearest-rank P95 and model-compute/input-preparation timings are also in each summary. No latency outliers are trimmed.



## What was timed

NVIDIA A800-SXM4-80GB, BF16, SDPA, one loaded model on benchmark GPU0, synchronized CUDA timing, eight CPU threads. The response timer includes image reading/preparation, text tokenization, transfer, model computation and result readout. Model loading, HTTP/network and queue time are excluded. The earlier single-letter-prompt experiment is historical and does not enter this natural-output comparison. No changed-prompt explanation group is included.

Native `generate` uses greedy decoding and normal EOS, with a 2,048-token emergency cap. Every cap hit is retained and disclosed; capped durations must not be described as complete-answer latency. Decision scoring uses one forward and returns a candidate distribution with zero generated text tokens. Native generation uses its normal within-call KV cache; the decision path does not allocate an unused decoding cache. Neither path reuses a cache between requests. The latency gain applies to this finite-choice output workflow; model weights and visual encoding are preserved. These ratios are specific to this implementation and hardware, not a guarantee for fully optimized deployment engines.

## Accuracy and actual output length

| Backbone | Decision correct, all repetitions | Native conservatively parsed-answer correct, all repetitions | Native unparsed outputs | Native median generated tokens | Native cap hits |
|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 90/408 (22.06%) | 285/408 (69.85%) | 84 | 240 | 3 |

| Backbone | Decision correct, first repetition | Native parsed correct, first repetition | Native/decision answer agreement | Stable prediction across three repetitions, both paths |
|---|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 30/136 | 95/136 | 23/136 | 272/272 |

Accuracy repetitions are not independent test questions. The timing runner's original parser does not recognize some Markdown/colon answer formats. The tables here use the separately published gold-blind post-hoc formatting audit (leading letter, unique explicit answer/choice label, or exact option text), which leaves original timing rows and timings unchanged. Natural-output parsing is conservative; ambiguous outputs remain unresolved, so parsed accuracy is not a human-adjudicated free-answer metric. The natural prompt differs from the historical single-letter accuracy protocol below. Per-question outputs and all unparsed/truncated answers remain in the raw evidence. The first-step raw-logit control runs once on every question and matches candidate decisions on **136/136** inputs; this checks the decision readout against native first-step logits, not equivalence of arbitrary generated answers or all backbone capabilities. Full/candidate logit differences and unchanged parameter-version checks are recorded in the summaries.

## Per-task quality and response time

Correctness uses the first repetition; latency uses each question's three-repetition medians. Native unparsed responses count as incorrect in this conservative parsed metric. The two exclusive-correct counts expose quality tradeoffs; they do not assert human-adjudicated correctness or statistical superiority. Capped questions in the last column refer to first-repetition native calls.

| Backbone / task | Questions | Decision correct | Native parsed correct | Decision-only / native-only correct | Native / decision observed median | Capped questions |
|---|---:|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it / ArxivQA | 20 | 3/20 | 11/20 | 1/9 | 15384.3/282.0 ms | 0 |
| Gemma-4-26B-A4B-it / CLEVR-HOPE | 20 | 9/20 | 17/20 | 2/10 | 9985.6/232.1 ms | 0 |
| Gemma-4-26B-A4B-it / FinQA | 20 | 1/20 | 12/20 | 0/11 | 12390.7/247.0 ms | 0 |
| Gemma-4-26B-A4B-it / Geometry3K | 20 | 2/20 | 20/20 | 0/18 | 20095.4/227.9 ms | 0 |
| Gemma-4-26B-A4B-it / Multimodal-Mind2Web | 12 | 3/12 | 2/12 | 3/2 | 13852.3/266.9 ms | 1 |
| Gemma-4-26B-A4B-it / ScreenSpot | 36 | 8/36 | 25/36 | 5/22 | 9868.7/253.6 ms | 0 |
| Gemma-4-26B-A4B-it / public_example_cards | 8 | 4/8 | 8/8 | 0/4 | 6231.1/236.2 ms | 0 |

## Actual generation length

| Backbone | Minimum tokens | Median tokens | Mean tokens | P95 tokens | Maximum tokens |
|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 27 | 240 | 288.7 | 615 | 2048 |

Lengths count all newly generated token IDs, including terminal special tokens. No minimum output length is imposed, and naturally short outputs are retained. These are measured lengths, not a requested explanation workload.

## EOS-completed subset (secondary diagnostic)

The primary table retains every question, including capped responses. This secondary table includes only questions for which all three native repetitions terminate below the cap; both paths are restricted to the same IDs. It is a completion-conditioned subset, not an unbiased replacement of the full benchmark.

| Backbone | EOS-completed questions / all | Native median | Decision median | Reduction |
|---|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 135/136 | 12491.3 ms | 248.2 ms | 98.0% |

## Shared single-letter prompt control

Both paths receive the same explicit single-letter request here. This is a separate minimal-generation control, not part of the neutral-prompt table. The cap is 32 new tokens; capped calls remain in observed-response timing. No capped response is claimed to be a complete answer.

| Backbone | Questions | Native observed median | Decision median | Reduction | Capped native calls |
|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 136 | 308.1 ms | 244.6 ms | 20.6% | 36/408 |

## Comparison with trained authors' reports

Qwen's corrected frozen-base scorer: RAVEN 178/300 (59.33%), against Decider's reported 80%; Visual7W 271/300 (90.33%), against 89%. Gemma's normalized-v6 input version, with shared gallery-position mapping and 455 original table-cell strings: 93/136 (68.38%), against Rune v3's reported 75.7% at 280 image tokens. Authors' trained weights were not run or downloaded for this experiment. These are non-paired references: exact author sample revisions/IDs for Qwen, and final rendering/candidate/crop pixels for Gemma, have not been matched. No non-inferiority or overall superiority is established.

Qwen's original colon-only answer prefix did not preserve token boundaries for 3,300/3,600 candidate checks; the corrected newline prefix passes all 3,600 checks. Original results are retained as historical diagnostic evidence. Gemma marker and legend correction changed 76/136 to 92/136; a subsequent table-only clipping fix reduced this to 90/136 (two right-to-wrong cases, zero wrong-to-right). The latest normalization preserves all questions/options/labels and adds shared observation metadata; it reaches 93/136. Original table strings come from structured dataset cells, not demonstrated automatic screenshot OCR. These input channels differ from the author presentation.

## Marker-readability limitations

Forty-eight reconstructed GUI screenshots preserve all candidate-target region pixels and have an explicit uniform explanation of labels/target boxes. Their 720 diagnostic probes yield isolated same-resolution letter recognition 237/240 (98.75%), full-screenshot coordinate-conditioned letter recognition 190/240 (79.17%), and target-to-label association 188/240 (78.33%). Coordinate probes also test localization and option matching; these failures are not all pure OCR failures. External input factors have **not** been completely excluded. Eight published cards retain the author's original pixels and are outside the 48-screenshot protection audit. One source FinQA item has an empty correct option and remains in the denominator, disclosed rather than silently removed.

These publicly exposed/reconstructed questions are diagnostic sets, not sealed blind tests. Candidate probabilities are relative preferences, not calibrated correctness probabilities. Avoid claiming additional-training-free superiority from latency alone.

[Raw measurements and protocols](../evidence/frozen-backbone/latency/) · [Accuracy records](../evidence/frozen-backbone/accuracy/) · [Input and marker diagnostics](../evidence/frozen-backbone/diagnostics/) · [Reproduction scripts](../experiments/frozen-backbone/)

Author references: [Decider vision model card](https://github.com/Mapika/decider/blob/e50e549b47e2da69223734fee4efa1ddd4528e93/MODEL_CARD_VISION.md), [Rune v3 model card](https://huggingface.co/surogate/rune-26b-a4b-GGUF).
