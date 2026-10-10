# Frozen-backbone accuracy against authors' reported trained models

This separate experiment uses official frozen **Qwen3.5-2B-Base** and **Gemma-4-26B-A4B-it**, without additional training or adapters. The trained Decider/Rune weights were not downloaded or run. Their reported scores are references, not our measurements. Qwen3.5 is now registered as the sixth console adapter; the benchmark runners record their own input protocols.

| Official backbone | Evaluation | Correct / questions | Frozen decision accuracy | Author trained-model report | Difference, percentage points |
|---|---|---:|---:|---:|---:|
| Qwen3.5-2B-Base | RAVEN | 178/300 | 59.33% | Decider 80% | −20.67 |
| Qwen3.5-2B-Base | Visual7W | 271/300 | 90.33% | Decider 89% | +1.33 |
| Gemma-4-26B-A4B-it | 128 public preview items + 8 published cards | 93/136 | 68.38% | Rune v3 75.7%, 280 image tokens | −7.32 |

All 736 questions were scored, with zero inference exceptions. This does **not** establish paired non-inferiority: author sample revisions/IDs for Qwen and final image/candidate/crop pixels for Gemma are not independently matched. Visual7W is numerically close to the reported reference; RAVEN remains 20.67 points below its reference; Gemma is 7.32 points below its reference.

## Reconstruction and input corrections

Qwen uses the pinned author's ordering/sampling procedure and Cauldron revision, not a selected high-scoring subset. Its first colon-only answer prefix failed 3,300/3,600 token-boundary checks; the corrected newline prefix passes all 3,600 checks. Historical predictions and the audit remain public.

Gemma's 136 questions reconstruct Rune's public 128-preview-plus-8-card scope. They are not Rune's private expanded 1,196-question dataset. Image budget is 280 tokens. Source questions, options and labels are retained; GUI distractors/crops and FinQA tables are deterministic reconstructions rather than verified author pixels. The eight published cards preserve their original hashes.

Historical Gemma accuracy was 76/136 (55.88%). Larger offset markers, hollow target boxes and a uniform legend raised it to 92/136 (67.65%). Restoring full unclipped FinQA table text subsequently lowered it to 90/136 (66.18%): two right-to-wrong cases and zero wrong-to-right cases. That historical experiment uses complete tables rather than selecting the earlier higher score. One source FinQA correct option is empty; the question is retained and disclosed.

The shared accuracy prompt requests the most suitable uppercase option letter. This accuracy protocol is separate from the later natural-output latency protocol, which removes that requirement from **both** paths and remeasures accuracy rather than reusing these scores.

## Latest shared observation normalization (v6)

The current result is **93/136 (68.38%)**, with no inference errors. All questions, options and gold labels are retained. Both inference paths receive the same normalized observations:

- GUI images contain the unmarked source screenshot and five equally formatted candidate-centered crops. A–E appear below crops, outside target pixels. Shared state explicitly maps options to the first through fifth crop and its click point, avoiding dependence on leader tracking or letter OCR.
- FinQA keeps complete table images and includes all **455 original dataset table-cell strings** as shared structured observations. This is a transcription from existing structured dataset cells, not a demonstrated automatic screenshot-OCR capability.

No answer or rationale is added to these observations. Compared with v4, FinQA rises from 5 to 8/20 and Mind2Web from 5 to 7/12; ScreenSpot decreases from 31 to 29/36. The net gain is three correct answers. Presentation changes limit comparison with the reported trained-model score. The latest Gemma predictions retain all 43 errors; the earlier 46-error ledger remains historical. Nine answers change from wrong to right and six from right to wrong. [Current 194-case ledger](../evidence/frozen-backbone/accuracy-error-ledger-v6.jsonl) retains Qwen’s 151 and Gemma’s 43 errors without causal attribution.

[Latest results](../evidence/frozen-backbone/accuracy/gemma-normalized-v6/) · [Shared-input audit](../evidence/frozen-backbone/diagnostics/normalized-v6/)

The gallery-v5 diagnostic also remains public: coordinate-conditioned identification scores 146/240 and click-center association 135/240, despite 240/240 isolated letters. With diagnostic-only shuffled labels and ordinal questions, full-gallery reading is 234/240 and association 236/240; the strict all-label gate still fails. These failures motivated explicit shared position metadata in v6 rather than assuming OCR/association were solved. [Gallery audits](../evidence/frozen-backbone/diagnostics/gemma-gallery-ordinal-audit-v5/).

## Historical marker meaning and recognition

A–E identify candidate click positions, not function names or confidence. A black letter inside a white/red circle connects by a red leader to a hollow red target box. The click is at the target box center, not at the letter circle. Distractors use the same style; no gold answer is included in the legend.

All 48 reconstructed GUI screenshots restore candidate-target rectangles plus a three-pixel buffer exactly, including the blue translation button's pixels. Markers are at least 28 processed font pixels and 44 processed diameter pixels at the actual 280-token resize. Pixel preservation does not prove recognition.

| Diagnostic, 48 screenshots × five labels | Correct / probes | Accuracy |
|---|---:|---:|
| Isolated letter, same canvas and resize | 237/240 | 98.75% |
| Full screenshot, coordinate-conditioned letter | 190/240 | 79.17% |
| Target-to-letter association | 188/240 | 78.33% |

Zero inference errors across all 720 probes. Full-screenshot checks include localization/association, not just OCR. Among the 12 wrong reconstructed GUI task answers, 10 fail at least one of the three **gold-label** diagnostic checks and two pass all three. This is an association, not a causal attribution, and a gold-label pass does not prove recognition of every option or correct task understanding. The eight original cards are outside the 48-image target-pixel audit. External factors remain unresolved.

The historical v4 ledger retains 197 errors: Qwen 122 RAVEN + 29 Visual7W; Gemma 46. No failed question is removed. Candidate probabilities are relative preferences and are not calibrated correctness probabilities. These public/reconstructed sets are diagnostic, not sealed blind tests.

## Completed minimal-answer latency control

Under **exactly the same image, question, state, options and one-letter prompt**, Qwen's 600 questions were measured three times per path on one A800. Native generation median: **118.3 ms**; decision scoring: **84.1 ms**; difference **34.1 ms**, reduction **28.9%**, speedup **1.41×** (paired bootstrap 95% interval 1.40–1.48×). Timers include preparation, synchronized model compute and readout; model load/network/queue are excluded. Actual native length was two new tokens (letter and EOS), with zero 32-token cap hits. All 600 per-question answers agree across both paths; first-step raw/candidate logits match exactly on all 600 controls. Qwen was measured across two sessions, with provenance retained. Optional causal-convolution/gated-delta-rule optimization packages were absent for both paths; results are runtime-specific.

This demonstrates reduced latency even when generation itself is minimal. It is separate from the multiple-choice generation experiment without an added output-length instruction; no long-answer speedup is inferred here. Gemma's normalized-v6 same-prompt control is also complete: **136 questions × three repetitions**, native observed response median **308.1 ms** versus decision **244.6 ms**, a **20.6%** reduction (1.26×). All 136 first-step candidate and full-vocabulary comparisons match exactly. However, native generation hits the **32-token cap on 36/408 calls (12/136 questions)**; the reported time includes those truncated calls and is not a full-answer completion-time claim. The current automatic parser scores 88/136 native answers versus 93/136 decision answers; post-hoc formatting audit will be reported separately, retaining every truncation. [Gemma raw short-answer control](../evidence/frozen-backbone/latency-short-answer/gemma/). The Qwen multiple-choice generation experiment without an added output-length instruction is complete and Gemma's is running; the combined report will include lengths, termination and accuracy.

[Accuracy records](../evidence/frozen-backbone/accuracy/) · [Error ledger](../evidence/frozen-backbone/accuracy-error-ledger.jsonl) · [Diagnostic records](../evidence/frozen-backbone/diagnostics/) · [Qwen minimal-answer timing evidence](../evidence/frozen-backbone/latency-short-answer/qwen/) · [Reproduction](../experiments/frozen-backbone/)

Author references: [Decider model card](https://github.com/Mapika/decider/blob/e50e549b47e2da69223734fee4efa1ddd4528e93/MODEL_CARD_VISION.md), [Rune v3 card](https://huggingface.co/surogate/rune-26b-a4b-GGUF).
