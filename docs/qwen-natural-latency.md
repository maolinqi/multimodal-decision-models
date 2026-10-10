# Qwen3.5: identical-input multiple-choice generation latency

Official frozen Qwen3.5-2B-Base; **600 questions (300 RAVEN + 300 Visual7W), three paired repetitions per path**. Both paths see the same image, state, question, candidates and neutral task prompt. The shared prompt imposes no one-letter response, brevity or explanation requirement. Original source questions are retained, including any source wording about choosing a letter.

| Mode | Native median | Decision median | Reduction | Paired speedup, 95% CI |
|---|---:|---:|---:|---:|
| Multiple-choice prompt, no added output-length instruction | 121.4 ms | 84.4 ms | 30.4% | 1.44× [1.43, 1.69] |
| Separate shared one-letter control | 118.3 ms | 84.1 ms | 28.9% | 1.41× [1.40, 1.48] |

Each question's three repetitions are first collapsed to its median. Confidence intervals resample 600 paired questions (2,000 resamples), not 1,800 independent samples. All timings are retained; no latency outliers are removed.

In the group without an added output-length instruction, native generation emits a median of **two new tokens**, with **zero 2,048-token cap hits**. Its Base model gives short answers under the multiple-choice format and plain `Answer:\n` slot. This is not ordinary conversational generation, and no general long-answer speedup is inferred. Of the 600 first-repetition calls, **346** emit two tokens; the remaining **254** emit 4-10 tokens (for example, `C: Afternoon`). There are **65** source questions containing the word "letter", retained identically in both paths. No explanation or minimum-length instruction is added. Both paths score **443/600 (73.83%)** on each repetition and agree on every answer. These scores belong to the neutral-prompt experiment and differ from the historical explicit-letter accuracy experiment. Deterministic post-hoc formatting parsing is gold-blind and is published separately; original text, parser results and measured rows remain available. Full-vocabulary and candidate logits match exactly on all **600 first-step controls**, with unchanged parameter versions.

A800-SXM4-80GB, BF16, SDPA, Transformers 5.19.0, eight CPU threads. Models are loaded before timing. The synchronized response timer includes image/text preparation, transfer, model compute and answer readout; it excludes model loading, network and queueing. Native generation uses greedy decoding and per-call KV cache; decision scoring generates no text. No cache is reused between calls. Optional Qwen causal-convolution/gated-delta-rule optimization packages are absent for both paths, so these ratios are specific to the recorded runtime. Other jobs on GPU1 share host CPU/storage.

The completed experiment contains **4,200 rows**: 1,800 native generation calls, 1,800 decision calls and 600 first-step controls. Gemma's natural-output experiment is still running and will be reported with termination, accuracy and output lengths before a combined comparison is claimed.

[Raw measurements, prompts hashes, outputs and audit](../evidence/frozen-backbone/latency/qwen/) · [Single-letter control](../evidence/frozen-backbone/latency-short-answer/qwen/) · [Reproduction](../experiments/frozen-backbone/)

[Raw aggregate audit](../evidence/frozen-backbone/published-latency-audit.json) can be regenerated with `python scripts/audit_paired_latency.py`; this checks all currently completed published groups without new inference.
