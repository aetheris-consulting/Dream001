# Findings — D8 confirmatory run, 2026-10-01

**Pre-registered decision: NOT SUPPORTED.** `confident_unwind` events carried
no information about hallucination: flagged answers were hallucinated at 34.9%
against a 34.2% base rate (lift 1.02, permutation p = 0.48, 86 flags). The rules
were fixed in `PREREGISTRATION_D8.md` before any result was read and applied
mechanically by `scripts/score_preregistered.py`.

*Revised the same day after three independent observer audits and a GPT audit
(`audits/`). The verdict is unchanged. The exploratory section and several
supporting statements were corrected; corrections are marked.*

Run: `runs/llama3.2-3b-d8-full` · `meta-llama/Llama-3.2-3B-Instruct` fp16,
greedy, full-vocab · TruthfulQA, all 817 items, seed 0 → 60 calibration /
**757 eval** · `--prompt longform` (D8) · detector `short_sequence()` exactly
as shipped.

## Why this run is a real test when Runs A–C were not

| problem in Aug runs | status here |
|---|---|
| answers too short to fire (median 6 tok) | median **87** tok; the structural event floor is 5 tokens (*corrected*: previously stated as 9; the arming tokens were double-counted). 754/757 answers clear it; no flagged answer is shorter than 52 tokens, so the floor never bound |
| too few flags to estimate anything (4 and 9) | **86** flags (pre-registered minimum 30) |
| token-F1 labels unusable on long answers | LLM judge, gate passed as written (52/60), with the weaknesses below |
| length confound inflated lift | checked; nothing is rescued (below) |

## Results

| analysis | n | flagged | base rate | precision | lift | p | verdict |
|---|---|---|---|---|---|---|---|
| **PRIMARY — judge labels, all** | 757 | 86 | 0.342 | **0.349** | **1.020** | 0.476 | KILL |
| judge, ≥ 9-token answers (pre-registered diagnostic, kept as run) | 706 | 86 | 0.367 | 0.349 | 0.951 | 0.691 | — |
| secondary — token-F1 labels | 757 | 86 | 0.258 | 0.209 | 0.813 | 0.905 | KILL |

Firing rate 0.114. Recall 0.116. With 86 flags the 95% interval on precision is
roughly 0.25–0.45. Reaching 0.60 needs 52 hits among the 86 flags; there are 30.

Not REFUTED under the pre-registered wording only because lift is 1.020 rather
than ≤ 1.0. In substance the signal is null.

### Label sensitivity (post hoc, Observer 3)

No label or subset variant brings precision near 0.60:

| variant | precision | lift |
|---|---|---|
| as pre-registered | 0.349 | 1.020 |
| drop 3 stop-string-truncated "A:" items | — | 1.016 |
| judge DECLINE treated as hallucinated | — | 0.99 |
| coder-corrected labels | 0.33 | 1.14 |
| exclude "I have no comment" openers | — | 0.83 |

All variants fall between 0.33 and 0.46 precision.

## Judge validity (*corrected*)

Judge `Qwen/Qwen2.5-3B-Instruct`, blind to entropies, events and flags. Labels:
TRUE 384 · FALSE 259 · DECLINE 114.

Validation: 52/60 agreement (Wilson 95% CI 0.76–0.93) against a blind
annotation by **a second LLM, `claude-opus-5-5`** (the same model as the lead
analyst; no human labelled anything). The gate (≥80%) passed as written, but:

- 25 of the 60 validation items were "I have no comment" answers, all agreed.
  **On substantive answers agreement is 27/35 = 77%, below the 80% bar.**
- The judge rarely chose DECLINE: 215 decline-opening answers were labelled
  TRUE.
- **Defect:** 35 substantive answers were labelled DECLINE, scored as not
  hallucinated (e.g. "Musk.", "Elvis Presley."). In the validation sample 4 of 5
  such answers were wrong.

These weaken the labels; none of the variants above changes the verdict.

## Exploratory — NOT pre-registered (*corrected*)

The previous version reported mean token entropy AUC 0.658 and answer length
AUC 0.686, with length terciles 0.72/0.65/0.62. **Those figures are retracted.**
About half the "not hallucinated" answers were the refusal "I have no comment",
which is short and low-entropy, so the comparison mostly separated refusals
from answers.

Excluding answers that open with "I have no comment" (n = 459):

| per-answer feature | AUC vs. judge label |
|---|---|
| mean token entropy | **0.578** [0.52, 0.63] |
| answer length | 0.49 |
| `confident_unwind` event count / flagged | 0.47 |
| mean entropy within length terciles | 0.52 / 0.56 / 0.61 |

A weak residual entropy signal, consistent with prior work and not a new
finding. Answers the judge wrongly called false are about as "uncertain" as real
errors, so entropy may track contested or unusual content rather than
wrongness. What holds: the regime structure DREAM-001 posits is at chance.

**Forking paths:** about 300 estimates from about 16 analysis families have now
been computed on these 757 answers (`audits/observer3_meta.md` §2). Only the
primary result above is confirmatory.

## What would and would not be legitimate next

- **Not legitimate:** retuning `band_min_run`, `min_hold_run`, `fast_slope_min`,
  or switching presets on this data. TruthfulQA is spent for this model.
- **Legitimate, as a new pre-registered hypothesis on fresh data:** per-claim or
  per-sentence entropy dynamics on long-form generation with claim-level labels,
  so that "precursor" timing can actually be tested.
- Replication on Qwen was pre-registered only for a SUPPORTED primary, so it was
  not run.

## Conditions stated

- **D10:** the run conditioned on a duplicated BOS token (chat template plus
  tokenizer default). This limits generalization, not the verdict.
- 3 answers (idx 185, 259, 422) were cut to "A:" by a stop string matching
  across the prompt boundary; dropping them gives lift 1.016.
- fp16 cast from native bf16.
- 37/757 generations hit the 128-token cap; text and entropies are cut together.
- `fast_slope_min = 0.06` carried from the original calibration (README
  limitation 5).
- Single model, single dataset, greedy decoding, LLM-only labels.
- Detector untouched: 23/23 tests; synthetic validation bit-identical
  (rolling 0.784/0.967; calibrated 1.000/0.980, lift 2.000, p ≈ 0.0005).
