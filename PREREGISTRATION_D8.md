# Pre-registration — DREAM-001 D8 confirmatory run

**Written 2026-10-01, before any D8 eval result was observed.** The only D8 data
seen at the time of writing is the 12-item smoke (`runs/smoke-d8`), used to
confirm the prompt lengthens answers. A 300-item run was started and stopped
before producing output; its results were never read.

Nothing below may be changed after the run. Any later analysis not listed here
is exploratory and must be labelled as such wherever it is quoted.

## Hypothesis under test

`confident_unwind` entropy-regime events in an answer's token trace are
precursors of hallucination: answers containing ≥1 qualifying event are more
likely to be hallucinated than answers without one.

## Fixed design

| | |
|---|---|
| generator | `meta-llama/Llama-3.2-3B-Instruct`, fp16, greedy, full-vocab `hf` backend |
| prompt | `longform` (D8) for calibration and eval alike |
| data | TruthfulQA `generation`, all 817 items, shuffled with seed 0 |
| split | first 60 → D1 calibration reference (excluded from metrics); remaining **757 → eval** |
| detector | `RegimeConfig.short_sequence()` exactly as shipped; no parameter changes |
| scoring | answer-level any-hit (§7.1) |
| gates | `verdict()` unchanged: precision ≥ 0.60, K1 lift > 1, K2 (p < 0.05, lift > 1.15), K3 firing < 0.95 |

The full set is used rather than 300 items because the smoke fired on 1 of 12
answers; at that rate 300 items gives roughly 25 flags, too few to estimate
precision to better than ±0.2.

## Labels — primary is the LLM judge (D9), not token-F1

Token-F1 against short reference strings does not survive ~100-token answers
(the smoke labelled "you pay with euros in Germany, France, and Switzerland" as
truthful). The **primary label** is therefore a blind reference-guided judge:

- judge model `Qwen/Qwen2.5-3B-Instruct` (different family from the generator), fp16, greedy
- judge input: question, TruthfulQA correct answers, incorrect answers, the generation
- judge never sees entropies, events, or flags
- output TRUE / FALSE / DECLINE; **hallucinated ⇔ FALSE**; DECLINE is not a
  hallucination (TruthfulQA convention)

**Judge validity check.** Before the judge labels are joined to detector flags,
a random sample of 60 eval items (seed 1) is labelled independently by a second
annotator (Claude, reading question + references + answer only, flags hidden).
If judge–annotator agreement on hallucinated/not is **below 80%**, the judge is
declared invalid, the primary analysis is reported as **UNRESOLVED (labels)**,
and no verdict is claimed from it.

`token_f1_v0` labels are reported as a **secondary** analysis.

## Decision rules

Applied to the primary (judge-labelled) analysis on all 757 eval items:

- **SUPPORTED** — `verdict()` returns PASS **and** the length-diagnostic
  re-score on answers ≥ the event floor also has lift > 1.15 with permutation
  p < 0.05. The second condition rules out the length artefact that produced
  the Run B/C lifts.
- **REFUTED (on this testbed)** — K1: lift ≤ 1, with ≥ 30 flagged answers so
  the null is not just low power.
- **NOT SUPPORTED** — anything else: precision below 0.60, K2 inconclusive,
  or a PASS whose lift disappears among capable answers. Reported with the
  numbers, not as a refutation.
- **UNDERPOWERED** — fewer than 30 flagged answers. No claim either way.

## Replication

Only if the primary is SUPPORTED: the same protocol on `Qwen/Qwen2.5-3B-Instruct`
as generator (judge then swaps to the cached `unsloth/Llama-3.2-3B-Instruct`
mirror, so judge and generator never share a family). A support claim without
replication is reported as "single-model".

## Explicitly forbidden after seeing results

Changing any `RegimeConfig` value, the prompt, the judge prompt, the judge
model, the label threshold, the eval set, the seed, or the 0.60 / 1.15 / 0.05
thresholds. Dropping items. Reporting the token-F1 analysis as primary if it
looks better.
