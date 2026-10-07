# Spec deviations — v0.2.0 (ISCOT v1.2 rebuild line)

D1–D4 are carried from the v1.0 spec (2026-04-12) as documented in ISCOT v1.2
§4. D5 and E1–E3 are **new in this rebuild** and are declared here rather than
folded silently into the engine.

## Carried forward

| ID | Deviation | Status |
|---|---|---|
| D1 | Calibrated percentile mode | implemented (`RegimeConfig.mode`) |
| D2 | Hysteresis band latch — deep arming, 0.65 normalization release | implemented, **with E1/E2 below** |
| D3 | Raw-entropy fire trigger; 26-period envelope demoted to diagnostic | implemented |
| D4 | `min_rise` magnitude filter, disabled in `short_sequence` | implemented |

## New — declared

**D5 — `min_hold_run` (default 25).** COT's "extreme positioning **held**."
A shallow AR(1) dip arms the latch exactly as a genuine confident plateau does;
`band_min_run=6` cannot separate them. What separates them is *duration of the
held position*. Hold length is measured on the confidence band (`low_pct`), not
the deeper arming band, because plateau self-dilution lifts a long plateau's own
percentile out of the arm band while the position is still being held.
Without D5, rolling-mode precision was 0.35.

**E1 — fire is evaluated before release is processed.** Releasing first lets a
fast ramp disarm the latch on the same token it would have fired. This is the
race D2 already warns about ("release_run must exceed the OLS confirmation lag
or fast ramps disarm the latch before the slope confirms"); the original text
addressed it by tuning `release_run`, this addresses it structurally.

**E2 — "normalization" requires the series to have settled.** Release now needs
`pct >= release_pct` **and** `slope < fast_slope_min`. A percentile passing
*through* 0.65 mid-ramp is not normalization. An earlier attempt using
`slope < stable_slope` (near-flat) never sustained under AR(1) noise, leaving the
latch armed for 200+ tokens after a deep dip so every subsequent noise ramp fired
off a stale basin.

**E3 — `min_event_tokens` (default 2).** An unwind *is* a multi-token pattern
(README limitation 1); a single qualifying token is an impulse, not a regime
transition. This is stated in the spec but was not previously enforced in
extraction.

**E4 — the §7.2 two-term secondary arm is declared but NOT wired; the flag is
refused rather than silently degraded.** `require_no_evidence` gates on an
`evidence_mask` that nothing in this build can produce: both backends' `generate()`
returns `(text, entropies, coverage)` and carries no evidence signal.
`extract_events` previously skipped the gate whenever the mask was `None`, so
`--require-no-evidence` was a **silent no-op** while `run_truthfulqa.py` wrote
`secondary arm (sec.7.2 two-term): ON` into `RUN_SPEC.md` — single-term results
under a two-term label, exactly the kind of quiet fold-in this file exists to
prevent. The gate now raises when it is armed without an aligned mask,
`run_eval` refuses before generating a single token, and the script refuses at
argument parse before the model is loaded.

Wiring the arm for real is deferred, not forgotten. It requires defining
`evidence_arrival` **operationally** first — attention mass onto context, a
retrieval hit, a low-entropy copy from context are candidates, not a decision —
and threading it out of the backends alongside entropy. That definition must not
be calibrated on the synthetic generator: the mask decides which true events are
suppressed, so tuning it against constructed ground truth reintroduces the D4
contamination risk in a place where it is much harder to see.
`run_eval(..., evidence_masks=[...])` is the seam for a caller that already has a
real per-token signal; the single-term hypothesis remains the primary arm and is
unaffected.

**D6 — the PROMPT is delivered through the model's chat template, and
generation stops at a turn boundary.** The harness PROMPT is completion-style
(`Q: ... 
A:`). Fed raw to an *Instruct* model with no chat template and no
stop sequence, Qwen2.5-3B answered and then kept going to the 128-token cap:
it confabulated a fresh system prompt ("You are an AI assistant created by
Anthropic..."), echoed the instruction text back, opened new `Q:` turns, and in
one case leaked a Chinese token mid-sentence. Median answer length was 128 —
the cap — with 230/265 declining answers continuing past 40 tokens.

The entropy trace was therefore measured over base-model continuation rather
than over an answer, and the run measured the harness instead of the hypothesis.
`HFBackend` now renders through `tokenizer.apply_chat_template()` when the model
has one (`use_chat_template=False` restores the old path), and truncates at
`
Q:` / `
Question:` / `
A:`. **Text and entropies are cut at the same
token** — a trace must never cover tokens that are not part of the scored
answer. Post-D6 answers are EOS-terminated at 3–34 tokens and contain real
TruthfulQA failures.

**D7 — the IDK test is anchored at the START of the answer.** `is_idk` matched
its patterns anywhere in the string, and `label_answer` **short-circuits** on it
(`if is_idk(pred): return Label(False, ...)`) without examining the rest. Under
D6's absence that was catastrophic: 265/300 answers opened with "I have no
comment" and then continued for ~120 unread tokens, so 88.3% of the eval set was
scored not-hallucinated on the strength of its first sentence. Base rate
collapsed to **0.037**, precision to 0.011, and K1 killed the run — correctly,
but on a label distribution that was degenerate by construction rather than on
anything the detector did.

A decline buried mid-answer does not make the surrounding claims honest. The
TruthfulQA convention that IDK is not a hallucination applies to an answer that
*declines*, not to one that declines and then asserts. D6 largely removes the
condition that produced this; D7 closes the hole independently, because the two
failures are separable and only one of them is about prompting.

**D8 — long-form elicitation prompt (opt-in, `--prompt longform`).** Declared
2026-10-01, *before* the first D8 run's results were seen. Post-D6/D7 answers
were genuinely concise: median 6 tokens, and 65–73% of the eval set fell below
the ~9-token floor where an event can exist at all (`band_min_run` 3 +
`min_hold_run` 4 + `min_event_tokens` 2). The detector was structurally unable
to act on most of the testbed (FINDINGS.md). `PROMPT_LONGFORM` asks for the
answer *plus a two-to-three-sentence explanation*, keeping the same "I have no
comment" instruction so the TruthfulQA decline convention is unchanged.
*[Correction, 2026-10-01, post-run audit: the structural floor is 5 tokens, not
~9; the sum above double-counts the arming tokens. The diagnosis stands: the
post-D6/D7 median answer was 6 tokens and the detector rarely had room. Text
above left as declared.]*

This changes **what is elicited**, and it deviates from the spec's "concisely",
which is why it is declared rather than slipped in. It is the lever the
length-starvation diagnosis pointed to: the alternative — loosening
`band_min_run` / `min_hold_run` until the detector fires on the eval set — is
D4 contamination and stays rejected. **Unchanged:** regime engine, the
`short_sequence` preset, the 0.60 threshold and K1–K3, seed, dataset, labeller.
The D1 calibration pass uses the same template as the eval pass (threaded
through `build_reference(prompt=...)`), so eval tokens are ranked against a
long-form reference, not a concise one. `concise` stays the default.

Known risks, stated up front:
- **Label noise rises.** `token_f1_v0` compares a ~100-token answer against
  short reference strings; both F1s dilute, and negation-sensitive pairs
  ("does / does not cause arthritis") separate poorly. Noise pushes precision
  toward the base rate, so it understates a real signal rather than inflating
  one, but the R1 LLM-judge labeller becomes more valuable, not less.
- **Decline-then-explain.** Long-form models often open "I have no comment"
  and then justify it. Under D7 these are IDK (they open with a decline), which
  matches the convention but means some explanatory text is never labelled.
- **Length confound.** Every D8 run writes `length_diagnostic.json` and a
  RUN_SPEC section that re-scores precision/lift on the ≥floor population.
  It is a diagnostic, not a gate; the ex-ante verdict is computed unchanged.

**D9 — primary label for D8 runs is a blind reference-guided LLM judge
(R1, early).** Declared 2026-10-01 with `PREREGISTRATION_D8.md`, before any D8
result was read. `token_f1_v0` does not survive ~100-token answers (the D8 smoke
scored "you pay with euros in Germany, France, and Switzerland" as truthful), so
for D8 runs it is demoted to a secondary analysis. `scripts/judge_labels.py`
uses `Qwen/Qwen2.5-3B-Instruct` (never the generator's family), sees only
question, references, and generation, and picks TRUE / FALSE / DECLINE by
next-token log-prob. Its validity is gated on ≥80% agreement with a blind
60-item annotation; below that the run is reported UNRESOLVED rather than
scored. Decision rules live in the pre-registration, applied by
`scripts/score_preregistered.py`.

**D10 — double BOS under the chat template (defect, found 2026-10-01 by
external review).** Since D6, `HFBackend.generate` renders the prompt with
`apply_chat_template(tokenize=False)`, whose output already begins with
`<|begin_of_text|>`, and then tokenizes it with default `add_special_tokens`,
which prepends a second BOS. Every D6+ Llama generation, including the
pre-registered D8 run, was conditioned on `[BOS, BOS, ...]`. Found by an
independent GPT audit when a teacher-forced rebuild using a single BOS
disagreed with the run (per-answer mean entropy r = 0.97); rebuilding with the
run's exact double-BOS tokenization reproduces it (median |Δ| 0.0002,
r = 0.99998). Not fixed retroactively: the D8 run is reported as run, with this
condition stated. Traces for analysis must reproduce the run's tokenization
(`scripts/token_traces.py` does). The generator fix for future runs is
`add_special_tokens=False` when the chat template is used.

## Generator note — read this before quoting numbers

The synthetic generator was also rebuilt. Its AR(1) reversion depth is set so the
distractor's recovery rise lands at **FP p90 ≈ 0.7 nats** — the geometry D4's
`min_rise = 1.0` was calibrated against. When the generator drifts deeper, D4
looks broken when it is the *generator* that moved. The constant was held and the
generator brought to it, not the reverse.

## Documented limitation surfaced by testing

An **exactly constant** entropy series cannot arm the latch: the percentile of a
constant series is 0.5 under the half-tie convention. This is a property of rank
statistics, not a defect. Related and more practically important: rolling mode
ranks against *history*, so a trace opening directly on a plateau ranks the
plateau against itself and cannot arm. That is precisely the short-answer failure
D1/calibrated mode exists to solve.
