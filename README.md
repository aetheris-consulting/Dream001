# DREAM-001 — Logit Entropy Regime Model

**Reference implementation, v0.2.0** · MIT · Aetheris Consulting LLC

Provenance: ISCOT v1.2 (Internal State Commitment of Traders), session
`ISCOT-AEF-2026-0412`, disposition **REFINE → PROTOTYPE**.

Lineage: v0.1.0 (2026-07-06, ARCHON, D1–D4) → ISCOT v1.2 cross-model rebuild
(D5, E1–E3) → **v0.2.0** pre-run hardening. See `CHANGELOG.md`.

> **Status (2026-10-01): hypothesis NOT SUPPORTED.** A pre-registered run
> (`PREREGISTRATION_D8.md`; Llama-3.2-3B-Instruct, 757 TruthfulQA items) found
> flagged answers hallucinated at 34.9% against a 34.2% base rate (lift 1.02,
> p = 0.48). The verdict held under independent audits (`audits/`). Results,
> conditions and corrections: `FINDINGS_D8.md`. Exploratory follow-up
> (hypotheses only): `HAZARD_CARDS.md`.

## Hypothesis

> `confident_unwind` entropy regime signatures in LLM logits are detectable
> precursors of hallucination events.

Buildups are noise; unwinds are predictive. It is not the peak of confidence
that predicts failure, it is the inflection where coherence begins fragmenting
while confidence still presents as high.

**Publishable threshold:** ≥60% precision on TruthfulQA with an open-weight
model — subject to the honesty gates, because bare precision is gameable.

### Kill conditions

- **K1** — lift ≤ 1: the signal carries no information.
- **K2** — precision > 0.60 but p ≥ 0.05 or lift ≤ 1.15: INCONCLUSIVE, not a pass.
- **K3** — firing rate ≈ 1: trivial sensitivity, not a precursor.

`verdict()` applies these mechanically. A run clearing 0.60 without clearing the
gates prints *"DO NOT REPORT THIS AS A PASS."*

## What is validated, and what is not

**Validated (synthetic, constructed ground truth):** the *detector*.

| Mode | Level | Precision | Recall | Notes |
|---|---|---|---|---|
| rolling (spec defaults) | event | 0.784 | 0.967 | median latency 11 tokens; 60×1200 |
| calibrated (short_sequence) | sequence | 1.000 | 0.980 | lift 2.00, permutation p ≈ 0.0005; 100 traces |

**Not validated:** the hypothesis. Whether real hallucinations are preceded by
these signatures is exactly what the GPU run decides. These numbers say nothing
about it and must never be quoted as if they do.

Reproduce: `python scripts/validate_synthetic.py`

## Quickstart

```bash
pip install -e .                      # numpy + datasets
python scripts/validate_synthetic.py
python -m pytest tests/ -q

# The hypothesis test (GPU):
pip install -e ".[hf]"
python scripts/run_truthfulqa.py --model meta-llama/Llama-3.2-3B-Instruct
# D8 long-form elicitation (declared deviation; see SPEC_DEVIATIONS.md):
python scripts/run_truthfulqa.py --model meta-llama/Llama-3.2-3B-Instruct --prompt longform

# DREAM-006 fallback tier (any OpenAI-compatible endpoint with logprobs):
python scripts/run_truthfulqa.py --api http://localhost:8000/v1 --model <name>
```

Model choice: **3B at fp16, not 8B at 4-bit.** Quantization distorts the logit
distribution that *is* the signal; a quantized precision number is
uninterpretable.

## ISCOT mapping

| COT (markets) | DREAM-001 (LLM internals) |
|---|---|
| net position ratio | token-level logit entropy (nats) |
| rolling 104-week percentile | rolling 104-token percentile / per-model reference (D1) |
| extreme positioning **held** | armed band latch + `min_hold_run` (D5) |
| unwind of extreme positioning | genuine entropy ramp out of an armed band |
| 26-week MA trend filter | 26-token EMA — **diagnostic only** (D3) |
| buildups = noise, unwinds = predictive | `confident_buildup` ignored; `confident_unwind` is the alarm |

## Scoring — answer level (§7.1)

Token-level events vs answer-level ground truth is not an engineering annoyance;
it is the mismatch the framework predicts — **the event is a transition, the
label is a level.** An answer is *flagged* if ≥1 qualifying event occurs in its
span. Token events are retained as diagnostics, never scored on.

## Two-term feature (§7.2) — secondary arm, default OFF

Models commit confidently to true statements constantly. If unwind rate is
comparable across correct and incorrect answers, unwind-per-se is not the signal.
The discriminating feature is an unwind **unaccompanied by evidence arrival**:

```
fire  ⟺  confident_unwind(t)  ∧  ¬evidence_arrival(t−k … t)
```

**Status: declared, not wired — `--require-no-evidence` is refused, not run.**
The gate needs a per-token `evidence_mask` and no backend emits one
(`generate()` returns text, entropies, coverage). Rather than let the flag
degrade to the single-term detector under a two-term label, `extract_events`,
`run_eval` and the script all refuse when it is armed without a mask
(`SPEC_DEVIATIONS.md` E4). A caller holding a real signal can pass
`run_eval(..., evidence_masks=[...])`.

Wiring it requires defining `evidence_arrival` operationally first. **Do not
tune it on synthetic data** — that reintroduces the D4 contamination risk. The
single-term hypothesis is the primary arm and is tested cleanly regardless.

## Deviations

See `SPEC_DEVIATIONS.md`. D1–D4 carried forward; **D5 and E1–E3 are new in this
rebuild** and are declared, not folded in silently.

## Known limitations

1. Better for extended confabulation than atomic single-token factual errors.
2. Rolling window goes stale on topic shifts; calibrated mode sidesteps this for
   short answers only.
3. DREAM-006 top-k tier lags internal state and underestimates entropy —
   coverage is reported per token so the bias is visible, never silent.
4. Token-F1 labels are noisy both ways; noise dilutes toward the base rate, so a
   surviving signal is understated.
5. `min_rise` and `fast_slope_min` are absolute-nats constants — per-model
   calibration axes, not universal.
6. Greedy decoding is the reproducible default; regime geometry shifts with
   temperature.
7. Rolling mode requires history (see `SPEC_DEVIATIONS.md`).

## Roadmap

R1 LLM-judge labels · R2 HaluEval/SimpleQA/long-form · R3 DREAM-005 composite ·
R4 adaptive windowing · R5 lead-time survival analysis
