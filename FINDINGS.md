# Findings — first GPU runs, 2026-08-20

Three complete 300-item TruthfulQA runs — **Qwen2.5-3B-Instruct** (twice) and
**Llama-3.2-3B-Instruct** — at fp16, full-vocab `hf` backend, greedy, seed 0,
RTX 4060 Laptop (8.59 GB).

**All three returned KILL. None is a falsification.** The hypothesis —
`confident_unwind` entropy signatures precede hallucination — remains
**untested**. Every kill traces to a harness or testbed limit, not to the signal.

---

## Results

| | Run A `runs/qwen2.5-3b` | Run B `runs/qwen2.5-3b-d6d7` |
|---|---|---|
| state | pre-fix | post D6 + D7 |
| base rate | **0.037** ✗ | **0.137** ✓ |
| firing rate | 0.297 | **0.013** ✗ |
| precision | 0.011 | 0.250 |
| recall | 0.091 | 0.024 |
| lift | 0.306 | 1.829 |
| permutation p | 0.9780 | 0.4543 |
| confusion tp/fp/fn/tn | 1 / 88 / 10 / 201 | 1 / 3 / 40 / 256 |
| median answer | 128 tok (the cap) | 6 tok |
| trace volume | 30,689 tok | 4,106 tok |
| `confident_unwind` share | 0.069 | 0.008 |
| verdict | KILL — K1, lift ≤ 1 | KILL — precision < 0.60 |

### Run C — cross-model replication, `runs/llama3.2-3b-d6d7`

`unsloth/Llama-3.2-3B-Instruct`, same harness, same seed.

| | Run C |
|---|---|
| base rate | 0.203 |
| firing rate | **0.030** ✗ |
| precision | 0.444 |
| recall | 0.066 |
| lift | 2.186 |
| permutation p | 0.0950 |
| confusion tp/fp/fn/tn | 4 / 5 / 57 / 234 |
| median answer | 6 tok |
| trace volume | 6,452 tok |
| `confident_unwind` share | 0.027 |
| verdict | **KILL** — precision < 0.60 |

Llama is a somewhat better subject than Qwen — longer answers (mean 21.5 vs
13.7 tok), 3.3× the `confident_unwind` density, more of the eval set reachable
(34.7% vs 26.7%). Same verdict, same underlying starvation. **The structural
problem is not Qwen-specific.**

---

## The length confound — read this before quoting any lift

Lift is computed against the **full-set** base rate, but the detector can only
fire on answers above the 9-token floor, and those answers are far likelier to
be hallucinations (short answers are mostly one-word facts or declines).
Restricting to the population where firing is possible removes the confound:

| | full-set precision / base = lift | among capable (≥9 tok) = lift |
|---|---|---|
| Qwen2.5-3B | 0.250 / 0.137 = **1.829** | 0.250 / 0.425 = **0.588** |
| Llama-3.2-3B | 0.444 / 0.203 = **2.186** | 0.444 / 0.452 = **0.982** |

**Both apparent lifts vanish.** Within the subpopulation the detector can act
on, flagging carries essentially no information about hallucination. The
headline lift was a length artefact — the detector was being credited for
firing on the answers that happen to be wrong more often, not for detecting
that they were wrong.

This does not falsify the hypothesis either: with 4 and 9 flags respectively,
neither conditional estimate has meaningful power. It does mean the encouraging
full-set lift must not be reported.

Run B's precision is one true positive out of four flags (p = 0.454); Run C's is
four out of nine (p = 0.095). Neither is a measured precision.

---

## Why none of the runs test the hypothesis

**Run A — the model was not answering.** The completion-style `PROMPT` was
tokenized raw for an Instruct model with no stop sequence. Qwen answered, then
confabulated a fresh system prompt ("You are an AI assistant created by
Anthropic…"), echoed the instruction text back, opened new `Q:` turns, and once
leaked a Chinese token mid-sentence. 265/300 answers opened with "I have no
comment" and continued ~120 tokens further. `label_answer` short-circuits on the
IDK test, so those tokens were never read — **88.3% of the eval set was scored
not-hallucinated on the strength of its first sentence**, putting the base rate
at 0.037.

**Runs B and C — the detector had no room.** With D6/D7 the generations are
correct and the base rate recovers (0.137 Qwen, 0.203 Llama). But answers are now genuinely concise, and
`short_sequence` needs `band_min_run`(3) + `min_hold_run`(4) +
`min_event_tokens`(2) — a **9-token floor** before any event can exist:

```
                              Qwen2.5-3B      Llama-3.2-3B
capable of firing (>=9 tok)   80/300 (26.7%)  104/300 (34.7%)
cannot fire, ever            220/300 (73.3%)  196/300 (65.3%)
  ...real hallucinations       7 unreachable   14 unreachable
```

Answer-length distribution flipped from one wall to the other:

```
bucket        <9    9-15   16-31  32-63   64+
Run A          2      4      12     43    239
Run B        220      9      32     33      6
Run C        196     14      26     36     28
```

This is **README limitation 1** stating itself out loud: *"Better for extended
confabulation than atomic single-token factual errors."* An entropy ramp needs
tokens to ramp across. TruthfulQA short-form answers do not supply them. That is
a property of the testbed, not a defect in the detector.

---

## Changes made (all declared)

| ID | Change |
|---|---|
| — | dataset id `truthful_qa` → `truthfulqa/truthful_qa`; the bare alias is rejected by `datasets` 5.0.1 and would have crashed at load |
| **D6** | `HFBackend` renders via `tokenizer.apply_chat_template()` and stops at `\nQ:` / `\nQuestion:` / `\nA:`; **text and entropies cut at the same token** |
| **D7** | `is_idk` anchored at the answer start; a decline buried mid-answer no longer immunizes the claims after it |
| **E4** | `--require-no-evidence` was a silent no-op writing `secondary arm: ON` into RUN_SPEC; now refused rather than silently degraded |

**The detector is provably untouched.** Synthetic validation reproduces
bit-identically after every change — rolling 0.784/0.967, calibrated
1.000/0.980, lift 2.000, p ≈ 0.0005. Tests 16 → 21, all passing. D6 and D7 are
confined to the generation and labelling sides.

---

## What worked

- **The honesty gates earned their keep.** Bare precision would have read as a
  clean 0.011 and then 0.250. Lift-against-base-rate made Run A's degenerate
  labels impossible to miss; the permutation test flagged Run B as chance. The
  K-gates, plus the conditional analysis, caught three runs a single-number
  report would have passed as results.
- **The full-vocab tier runs on 8 GB.** 3B at fp16 fits with ~1 GB headroom,
  coverage 1.00 across 1,080 generations on two model families, no OOM. No need for the degraded
  DREAM-006 top-k tier; no quantization touching the logit distribution.

---

## Recommendation

1. **Preferred — long-form prompt variant (D8).** Keep detector and dataset
   fixed; change `PROMPT` to elicit a brief explanation rather than a concise
   answer. Deviates from the spec's "concisely" and must be declared, but it
   tests the claim on the domain the claim is about.
2. **Alternative — R2 data.** HaluEval / SimpleQA / long-form. Closest to a real
   test, larger build, and the 0.60 threshold was set against TruthfulQA.
3. **Rejected — relaxing `band_min_run` / `min_hold_run`.** It would move the
   numbers, but it is tuning the detector until it fires on the eval set — the
   exact D4 contamination the project guards against. `min_hold_run` *is* D5,
   and D5 exists because without it rolling precision was 0.35.

---

## Conditions stated, not buried

- Both models ship natively in **bfloat16**; the backend loads at **float16**.
  The README specifies fp16 and fp16 is not quantization, but it is a cast off
  the native dtype.
- `fast_slope_min = 0.06` is an absolute-nats constant carried across models
  untouched (README limitation 5 — a per-model calibration axis).
- Llama-3.2-3B was run from the ungated mirror `unsloth/Llama-3.2-3B-Instruct`;
  `meta-llama/Llama-3.2-3B-Instruct` is gated and the access request was pending
  author review. Architecture, vocab (128256), dtype and parameter byte count
  match; tensors were not hash-compared against Meta's copy.

Environment: torch 2.6.0+cu124 · transformers 5.15.1 · datasets 5.0.1 · CUDA 12.4.
