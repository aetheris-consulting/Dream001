# Changelog

## [Unreleased]

- **D8 confirmatory run: NOT SUPPORTED** (`runs/llama3.2-3b-d8-full`,
  `FINDINGS_D8.md`). 757 eval items, 86 flags, judge-validated labels:
  precision 0.349 vs base 0.342, lift 1.020, p 0.476. Exploratory: ~~mean
  entropy predicts hallucination (AUC 0.66)~~ **corrected:** excluding
  "I have no comment" answers, mean-entropy AUC is 0.578 [0.52, 0.63] and
  `confident_unwind` events sit at 0.47. The 0.66 figure mostly separated
  refusals from answers.
- **Audit corrections (same day).** Independent GPT audit and three observer
  audits (`audits/`). Found and declared: D10 double BOS; judge labelled 215
  decline-openers TRUE and 35 substantive answers DECLINE; 3 stop-string
  truncations to "A:"; the structural event floor is 5 tokens, not 9 (the arming
  tokens were double-counted). Verdict unchanged under every label variant.
  FINDINGS_D8.md and HAZARD_CARDS.md revised; the headline-vs-explanation
  "rationalization" claim withdrawn as a comparator artefact.
  `scripts/token_traces.py` (run-exact trace rebuild) and
  `scripts/stage_localization.py` added.
- **D9 judge + pre-registration tooling.** `PREREGISTRATION_D8.md`,
  `scripts/judge_labels.py`, `scripts/score_preregistered.py`.

- **D8 — `--prompt longform` (opt-in, declared 2026-10-01 before results).**
  Elicits answer + 2–3 sentence explanation so traces clear the ~9-token event
  floor; the concise spec prompt stays the default. `build_reference` and
  `run_eval` take a `prompt=` argument so the D1 reference is always built from
  the eval template. Smoke (12 items, Llama-3.2-3B): median answer 6 → 99
  tokens, 11/12 above the floor. See `SPEC_DEVIATIONS.md` D8.
- **Length-confound diagnostic written by every run.** `length_diagnostic.json`
  plus a RUN_SPEC section re-scoring precision/lift on answers at or above the
  event floor, and counting generations that hit the token cap. Diagnostic
  only; the verdict is untouched. Formalizes the manual correction in
  FINDINGS.md that erased the Runs B/C apparent lifts.
- Tests: 21 → 23. Synthetic validation reproduces exactly (rolling
  0.784/0.967; calibrated 1.000/0.980, lift 2.000, p ≈ 0.0005).

- **Generations were base-model continuation, not answers (D6).** The
  completion-style `PROMPT` was tokenized raw for *Instruct* models with no stop
  sequence, so Qwen2.5-3B answered and then ran to the 128-token cap —
  confabulating a fake system prompt ("You are an AI assistant created by
  Anthropic..."), echoing the instruction text, opening new `Q:` turns. Median
  answer length was 128, i.e. the cap. `HFBackend` now renders via
  `tokenizer.apply_chat_template()` and stops at `
Q:` / `
Question:` /
  `
A:`, cutting **text and entropies at the same token** so a trace never
  covers non-answer tokens. Post-fix answers are EOS-terminated at 3–34 tokens.
  `use_chat_template=False` restores the previous behavior.
- **`is_idk` matched anywhere in the answer; it is now anchored at the start
  (D7).** `label_answer` short-circuits on it, so an answer that opened with a
  decline and continued for ~120 tokens was scored not-hallucinated with its
  body unread. Combined with D6 this put **265/300 answers in the IDK bucket,
  base rate 0.037**, and produced a K1 KILL on a label distribution that was
  degenerate by construction. The two failures are separable, so both are fixed.
- Run `runs/qwen2.5-3b` (pre-D6/D7, Qwen2.5-3B-Instruct fp16, 300 items) is
  **retained but must not be quoted**: precision 0.011, lift 0.306, p 0.978,
  VERDICT KILL. It measured the harness, not the hypothesis. See
  `SPEC_DEVIATIONS.md` D6/D7.
- Synthetic validation reproduces exactly after D6/D7 (rolling 0.784/0.967;
  calibrated 1.000/0.980, lift 2.000, p ≈ 0.0005) — the detector is untouched;
  both deviations are on the generation and labelling sides only.

- **`--require-no-evidence` was a silent no-op; it is now refused (E4).** The
  §7.2 two-term gate in `extract_events` only ran `if evidence_mask is not
  None`, and `run_eval` never passed one — so the flag suppressed nothing while
  `run_truthfulqa.py` wrote `secondary arm (sec.7.2 two-term): ON` into
  `RUN_SPEC.md`. Any run made with the flag recorded **single-term results under
  a two-term label**. The gate now raises when armed without an aligned mask,
  `run_eval` refuses before generating, and the script refuses at argument parse
  before loading the model. No backend emits an evidence signal, so the arm
  stays unavailable until `evidence_arrival` is defined operationally — and it
  must not be calibrated on synthetic data (D4). See `SPEC_DEVIATIONS.md` E4.
- `run_eval(..., evidence_masks=...)` added as the seam for a caller that has a
  real per-token evidence signal. Defaults to `None`; the single-term primary
  arm is unchanged.
- Tests: 16 → 21. Synthetic validation reproduces exactly (rolling 0.784/0.967;
  calibrated 1.000/0.980, lift 2.000, p ≈ 0.0005) — the primary arm is untouched.

## [0.2.0] — 2026-08-19

Pre-run hardening on the v1.2 rebuild. Revalidated after patching: 16/16
tests; synthetic validation reproduced exactly (rolling 0.784/0.967;
calibrated 1.000/0.980, lift 2.000, permutation p ≈ 0.0005).

- Seeded shuffle of TruthfulQA before the calibration/eval split (`--seed`,
  recorded in `RUN_SPEC.md`). The dataset is category-grouped; taking the
  first N rows for calibration skewed the D1 reference distribution toward
  the leading categories.
- `datasets` moved from the `[hf]` extra to base dependencies — the
  DREAM-006 API tier needs data loading too and previously crashed at
  `load_items` without the GPU extra installed.
- Version honesty: this line is 0.2.0. The v0.1.0 name is reserved for the
  original 2026-07-06 build, to be archived under `v0.1.0-original` when
  the original artifact is restored to the repo.

## [v1.2 rebuild] — 2026-08 (imported as the root commit)

Ground-up rebuild against ISCOT v1.2 in a cross-model session (spec and
rebuild: GPT; execution audit and packaging: Claude). New declared
deviations D5 and E1–E3 (`SPEC_DEVIATIONS.md`). Closes both v0.1.0 gaps:
HF full-vocab in-step entropy backend, and answer-level token-F1 labelling
with the IDK convention. The synthetic generator was also rebuilt —
synthetic numbers are not comparable across generator versions.

## [0.1.0] — 2026-07-06

Original reference implementation (ARCHON role, session
ISCOT-AEF-2026-0412). D1–D4. Synthetic validation: rolling event
0.971/0.917; calibrated sequence 0.962/1.000, lift 1.923, p ≈ 0.0005.
Never reached the remote — GitHub connector outage; artifact pending
recovery, to be archived under tag `v0.1.0-original`.
