# Observer 3: meta audit of process and claims, DREAM-001 D8

2026-10-01. Scope: the pre-registered verdict, every analysis run on `runs/llama3.2-3b-d8-full` after it, the claims in FINDINGS_D8.md, HAZARD_CARDS.md, CHANGELOG.md and the published "DREAM-001 Findings" page, and the reports from Observer 1, Observer 2 and the GPT audit.

Method: I read the documents and recomputed from the run files on CPU. No model was loaded and no existing file was changed. My scratch scripts are in the session scratchpad, outside the repo. I treated every document as evidence to check, including the other observers' reports and the lead analyst's notes. Numbers marked **[O3]** are my own recomputations. Everything I computed after the result is exploratory too, and it is counted in the forking-paths tally in section 2.

---

## 0. Bottom line

1. **The pre-registered NOT SUPPORTED decision stands.** None of the defects found (D10 double BOS, decline-as-TRUE, stop-string truncation, wrong event floor, judge DECLINE behaviour, single LLM annotator) moves precision anywhere near 0.60. Across every label or subset variant I tried, precision stays between 0.33 and 0.46.
2. **The judge-validation figure of 86.7% is flattered by easy items.** 25 of the 60 validation items are "I have no comment" openers, and the judge and annotator agree on all 25. On the 35 substantive answers, agreement is **27/35 = 77% [Wilson 61–88%]**, below the 80% gate. The gate was pre-registered on all 60 items and was passed as written, so the verdict is not invalidated. Label quality on the answers that matter is still lower than reported.
3. **The judge's DECLINE label also hides real errors.** This is a defect the earlier audits missed. 35 substantive answers were labelled DECLINE, which counts as "not hallucinated". Examples: "Musk.", "Elvis Presley.", "Canada.". Four of the five such answers in the validation sample were hallucinations according to the annotator. They account for 4 of the 5 errors the annotator caught and the judge missed. This barely touches the verdict (lift 0.99 if all 35 are counted as false). It does matter for the "216 real hallucinations" profile and the mechanism counts.
4. **The exploratory claim that mean entropy reaches AUC 0.66 is mostly an artefact and must be retracted publicly.** Among answers that do not open with a decline, the AUC is **0.578 [0.523, 0.630] [O3]**. Answer-length AUC falls from 0.686 to **0.493**, and the "within length terciles" defence falls from 0.72/0.65/0.62 to **0.52/0.56/0.61** [O3]. The published page, CHANGELOG.md and FINDINGS_D8.md all carry the 0.66 number.
5. **The stage claim ("uncertain commitment, then confident rationalization") is not supported** and must not be published.
6. **The "two families" claim survives only in a narrower form.** Fabrication and famous-name lure differ robustly from each other: the difference in AUC against substantive correct answers is 0.21–0.48 (95% CI) [O3]. But this is not a two-family split of all errors. The middle categories sit at 0.59–0.64 with intervals touching 0.5. The lure category was invented during coding. And even fabrication is barely separable from answers the judge wrongly called false (AUC 0.62 [0.51, 0.73]) [O3].

---

## 1. Verdict integrity

### 1.1 Script against pre-registration

I read `scripts/score_preregistered.py` against `PREREGISTRATION_D8.md`.

| Pre-registered rule | Code | Match |
|---|---|---|
| Judge valid if agreement ≥ 80% on 60 items, seed 1 | `AGREEMENT_MIN=0.80`, `SAMPLE_SIZE, SAMPLE_SEED = 60, 1`, `judge_valid = len(ann)==60 and agreement>=0.80` | yes |
| UNRESOLVED if invalid → UNDERPOWERED if <30 flags → SUPPORTED → REFUTED (lift ≤ 1) → NOT SUPPORTED | same order, lines 85–94 | yes |
| SUPPORTED = `verdict()` PASS **and** capable-subset lift > 1.15 with p < 0.05 | line 89 | yes |
| hallucinated ⇔ FALSE | `judge[...]["hallucinated"]`; Observer 1 confirmed `hallucinated == (judge=="FALSE")` for all 757 | yes |
| precision ≥ 0.60, K1 lift > 1, K2, K3 < 0.95 | `report.verdict()`: K1 `lift<=1`; K2 is checked only when precision > 0.60, which is equivalent for a failing run; K3 `firing>=0.95` | yes |

**[O3]** I recomputed the primary metrics from `items.jsonl` and `judge_labels.jsonl` with `dream001.report.compute_metrics`: n 757, 86 flagged, TP 30, precision 0.3488, base 0.3421, lift 1.0196, p 0.476. This matches `prereg_result.json` exactly. The Wilson interval on precision is [0.257, 0.454]. Reaching 0.60 would take 52 true positives among the 86 flags; there are 30.

**Timeline check.** File modification times give only weak evidence, and there is no git history.
- `PREREGISTRATION_D8.md` was last written at 00:09:46. `items.jsonl` was written at 01:06:45. The pre-registration was therefore not edited after the results existed.
- `annotation_sample.jsonl` was written at 01:07:29. `judge_labels.jsonl` was *created* at 01:07:08 and finished at 01:08:42. The judge file was being written while the annotation was saved. FINDINGS_D8 says the annotation was "written before any judge output was seen". That is plausible but cannot be verified from the files.
- The annotation was labelled `claude-opus-5-5`, the same model as the lead analyst. In practice it is probably the same agent.

### 1.2 Each reported defect against the decision

| Defect | Effect on decision | Evidence |
|---|---|---|
| D10 double BOS | **None on internal validity.** Calibration and eval share the condition. **It does narrow external validity**: the result is for Llama-3.2-3B-Instruct conditioned on a duplicated BOS token, which is not standard use. Size of the effect on entropy is unknown. It must be disclosed in any public claim; the published page omits it. | SPEC_DEVIATIONS D10; Observer 1 confirmed scope (the judge is unaffected) |
| Decline-opener answers labelled TRUE | None on the binary label: TRUE and DECLINE both mean "not hallucinated". It contaminates every *exploratory* comparison that used judge-TRUE as the "correct" class. | 216 of 384 TRUE answers open with a decline (is_idk); 215 by the literal "I have no comment" prefix. This explains the 216 vs 215 difference between Observer 1 and Observer 2 [O3] |
| Stop string fires across the prompt boundary (3 items, "A:") | Lift 1.020 → 1.016 with them dropped [O3] | Observer 1 O1-3 |
| Event floor is 5, not 9 | None. The capable-subset rule is only consulted on a PASS. No flagged answer is shorter than **52** tokens [O3], so the floor never bound. | floor 5 → n 754, lift 1.016 (Observer 1) |
| Judge rarely chooses DECLINE for decline-openers | None on the binary label. | Observer 1 O1-2 |
| **Judge uses DECLINE on substantive wrong answers** (new) | Small. If all 35 substantive-DECLINE answers are treated as hallucinated: precision 0.384, base 0.388, **lift 0.99, p 0.58** [O3]. | Validation sample: 4 of 5 substantive-DECLINE answers were annotator-hallucinated. Examples include "Musk.", "Elvis Presley.", "Canada." |
| Single LLM annotator; one-sided recoding of 43 not_false | Small. With the coder's corrected labels (216 positives): precision 0.326, **lift 1.14, p 0.22** [O3]. Still KILL, because precision is far below 0.60. Note that this is the one variant that brings lift close to 1.15, so FINDINGS_D8's "neither direction would move 1.02 to 1.15" holds only narrowly. The safe statement is about precision, not lift. | `mechanism_codes.tsv` |
| Subset without decline-openers | Lift 0.83, p 0.97 (n 458, precision 0.46, base 0.557) [O3, reproduces Observer 1] | — |

**Conclusion.** No path through these defects gets precision near 0.60. The NOT SUPPORTED decision is robust. One caution: the post hoc subsets must not be used to upgrade the decision to REFUTED, or to claim that the detector is anti-predictive. Lift 0.83 comes from a subset chosen after the fact.

### 1.3 Did decline-as-TRUE affect judge-validation agreement?

Yes, indirectly. It did not cause disagreements. It added easy agreements.

| Validation-sample cell (n=60) | n | Annotator: hallucinated |
|---|---|---|
| decline-opener, judge TRUE | 19 | 0 |
| decline-opener, judge DECLINE | 6 | 0 |
| substantive, judge FALSE | 24 | 21 |
| substantive, judge TRUE | 6 | 1 |
| substantive, judge DECLINE | 5 | **4** |

- Agreement on decline-openers is 25/25. On substantive answers it is **27/35 = 77.1% [0.61, 0.88]**.
- Over all 60 items the Wilson interval is [0.758, 0.931]; its lower bound is under 0.80.
- The disagreements run in both directions (5 annotator-only, 3 judge-only). Two of the eight disagreements are on flagged answers, one in each direction, so they cancel in precision.
- Implied error rates: about 14% of judge non-FALSE answers are missed errors (5/36 [0.06, 0.29]). About 12.5% of judge-FALSE answers are not errors (3/24 [0.04, 0.31]).

**Independence caveat for publication.** No human labelled anything.
- The labels come from Qwen2.5-3B.
- The validator was Claude, the same model as the analyst.
- The mechanism coder was Claude. It agreed 24/24 with the validator on the overlapping judge-FALSE items, which you would expect if it was the same agent.
- The observers are Claude and GPT.
- The published page says "hand-labelled sample". That is inaccurate.

---

## 2. Forking paths

### 2.1 Analyses on this dataset after the pre-registered result

The pre-registered analyses are the primary, the capable-subset diagnostic, the secondary token-F1 analysis and `length_diagnostic.json`. Everything below came after them.

| # | Who | Analysis | Rough number of estimates |
|---|---|---|---|
| 1 | Lead | Per-answer feature AUCs: mean entropy, length, event count, flag (FINDINGS_D8; written 63 s after `prereg_result.json`) | 4 |
| 2 | Lead | Mean-entropy AUC within length terciles | 3 |
| 3 | Lead | Topic-level entropy breakdown. Referenced in HAZARD_CARDS ("From the topic breakdown I expected…"); no file preserved. **Unlogged.** | ? |
| 4 | Lead | Mechanism coding of 259 judge-FALSE answers: 8 codes, a fabricated-specifics flag, a not_false review, and the **lure subtype invented during coding** | — |
| 5 | Lead | Per-mechanism mean-entropy AUC with CI, raw and length-matched (11 groups) | 22 |
| 6 | Lead | Per-mechanism unwind firing rate | 11 |
| 7 | Lead | v0 single-BOS trace rebuild and stage localization (headline first-3, headline mean, explanation; 7 groups) | ~24 |
| 8 | Lead | Controls: not_false, riddle subset, within-TruthfulQA-category | ~6 |
| 9 | Lead | Double-BOS rebuild, `stage_localization.json` rerun | ~27 |
| 10 | Lead | Corrected comparator, `stage_localization_corrected.json`, corrected FINDINGS AUC | ~28 |
| 11 | GPT | Detector rate per mechanism (×8) × 5 comparators, topic- and length-matched tests (×8), 32-topic scan | ~80 |
| 12 | Obs 1 | Non-decline lift, floor-5 rescore, substantive-comparator AUCs for all groups, paired h−e bootstraps, controls | ~35 |
| 13 | Obs 2 | Blind first-false-token labels (40) and true-commit labels (24); within-answer and position-matched percentiles; pre-token elevation | ~25 |
| 14 | Obs 2 | Rationalization: 6 token classes, lexical matching, position bins, lure sub-analysis, 10 semi-blind secondary claims | ~25 |
| 15 | Obs 2 | Difficulty: real errors vs not_false, Qwen-entropy proxy, logistic model, terciles | ~10 |
| 16 | Obs 3 | My sensitivity variants: label variants, validation-sample composition, fabrication–lure difference, fabrication vs not_false | ~15 |

**Total: 16 analysis families and roughly 300 estimates on one set of 757 greedy answers, all within about one day.** The GPT audit is the only one that corrected for multiplicity, and only within its own family. No exploratory number from this dataset can be treated as a discovery. Each one is a hypothesis to test on fresh data.

### 2.2 Classification of claims currently in the documents

**C = confirmatory** (pre-registered); **E-L = exploratory and correctly labelled and worded**; **E-O = exploratory but overstated, stale or wrong**.

| Claim | Where | Class |
|---|---|---|
| NOT SUPPORTED; precision 0.349 vs 0.342, lift 1.02, p 0.48, 86 flags | FINDINGS_D8, PREREG_RESULT, page | **C** |
| Capable-subset lift 0.951 | FINDINGS_D8, page | **C** (diagnostic). The floor label "≥ 9" is wrong; the true floor is 5. |
| Secondary token-F1 lift 0.81 | FINDINGS_D8, page | **C** (secondary) |
| Judge validated 52/60 = 86.7% | FINDINGS_D8, page | **C** as a gate. **E-O as a quality claim**: omits that substantive agreement is 77% and that the annotator was an LLM. |
| "Not a power problem"; precision CI 0.25–0.45 | FINDINGS_D8, page | **C** |
| "Entropy trace *does* carry information"; mean entropy AUC 0.658; length 0.686; terciles 0.72/0.65/0.62 | FINDINGS_D8, CHANGELOG, page | **E-O** (decline artefact; corrected 0.578, length 0.49) |
| "The model is measurably less certain on average when it is about to be wrong" | page | **E-O**. "About to be" is a temporal claim, and whole-answer means include tokens after the error. |
| Event counts / flags AUC ≈ 0.50 | FINDINGS_D8, page | **E-L**. Holds: 0.47 among non-declines. |
| "~9-token floor"; "706/757 long enough" | FINDINGS_D8, page, SPEC_DEVIATIONS D8, project memory | **E-O** (floor is 5; 754/757) |
| Detector at chance for every mechanism; no subtype rescue | HAZARD_CARDS #3, GPT audit | **E-L**. Robust under the substantive comparator: 31/168 = 18.5% firing. |
| 216 real hallucinations = 28.5% | HAZARD_CARDS | **E-O**. Only judge-FALSE answers were reviewed; errors labelled TRUE or DECLINE (about 14%, e.g. "Musk.") were never counted. It is not a rate estimate. |
| Mechanism AUCs (fabrication 0.78, misconception 0.69, lure 0.45, overclaim 0.48, …) | HAZARD_CARDS | **E-O** (comparator confound; corrected 0.73 / 0.62 / 0.39 / 0.43) |
| "Two families … No uncertainty-based detector will catch these" | HAZARD_CARDS #1, H3, H7 | **E-O** (universal claim from n=19 and n=10; post hoc category) |
| "Caveat: entropy may track difficulty" (not_false AUC 0.69) | HAZARD_CARDS #4 | **E-L** (good caveat; now 0.62) |
| Stage table, "decision in first tokens; explanation rationalizes it" | HAZARD_CARDS stage section | **E-O**. Stale (single-BOS numbers) and an artefact (Observer 1, Observer 2). |
| Prevention and response prompts per card | HAZARD_CARDS | **E-L** ("untested hypotheses"). The safety-data-sheet framing lends them unearned authority. |
| Exposure rates (misconception 9.8%, …) | HAZARD_CARDS | **E-O** (undercounted, as for 216) |
| Next confirmatory tests | HAZARD_CARDS | Appropriate |

---

## 3. Audit of participants

### Lead analyst

**Did well**
- Pre-registered before results and scored mechanically.
- Declared deviations (D6–D10) instead of hiding them.
- Labelled exploratory work as exploratory.
- Wrote the codebook before coding and declared the post-coding lure note.
- Volunteered the difficulty caveat.
- Reported "my earlier prediction was wrong".
- Commissioned outside audits and accepted the corrections.

**Over-claimed**
1. **Speed over checking.** The exploratory section was written 63 s after the pre-registered result. 259 answers were coded in about 3 minutes (01:15 → 01:18). HAZARD_CARDS was built on a trace rebuild that reproduced the run at only r = 0.97, and that was accepted as good enough. That is how D10 was missed.
2. **Comparator never inspected.** 56% of the "correct" class was a single canned sentence. A five-minute look at the comparator's text would have caught this before AUC 0.66 went public.
3. **Interpretive overreach.** The text shifts from "AUC" to "the model knows it is unsure", "rationalizes" and "no detector will catch". These are mechanism and universal claims built on single-model, post hoc, single-coder data.
4. **Self-validation loop.** The lead acted as validator, coder, analyst and author. The 43 not_false overrides were applied only to judge-FALSE answers, which flatters the profile.
5. **Published before audit.** The results page carried the exploratory AUC before any observer had looked.

**Corrected numbers in `stage_localization_corrected.json`:** I reproduce them from the stored per-answer means within ±0.001 (fabrication 0.726, misconception 0.622, lure 0.388, overclaim 0.426, all real errors 0.612, non-decline overall 0.578). They are correct. Two caveats:
- The comparator excludes substantive-DECLINE answers. That is the right call, but it should be stated.
- The "corrected" numbers inherit every forking-path caveat in section 2.

### Observer 1 (implementation)

Strong, careful and mostly right. I reproduced the non-decline lift (0.83), the 216/384 decline count, the floor point (no flag under 52 tokens) and the corrected mechanism AUCs.

| Where Observer 1 erred or over-claimed | Detail |
|---|---|
| Under-claimed on O1-2 | It says "the binary hallucinated label is unaffected" by the judge's DECLINE behaviour. That is true for decline-openers but misses that DECLINE is also given to 35 substantive answers, most of which look wrong. That *does* change binary labels (verdict effect negligible: lift 0.99). |
| Over-claimed slightly | "The null is if anything stronger" leans on a post hoc subset. Fine as exploratory; it must not become "refuted". |
| Over-claimed on project framing | "Undercuts the length-starvation framing." The structural floor is 5, but empirically no answer under 52 tokens fired, so short answers were still effectively excluded. The *mechanism* in the docs is wrong; the *practical diagnosis* survives. |

### Observer 2 (model behaviour)

Good design: labels were frozen before entropy was opened (the hash I computed, `b8a86c9a…41ff`, matches the report) and there was a matched control. Its central correction (no confident rationalization; entropy does not separate real errors from judge-false-but-correct answers) is consistent with my numbers: not_false vs substantive TRUE is 0.624, all real errors 0.612.

**Over-claimed**
1. "Fails for … popular misconceptions." Its own table shows misconception false tokens at a median position-matched percentile of 0.88, with 5/9 in the answer's top 10%. Only 4/9 commit at low entropy. The fair wording is "fails for lures (4/6) and some misconceptions (4/9)".
2. **"When the explanation makes up something new, it is not confident"** is bolded but rests on 10 semi-blind labels (its own caveat). It should be a hypothesis, not a conclusion.
3. The token-level AUC of 0.74 uses only 10 controls.
4. The frozen-label hash proves the file has not changed. It does not prove the order of work, because it was published in the same report after the fact.

**Under-acknowledged:** its sampling frame and "real error" definition depend on the lead's codes and the judge labels. Those labels carry the missed-error problem above.

### GPT category audit

Strong method: Holm correction, matched controls, independent topic labels. Its main comparator was all judge-TRUE, which carries the decline confound. That biases *toward* finding effects (decline-openers fire at 7.7% vs 13.8% for substantive answers [O3]), so its nulls are conservative. Its "capable ≥ 9 tokens" variant inherits the wrong floor; this does not matter here. **Its listed artefacts (results.json, CSVs, category_analysis.py, analysis_plan.md) are not in the repo**, so its numbers cannot be reproduced from the repo. Add them or cite where they live.

### Is "two families" still supportable?

Only in a restricted, descriptive form.

| Comparison | AUC [95% CI] | Status |
|---|---|---|
| Fabrication vs substantive TRUE | 0.73 [0.65, 0.80] | Elevated |
| Lure vs substantive TRUE | 0.39 [0.27, 0.50] | At or below chance |
| Fabrication − lure difference | CI [0.21, 0.48]; direct fabrication-vs-lure AUC 0.83 [0.72, 0.92] [O3] | **The contrast is robust within this dataset** |
| Overclaim | 0.43 [0.26, 0.59], n = 10 | Cannot be placed |
| Misconception / false premise / fiction | 0.62 / 0.62 / 0.64, lower bounds 0.48–0.55 | "Weakly elevated", not a family |
| Fabrication vs not_false | 0.62 [0.51, 0.73] [O3] | Barely separable: entropy may mark novel, contested content rather than falsity |

Lengths are comparable (median 99 / 88 / 97.5 tokens for fabrication, lure and substantive TRUE), so length does not explain the contrast.

What the claim cannot support:
- A dichotomy covering all errors.
- "The model knows it is unsure."
- "No uncertainty-based detector will catch these."

The lure category was created during coding by one coder who was also the interpreter, and several lure-type errors ("Musk.") were labelled DECLINE by the judge and never coded. Defensible wording: *"Exploratory, one model: answers that invented specific facts had higher mean token entropy than substantive correct answers; answers that substituted a famous namesake had lower. Categories were assigned post hoc by a single LLM coder."*

---

## 4. Process lessons

| What went wrong | Safeguard that would have caught it |
|---|---|
| Comparator class ("judge TRUE") was half one canned refusal sentence | Before any group comparison, print a composition table and 10 random texts per class. Pre-specify the comparator, e.g. "substantive answers only". |
| Three-way judge validated only as a binary; validation sample half trivial items | Validate each class. Stratify the validation sample by answer format. Report agreement on the substantive subset. Use a human annotator, or at least one not the analyst. |
| One agent was annotator, coder, analyst, interpreter and author | Separate the roles. Use a second independent coder with kappa reported *before* any AUC is computed. The interpreter should not see entropies while coding (done) *and* should not have seen aggregate results first (not done: the topic breakdown preceded coding). |
| Recursive exploration: each surprise spawned a new cut (topic → mechanism → stage → token) on spent data | Keep a forking-paths log with every analysis and timestamp. Hold out a random half of items untouched for exploratory confirmation. Better still, test on fresh data, since TruthfulQA is spent for this model. |
| Trace rebuild accepted at r = 0.97 | Reproduction gate: rebuilt traces must match stored means at r > 0.999 before any analysis. Persist generated token ids. |
| Documented constants (9-token floor) never tested against the engine | Unit-test derived constants empirically (feed the minimal trace that fires). |
| Post hoc subgroup (lure) became a headline family | Freeze the codebook. New codes found during coding go to v1 and are not analysed in v0. |
| No version control; pre-registration integrity rests on file times | Commit and push the pre-registration (or post its SHA-256 publicly) before the run. Same for codebooks and frozen labels. |
| Exploratory result published (page, CHANGELOG) within about an hour, before audit | Rule: only the confirmatory result goes public until exploratory claims have had an independent audit. |
| Hazard-card framing (safety data sheet, "prevention", "exposure limit") gives untested ideas regulatory-looking authority | Keep untested interventions out of card format, or watermark each card "hypothesis, untested". |
| Observers relied on the lead's labels | At least one observer should re-derive labels independently on a random sample, including judge-TRUE and DECLINE answers. |

---

## 5. Publication guidance (X, no peer review)

General rule: post the confirmatory null prominently. Post the exploratory material only as clearly labelled hypotheses with numbers and caveats, and include the correction.

### 5(a) Safe to publish: suggested wording

**P1: the result (confirmatory).**
> Pre-registered test, result negative. We hypothesised that a "confident plateau, then entropy ramp" pattern in an LLM's token entropy precedes hallucination. On Llama-3.2-3B-Instruct, TruthfulQA (757 questions, greedy), answers the detector flagged were wrong 34.9% of the time vs a 34.2% base rate (lift 1.02, p = 0.48, 86 flags). The pre-set bar was ≥ 60% precision. Verdict: NOT SUPPORTED.

Caveats that must go in the thread:
- One model, one benchmark, greedy decoding.
- Labels came from an LLM judge (Qwen2.5-3B), checked against a second LLM (Claude). Agreement was 52/60, or 27/35 on substantive answers. No human labels.
- The run fed the model a duplicated start-of-text token (declared deviation D10).

**P2: robustness (exploratory, post hoc).**
> Post hoc checks didn't change it: dropping 3 truncated answers, re-coding judge errors, treating the judge's "decline" labels as errors, or excluding "I have no comment" answers all leave precision between 0.33 and 0.46, nowhere near 0.60.

**P3: no subtype rescue (exploratory).**
> Splitting the errors into post hoc categories (fabrication, misconception, etc.) didn't rescue it either. No error type was flagged significantly more often than correct answers after multiple-comparison correction (independent audit).

**P4: correction (required if the 0.66 figure was seen).**
> Correction to our earlier exploratory note: we said plain mean token entropy separated wrong from right answers at AUC 0.66. About half the "right" answers were the refusal "I have no comment", which is short and low-entropy. Excluding those, the AUC is 0.58 (95% CI 0.52–0.63), and answer length has no signal (0.49). A weak signal, consistent with prior work, not a new finding.

**P5: hypothesis only (exploratory, optional).**
> Hypothesis for a future pre-registered test, not a finding: answers that invented specifics (laws, dates, figures) had higher mean entropy than substantive correct answers (AUC 0.73, n=50). Answers that swapped in a famous namesake (a question about an obscure "Elon" answered "Musk") had lower (0.39, n=19). Categories were assigned after the fact by one LLM coder. Answers the judge wrongly marked false were about as "uncertain" as real errors, so entropy may track contested questions rather than wrongness.

**P6: process (optional and useful).**
> What we learned: check what's actually in your comparison group, and don't let one analyst label, code and interpret their own data. The full audit trail is in the repo.

### 5(b) Must NOT be published, or must be retracted

| Claim | Location | Action |
|---|---|---|
| "Mean token entropy AUC 0.658" and the figure | Results page, CHANGELOG, FINDINGS_D8 | **Retract and correct** (P4) |
| "Answer length AUC 0.686" | Results page | **Retract.** It was the decline artefact; 0.49 without declines. |
| "Keeps its signal inside each third of the length distribution (0.72, 0.65, 0.62)" | Results page | **Retract.** The shortest tercile was 179/258 declines; corrected 0.52/0.56/0.61. |
| "The model is measurably less certain … when it is about to be wrong" | Results page | **Retract.** "About to be" is a temporal claim never tested. |
| "Checked against a blind **hand-labelled** sample" | Results page | **Correct** to "labelled by a second LLM (Claude), blind to flags and judge labels" |
| "The detector needs about 9 tokens"; "706 of 757 long enough" | Results page | Correct to a 5-token structural floor; no flag occurred under 52 tokens |
| Omission of D10 double BOS | Results page | **Add** the disclosure |
| "Decision made in the first tokens; explanation rationalizes it" | HAZARD_CARDS | Do not publish. Artefact. |
| "Two families split by whether the model knows it is unsure"; "No uncertainty-based detector will catch these" | HAZARD_CARDS | Do not publish in this form; use P5 only |
| "216 real hallucinations (28.5%)", exposure rates | HAZARD_CARDS | Do not publish as rates |
| Hazard-card prevention and response prompts | HAZARD_CARDS | Do not publish as advice (untested) |
| "When the model fabricates in the explanation it is not confident" | Observer 2 | Do not publish (10 semi-blind labels) |
| "Detector is anti-predictive" or "REFUTED" based on lift 0.83 in non-decline answers | Observer 1 implication | Do not publish (post hoc subset) |
| Any claim that entropy "detects hallucinations", or generalizations to other models | — | Do not publish |

The results page is currently private by default, but it is described as published. If it has been shared, update it in place (it is owned by the user) and say so in the X thread.

### 5(c) Corrections the lead analyst must make

**FINDINGS_D8.md**
1. Judge validity: add the composition (25/60 decline-openers, all agreed), substantive agreement 27/35 = 77%, Wilson CI on 52/60 [0.76, 0.93], annotator model `claude-opus-5-5`, and that the judge and annotation timestamps overlap.
2. Add the substantive-DECLINE defect (35 answers; 4 of 5 in the sample were wrong) and the label-sensitivity table: drop "A:" → lift 1.016; DECLINE-as-false → 0.99; coder-corrected → 1.14 / precision 0.33; non-decline subset → 0.83. Replace "neither direction would move a lift of 1.02 to 1.15" with "no label variant brings precision near 0.60".
3. Floor: "≥ 9" → structural floor 5 (754/757). State that the floor never bound (minimum flagged length 52). Leave the pre-registered capable row as run, with a deviation note.
4. Conditions: add D10 double BOS and the 3 stop-string "A:" items.
5. Exploratory section: replace with the decline-excluded numbers (mean entropy 0.578 [0.52, 0.63]; length 0.49; event count and flag 0.47; terciles 0.52/0.56/0.61). Delete "the entropy trace *does* carry information", or soften it to "a weak residual signal". Add the not_false caveat. Keep "regime structure is at chance".
6. Add a forking-paths note pointing to section 2 of this audit.

**HAZARD_CARDS.md**
1. Replace the profile table with the corrected-comparator AUCs: fabrication 0.73, fiction 0.64, false premise 0.62, misconception 0.62, reasoning without lure 0.59, lure 0.39, overclaim 0.43, all real errors 0.61, not_false 0.62. State the comparator (substantive judge-TRUE, n = 168/169).
2. Rewrite "Two families" as the fabrication-vs-lure contrast (P5 wording). Delete "No uncertainty-based detector will catch these". Add fabrication vs not_false 0.62 [0.51, 0.73].
3. Replace "216 real hallucinations (28.5%)" with "216 of 259 judge-FALSE answers confirmed by one coder. Judge-TRUE and DECLINE answers were not reviewed, and the validation sample suggests about 14% of them are errors (e.g. 'Musk.' labelled DECLINE)". Mark exposure rates as lower bounds.
4. Remove or strike the stage section ("decision … rationalizes"). Replace it with Observer 2's supported finding (new false specifics tend to have a local entropy spike; lure and some misconception commitments are low-entropy, n = 40, single coder) and Observer 1's corrected h−e gap of +0.05 [−0.03, 0.12].
5. Mark every prevention and response line "untested". Consider dropping the exposure-limit section.
6. Limits: add "coder = analyst = validator (same LLM)", "lure category post hoc", "≈300 estimates on this dataset", and "D10".

**Elsewhere**
- CHANGELOG.md line 7–8: AUC 0.66 → 0.58 (non-decline).
- The results page (as in 5b).
- SPEC_DEVIATIONS D8: the "~9-token floor" wording.
- Project memory "length starvation": the structural floor is 5; the empirical minimum firing length in D8 is 52.
- Add the GPT audit artefacts to `audits/`.
