# DREAM-001: category-specific scoring of the existing detector

2026-10-01 · Exploratory · Existing D8 generations and flags only

No failure category showed a reliable positive detector association after correction for eight comparisons.

This tests whether the unchanged confident_unwind alarm is more frequent for a particular error type than for correct answers. It does not run new generations, change prompts, retune the detector, or test the hazard-card prevention clauses. The original NOT SUPPORTED result remains intact.

## Failure categories

Correct-answer comparator: 45/384 judge-TRUE answers flagged (11.7%; Wilson 95% interval 8.9%–15.3%). These are judge labels, not independently audited ground truth.

| Failure category | Flags / errors | Error flag rate | 95% interval | Rate ratio vs correct | One-sided exact p | Holm-adjusted p |
|---|---:|---:|---|---:|---:|---:|
| Fabrication | 7/50 | 14.0% | 7.0%–26.2% | 1.19 | 0.390 | 1.000 |
| Misconception | 10/74 | 13.5% | 7.5%–23.1% | 1.15 | 0.393 | 1.000 |
| Famous-name lure | 2/19 | 10.5% | 2.9%–31.4% | 0.90 | 0.674 | 1.000 |
| Reasoning excluding lure | 4/34 | 11.8% | 4.7%–26.6% | 1.00 | 0.584 | 1.000 |
| False premise | 0/15 | 0.0% | 0.0%–20.4% | 0.00 | 1.000 | 1.000 |
| Fiction as fact | 2/13 | 15.4% | 4.3%–42.2% | 1.31 | 0.469 | 1.000 |
| Overclaim | 3/10 | 30.0% | 10.8%–60.3% | 2.56 | 0.110 | 0.881 |
| Indexical | 0/1 | 0.0% | 0.0%–79.3% | 0.00 | 1.000 | 1.000 |

A rate ratio of 1 means the alarm is equally common in that error group and the correct-answer comparator. Higher than 1 is the direction the original detector claims. The exact test holds group sizes and total flags fixed. Holm correction accounts for looking across all eight categories. Indexical has one example and cannot support a useful performance estimate.

Comparison-set precision, lift, and binary-flag AUC are available in the CSV/JSON. Because error subtypes were selected after outcomes were known and paired with all correct answers, their comparison-set precision is not a deployment estimate or a new application of the original 60% acceptance gate.

## Robustness checks

| Comparator or matching rule | Categories with Holm p < 0.05 | Smallest adjusted p |
|---|---|---:|
| judge_true | None | 0.881 |
| judge_true_capable | None | 0.989 |
| judge_true_plus_reviewed_not_false | None | 0.759 |
| all_non_false_after_partial_review | None | 0.704 |
| topic_length_matched_true | None | 1.000 |

The capable comparator restricts correct answers to at least 9 tokens, the detector's minimum event opportunity. Reviewed not_false combines true answers and declines reclassified by the single coder. The all-non-false comparator additionally includes original judge declines. These definitions are separate sensitivity analyses, not interchangeable truth sets.

The matching check compares cases and correct controls within the same original TruthfulQA topic and fixed answer-length bin. It tests the sum of flagged cases against the exact conditional distribution across those strata. Unsupported strata are excluded with coverage shown below; all other analyses retain their specified full groups.

| Category | Matched errors / total | Matched control rows | Flags in matched errors | Case-weighted control flag rate | Conditional p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Fabrication | 42/50 | 148 | 6 | 18.5% | 0.881 | 1.000 |
| Misconception | 71/74 | 169 | 9 | 9.8% | 0.573 | 1.000 |
| Famous-name lure | 13/19 | 10 | 1 | 3.8% | 0.667 | 1.000 |
| Reasoning excluding lure | 32/34 | 122 | 4 | 10.0% | 0.356 | 1.000 |
| False premise | 15/15 | 80 | 0 | 14.1% | 1.000 | 1.000 |
| Fiction as fact | 12/13 | 27 | 2 | 6.5% | 0.333 | 1.000 |
| Overclaim | 7/10 | 12 | 2 | 4.8% | 0.300 | 1.000 |
| Indexical | 1/1 | 5 | 0 | 40.0% | 1.000 | 1.000 |

## Categories known from the question

A separate check scores the unchanged flags within all 38 TruthfulQA topics. Unlike failure mechanisms, these topic labels exist independently of the generated answer. The test uses the original judge FALSE versus TRUE/DECLINE labels and corrects across 32 topics with both outcomes. All 757 answers and 86 flags are accounted for. Topic-level matches after Holm correction: none.

The six smallest unadjusted topic p-values are shown only for completeness; they are selected after scoring, not confirmatory discoveries. Full topic results, including groups without both outcomes, are in input_topic_scores.csv.

| Question topic | Answers | Errors flagged / errors | Non-errors flagged / non-errors | Precision | Lift | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proverbs | 17 | 2/7 | 0/10 | 100.0% | 2.43 | 0.154 | 1.000 |
| Health | 54 | 2/17 | 1/37 | 66.7% | 2.12 | 0.230 | 1.000 |
| Misconceptions | 94 | 6/34 | 8/60 | 42.9% | 1.18 | 0.390 | 1.000 |
| Education | 9 | 2/6 | 0/3 | 100.0% | 1.50 | 0.417 | 1.000 |
| Sociology | 50 | 2/17 | 2/33 | 50.0% | 1.47 | 0.420 | 1.000 |
| Myths and Fairytales | 19 | 1/8 | 0/11 | 100.0% | 2.38 | 0.421 | 1.000 |

## Interpretation and limits

- Category narrowing asks a valid exploratory question; it does not by itself give the existing alarm a predictive signal.
- No significance is evidence of insufficient support in these samples, not proof that every subtype has exactly zero effect. Small categories have wide intervals.
- The 216 retained errors come from reviewing only the original judge-FALSE set. Missed errors among judge-TRUE/DECLINE answers remain possible. The coder and judge can both be wrong.
- Failure labels describe outputs after adjudication; this analysis does not prove those categories can be routed accurately before an error occurs. Topic and length matching reduce two known confounds but cannot remove all selection or labeling effects.
- These data have already been explored. Multiple-comparison adjustment does not make the analysis confirmatory, undo selection of the taxonomy, or establish generalization to another model/task.
- Event presence is tested at answer level. Lead time before a false claim is not measured.
- Hazard-card prevention and response controls remain untested by this scoring run.

## Reproduction and validation

Run category_analysis.py using the existing DREAM-001 virtual environment; default paths point to the inspected local repo and cached TruthfulQA Arrow file. Override --repo, --dataset-arrow, --plan, and --out as needed. The script performs no network calls and no model inference.

Checks passed: unique row IDs and complete joins; taxonomy covers exactly the 259 judge-FALSE answers; error partition totals 216; saved flags agree with event presence; the cached dataset reproduces the exact seeded 60/757 split; original primary counts, precision and lift reproduce; exact tests and conditional convolution agree with independently enumerated small examples; all input hashes are unchanged after scoring.

Artifacts: results.json (full statistics, strata, validation and source hashes), category_scores.csv (all mechanism comparisons), input_topic_scores.csv (all question topics), scored_answers.csv (757-row audit ledger), analysis_plan.md (scope and methods written before this scoring pass), and category_analysis.py (executable source).

Primary source files:

- C:\Users\T G\Downloads\dream001-0.2.0_1\runs\llama3.2-3b-d8-full\items.jsonl
- C:\Users\T G\Downloads\dream001-0.2.0_1\runs\llama3.2-3b-d8-full\judge_labels.jsonl
- C:\Users\T G\Downloads\dream001-0.2.0_1\runs\llama3.2-3b-d8-full\mechanism_codes.tsv
- C:\Users\T G\Downloads\dream001-0.2.0_1\runs\llama3.2-3b-d8-full\prereg_result.json
- C:\Users\T G\Downloads\dream001-0.2.0_1\runs\llama3.2-3b-d8-full\annotation_sample.jsonl
- C:\Users\T G\Downloads\dream001-0.2.0_1\runs\llama3.2-3b-d8-full\RUN_SPEC.md
- C:\Users\T G\Downloads\dream001-0.2.0_1\TAXONOMY_CODEBOOK.md
- C:\Users\T G\Downloads\dream001-0.2.0_1\PREREGISTRATION_D8.md
