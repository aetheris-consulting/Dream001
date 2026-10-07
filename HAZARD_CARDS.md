# Hallucination hazard cards — Llama-3.2-3B-Instruct (exploratory v0, revised)

> **HYPOTHESES, NOT FINDINGS.** Built 2026-10-01 on `runs/llama3.2-3b-d8-full`,
> data already analysed for DREAM-001. Categories were assigned after the fact by
> a single LLM coder (`claude-opus-5-5`), who was also the analyst and the
> judge-validation annotator. Every prevention and response line is **untested**.
> Revised the same day after independent audits (`audits/observer1_*`,
> `observer2_*`, `observer3_meta.md`); the first version's stage section and
> "two families" wording were wrong and are withdrawn.

A card set belongs to one model, prompt and decoding setup, as a safety data
sheet belongs to one product formulation.

## What was coded

216 of the 259 judge-FALSE answers were confirmed as errors by one coder; 43
were not false. Judge-TRUE and DECLINE answers were **not reviewed**, and the
validation sample suggests some are errors (e.g. "Musk." labelled DECLINE), so
every count below is a lower bound. Codebook: `TAXONOMY_CODEBOOK.md`; codes:
`mechanism_codes.tsv`.

## Profile

Comparator: **substantive** judge-TRUE answers, i.e. those not opening with
"I have no comment" (n = 169). The first version compared against all 384
judge-TRUE answers, 215 of which were that refusal phrase; its AUCs were
inflated and are withdrawn.

| mechanism | n | mean-entropy AUC vs. substantive correct [95% CI] |
|---|---|---|
| fabrication | 50 | **0.73** [0.64, 0.80] |
| fiction as fact | 13 | 0.64 [0.51, 0.75] |
| misconception | 74 | 0.62 [0.55, 0.69] |
| false premise | 15 | 0.62 [0.50, 0.73] |
| reasoning (excl. lure) | 34 | 0.59 [0.49, 0.68] |
| overclaim | 10 | 0.43 [0.26, 0.59] — cannot be placed |
| famous-name lure | 19 | **0.39** [0.27, 0.51] |
| *all confirmed errors* | 216 | 0.61 [0.56, 0.67] |
| *judge-false but actually correct* | 43 | 0.63 [0.54, 0.72] |

The unwind detector fires at the same rate on every error type as on correct
answers (independent GPT audit, Holm-corrected: no category significant).

## What it supports, and what it does not

**Supportable, as a hypothesis (one model, post hoc categories):** answers that
invented specific facts had higher mean token entropy than substantive correct
answers (0.73, n = 50); answers that substituted a famous namesake had lower
(0.39, n = 19). The direct fabrication-vs-lure contrast is AUC 0.83 [0.72, 0.92],
and their lengths are similar. Token-level spot checks (Observer 2, n = 40,
single coder) agree: new false specifics tend to show a local entropy spike;
lure and some misconception commitments are low-entropy ("Gates": 0.03 nats).

**Not supported:**
- A dichotomy covering all errors. Misconception, false premise and fiction are
  only weakly elevated (lower bounds 0.48–0.55).
- "The model knows when it is unsure", or "no uncertainty-based detector will
  catch these".
- That entropy tracks *wrongness*: fabrication barely separates from
  judge-false-but-correct answers (0.62 [0.51, 0.73]). Entropy may mark novel
  or contested content.
- **Withdrawn:** "the decision is made in the first tokens and the explanation
  rationalizes it." That was an artefact of the refusal-heavy comparator.
  Corrected headline-minus-explanation gap: +0.05 [−0.03, 0.12].

## Cards

All prevention and response entries are **untested ideas**.

### H1 · Fabrication — invented specifics
- **Identification:** dates, statutes, figures, attributions or mechanisms that
  are false ("the Highway Act 1835 prohibits driving barefoot").
- **Detection (exploratory):** the most entropy-elevated type, 0.73.
- **Prevention (untested):** "Do not state specific names, dates, numbers or
  laws unless you are certain of them."
- **Response (untested):** verify high-entropy spans containing numerals or
  proper nouns.

### H2 · Misconception — popular false belief
- **Identification:** repeats a widely held myth (Einstein flunked math).
- **Detection (exploratory):** weak, 0.62.
- **Prevention (untested):** "Consider whether the popular answer is actually
  true."
- **Response (untested):** check against a curated misconception list; this type
  is enumerable.

### H3 · Famous-name lure — *category created during coding; v1 candidate*
- **Identification:** the question describes an obscure entity; the answer
  completes to the famous namesake (Elon → Musk), then recites facts about it.
- **Detection (exploratory):** **lower** entropy than correct answers, 0.39.
  Entropy-based detection would not catch it.
- **Prevention (untested):** "Check every detail in the question against your
  answer."
- **Response (untested):** constraint check against each clue in the question.

### H4 · Reasoning (other)
- **Identification:** tautologies answered "No", ignored constraints, probability
  errors, self-contradiction, idioms read literally.
- **Detection (exploratory):** weak, 0.59.

### H5 · False premise
- **Identification:** goes along with a false presupposition in the question.
- **Detection (exploratory):** weak, 0.62.
- **Prevention (untested):** "If the question assumes something false, say so."

### H6 · Fiction as fact
- **Identification:** fairy tales, superstition or astrology stated as
  real-world fact.
- **Detection (exploratory):** 0.64, n = 13.

### H7 · Overclaim
- **Identification:** contested evidence stated as settled.
- **Detection:** n = 10, cannot be placed.

## Limits

- Coder = analyst = validator (the same LLM). No human labels. No second coder
  yet.
- Categories post hoc; the lure category was invented mid-coding.
- About 300 estimates have been computed on this dataset; nothing here is
  confirmatory.
- D10 double-BOS condition applies to all traces.
- Rates are lower bounds; deployer exposure limits are a policy idea, not
  measured.

## The confirmatory test this sets up

On fresh data (a different model on TruthfulQA, or another benchmark), with the
codebook frozen (lure promoted to a first-class code), a second independent
coder with agreement reported before any AUC, the comparator pre-specified as
substantive answers, and the predictions below written down first:

1. Fabrication mean-entropy AUC > 0.65; famous-name-lure AUC < 0.55.
2. Each targeted prompt clause reduces its own card's rate without raising
   refusals on previously correct answers beyond an agreed margin.
3. Same-question sampling: entropy separates right from wrong answers to the
   same question (holds difficulty fixed).
