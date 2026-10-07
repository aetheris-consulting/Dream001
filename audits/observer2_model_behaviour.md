# Observer 2: model-behaviour audit of "uncertain commitment, confident rationalization"

Run: `runs/llama3.2-3b-d8-full` (Llama-3.2-3B-Instruct, greedy, 757 TruthfulQA items). CPU only. No models loaded, no run files modified.
Coder: Observer 2 (single coder). Date: 2026-10-01.

## Verdict

**The claim weakens a lot. In its strong form it fails.**

| Part of the claim | Status |
|---|---|
| The first false claim is usually made in the first sentence | **Supported**, 37/40 sampled errors. Expected, since the prompt asks for the answer first. |
| Entropy is elevated *at* the false token for errors where the model produces new content (fabrication, false premise, fiction-as-fact, most misconceptions) | **Supported** within the answer and against correct answers at the same position. It is local: tokens before the false token are not elevated. |
| "Uncertain at commitment" as a general property of errors | **Fails for famous-name lures and popular misconceptions.** These commit with low entropy ("Gates" 0.03 nats, "United States" 0.08, "Musk" 0.04, "Drink water" 0.33). |
| "Confident rationalization": explanations of errors are low-entropy | **Not supported.** On matched material, error explanations are slightly *higher* entropy than correct explanations. New false claims made inside explanations spike in entropy. The original headline-vs-explanation gap mostly comes from the correct-answer baseline being 56% "I have no comment" refusals. |
| Entropy tracks *wrongness* rather than question or answer type | **Not supported.** Real errors are indistinguishable from judge-FALSE-but-actually-correct answers (headline AUC 0.52). |

**What the data do support:** when the model writes a specific new false entity, number or law, the token that commits to it tends to be high-entropy. That spike is local; the answer was not elevated before it. Everything beyond that is speculative: "rationalizes", "uncertain at the moment of commitment" as a mechanism, and the causal reading.

## 1. A composition confound the original analysis missed

The original AUCs compare errors against **all** judge-TRUE answers. Of the 384 TRUE answers, 215 (56%) begin "I have no comment". These are formulaic and low-entropy. Only 2 of 216 real errors are refusals. I re-ran the comparison against the 169 substantive (non-refusal) TRUE answers. Real errors have mean headline length 13.6 tokens and substantive TRUE 13.2, so length is matched.

| Comparison | headline first-3 | headline mean | explanation mean |
|---|---|---|---|
| real errors vs all TRUE (original) | 0.78 [0.74, 0.81] | 0.79 [0.75, 0.83] | 0.63 [0.58, 0.67] |
| real errors vs **substantive TRUE** | 0.62 [0.56, 0.67] | 0.64 [0.58, 0.69] | 0.59 [0.53, 0.65] |
| famous-name lure (n=19) vs substantive TRUE | — | 0.62 [0.50, 0.74] | 0.35 [0.23, 0.46] |
| not_false (n=43) vs substantive TRUE | — | 0.63 [0.55, 0.71] | — |

- The headline-minus-explanation gap shrinks from 0.16 to 0.05 AUC.
- Most of the "headline" effect was refusal versus non-refusal.
- The headline/explanation split is off by one: the cut includes the following whitespace, so when the period is a bare `.` token the headline swallows the next word. This happens in 32/216 errors and 11/169 substantive TRUE.
  - 13 error headlines end inside a numbered list ("…including:\n\n1. Denmark"). No substantive TRUE headline does.
  - Fixing the off-by-one changes the AUC only from 0.637 to 0.633.

## 2. First false tokens (blind)

### Method
- **Sample.** I drew 40 real errors (code ≠ not_false), stratified by mechanism with seed 2026: fabrication 10, misconception 9, lure 6, false_premise 5, overclaim 4, reasoning (non-lure) 3, fiction_as_fact 3.
- **Blind labelling.** For each I read the question, the TruthfulQA references and the indexed token list only, with no entropy loaded. I labelled the *decisive* token: the first token after which no true continuation of the claim is plausible. Where a premise echo or clause start comes materially earlier, I recorded it as `first_false_tok_alt`. Each label has a confidence: 32 high, 6 medium, 2 low.
- **Labels frozen first.** They were written to `audits/observer2_first_false_tokens.jsonl` (sha256 `b8a86c9a…41ff`) before any entropy was opened.
- **Matched control.** Before looking at entropy, I also labelled the commitment token of 24 random judge-TRUE answers (`observer2_true_commit_tokens.jsonl`). Ten are substantive; 14 are refusals.
- **Measurements.** `entropy[k]` is the entropy of the distribution that emitted token k (confirmed in `scripts/token_traces.py`). For each label I measured:
  - its percentile within its own answer (end-of-turn excluded);
  - its percentile among substantive-TRUE answers at the same absolute position;
  - for errors with the false token at position ≥ 2, the same position-matched percentile for every token *before* it.

### Results by mechanism

| group | n | false token in headline | at pos 0 | median entropy at false token (nats) | median within-answer percentile | top-10% of answer | median percentile vs TRUE at same position |
|---|---|---|---|---|---|---|---|
| fabrication | 10 | 8 | 2 | 2.15 | 0.93 | 5/10 | 0.92 |
| false_premise | 5 | 4 | 0 | 2.10 | 0.91 | 4/5 | 0.96 |
| fiction_as_fact | 3 | 3 | 0 | 3.13 | 0.99 | 2/3 | 0.99 |
| misconception | 9 | 9 | 3 | 1.68 | 0.91 | 5/9 | 0.88 |
| overclaim | 4 | 4 | 2 | 1.18 | 0.76 | 0/4 | 0.47 |
| reasoning | 3 | 3 | 2 | 1.23 | 0.90 | 1/3 | 0.50 |
| **lure** | 6 | 6 | 1 | **0.42** | **0.51** | **0/6** | **0.46** |
| ALL | 40 | **37** | 10 | 1.54 | 0.86 | 17/40 | 0.87 |
| TRUE substantive commitment tokens | 10 | — | 4 | 0.69 | 0.64 | 0/10 | — |

- **Comparison with correct answers.** Entropy at the false token is higher than at correct-answer commitment tokens: AUC 0.74 [0.61, 0.86], with only 10 controls. The false token is the single highest-entropy token of its answer in only 2/40 cases.
- **No elevation beforehand.** Across 325 tokens that precede the false token (errors with false token ≥ 2), the median percentile against TRUE at the same position is **0.51**. Entropy is not elevated before the false token, so the spike is local rather than a whole-answer state.
- **Low-entropy commitments.** 9/40 false tokens are below the median entropy of correct commitments (0.69):
  - 4 of 6 lures (3 Musk 0.04, 600 Gates 0.03, 81 New England 0.24, 575 York 0.61);
  - 4 of 9 misconceptions (567 United States 0.08, 617 water 0.33, 703 Edison 0.45, 577 British 0.61);
  - 1 overclaim (319, 0.12).
  - In these cases the uncertainty in the headline sits on the *format* token before the name ("Bill" 2.12, "New" 1.68, "Drink" 2.40), not on the false content.
- **Where the false token lands.** "First false token in the explanation" is rare: 3/40.
  - Two are fabrications where the headline is plausibly true and a number is invented in the explanation: 88 tea "4.5 kg", and 589 "one-fifth of a circle" (low confidence).
  - One is a false-premise list item: 150 conveyancing.
  - If my earlier `alt` labels are used for the false-premise cases (premise echo), all three of those land in the headline.
- **Does the headline elevation come only from the false token?**
  - In the sample, the headline excluding the false token ±1 still separates from substantive TRUE (0.65, versus 0.69 with it).
  - Across all 216 errors, dropping each headline's single highest-entropy token moves the AUC from 0.64 to 0.60.
  - So the elevation is not one token, but it is modest.

## 3. Rationalization check

Explanation tokens were classed as punctuation, function word, subword continuation, content word (word-initial) or number. Content words were further split into copied (the word already appears in the question or earlier in the answer) and novel. All comparisons are real errors vs substantive TRUE.

| explanation token class | AUC real vs TRUE-sub | mean entropy real / TRUE-sub | lure vs TRUE-sub |
|---|---|---|---|
| punctuation | 0.55 [0.49, 0.61] | 0.52 / 0.48 | 0.59 |
| function words | 0.58 [0.52, 0.64] | 0.75 / 0.69 | 0.42 |
| subword continuations | 0.55 [0.48, 0.62] | 0.19 / 0.18 | 0.47 |
| content words | 0.58 [0.52, 0.64] | 1.03 / 0.94 | **0.36** |
| content, copied | 0.55 [0.50, 0.61] | 0.79 / 0.72 | 0.44 |
| content, novel | 0.58 [0.53, 0.64] | 1.13 / 1.03 | 0.38 |

- **Same mix of token types.** Explanations of errors and correct answers have the same token-class composition (content 43.6% vs 43.6%, function 37.6% vs 38.5%), so "formulaic text" does not differ by correctness.
- **Lexically matched.** Over the 336 content-word types appearing at least 3 times in both groups' explanations, errors are **+0.10 nats higher on the same word** [0.05, 0.16], and higher on 57% of types.
- **By position in the explanation:** tokens 0–10 AUC 0.59, 10–30 0.60, 30–60 0.58, 60+ 0.51.
- **Relative to the answer's own headline:** the explanation-minus-headline AUC is 0.42 [0.37, 0.48]. The *drop* is bigger for errors only because their headline starts higher.
- **New false claims inside explanations.** I labelled 10 such claims after seeing per-answer aggregate entropies but not these tokens, so they are not fully blind. 8/10 sit at or above the 85th percentile of their answer. Examples:
  - 13 invented "Offences Against the Person Act 1861": 3.67 nats.
  - 203 invented "letter to his brother John": 3.01.
  - 600 Gates "former mayor of Seattle": 2.77.
  - 575 NYC "associated with Hungarians": 2.41.
  - Only 81 "nicknamed the Pilgrims" (0.54) and 251 "Equal Pay Act" (1.08) are modest.
  - **When the explanation makes up something new, it is not confident.**
- **Lures.** Lure explanations are low-entropy. They also have the highest copied-content share (0.35 vs 0.28) and mostly recite true, well-known facts about the wrongly chosen famous entity (Gates co-founded Microsoft; the Red Sox and Fenway). That is fluent recall about a confidently chosen entity, not confident rationalization after an uncertain choice. The lure commitment itself was already low-entropy.

**Conclusion for task 3.** Explanations are lower-entropy than headlines for everyone, which is formulaic and position-driven. Error explanations are not more "confident" than correct ones on matched material. The rationalization reading is not supported.

## 4. Difficulty

| test | result |
|---|---|
| (a) real errors vs **judge-FALSE-but-not-false** (both flagged by the judge; only one is actually wrong) | h_first3 0.53 [0.43, 0.63], h_mean **0.52** [0.43, 0.60], e_mean 0.49 [0.40, 0.58] |
| (b) independent model: Qwen2.5-3B mean entropy on the same 300 questions (`runs/qwen2.5-3b-d6d7`, same idx/question order) predicting Llama error | AUC 0.46 [0.37, 0.55]. Correlation with Llama headline entropy is 0.14. In a logistic model, Llama headline mean gives +0.42 ± 0.19 per SD and Qwen entropy −0.01 ± 0.18. Within Qwen-entropy terciles, Llama headline AUC is 0.70 / 0.57 / 0.58 (n ≈ 45 each, CIs span 0.5). |
| (c) entropy before the false token, matched by position to TRUE | median percentile 0.51 (no elevation) |

**Reading the three tests:**
- **(a) is the cleanest test available, and it fails to separate.** Entropy is just as elevated in correct answers to contested or flagged questions as in real errors. This is consistent with entropy tracking "substantive, non-formulaic answer to a hard or contested question" rather than falsity. Caveats: n = 43, and the not_false answers are hedged or borderline, so they are not clean correct answers.
- **(b) rules out only generic question difficulty as another model sees it.** The Qwen run uses a different prompt and noisy token-F1 labels, so it is a weak difficulty proxy.
- **(c) argues against a whole-answer "hard question" state that starts at the prompt.** But under greedy decoding a hard question will still show up as entropy at the decision token, so (c) cannot separate "hard" from "wrong".

**The cleanest test, which was not run because it needs the model.** For each question, teacher-force or sample the model's own continuations from the prefix just before the decisive token. Compare decision-token entropy and outcome *within the same question*, across samples (T > 0) or across forced correct versus incorrect reference answers. A within-question contrast holds difficulty fixed. It needs model forward passes; a 3B model on CPU is feasible but slow, and it was out of scope here.

## Limitations

- **Single coder.** One coder (me) labelled every token. There was no inter-rater check, and decisive-token placement is uncertain by ±1–2 tokens. 8/40 labels are medium or low confidence, including 589, 519, 181 and 144.
- **Sample size.** 40 errors and 10 substantive correct commitment tokens give per-mechanism n of 3 to 10. Per-mechanism numbers are descriptive only.
- **Secondary-claim labels not fully blind.** Those 10 labels were made after seeing aggregate per-answer entropies.
- **Mechanism codes and judge labels taken as given.** Several codes are borderline (for example 144 chameleons, 519 "often characterized as").
- **Classifier limits.** The function-word list and token classifier are heuristic.
- **Greedy decoding.** One answer per question means question difficulty and answer wrongness cannot be separated with this data.

## Files (all new, under `audits/`)

- `observer2_first_false_tokens.jsonl`: blind first-false-token labels (40).
- `observer2_true_commit_tokens.jsonl`: blind commitment-token labels for 24 judge-TRUE answers.
- `observer2_secondary_false_tokens.jsonl`: false claims inside explanations (10, semi-blind).
- `observer2_analysis.py`: reproduces every number above. Run with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe audits/observer2_analysis.py`.
- `observer2_results.json`: full outputs, including per-row first-false-token data.
