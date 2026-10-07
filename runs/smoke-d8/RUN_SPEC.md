# RUN_SPEC

- model: `meta-llama/Llama-3.2-3B-Instruct`
- backend: hf full-vocab
- temperature: 0.0 (greedy)
- seed: 0 (calibration/eval shuffle)
- eval items: 12 | calibration items: 12 (excluded)
- preset: short_sequence (D1 calibrated, D4 min_rise disabled)
- prompt: longform (D8 declared deviation -- long-form elicitation)
- secondary arm (sec.7.2 two-term): off (not runnable -- no evidence source; SPEC_DEVIATIONS.md E4)
- label method: token_f1_v0 (noisy both directions; dilutes toward base rate)

## Result
precision 0.0000 | recall 0.0000 | lift 0.0000
base rate 0.2500 | firing rate 0.0833 | p 1.0000

**VERDICT: KILL**

## Length diagnostic (not a gate)
11/12 answers >= 9-token event floor; median answer
99 tokens; 1 hit the generation cap.
Among capable answers: precision 0.0000 | base rate 0.2727 | lift 0.0000 | p 1.0000 | flagged 1/11
