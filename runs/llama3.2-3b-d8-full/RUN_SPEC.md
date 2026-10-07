# RUN_SPEC

- model: `meta-llama/Llama-3.2-3B-Instruct`
- backend: hf full-vocab
- temperature: 0.0 (greedy)
- seed: 0 (calibration/eval shuffle)
- eval items: 757 | calibration items: 60 (excluded)
- preset: short_sequence (D1 calibrated, D4 min_rise disabled)
- prompt: longform (D8 declared deviation -- long-form elicitation)
- secondary arm (sec.7.2 two-term): off (not runnable -- no evidence source; SPEC_DEVIATIONS.md E4)
- label method: token_f1_v0 (noisy both directions; dilutes toward base rate)

## Result
precision 0.2093 | recall 0.0923 | lift 0.8125
base rate 0.2576 | firing rate 0.1136 | p 0.9045

**VERDICT: KILL**

## Length diagnostic (not a gate)
706/757 answers >= 9-token event floor; median answer
87 tokens; 37 hit the generation cap.
Among capable answers: precision 0.2093 | base rate 0.2762 | lift 0.7578 | p 0.9520 | flagged 86/706
