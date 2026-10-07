# RUN_SPEC

- model: `unsloth/Llama-3.2-3B-Instruct`
- backend: hf full-vocab
- temperature: 0.0 (greedy)
- seed: 0 (calibration/eval shuffle)
- eval items: 300 | calibration items: 60 (excluded)
- preset: short_sequence (D1 calibrated, D4 min_rise disabled)
- secondary arm (sec.7.2 two-term): off (not runnable -- no evidence source; SPEC_DEVIATIONS.md E4)
- label method: token_f1_v0 (noisy both directions; dilutes toward base rate)

## Result
precision 0.4444 | recall 0.0656 | lift 2.1858
base rate 0.2033 | firing rate 0.0300 | p 0.0950

**VERDICT: KILL**
