# RUN_SPEC

- model: `Qwen/Qwen2.5-3B-Instruct`
- backend: hf full-vocab
- temperature: 0.0 (greedy)
- seed: 0 (calibration/eval shuffle)
- eval items: 300 | calibration items: 60 (excluded)
- preset: short_sequence (D1 calibrated, D4 min_rise disabled)
- secondary arm (sec.7.2 two-term): off (not runnable -- no evidence source; SPEC_DEVIATIONS.md E4)
- label method: token_f1_v0 (noisy both directions; dilutes toward base rate)

## Result
precision 0.2500 | recall 0.0244 | lift 1.8293
base rate 0.1367 | firing rate 0.0133 | p 0.4543

**VERDICT: KILL**
