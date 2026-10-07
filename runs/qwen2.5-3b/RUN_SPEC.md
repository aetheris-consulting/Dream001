# RUN_SPEC

- model: `Qwen/Qwen2.5-3B-Instruct`
- backend: hf full-vocab
- temperature: 0.0 (greedy)
- seed: 0 (calibration/eval shuffle)
- eval items: 300 | calibration items: 60 (excluded)
- preset: short_sequence (D1 calibrated, D4 min_rise disabled)
- secondary arm (sec.7.2 two-term): off
- label method: token_f1_v0 (noisy both directions; dilutes toward base rate)

## Result
precision 0.0112 | recall 0.0909 | lift 0.3064
base rate 0.0367 | firing rate 0.2967 | p 0.9780

**VERDICT: KILL**
