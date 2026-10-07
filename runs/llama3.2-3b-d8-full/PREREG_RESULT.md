# Pre-registered result — llama3.2-3b-d8-full

**DECISION: NOT SUPPORTED** (rules: PREREGISTRATION_D8.md)

Judge validity: 52/60 agreement with blind annotation = 0.867
(threshold 0.8) -> VALID.
Judge labels: {'TRUE': 384, 'FALSE': 259, 'DECLINE': 114}

| analysis | n | flagged | base rate | precision | recall | lift | p | verdict |
|---|---|---|---|---|---|---|---|---|
| PRIMARY judge, all | 757 | 86 | 0.342 | 0.349 | 0.116 | 1.020 | 0.4763 | KILL |
| judge, >= 9 tok (diagnostic) | 706 | 86 | 0.367 | 0.349 | 0.116 | 0.951 | 0.6912 | — |
| secondary token-F1, all | 757 | 86 | 0.258 | 0.209 | 0.092 | 0.813 | 0.9045 | KILL |

Primary verdict reasons: precision 0.349 below ex-ante threshold 0.6 (set 2026-04-12)
