"""
Metrics and honesty gates.

Bare precision is gameable: on a task where the model hallucinates 55% of the
time, a coin-flip detector scores ~55% precision. Every report therefore
carries base rate, lift, firing rate, and a label-permutation p-value.

K1 -- precision <= base rate (lift <= 1): the signal carries no information.
K2 -- precision > 0.60 but p >= 0.05 or lift <= 1.15: INCONCLUSIVE, not a pass.
K3 -- firing rate ~= 1: trivial sensitivity, not a precursor.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence, Dict, Any, Optional

import numpy as np

PRECISION_THRESHOLD = 0.60
LIFT_THRESHOLD = 1.15
P_THRESHOLD = 0.05
FIRING_RATE_CEILING = 0.95


@dataclass
class Metrics:
    n: int
    n_flagged: int
    n_positive: int          # ground-truth hallucinations
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    base_rate: float
    lift: float
    firing_rate: float
    permutation_p: float
    n_permutations: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_metrics(
    flags: Sequence[bool],
    labels: Sequence[bool],
    n_permutations: int = 2000,
    seed: int = 0,
) -> Metrics:
    """
    flags  -- detector said "this answer is hallucinated"
    labels -- ground truth "this answer is hallucinated"
    """
    f = np.asarray(list(flags), dtype=bool)
    y = np.asarray(list(labels), dtype=bool)
    if f.shape != y.shape:
        raise ValueError(f"length mismatch: {f.shape} vs {y.shape}")
    n = int(f.size)
    if n == 0:
        raise ValueError("empty evaluation set")

    tp = int((f & y).sum())
    fp = int((f & ~y).sum())
    fn = int((~f & y).sum())
    tn = int((~f & ~y).sum())

    n_flagged = tp + fp
    n_positive = tp + fn
    precision = tp / n_flagged if n_flagged else 0.0
    recall = tp / n_positive if n_positive else 0.0
    base_rate = n_positive / n
    lift = precision / base_rate if base_rate > 0 else 0.0
    firing_rate = n_flagged / n

    # label permutation: how often does a shuffled label set do this well?
    rng = np.random.default_rng(seed)
    if n_flagged == 0:
        p = 1.0
    else:
        ge = 0
        for _ in range(n_permutations):
            perm = rng.permutation(y)
            ge += int((f & perm).sum() / n_flagged >= precision)
        p = (ge + 1) / (n_permutations + 1)

    return Metrics(
        n=n, n_flagged=n_flagged, n_positive=n_positive,
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall,
        base_rate=base_rate, lift=lift, firing_rate=firing_rate,
        permutation_p=float(p), n_permutations=n_permutations,
    )


@dataclass
class Verdict:
    passed: bool
    k1_failed: bool
    k2_failed: bool
    k3_failed: bool
    label: str
    reasons: list

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verdict(m: Metrics) -> Verdict:
    """Applies K1-K3 mechanically. Clearing 0.60 is necessary, never sufficient."""
    reasons = []

    k1 = m.lift <= 1.0
    if k1:
        reasons.append(
            f"K1: lift {m.lift:.3f} <= 1 (precision {m.precision:.3f} vs "
            f"base rate {m.base_rate:.3f}) -- signal carries no information"
        )

    k3 = m.firing_rate >= FIRING_RATE_CEILING
    if k3:
        reasons.append(
            f"K3: firing rate {m.firing_rate:.3f} -- detector fires on nearly "
            "everything; trivial sensitivity, not a precursor"
        )

    k2 = (m.precision > PRECISION_THRESHOLD) and (
        m.permutation_p >= P_THRESHOLD or m.lift <= LIFT_THRESHOLD
    )
    if k2:
        reasons.append(
            f"K2: precision {m.precision:.3f} clears {PRECISION_THRESHOLD} but "
            f"p={m.permutation_p:.4f} / lift={m.lift:.3f} -- threshold is doing "
            "no work; INCONCLUSIVE"
        )

    gates_clear = not (k1 or k2 or k3)
    passed = gates_clear and m.precision >= PRECISION_THRESHOLD

    if passed:
        label = "PASS"
    elif k1 or k3:
        label = "KILL"
    elif k2:
        label = "INCONCLUSIVE"
    elif m.precision < PRECISION_THRESHOLD:
        label = "KILL"
        reasons.append(
            f"precision {m.precision:.3f} below ex-ante threshold "
            f"{PRECISION_THRESHOLD} (set 2026-04-12)"
        )
    else:
        label = "INCONCLUSIVE"

    return Verdict(
        passed=passed, k1_failed=k1, k2_failed=k2, k3_failed=k3,
        label=label, reasons=reasons,
    )


def format_report(m: Metrics, v: Verdict, config: Optional[dict] = None) -> str:
    lines = [
        "=" * 66,
        "DREAM-001 -- logit entropy regime model",
        "hypothesis: confident_unwind precedes hallucination",
        "=" * 66,
        "",
        f"  n answers          {m.n}",
        f"  flagged            {m.n_flagged}   (firing rate {m.firing_rate:.3f})",
        f"  ground-truth pos   {m.n_positive}   (base rate    {m.base_rate:.3f})",
        "",
        f"  precision          {m.precision:.3f}      <-- threshold {PRECISION_THRESHOLD}",
        f"  recall             {m.recall:.3f}",
        f"  lift               {m.lift:.3f}      <-- must exceed {LIFT_THRESHOLD}",
        f"  permutation p      {m.permutation_p:.4f}     <-- must be < {P_THRESHOLD}"
        f"  ({m.n_permutations} shuffles)",
        "",
        f"  confusion          tp={m.tp} fp={m.fp} fn={m.fn} tn={m.tn}",
        "",
        "-" * 66,
        f"  VERDICT: {v.label}",
    ]
    for r in v.reasons:
        lines.append(f"    - {r}")
    if not v.passed and m.precision >= PRECISION_THRESHOLD:
        lines.append("")
        lines.append("  *** DO NOT REPORT THIS AS A PASS ***")
    lines.append("-" * 66)
    if config:
        lines.append("  config echo:")
        for k, val in sorted(config.items()):
            lines.append(f"    {k} = {val}")
    return "\n".join(lines)
