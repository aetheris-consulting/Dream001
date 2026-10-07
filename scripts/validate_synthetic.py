#!/usr/bin/env python3
"""
Synthetic validation -- DETECTOR MECHANICS ONLY.

Reproduces the two validation modes:
  rolling    (spec defaults)  event-level P/R + detection latency, 60x1200
  calibrated (short_sequence) sequence-level P/R + lift + permutation p, 100

THIS DOES NOT TEST THE HYPOTHESIS. Ground truth here is constructed. Whether
real hallucinations are preceded by these signatures is what the GPU run
decides. Never quote these numbers as evidence for the hypothesis.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dream001 import (RegimeConfig, RegimeEngine, extract_events,
                      compute_metrics, verdict)
from dream001.backends import SyntheticBackend


def validate_rolling(n_traces=60, n_tokens=1200, tol=40, seed0=0):
    """Event-level. A detection is a TP if it falls within tol tokens AFTER onset."""
    tp = fp = fn = 0
    latencies = []
    for k in range(n_traces):
        tr = SyntheticBackend(seed=seed0 + k).trace(n=n_tokens)
        eng = RegimeEngine(RegimeConfig.spec_default())
        states = eng.run(tr.entropy)
        events = extract_events(states)

        unmatched = list(tr.true_unwind_onsets)
        for ev in events:
            hit = None
            for on in unmatched:
                if on - 5 <= ev.start <= on + tol:
                    hit = on
                    break
            if hit is not None:
                tp += 1
                latencies.append(ev.start - hit)
                unmatched.remove(hit)
            else:
                fp += 1
        fn += len(unmatched)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "mode": "rolling", "level": "event",
        "n_traces": n_traces, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "median_latency_tokens": int(np.median(latencies)) if latencies else None,
    }


def validate_calibrated(n_traces=100, seed0=1000, hallucination_rate=0.5):
    """
    Sequence-level, short answers. D1: rolling window cannot fill, so percentiles
    come from a per-model reference distribution built from CALIBRATION traces
    that are excluded from metrics.
    """
    rng = np.random.default_rng(seed0)

    ref = []
    for k in range(40):
        b = SyntheticBackend(seed=seed0 + 5000 + k)
        ref.extend(b.short_answer(n=int(rng.integers(24, 44)),
                                  hallucinate=bool(rng.random() < hallucination_rate)))
    reference = np.asarray(ref)

    cfg = RegimeConfig.short_sequence()
    flags, labels = [], []
    for k in range(n_traces):
        halluc = bool(rng.random() < hallucination_rate)
        b = SyntheticBackend(seed=seed0 + k)
        h = b.short_answer(n=int(rng.integers(24, 44)), hallucinate=halluc)
        eng = RegimeEngine(cfg, reference=reference)
        states = eng.run(h)
        events = extract_events(states, config=cfg)
        flags.append(len(events) > 0)
        labels.append(halluc)

    m = compute_metrics(flags, labels, n_permutations=2000, seed=7)
    v = verdict(m)
    out = m.to_dict()
    out.update({"mode": "calibrated", "level": "sequence", "verdict": v.label})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    nt = 12 if a.quick else 60
    ns = 40 if a.quick else 100

    r = validate_rolling(n_traces=nt)
    c = validate_calibrated(n_traces=ns)

    print("\n" + "=" * 62)
    print("DREAM-001 synthetic validation -- DETECTOR MECHANICS ONLY")
    print("=" * 62)
    print(f"\n[rolling / event-level]  {r['n_traces']} traces x 1200 tokens")
    print(f"  precision {r['precision']:.3f}   recall {r['recall']:.3f}"
          f"   tp={r['tp']} fp={r['fp']} fn={r['fn']}")
    print(f"  median detection latency: {r['median_latency_tokens']} tokens")
    print(f"\n[calibrated / sequence-level]  {c['n']} short answers")
    print(f"  precision {c['precision']:.3f}   recall {c['recall']:.3f}"
          f"   lift {c['lift']:.3f}   p {c['permutation_p']:.4f}")
    print(f"  base rate {c['base_rate']:.3f}   firing rate {c['firing_rate']:.3f}")
    print(f"  gate verdict on synthetic: {c['verdict']}")
    print("\n  These numbers validate the DETECTOR. The hypothesis is untested.")
    print("=" * 62 + "\n")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rolling": r, "calibrated": c}, f, indent=2)
