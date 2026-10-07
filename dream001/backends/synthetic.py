"""
Synthetic ground-truth generator.

VALIDATES THE DETECTOR, NOT THE HYPOTHESIS. Numbers produced here say nothing
about whether real hallucinations are preceded by these signatures and must
never be quoted as if they do.

Trace grammar:
  baseline AR(1) wander
  + planted episodes: confident plateau -> sharp entropy ramp (TRUE unwind)
  + adversarial distractors, each defeating a naive detector:
      uncertain episode         high-entropy excursion; must not fire
      single-token spike        impulse, not a ramp
      AR(1) deep mean reversion recovers own dip only; defeated by min_rise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SyntheticTrace:
    entropy: np.ndarray
    true_unwind_onsets: List[int]
    distractor_spans: List[Tuple[int, int, str]]


class SyntheticBackend:
    def __init__(self, seed: int = 0, base: float = 1.3, sigma: float = 0.12, phi: float = 0.72):
        self.rng = np.random.default_rng(seed)
        self.base, self.sigma, self.phi = base, sigma, phi

    def _ar1(self, n: int) -> np.ndarray:
        e = np.zeros(n)
        e[0] = self.base
        for i in range(1, n):
            e[i] = self.base + self.phi * (e[i - 1] - self.base) + self.rng.normal(0, self.sigma)
        return e

    def trace(self, n: int = 1200, n_episodes: int = 3) -> SyntheticTrace:
        h = self._ar1(n)
        onsets: List[int] = []
        spans: List[Tuple[int, int, str]] = []

        slots = list(np.linspace(0.10, 0.86, n_episodes * 2))
        self.rng.shuffle(slots)

        for k in range(n_episodes):
            s = int(slots[k] * n)
            plateau = int(self.rng.integers(45, 80))
            ramp = int(self.rng.integers(18, 30))
            if s + plateau + ramp + 40 >= n:
                continue
            floor = self.base - self.rng.uniform(0.75, 0.95)
            h[s:s + plateau] = floor + self.rng.normal(0, 0.06, plateau)
            onset = s + plateau
            peak = floor + self.rng.uniform(1.6, 2.1)
            h[onset:onset + ramp] = np.linspace(floor, peak, ramp) + self.rng.normal(0, 0.07, ramp)
            decay = min(60, n - (onset + ramp))
            h[onset + ramp:onset + ramp + decay] = (
                np.linspace(peak, self.base + 0.15, decay) + self.rng.normal(0, 0.09, decay)
            )
            onsets.append(int(onset))
            spans.append((s, onset + ramp, "true_unwind"))

        for k in range(n_episodes, len(slots)):
            s = int(slots[k] * n)
            kind = ["uncertain_episode", "single_token_spike", "ar1_reversion"][k % 3]
            if kind == "uncertain_episode":
                w = min(int(self.rng.integers(45, 70)), n - s - 1)
                if w < 10:
                    continue
                bump = np.concatenate([np.linspace(0, 1.0, w // 2),
                                       np.linspace(1.0, 0, w - w // 2)])
                h[s:s + w] += bump + self.rng.normal(0, 0.08, w)
                spans.append((s, s + w, kind))
            elif kind == "single_token_spike":
                if s + 1 < n:
                    h[s] += self.rng.uniform(1.2, 1.6)
                    spans.append((s, s + 1, kind))
            else:
                # Dip depth is set so this distractor's recovery rise lands at
                # FP p90 ~ 0.7 nats -- the geometry D4's min_rise=1.0 was
                # calibrated against (README, spec deviations log). A deeper dip
                # would push FP rise past the constant and make D4 look broken
                # when it is the GENERATOR that drifted.
                w = min(int(self.rng.integers(25, 40)), n - s - 1)
                if w < 10:
                    continue
                dip = self.rng.uniform(0.28, 0.42)
                half = w // 2
                h[s:s + half] -= dip
                h[s + half:s + w] -= np.linspace(dip, 0, w - half)
                spans.append((s, s + w, kind))

        return SyntheticTrace(np.clip(h, 0.01, None), sorted(onsets), spans)

    def short_answer(self, n: int = 34, hallucinate: bool = False):
        """
        Short-sequence trace for calibrated-mode (D1) validation. Mirrors a
        TruthfulQA-length answer: sub-window, so the rolling window can never
        fill. Ground truth is CONSTRUCTED -- this validates plumbing only.
        """
        h = self._ar1(n)
        if hallucinate and n >= 16:
            s = int(n * 0.30)
            plateau = max(4, int(n * 0.22))
            floor = self.base - self.rng.uniform(0.75, 0.95)
            h[s:s + plateau] = floor + self.rng.normal(0, 0.05, plateau)
            onset = s + plateau
            ramp = min(max(5, int(n * 0.22)), n - onset)
            if ramp >= 3:
                peak = floor + self.rng.uniform(1.6, 2.0)
                h[onset:onset + ramp] = (np.linspace(floor, peak, ramp)
                                         + self.rng.normal(0, 0.06, ramp))
                if onset + ramp < n:
                    h[onset + ramp:] = self.base + self.rng.normal(0, 0.12, n - onset - ramp)
        return np.clip(h, 0.01, None)
