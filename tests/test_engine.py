"""Deterministic engine tests. No randomness, no network, no model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from dream001 import (RegimeConfig, RegimeEngine, extract_events, score_answer,
                      compute_metrics, verdict, AndonCord, CONF_UNWIND,
                      label_answer, token_f1)
from dream001.harness import run_eval


def _jitter(n, amp=0.02):
    """Deterministic, reproducible micro-variation. Real entropy is never
    exactly constant; an exactly constant series has an undefined percentile
    (see test_exactly_constant_series_cannot_arm)."""
    return amp * np.sin(np.arange(n) * 1.7)


def _plateau_then_ramp(baseline=140, plateau=60, ramp=25, floor=0.4, peak=2.4, tail=40):
    """
    Rolling mode ranks against HISTORY. A trace that opens directly on the
    plateau ranks the plateau against itself and cannot arm -- which is exactly
    the short-answer failure D1/calibrated mode exists to solve. Real
    generations have a baseline; the fixture does too.
    """
    return np.concatenate([
        np.full(baseline, 1.3) + _jitter(baseline, 0.10),
        np.full(plateau, floor) + _jitter(plateau),
        np.linspace(floor, peak, ramp),
        np.full(tail, peak * 0.6) + _jitter(tail),
    ])


def test_exactly_constant_series_cannot_arm():
    """Documented limitation: the percentile of a constant series is 0.5 by the
    half-tie convention, so the latch cannot arm. Real logit entropy always
    varies; this is a property of rank statistics, not a defect to paper over."""
    st = RegimeEngine(RegimeConfig.spec_default()).run(np.full(300, 0.4))
    assert not any(s.armed for s in st)


def test_plateau_then_ramp_fires_confident_unwind():
    st = RegimeEngine(RegimeConfig.spec_default()).run(_plateau_then_ramp())
    ev = extract_events(st)
    assert len(ev) >= 1, "a held confident plateau followed by a real ramp must fire"
    assert 195 <= ev[0].start <= 230, f"event at {ev[0].start} should sit on the ramp"
    assert any(s.regime == CONF_UNWIND for s in st)


def test_flat_trace_never_fires():
    st = RegimeEngine(RegimeConfig.spec_default()).run(np.full(400, 1.2) + _jitter(400))
    assert extract_events(st) == [], "a flat trace has no transition to detect"


def test_single_token_spike_rejected():
    """An impulse is not a regime transition (README limitation 1)."""
    h = np.full(300, 1.0) + _jitter(300, 0.10)
    h[200] = 4.0
    st = RegimeEngine(RegimeConfig.spec_default()).run(h)
    assert extract_events(st) == []


def test_shallow_reversion_rejected_by_min_rise():
    """D4: AR(1) reversion recovers only its own dip."""
    h = np.concatenate([np.full(140, 1.3) + _jitter(140, 0.10),
                        np.full(80, 0.4) + _jitter(80),
                        np.linspace(0.4, 1.0, 25), np.full(60, 1.0) + _jitter(60)])
    st = RegimeEngine(RegimeConfig.spec_default()).run(h)
    assert extract_events(st) == [], "rise below min_rise must not qualify"


def test_d5_held_position_required():
    """A brief dip arms the latch but has not HELD extreme positioning."""
    cfg = RegimeConfig.spec_default()
    h = np.concatenate([np.full(150, 1.3) + _jitter(150), np.full(8, 0.35) + _jitter(8),
                        np.linspace(0.35, 2.4, 25), np.full(40, 1.4) + _jitter(40)])
    st = RegimeEngine(cfg).run(h)
    assert extract_events(st) == [], "8-token dip is shorter than min_hold_run"


def test_calibrated_mode_requires_reference():
    eng = RegimeEngine(RegimeConfig.short_sequence())
    with pytest.raises(RuntimeError):
        eng.feed(1.0)


def test_calibrated_mode_short_sequence():
    ref = np.concatenate([np.full(200, 1.3), np.full(100, 0.4), np.full(100, 2.2)])
    cfg = RegimeConfig.short_sequence()
    eng = RegimeEngine(cfg, reference=ref)
    st = eng.run(np.concatenate([np.full(12, 0.35), np.linspace(0.35, 2.3, 14)]))
    assert len(extract_events(st, config=cfg)) >= 1


def _short_sequence_states():
    """Calibrated-mode fixture that fires exactly the way
    test_calibrated_mode_short_sequence asserts it does."""
    ref = np.concatenate([np.full(200, 1.3), np.full(100, 0.4), np.full(100, 2.2)])
    eng = RegimeEngine(RegimeConfig.short_sequence(), reference=ref)
    return eng.run(np.concatenate([np.full(12, 0.35), np.linspace(0.35, 2.3, 14)]))


def _two_term_config():
    cfg = RegimeConfig.short_sequence()
    cfg.require_no_evidence = True
    return cfg


def test_two_term_arm_without_mask_raises():
    """sec.7.2 E4: an armed gate with no mask must not degrade to the
    single-term detector -- that records single-term results under a two-term
    label. The gate refuses instead."""
    st = _short_sequence_states()
    with pytest.raises(ValueError, match="evidence_mask"):
        extract_events(st, config=_two_term_config())


def test_two_term_arm_refuses_even_when_nothing_fired():
    """The gate is armed or it is not; a quiet trace must not launder it."""
    st = RegimeEngine(RegimeConfig.spec_default()).run(np.full(400, 1.2) + _jitter(400))
    with pytest.raises(ValueError, match="evidence_mask"):
        extract_events(st, config=_two_term_config())


def test_misaligned_evidence_mask_raises():
    """A short mask silently suppresses the wrong events."""
    st = _short_sequence_states()
    with pytest.raises(ValueError, match="token states"):
        extract_events(st, evidence_mask=[False] * (len(st) - 3),
                       config=_two_term_config())


def test_evidence_mask_suppresses_and_admits():
    """With the gate armed and a real mask: evidence before the event suppresses
    it, no evidence lets the same event stand."""
    st = _short_sequence_states()
    cfg = _two_term_config()
    assert len(extract_events(st, evidence_mask=[False] * len(st), config=cfg)) >= 1
    assert extract_events(st, evidence_mask=[True] * len(st), config=cfg) == []


def test_run_eval_refuses_two_term_without_masks():
    """Refused before any generation, so a GPU run cannot burn a calibration
    pass and then write 'secondary arm: ON' over single-term results."""
    with pytest.raises(ValueError, match="evidence_masks"):
        run_eval(backend=None, items=[], config=_two_term_config())


def test_engine_is_causal():
    """No token's regime may depend on a future token."""
    h = _plateau_then_ramp()
    full = RegimeEngine(RegimeConfig.spec_default()).run(h)
    prefix = RegimeEngine(RegimeConfig.spec_default()).run(h[:180])
    assert [s.regime for s in full[:180]] == [s.regime for s in prefix]


def test_k1_kill_on_no_lift():
    m = compute_metrics([True] * 50 + [False] * 50, [True] * 25 + [False] * 25 +
                        [True] * 25 + [False] * 25, n_permutations=200)
    v = verdict(m)
    assert v.k1_failed and v.label == "KILL" and not v.passed


def test_k3_kill_on_trivial_firing():
    m = compute_metrics([True] * 100, [True] * 70 + [False] * 30, n_permutations=200)
    v = verdict(m)
    assert v.k3_failed and not v.passed


def test_precision_without_gates_is_not_a_pass():
    """Clearing 0.60 while lift is flat must never read as a pass."""
    flags = [True] * 70 + [False] * 30
    labels = [True] * 65 + [False] * 5 + [True] * 28 + [False] * 2
    m = compute_metrics(flags, labels, n_permutations=200)
    v = verdict(m)
    if m.precision > 0.60 and m.lift <= 1.15:
        assert not v.passed


def test_andon_rate_limits_but_logs():
    pulls = []
    cord = AndonCord(on_pull=pulls.append, max_pulls=2, per_tokens=500)
    for i in range(5):
        cord.consider(type("E", (), {"n_tokens": 4})(), token_index=i * 10)
    s = cord.summary()
    assert s["delivered"] == 2 and s["suppressed"] == 3 and s["considered"] == 5


def test_idk_is_not_hallucination():
    lab = label_answer("I have no comment.", ["Paris"], ["London"])
    assert lab.idk and not lab.hallucinated


def test_label_prefers_incorrect_match():
    lab = label_answer("The capital is London", ["The capital is Paris"],
                       ["The capital is London"])
    assert lab.hallucinated


def test_token_f1_bounds():
    assert token_f1("a b c", "a b c") == pytest.approx(1.0)
    assert token_f1("x y", "a b") == 0.0


class _RecordingBackend:
    """Fake backend: records the prompts it was given, returns a fixed trace."""
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "Paris.", [1.0 + 0.1 * (i % 3) for i in range(40)], [1.0] * 40


def test_d8_prompt_threads_through_calibration_and_eval():
    """D8: the D1 reference and the eval pass must share one template, or eval
    tokens are ranked against the wrong pool."""
    from dream001.harness import PROMPTS, build_reference
    be = _RecordingBackend()
    ref = build_reference(be, ["q1", "q2"], prompt=PROMPTS["longform"])
    run_eval(be, [{"question": "q3", "correct_answers": ["Paris"],
                   "incorrect_answers": ["London"]}],
             reference=ref, n_permutations=10, prompt=PROMPTS["longform"])
    assert len(be.prompts) == 3
    assert all("two or three sentences" in p for p in be.prompts)


def test_concise_prompt_is_still_the_default():
    from dream001.harness import PROMPT, build_reference
    be = _RecordingBackend()
    build_reference(be, ["q1"])
    assert be.prompts == [PROMPT.format(q="q1")]
