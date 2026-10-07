"""
Event extraction and answer-level rollup.

ISCOT v1.2 sec.7.1: token-level events vs answer-level ground truth is not an
engineering annoyance, it is the mismatch predicted by sec.2 -- the event is a
transition, the label is a level. Do NOT force token-wise alignment. Score at
the answer level (any-hit); retain token events as diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Dict, Any

from .regime import TokenState, CONF_UNWIND, RegimeConfig


@dataclass
class UnwindEvent:
    start: int
    end: int
    peak_slope: float
    rise: Optional[float]
    entry_percentile: Optional[float]
    peak_entropy: float
    basin: Optional[float]
    n_tokens: int
    envelope_confirmed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_events(
    states: Sequence[TokenState],
    merge_gap: int = 3,
    min_event_tokens: int = 2,
    evidence_mask: Optional[Sequence[bool]] = None,
    config: Optional[RegimeConfig] = None,
) -> List[UnwindEvent]:
    """
    Collapse runs of confident_unwind tokens into events.

    min_event_tokens: an unwind IS a multi-token pattern (README limitation 1);
    a single-token qualifying sample is an impulse, not a regime transition.
    Set to 1 to disable.

    evidence_mask (sec.7.2, OPTIONAL, default off): a per-token boolean that is
    True where evidence arrived (attention mass onto context, retrieval hit,
    low-entropy copy from context). When config.require_no_evidence is set, an
    event is suppressed if evidence arrived within evidence_lookback tokens of
    its start. This is a declared SECONDARY ARM -- the single-term hypothesis
    must still be testable cleanly, and this gate must never be tuned on
    synthetic data (D4 contamination risk).

    When require_no_evidence is set the mask is MANDATORY and must be aligned
    token-for-token with `states`. An armed gate with nothing to gate on is a
    PROVENANCE failure, not a convenience: it would score the single-term
    detector while letting the caller record the result under a two-term label
    (SPEC_DEVIATIONS.md E4). It raises instead.
    """
    cfg = config or RegimeConfig()

    # Checked before the no-hits early return: the gate is either armed or it is
    # not, independently of whether this particular trace happened to fire.
    if cfg.require_no_evidence:
        if evidence_mask is None:
            raise ValueError(
                "require_no_evidence is set but no evidence_mask was supplied. "
                "The sec.7.2 secondary arm must not degrade silently to the "
                "single-term detector -- that records single-term results under "
                "a two-term label. Supply a per-token evidence_mask, or leave "
                "require_no_evidence off (SPEC_DEVIATIONS.md E4)."
            )
        if len(evidence_mask) != len(states):
            raise ValueError(
                f"evidence_mask has {len(evidence_mask)} entries but there are "
                f"{len(states)} token states; a misaligned mask suppresses the "
                "wrong events"
            )

    hits = [i for i, s in enumerate(states) if s.regime == CONF_UNWIND]
    if not hits:
        return []

    groups: List[List[int]] = [[hits[0]]]
    for i in hits[1:]:
        if i - groups[-1][-1] <= merge_gap:
            groups[-1].append(i)
        else:
            groups.append([i])

    events: List[UnwindEvent] = []
    for g in groups:
        if len(g) < min_event_tokens:
            continue
        seg = [states[i] for i in g]
        start, end = g[0], g[-1]

        if cfg.require_no_evidence:
            lo = max(0, start - cfg.evidence_lookback)
            if any(bool(e) for e in evidence_mask[lo:start + 1]):
                continue  # confidence collapsed, but something arrived to justify it

        rises = [s.rise for s in seg if s.rise is not None]
        events.append(
            UnwindEvent(
                start=start,
                end=end,
                peak_slope=max((s.slope or 0.0) for s in seg),
                rise=max(rises) if rises else None,
                entry_percentile=seg[0].percentile,
                peak_entropy=max(s.entropy for s in seg),
                basin=seg[0].basin,
                n_tokens=len(g),
                envelope_confirmed=any(bool(s.envelope_confirms) for s in seg),
            )
        )
    return events


@dataclass
class AnswerVerdict:
    """sec.7.1 -- an answer is FLAGGED if >=1 qualifying unwind event occurs."""
    flagged: bool
    n_events: int
    first_event_token: Optional[int]
    n_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score_answer(states: Sequence[TokenState], events: Sequence[UnwindEvent]) -> AnswerVerdict:
    return AnswerVerdict(
        flagged=len(events) > 0,
        n_events=len(events),
        first_event_token=events[0].start if events else None,
        n_tokens=len(states),
    )
