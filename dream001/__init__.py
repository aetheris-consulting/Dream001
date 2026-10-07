"""
DREAM-001 -- Logit Entropy Regime Model
Reference implementation. MIT. Aetheris Consulting LLC.

Provenance: ISCOT v1.2 (Internal State Commitment of Traders), session
ISCOT-AEF-2026-0412, disposition REFINE -> PROTOTYPE.

HYPOTHESIS (untested):
    confident_unwind entropy regime signatures in LLM logits are detectable
    precursors of hallucination events.

Synthetic validation exercises the DETECTOR. It says nothing about the
hypothesis. Do not quote synthetic numbers as evidence for it.
"""

__version__ = "0.2.0"

from .regime import (  # noqa: F401
    RegimeConfig, RegimeEngine, TokenState,
    CONF_UNWIND, CONF_BUILDUP, CONF_STABLE,
    UNC_UNWIND, UNC_BUILDUP, UNC_STABLE,
    NEUTRAL, WARMUP, ALL_REGIMES,
)
from .events import UnwindEvent, AnswerVerdict, extract_events, score_answer  # noqa: F401
from .andon import AndonCord, PullRecord, MODE_BUDGETS  # noqa: F401
from .report import (  # noqa: F401
    Metrics, Verdict, compute_metrics, verdict, format_report,
    PRECISION_THRESHOLD, LIFT_THRESHOLD, P_THRESHOLD,
)
from .labels import Label, label_answer, token_f1, is_idk  # noqa: F401

__all__ = [
    "RegimeConfig", "RegimeEngine", "TokenState", "ALL_REGIMES",
    "CONF_UNWIND", "CONF_BUILDUP", "CONF_STABLE",
    "UNC_UNWIND", "UNC_BUILDUP", "UNC_STABLE", "NEUTRAL", "WARMUP",
    "UnwindEvent", "AnswerVerdict", "extract_events", "score_answer",
    "AndonCord", "PullRecord", "MODE_BUDGETS",
    "Metrics", "Verdict", "compute_metrics", "verdict", "format_report",
    "Label", "label_answer", "token_f1", "is_idk",
    "__version__",
]
