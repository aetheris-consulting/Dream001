"""
WEOF Andon Cord pull hook.

Constraint priority: Safety > Security > Performance > Cost.
Over-pulling degrades operator trust, so pulls are rate limited -- but a
suppressed pull is still LOGGED. A silently dropped alarm is worse than a
noisy one.

Drift budget is mode-dependent (ISCOT v1.2 sec.8): low unwind threshold in
execution mode, high in exploration mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any

MODE_BUDGETS: Dict[str, int] = {
    "execution": 5,      # tight: an unwind mid-execution is expensive
    "analysis": 3,
    "exploration": 1,    # loose: wandering is the point
}


@dataclass
class PullRecord:
    token_index: int
    delivered: bool
    reason: str
    payload: Any = None


class AndonCord:
    def __init__(
        self,
        on_pull: Optional[Callable[[Any], None]] = None,
        max_pulls: int = 3,
        per_tokens: int = 500,
        mode: str = "execution",
    ):
        if mode not in MODE_BUDGETS:
            raise ValueError(f"unknown mode {mode!r}; expected {list(MODE_BUDGETS)}")
        self.on_pull = on_pull
        self.max_pulls = max_pulls
        self.per_tokens = per_tokens
        self.mode = mode
        self.log: List[PullRecord] = []
        self._window_start = 0
        self._pulls_in_window = 0

    @property
    def budget(self) -> int:
        return MODE_BUDGETS[self.mode]

    def consider(self, event, token_index: int) -> PullRecord:
        """Offer an unwind event to the cord. Returns the record either way."""
        if token_index - self._window_start >= self.per_tokens:
            self._window_start = token_index
            self._pulls_in_window = 0

        if getattr(event, "n_tokens", 1) < self.budget and self.mode == "exploration":
            rec = PullRecord(token_index, False, "below exploration-mode drift budget", event)
            self.log.append(rec)
            return rec

        if self._pulls_in_window >= self.max_pulls:
            rec = PullRecord(token_index, False, "rate limited (suppressed, logged)", event)
            self.log.append(rec)
            return rec

        self._pulls_in_window += 1
        rec = PullRecord(token_index, True, "HALT/ESCALATE", event)
        self.log.append(rec)
        if self.on_pull is not None:
            self.on_pull(event)
        return rec

    def summary(self) -> Dict[str, int]:
        return {
            "considered": len(self.log),
            "delivered": sum(1 for r in self.log if r.delivered),
            "suppressed": sum(1 for r in self.log if not r.delivered),
        }
