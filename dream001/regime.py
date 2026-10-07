"""
DREAM-001 regime engine.

Implements the ISCOT domain transfer (COT market regime architecture ->
token-level logit entropy) with declared spec deviations D1-D4.

Regime taxonomy (ISCOT v1.2 sec.1):
    confident_buildup   low entropy pct, falling   -- noise
    confident_stable    low entropy pct, flat
    confident_unwind    low entropy pct band, rising -- THE ALARM
    uncertain_buildup   high entropy pct, rising   -- noise
    uncertain_stable    high entropy pct, flat
    uncertain_unwind    high entropy pct, falling  -- weak signal
    neutral             mid band
    warmup              insufficient history

D2 (hysteresis latch) means the "confident" side is a *latched* state, not a
per-token membership test: the band ARMS after a sustained deep run and
RELEASES only on true normalization. confident_unwind is therefore emitted
while the latch is armed and raw entropy is physically ramping (D3).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import numpy as np

WARMUP = "warmup"
NEUTRAL = "neutral"
CONF_BUILDUP = "confident_buildup"
CONF_STABLE = "confident_stable"
CONF_UNWIND = "confident_unwind"
UNC_BUILDUP = "uncertain_buildup"
UNC_STABLE = "uncertain_stable"
UNC_UNWIND = "uncertain_unwind"

ALL_REGIMES = (
    WARMUP, NEUTRAL,
    CONF_BUILDUP, CONF_STABLE, CONF_UNWIND,
    UNC_BUILDUP, UNC_STABLE, UNC_UNWIND,
)


@dataclass
class RegimeConfig:
    """
    Every constant here is a declared calibration axis. Constants in absolute
    nats (min_rise, fast_slope_min) are model- and temperature-dependent and
    must not be silently carried between models (README limitation 6).
    """

    # --- percentile machinery -------------------------------------------
    mode: str = "rolling"            # "rolling" | "calibrated"  (D1)
    window: int = 104                # COT 104-week analog
    low_pct: float = 0.20
    high_pct: float = 0.80

    # --- D2: hysteresis band latch --------------------------------------
    arm_pct: Optional[float] = None  # default low_pct/2 -- AR(1) makes shallow runs cheap
    band_min_run: int = 6            # consecutive tokens at arm depth to ARM
    release_pct: float = 0.65        # true normalization; plateau dilution tops ~0.55
    release_run: int = 5             # must exceed OLS confirmation lag
    min_hold_run: int = 25           # D5: COT "extreme positioning HELD" before an unwind counts

    # --- D3: raw-entropy fire trigger -----------------------------------
    slope_window: int = 5            # OLS window on raw entropy, nats/token
    fast_slope_min: float = 0.06     # nats/token
    ema_span: int = 26               # competence envelope -- DIAGNOSTIC ONLY

    # --- D4: magnitude filter -------------------------------------------
    min_rise: float = 1.0            # nats above pre-event basin; 0.0 disables

    # --- flat-band tolerance for buildup/stable/unwind on the raw slope --
    stable_slope: float = 0.02

    # --- sec.7.2 optional two-term feature (declared secondary arm) ------
    require_no_evidence: bool = False
    evidence_lookback: int = 8

    def __post_init__(self):
        if self.arm_pct is None:
            self.arm_pct = self.low_pct / 2.0
        if self.mode not in ("rolling", "calibrated"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if not 0.0 < self.arm_pct <= self.low_pct:
            raise ValueError("arm_pct must be in (0, low_pct]")
        if self.release_pct <= self.low_pct:
            raise ValueError("release_pct must exceed low_pct (D2 hysteresis)")

    # ---- presets --------------------------------------------------------
    @classmethod
    def spec_default(cls) -> "RegimeConfig":
        """Rolling mode, spec defaults. Long generations."""
        return cls()

    @classmethod
    def short_sequence(cls) -> "RegimeConfig":
        """
        D1 + D4: calibrated percentiles for sub-window answers, and min_rise
        DISABLED so a synthetic-tuned absolute-nats constant cannot contaminate
        the real experiment (ISCOT v1.2 sec.4, D4).
        """
        return cls(
            mode="calibrated",
            band_min_run=3,
            release_run=3,
            release_pct=0.60,
            min_hold_run=4,
            slope_window=4,
            min_rise=0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenState:
    """Per-token engine output. Everything the scorer or a diagnostic needs."""
    index: int
    entropy: float
    percentile: Optional[float]
    regime: str
    armed: bool
    slope: Optional[float]          # raw-entropy OLS slope, nats/token (D3)
    ema: Optional[float]            # 26-span competence envelope (diagnostic)
    envelope_confirms: Optional[bool]
    basin: Optional[float]          # entropy floor of the current armed episode
    rise: Optional[float]           # entropy - basin
    coverage: Optional[float] = None  # top-k probability mass (DREAM-006 tier)


def _ols_slope(y: np.ndarray) -> float:
    """Least-squares slope over unit-spaced x. Physical units per token."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0.0:
        return 0.0
    return float(((x - xm) * (y - y.mean())).sum() / denom)


class RegimeEngine:
    """
    Streaming, O(window) per token. Feed entropy one token at a time.

    The engine is deliberately causal: no token's regime depends on any future
    token. Detection latency is therefore a real quantity and is reported.
    """

    def __init__(
        self,
        config: Optional[RegimeConfig] = None,
        reference: Optional[np.ndarray] = None,
    ):
        self.cfg = config or RegimeConfig()
        self.reset()
        self._reference: Optional[np.ndarray] = None
        if reference is not None:
            self.set_reference(reference)

    # ---- calibrated mode (D1) ------------------------------------------
    def set_reference(self, reference) -> None:
        """
        Per-model reference entropy distribution, built from calibration
        generations that are EXCLUDED from metrics. Faithful analog of COT's
        percentile-over-long-history, which was never computed within a trade.
        """
        ref = np.asarray(list(reference), dtype=np.float64)
        ref = ref[np.isfinite(ref)]
        if ref.size < 32:
            raise ValueError(
                f"reference distribution too small ({ref.size}); "
                "calibrated mode needs a real per-model distribution"
            )
        self._reference = np.sort(ref)

    def reset(self) -> None:
        self._hist: List[float] = []
        self._i = 0
        self._ema: Optional[float] = None
        self._armed = False
        self._deep_run = 0
        self._shallow_run = 0
        self._hold_run = 0
        self._release_run = 0
        self._basin: Optional[float] = None
        self.states: List[TokenState] = []

    # ---- percentile -----------------------------------------------------
    def _percentile(self, h: float) -> Optional[float]:
        if self.cfg.mode == "calibrated":
            if self._reference is None:
                raise RuntimeError(
                    "calibrated mode requires set_reference(); refusing to "
                    "silently fall back to rolling"
                )
            # strict-less + half-ties, matches scipy 'mean' convention
            lo = np.searchsorted(self._reference, h, side="left")
            hi = np.searchsorted(self._reference, h, side="right")
            return float((lo + hi) / 2.0 / self._reference.size)

        if len(self._hist) < max(8, self.cfg.window // 8):
            return None
        w = np.asarray(self._hist[-self.cfg.window:], dtype=np.float64)
        return float((w < h).mean() + 0.5 * (w == h).mean())

    # ---- main step ------------------------------------------------------
    def feed(self, entropy: float, coverage: Optional[float] = None) -> TokenState:
        cfg = self.cfg
        h = float(entropy)
        idx = self._i
        self._i += 1

        # competence envelope -- diagnostic only (D3)
        alpha = 2.0 / (cfg.ema_span + 1.0)
        self._ema = h if self._ema is None else alpha * h + (1 - alpha) * self._ema

        pct = self._percentile(h)
        self._hist.append(h)

        # raw-entropy slope (D3): the alarm fires on physics, not on rank
        slope = None
        if len(self._hist) >= 2:
            seg = np.asarray(self._hist[-cfg.slope_window:], dtype=np.float64)
            slope = _ols_slope(seg)

        # --- D2: hysteresis latch ---------------------------------------
        # ORDER MATTERS. The fire test is evaluated against the latch state as
        # of THIS token, and release is processed only afterwards. Releasing
        # first lets a fast ramp disarm the latch on the same token it would
        # have fired -- the exact race D2 warns about ("release_run must exceed
        # the OLS confirmation lag or fast ramps disarm the latch before the
        # slope confirms").
        if pct is not None:
            if pct <= cfg.arm_pct:
                self._deep_run += 1
            else:
                self._deep_run = 0

            # hold length is measured on the CONFIDENCE BAND (low_pct), not the
            # deeper arming band: plateau self-dilution lifts a long plateau's
            # own percentile out of the arm band while the position is still
            # very much being held.
            if pct <= cfg.low_pct:
                self._shallow_run += 1
            else:
                self._shallow_run = 0

            if not self._armed and self._deep_run >= cfg.band_min_run:
                self._armed = True
                self._release_run = 0
                self._hold_run = self._shallow_run
                self._basin = float(min(self._hist[-cfg.band_min_run:]))
            elif self._armed and pct <= cfg.low_pct:
                self._hold_run += 1

            if self._armed:
                self._basin = min(self._basin, h) if self._basin is not None else h

        rise = None if self._basin is None else h - self._basin
        armed_now = self._armed
        armed_run_now = self._hold_run

        # release is deferred until after classification, and requires TRUE
        # normalization: the percentile is high AND the series has settled.
        # A percentile passing through 0.65 mid-ramp is not normalization.
        if pct is not None and self._armed:
            # "settled" means NOT MID-RAMP, not "perfectly flat". Requiring a
            # near-zero slope under AR(1) noise never sustains for release_run
            # tokens, so the latch stays armed for hundreds of tokens after a
            # deep dip and every noise ramp fires off a stale basin.
            settled = slope is None or slope < cfg.fast_slope_min
            if pct >= cfg.release_pct and settled:
                self._release_run += 1
                if self._release_run >= cfg.release_run:
                    self._armed = False
                    self._release_run = 0
                    self._hold_run = 0
                    self._basin = None
                    self._deep_run = 0
            else:
                self._release_run = 0

        envelope_confirms = None
        if self._ema is not None and len(self._hist) > cfg.ema_span:
            envelope_confirms = bool(h > self._ema)

        regime = self._classify(pct, slope, rise, armed_now, armed_run_now)

        st = TokenState(
            index=idx,
            entropy=h,
            percentile=pct,
            regime=regime,
            armed=armed_now,
            slope=slope,
            ema=self._ema,
            envelope_confirms=envelope_confirms,
            basin=self._basin,
            rise=rise,
            coverage=coverage,
        )
        self.states.append(st)
        return st

    def _classify(
        self, pct: Optional[float], slope: Optional[float], rise: Optional[float],
        armed: bool = False, hold_run: int = 0,
    ) -> str:
        cfg = self.cfg
        if pct is None:
            return WARMUP
        s = 0.0 if slope is None else slope

        # confident side is LATCHED (D2), not a per-token band test
        if armed:
            # D5: COT's "extreme positioning HELD". A shallow AR(1) dip arms the
            # latch just as a real confident plateau does; what separates them is
            # duration. Firing before the position has been held is reading noise.
            if hold_run < cfg.min_hold_run:
                return CONF_BUILDUP if s <= -cfg.stable_slope else CONF_STABLE
            if s >= cfg.fast_slope_min:
                if cfg.min_rise > 0.0 and (rise is None or rise < cfg.min_rise):
                    return CONF_BUILDUP if s < -cfg.stable_slope else CONF_STABLE
                return CONF_UNWIND
            if s <= -cfg.stable_slope:
                return CONF_BUILDUP
            return CONF_STABLE

        if pct >= cfg.high_pct:
            if s >= cfg.stable_slope:
                return UNC_BUILDUP
            if s <= -cfg.stable_slope:
                return UNC_UNWIND
            return UNC_STABLE

        if pct <= cfg.low_pct:
            return CONF_BUILDUP if s <= cfg.stable_slope else CONF_STABLE
        return NEUTRAL

    # ---- batch convenience ---------------------------------------------
    def run(self, entropies) -> List[TokenState]:
        self.reset()
        for h in entropies:
            self.feed(h)
        return self.states
