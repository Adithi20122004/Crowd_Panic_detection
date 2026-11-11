"""
r_skef_core/calibration.py -- RECONSTRUCTED baseline calibration / adaptive thresholds.

NOT recovered historical source. Written to satisfy the interface detector.py uses:

    calibrator = BaselineCalibrator(cfg, reference_prior)       # prior: dict or None
    t_on, t_off = calibrator.t_on, calibrator.t_off             # read BEFORE the frame's FSM update
    t_on_c, t_off_c, should_evaluate = calibrator.step(ema_score, pred_label,
                                                       consecutive_exceed, g_active)

Interface facts taken from detector.py:
  * `.t_on` / `.t_off` must exist from frame 1 and are what the alarm FSM compares against.
  * step() is called once per frame AFTER the FSM decision for that frame.
  * detector.py reports startup_frames_discarded = cfg.calibration_frames for "fixed_150" and 0
    otherwise, so should_evaluate is False for exactly the first calibration_frames frames under
    "fixed_150" and True for every frame under the other two strategies.

Shared threshold formulas are the ones rskef_comprehensive_optimization.py uses for its reference
prior (median / MAD*1.4826 / floors / 0.65 factor on the t_off floor):
    sigma = max(1.4826 * MAD, cfg.min_sigma_floor)
    t_on  = max(median + k_on  * sigma, cfg.min_threshold_floor)
    t_off = max(median + k_off * sigma, cfg.min_threshold_floor * 0.65)

CAUSALITY: every threshold in force at frame t is computed only from scores of frames < t.
step() returns the thresholds that were in force for the frame just decided (so logged records show
what the FSM actually used) and only then absorbs the current score.

RECONSTRUCTED DESIGN DECISIONS (none historically verified):
  * fixed_150: collect the EMA scores of the first cfg.calibration_frames frames, then freeze.
    Until then t_on = t_off = +inf (no alarms) and should_evaluate = False.
  * pure_causal: rolling window (length cfg.calibration_frames) of EMA scores from "calm" frames
    only; thresholds stay +inf (no alarms) until the window is full, but frames are still evaluated.
  * hybrid_reference: starts exactly at the reference prior's t_on / t_off, then blends the prior's
    median/sigma with the rolling calm-window statistics, linearly, with online weight
    len(window)/cfg.calibration_frames (reaches 1.0 when the window is full).
  * "Calm" frame = pred_label == 0, consecutive_exceed == 0 and g_active <= g_0 + 0.15. The 0.15 margin
    duplicates the literal in detector.py's alarm-release rule; it is not a config field.
"""
import math
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

STRATEGIES = ("fixed_150", "pure_causal", "hybrid_reference")
MAD_TO_SIGMA = 1.4826        # same literal as the optimization script
T_OFF_FLOOR_FACTOR = 0.65    # same literal as the optimization script
GATE_ANCHOR_MARGIN = 0.15    # same literal as detector.py's release rule
_PRIOR_KEYS = ("median", "sigma", "t_on", "t_off")


def robust_stats(scores, min_sigma_floor: float) -> Tuple[float, float, float]:
    """(median, MAD, sigma) with the sigma floor applied."""
    arr = np.asarray(scores, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = max(mad * MAD_TO_SIGMA, float(min_sigma_floor))
    return med, mad, sigma


def thresholds_from_stats(med: float, sigma: float, k_on: float, k_off: float,
                          min_threshold_floor: float) -> Tuple[float, float]:
    t_on = max(med + k_on * sigma, min_threshold_floor)
    t_off = max(med + k_off * sigma, min_threshold_floor * T_OFF_FLOOR_FACTOR)
    return float(t_on), float(t_off)


class BaselineCalibrator:
    def __init__(self, cfg, reference_prior: Optional[Dict[str, float]] = None):
        self.strategy = cfg.calibration_strategy
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown calibration_strategy {self.strategy!r}; expected one of {STRATEGIES}")
        # Snapshot parameters so later mutation of cfg cannot change a running calibrator.
        self._k_on = float(cfg.k_on)
        self._k_off = float(cfg.k_off)
        self._sigma_floor = float(cfg.min_sigma_floor)
        self._thr_floor = float(cfg.min_threshold_floor)
        self._g0 = float(cfg.g_0)
        self._n = int(cfg.calibration_frames)
        if self._n < 1:
            raise ValueError("cfg.calibration_frames must be >= 1 for every calibration strategy")

        self.frames_seen = 0
        self.t_on = math.inf
        self.t_off = math.inf
        self.calibrated = False
        self._fixed_scores = []
        self._window = deque(maxlen=self._n)
        self._prior = None

        if self.strategy == "hybrid_reference":
            if not reference_prior or any(k not in reference_prior for k in _PRIOR_KEYS):
                raise ValueError(
                    "hybrid_reference requires reference_prior with keys "
                    f"{_PRIOR_KEYS}; got {reference_prior!r}"
                )
            self._prior = {k: float(reference_prior[k]) for k in _PRIOR_KEYS}
            self.t_on, self.t_off = self._prior["t_on"], self._prior["t_off"]
            self.calibrated = True

    # -- helpers ---------------------------------------------------------------------------
    def _is_calm(self, pred_label, consecutive_exceed, g_active) -> bool:
        return (int(pred_label) == 0 and int(consecutive_exceed) == 0
                and float(g_active) <= self._g0 + GATE_ANCHOR_MARGIN)

    def _refresh_rolling(self) -> None:
        n_win = len(self._window)
        if self.strategy == "pure_causal":
            if n_win >= self._n:
                med, _, sigma = robust_stats(self._window, self._sigma_floor)
                self.t_on, self.t_off = thresholds_from_stats(
                    med, sigma, self._k_on, self._k_off, self._thr_floor)
                self.calibrated = True
            return
        # hybrid_reference
        if n_win == 0:
            self.t_on, self.t_off = self._prior["t_on"], self._prior["t_off"]
            return
        w = min(1.0, n_win / self._n)
        med_o, _, sigma_o = robust_stats(self._window, self._sigma_floor)
        med = (1.0 - w) * self._prior["median"] + w * med_o
        sigma = max((1.0 - w) * self._prior["sigma"] + w * sigma_o, self._sigma_floor)
        self.t_on, self.t_off = thresholds_from_stats(
            med, sigma, self._k_on, self._k_off, self._thr_floor)

    # -- detector interface ----------------------------------------------------------------
    def step(self, ema_score: float, pred_label: int, consecutive_exceed: int,
             g_active: float) -> Tuple[float, float, bool]:
        self.frames_seen += 1
        in_force = (self.t_on, self.t_off)  # what the FSM just used for this frame
        score = float(ema_score)

        if self.strategy == "fixed_150":
            if not self.calibrated:
                self._fixed_scores.append(score)
                if len(self._fixed_scores) >= self._n:
                    med, _, sigma = robust_stats(self._fixed_scores, self._sigma_floor)
                    self.t_on, self.t_off = thresholds_from_stats(
                        med, sigma, self._k_on, self._k_off, self._thr_floor)
                    self.calibrated = True
            should_evaluate = self.frames_seen > self._n
        else:
            if self._is_calm(pred_label, consecutive_exceed, g_active):
                self._window.append(score)
            self._refresh_rolling()
            should_evaluate = True

        return float(in_force[0]), float(in_force[1]), should_evaluate