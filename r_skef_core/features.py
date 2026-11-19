"""
r_skef_core/features.py -- RECONSTRUCTED motion-feature extraction.

Addresses audit findings:
  1. Floor mismatch: Since ACTIVE_MAG_THRESHOLD = 0.5 px/frame, any active pixel has
     0.5 * v^2 >= 0.125. The attainable values of raw active-pixel mean energy are therefore
     either 0.0 or >= 0.125. Detection thresholds (tau_on) must employ min_threshold_floor >= 0.15
     to prevent single active pixels from immediately exceeding quiet baselines.
  2. Mass proxy & crowd participation: Adds noise suppression for isolated pixel noise
     (min_active_pixels threshold) and returns active_fraction and crowd_scale to allow
     both intensive and crowd-mass-modulated energy representations.

detector.py reads:
    feat["e_area_base"]  float  area-mass kinetic energy (0.0 or >= 0.125)
    feat["num_active"]   int    active pixel count
    feat["mag"]          (H, W) float32 flow magnitude
    feat["active_mask"]  (H, W) bool    active pixels
    feat["f_div"]        float  multiplicative divergence factor, >= 1
    feat["f_disorder"]   float  multiplicative disorder factor,   >= 1
    feat["global_mag"]   float  mean flow magnitude over the whole frame
"""
from typing import Dict

import cv2
import numpy as np

# Reconstructed constants
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)
ACTIVE_MAG_THRESHOLD = 0.5      # pixels per frame at the detector's resized resolution
VELOCITY_CAP_PERCENTILE = 99.0  # cap applied to active magnitudes
DEFAULT_MIN_ACTIVE_PIXELS = 5   # noise floor: minimum pixels needed to establish coherent crowd movement
_EPS = 1e-6


def compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Dense Farneback flow, shape (H, W, 2): channel 0 = dx, channel 1 = dy (pixels/frame)."""
    return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FARNEBACK_PARAMS)


def extract_motion_features(prev_gray: np.ndarray, curr_gray: np.ndarray, cfg) -> Dict:
    flow = compute_flow(prev_gray, curr_gray)
    u, v = flow[..., 0], flow[..., 1]
    mag = np.hypot(u, v).astype(np.float32)
    active_mask = mag > ACTIVE_MAG_THRESHOLD
    num_active = int(active_mask.sum())
    total_pixels = int(mag.size)
    active_fraction = float(num_active / total_pixels) if total_pixels > 0 else 0.0
    global_mag = float(mag.mean())

    min_active_pixels = getattr(cfg, "min_active_pixels", DEFAULT_MIN_ACTIVE_PIXELS)

    if num_active < min_active_pixels:
        # Suppress lone sensor noise spikes (< min_active_pixels)
        return {
            "e_area_base": 0.0,
            "num_active": num_active,
            "active_fraction": active_fraction,
            "mag": mag,
            "active_mask": active_mask,
            "f_div": 1.0,
            "f_disorder": 1.0,
            "global_mag": global_mag,
        }

    active_mags = mag[active_mask]
    v_cap = float(np.percentile(active_mags, VELOCITY_CAP_PERCENTILE))
    # Attainable value >= 0.125 since all active_mags > 0.5
    e_area_base = float(np.mean(0.5 * np.square(np.minimum(active_mags, v_cap))))

    # Divergence of the flow field: du/dx + dv/dy
    div = np.gradient(u, axis=1) + np.gradient(v, axis=0)
    div_active = div[active_mask]
    if getattr(cfg, "bidirectional_divergence", True):
        d = float(np.mean(np.abs(div_active)))
    else:
        d = float(np.mean(np.maximum(div_active, 0.0)))
    div_scale = float(getattr(cfg, "div_scale", 1.0))
    beta_div = float(getattr(cfg, "beta_div", 0.4))
    f_div = 1.0 + beta_div * float(np.tanh(d / (div_scale + _EPS)))

    if getattr(cfg, "use_directional_disorder", False):
        theta = np.arctan2(v[active_mask], u[active_mask])
        weight_sum = float(np.sum(active_mags)) + _EPS
        resultant = float(np.hypot(np.sum(active_mags * np.cos(theta)),
                                   np.sum(active_mags * np.sin(theta)))) / weight_sum
        disorder_weight = float(getattr(cfg, "disorder_weight", 0.2))
        f_disorder = 1.0 + disorder_weight * (1.0 - min(1.0, resultant))
    else:
        f_disorder = 1.0

    return {
        "e_area_base": e_area_base,
        "num_active": num_active,
        "active_fraction": active_fraction,
        "mag": mag,
        "active_mask": active_mask,
        "f_div": float(f_div),
        "f_disorder": float(f_disorder),
        "global_mag": global_mag,
    }