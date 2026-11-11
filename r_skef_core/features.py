"""
r_skef_core/features.py -- RECONSTRUCTED motion-feature extraction.

NOT recovered historical source. Written to satisfy the interface detector.py uses:

    feat = extract_motion_features(motion_prev, curr_gray, cfg)

detector.py reads exactly these keys:
    feat["e_area_base"]  float  area-mass kinetic energy          (mass_representation == "area_mass")
    feat["num_active"]   int    active pixel count (area-mass "crowd count")
    feat["mag"]          (H, W) float32 flow magnitude            (density.py: compute_density_kinetic_energy)
    feat["active_mask"]  (H, W) bool    active pixels             (density.py)
    feat["f_div"]        float  multiplicative divergence factor, >= 1
    feat["f_disorder"]   float  multiplicative disorder factor,   >= 1 (exactly 1.0 when disabled)
    feat["global_mag"]   float  mean flow magnitude over the whole frame (logged only)

Optical flow is dense Farneback between `motion_prev` (already GMC-aligned if GMC is on)
and `curr_gray`.

RECONSTRUCTED DESIGN DECISIONS (formulas and constants below are NOT historically verified;
the detector only fixes the keys, shapes and the multiplicative use of f_div / f_disorder):
  * Farneback parameters: the common OpenCV tutorial values (FARNEBACK_PARAMS).
  * Active pixels: flow magnitude > ACTIVE_MAG_THRESHOLD pixels/frame (fixed).      [HIGH impact]
  * e_area_base: mean over active pixels of 0.5 * v_clipped^2 (unit mass per active pixel),
    with v capped at the 99th percentile of active magnitudes, mirroring density.py's
    treatment. This is the area-mass analogue of the "csrnet_active_mass" local energy,
    without the crowd-floor gate.                                                     [HIGH impact]
  * f_div = 1 + beta_div * tanh(d / div_scale), d = mean divergence of the flow over active pixels
    (positive part only, unless cfg.bidirectional_divergence, then absolute value).    [MEDIUM impact]
  * f_disorder = 1 + disorder_weight * (1 - R) when cfg.use_directional_disorder, else exactly 1.0;
    R is the magnitude-weighted resultant length of flow directions over active pixels
    (circular variance).                                                               [MEDIUM impact]
Only existing RskefConfig fields are read: beta_div, div_scale, bidirectional_divergence,
use_directional_disorder, disorder_weight.
"""
from typing import Dict

import cv2
import numpy as np

# Reconstructed constants (not config fields; config.py defines none for these).
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)
ACTIVE_MAG_THRESHOLD = 0.5      # pixels per frame at the detector's resized resolution
VELOCITY_CAP_PERCENTILE = 99.0  # same cap density.py applies to active magnitudes
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
    global_mag = float(mag.mean())

    if num_active == 0:
        return {
            "e_area_base": 0.0, "num_active": 0, "mag": mag, "active_mask": active_mask,
            "f_div": 1.0, "f_disorder": 1.0, "global_mag": global_mag,
        }

    active_mags = mag[active_mask]
    v_cap = float(np.percentile(active_mags, VELOCITY_CAP_PERCENTILE))
    e_area_base = float(np.mean(0.5 * np.square(np.minimum(active_mags, v_cap))))

    # Divergence of the flow field: du/dx + dv/dy.
    div = np.gradient(u, axis=1) + np.gradient(v, axis=0)
    div_active = div[active_mask]
    if cfg.bidirectional_divergence:
        d = float(np.mean(np.abs(div_active)))
    else:
        d = float(np.mean(np.maximum(div_active, 0.0)))
    f_div = 1.0 + float(cfg.beta_div) * float(np.tanh(d / float(cfg.div_scale)))

    if cfg.use_directional_disorder:
        theta = np.arctan2(v[active_mask], u[active_mask])
        weight_sum = float(np.sum(active_mags)) + _EPS
        resultant = float(np.hypot(np.sum(active_mags * np.cos(theta)),
                                   np.sum(active_mags * np.sin(theta)))) / weight_sum
        f_disorder = 1.0 + float(cfg.disorder_weight) * (1.0 - min(1.0, resultant))
    else:
        f_disorder = 1.0

    return {
        "e_area_base": e_area_base, "num_active": num_active, "mag": mag,
        "active_mask": active_mask, "f_div": float(f_div), "f_disorder": float(f_disorder),
        "global_mag": global_mag,
    }