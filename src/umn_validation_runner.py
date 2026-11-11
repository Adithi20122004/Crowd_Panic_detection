"""
========================================================================================
RSKEF UMN CONTROLLED ABLATION ADAPTER (RECONSTRUCTED COMPONENTS)
File: umn_validation_runner.py
Directory: C:\\crowd_project\\Research_Validation\\src\\umn_validation_runner.py
           C:\\crowd_project\\Research_Validation\\umn_validation_runner.py

Confirmed r_skef_core APIs:
  - r_skef_core.features:
      compute_flow(prev_gray, curr_gray) -> np.ndarray
      extract_motion_features(prev_gray, curr_gray, cfg) -> Dict
  - r_skef_core.gmc:
      GlobalMotionCompensator(max_features, ransac_threshold)
      estimate_and_warp(prev_gray, curr_gray) -> Tuple[np.ndarray, Dict]
  - r_skef_core.calibration:
      robust_stats(scores, min_sigma_floor) -> Tuple[float, float, float]
      thresholds_from_stats(med, sigma, k_on, k_off, min_threshold_floor) -> Tuple[float, float]
  - r_skef_core.density:
      compute_density_kinetic_energy(density_map, mag, active_mask, normalization_mode, crowd_floor)
      estimate_density_map(frame_bgr, model, device)
  - r_skef_core.detector:
      run_rskef_detector(video_path, gt_ranges, cfg, reference_prior, csrnet_model)

Key Integrations & Guarantees:
  1. Complete RskefConfig resolution: Audits and explicitly initializes every factory field
     (bidirectional_divergence, div_scale, jerk, surge, g_0, ema_alpha, etc.) to eliminate all
     possible UNVERIFIED ValueError exceptions.
  2. Startup Property Pre-Flight Audit: Tests reading every property on RskefConfig before any
     video frames are processed, reporting all missing/unresolved properties together.
  3. C6 GMC Pipeline: estimate_and_warp() executes BEFORE feature extraction. The affine
     estimator is reconstructed; its accuracy is not independently validated.
  4. C1-C6 use reconstructed extract_motion_features(), which calculates Farneback flow
     once per energy observation.
  5. This custom adapter owns calibration; BaselineCalibrator is not invoked.
  6. The smoke test is a 650-frame plumbing check, not benchmark validation.
  7. Pass a new output base under Results for every run. Existing outputs are protected.
========================================================================================
"""

import os
import sys
import copy
import time
import math
import json
import hashlib
import inspect
import argparse
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure local validation root is on sys.path
VALIDATION_ROOT = r"C:\crowd_project\Research_Validation"
RESULTS_ROOT = os.path.join(VALIDATION_ROOT, "Results")
if VALIDATION_ROOT not in sys.path:
    sys.path.insert(0, VALIDATION_ROOT)


# ======================================================================================
# 1. EXPLICIT EXPERIMENTAL PARAMETER PROVENANCE REGISTRY
# ======================================================================================
# Audited against all factory-instantiated parameters in r_skef_core.config.RskefConfig.__init__
EXPERIMENTAL_PROVENANCE_REGISTRY = {
    # Preprocessing & Frame Geometry
    "resize_width": {
        "value": 320,
        "status": "EXPERIMENTAL_PREPROCESSING_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit config value; adapter processes decoded native frames and does not resize them.",
        "justification": "Retained in the resolved config for provenance; not used by this adapter."
    },
    "resize_height": {
        "value": 240,
        "status": "EXPERIMENTAL_PREPROCESSING_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit config value; adapter processes decoded native frames and does not resize them.",
        "justification": "Retained in the resolved config for provenance; not used by this adapter."
    },
    # Motion Feature & Divergence Parameters
    "bidirectional_divergence": {
        "value": True,
        "status": "EXPERIMENTAL_FEATURE_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit experimental config value; original default unrecovered.",
        "justification": "The adapter uses only e_area_base and does not score the divergence multiplier."
    },
    "div_scale": {
        "value": 1.0,
        "status": "EXPERIMENTAL_FEATURE_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit experimental config value; original default unrecovered.",
        "justification": "The adapter uses only e_area_base and does not score the divergence multiplier."
    },
    "beta_div": {
        "value": 0.4,
        "status": "STATIC_CONFIG_DEFAULT",
        "affects_algorithm": False,
        "provenance": "Reconstructed config default; not a verified historical parameter.",
        "justification": "The adapter uses only e_area_base and does not score the divergence multiplier."
    },
    "disorder_weight": {
        "value": 0.2,
        "status": "EXPERIMENTAL_FEATURE_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit experimental config value; original default unrecovered.",
        "justification": "Not used by this adapter's scalar energy score."
    },
    "jerk_delta_frames": {
        "value": 3,
        "status": "EXPERIMENTAL_FEATURE_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit experimental config value; original default unrecovered.",
        "justification": "The custom adapter does not invoke the detector jerk gate."
    },
    "surge_hold_frames": {
        "value": 15,
        "status": "EXPERIMENTAL_FEATURE_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit experimental config value; original default unrecovered.",
        "justification": "The custom adapter does not invoke the detector surge gate."
    },
    "theta_jerk": {
        "value": 0.5,
        "status": "EXPERIMENTAL_FEATURE_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit experimental config value; original default unrecovered.",
        "justification": "The custom adapter does not invoke the detector jerk gate."
    },
    # GMC Camera Ego-Motion Parameters
    "gmc_max_features": {
        "value": 400,
        "status": "EXPERIMENTAL_GMC_PARAMETER",
        "affects_algorithm": True,
        "provenance": "Explicit validation choice; original default unrecovered.",
        "justification": "Feature-count setting for the reconstructed GMC estimator; not a recovered historical parameter."
    },
    "gmc_ransac_threshold": {
        "value": 3.0,
        "status": "EXPERIMENTAL_GMC_PARAMETER",
        "affects_algorithm": True,
        "provenance": "Explicit validation choice; original default unrecovered.",
        "justification": "RANSAC reprojection threshold for the reconstructed GMC estimator."
    },
    # Statistical Calibration & Threshold Parameters
    "calib_window": {
        "value": 90,
        "status": "PROTOCOL_PARAMETER",
        "affects_algorithm": True,
        "provenance": "Validation protocol choice; not a recovered historical RSKEF parameter.",
        "justification": "The adapter reserves the first 90 flow-energy observations in each clip. Candidate ground truth is provisional."
    },
    "k_on": {
        "value": 3.2,
        "status": "STATISTICAL_PARAMETER",
        "affects_algorithm": True,
        "provenance": "Explicit validation parameter, matching the reconstructed config default. Not verified as historical RSKEF.",
        "justification": "Upper threshold multiplier used by the adapter; no false-alarm probability guarantee is asserted."
    },
    "k_off": {
        "value": 1.8,
        "status": "STATISTICAL_PARAMETER",
        "affects_algorithm": True,
        "provenance": "Explicit validation parameter, matching the reconstructed config default. Not verified as historical RSKEF.",
        "justification": "Lower threshold multiplier used by the adapter."
    },
    "min_sigma_floor": {
        "value": 1e-4,
        "status": "NUMERICAL_STABILITY_PARAMETER",
        "affects_algorithm": True,
        "provenance": "Safety floor for robust_stats().",
        "justification": "Prevents division by zero or degenerate scale estimates during completely motionless video segments."
    },
    "min_threshold_floor": {
        "value": 0.05,
        "status": "NUMERICAL_STABILITY_PARAMETER",
        "affects_algorithm": True,
        "provenance": "Safety floor for thresholds_from_stats().",
        "justification": "Ensures positive detection thresholds even under ultra-low baseline kinetic energy."
    },
    "g_0": {
        "value": 1.0,
        "status": "EXPERIMENTAL_CALIBRATION_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit experimental config value; original default unrecovered.",
        "justification": "The custom adapter does not use this detector-gate parameter."
    },
    "ema_alpha": {
        "value": 0.1,
        "status": "EXPERIMENTAL_FILTERING_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit experimental config value; original default unrecovered.",
        "justification": "The custom adapter does not invoke the detector EMA."
    },
    "max_alarm_unanchored_frames": {
        "value": 90,
        "status": "EXPERIMENTAL_PERSISTENCE_PARAMETER",
        "affects_algorithm": False,
        "provenance": "Explicit experimental config value; original default unrecovered.",
        "justification": "The custom adapter does not use the detector's unanchored-alarm timeout."
    },
    "k_min": {
        "value": 5,
        "status": "PERSISTENCE_PARAMETER",
        "affects_algorithm": True,
        "provenance": "Temporal persistence counter.",
        "justification": "Requires 5 consecutive frames (166.7 ms at 30 FPS) above tau_on to trigger alarm, filtering single-frame sensor spikes."
    }
}


# ======================================================================================
# 2. RSKEF PRE-FLIGHT VALIDATION & DYNAMIC BINDING SYSTEM
# ======================================================================================

class RSKEFEnvironment:
    """
    Validates, configures, and dynamically binds the confirmed r_skef_core APIs.
    Performs comprehensive RskefConfig resolution to eliminate all UNVERIFIED errors.
    """
    def __init__(self):
        self.r_config = None
        self.r_features = None
        self.r_calib = None
        self.r_gmc = None
        self.r_density = None
        self.r_detector = None

        self.fn_compute_flow = None
        self.fn_extract_motion_features = None
        self.fn_robust_stats = None
        self.fn_thresholds_from_stats = None
        self.gmc_class = None
        self.fn_run_rskef_detector = None

        self.cfg_instance = None
        self.cfg_c6 = None
        self.feature_extractor_status = "reconstructed; not recovered historical source"
        self.validation_log: List[str] = []

        self._load_and_validate()

    def _load_and_validate(self):
        # 1. Module Imports
        modules_to_import = [
            ("r_skef_core.config", "r_config"),
            ("r_skef_core.features", "r_features"),
            ("r_skef_core.calibration", "r_calib"),
            ("r_skef_core.gmc", "r_gmc"),
            ("r_skef_core.density", "r_density"),
            ("r_skef_core.detector", "r_detector"),
        ]
        for mod_path, attr_name in modules_to_import:
            try:
                mod = __import__(mod_path, fromlist=[mod_path.split(".")[-1]])
                setattr(self, attr_name, mod)
                self.validation_log.append(f"[OK] Imported {mod_path}")
            except ImportError as e:
                raise ImportError(
                    f"[RSKEF Integration Failure] Could not import '{mod_path}': {e}\n"
                    f"Ensure you ran 'python inspect_and_bridge_rskef.py' to copy r_skef_core "
                    f"into {VALIDATION_ROOT}\\r_skef_core."
                )

        # 2. Validate r_skef_core.features.compute_flow
        if not hasattr(self.r_features, "compute_flow") or not callable(getattr(self.r_features, "compute_flow")):
            available = [n for n, o in inspect.getmembers(self.r_features, inspect.isfunction)]
            raise RuntimeError(
                f"[RSKEF Integration Failure] r_skef_core.features does not export 'compute_flow'.\n"
                f"Available functions in r_features: {available}"
            )
        self.fn_compute_flow = getattr(self.r_features, "compute_flow")
        self.validation_log.append(
            "[AUDIT] Bound reconstructed r_features.compute_flow; the adapter invokes "
            "extract_motion_features once per observation, which calls compute_flow once."
        )

        # 3. Validate r_skef_core.features.extract_motion_features
        if not hasattr(self.r_features, "extract_motion_features") or not callable(getattr(self.r_features, "extract_motion_features")):
            available = [n for n, o in inspect.getmembers(self.r_features, inspect.isfunction)]
            raise RuntimeError(
                f"[RSKEF Integration Failure] r_skef_core.features does not export 'extract_motion_features'.\n"
                f"Available functions in r_features: {available}"
            )
        self.fn_extract_motion_features = getattr(self.r_features, "extract_motion_features")
        self.validation_log.append(
            "[AUDIT] Bound reconstructed r_features.extract_motion_features(prev_gray, curr_gray, cfg)."
        )

        # 4. Validate r_skef_core.calibration.robust_stats
        if not hasattr(self.r_calib, "robust_stats") or not callable(getattr(self.r_calib, "robust_stats")):
            available = [n for n, o in inspect.getmembers(self.r_calib, inspect.isfunction)]
            raise RuntimeError(
                f"[RSKEF Integration Failure] r_skef_core.calibration does not export 'robust_stats'.\n"
                f"Available functions in r_calib: {available}"
            )
        self.fn_robust_stats = getattr(self.r_calib, "robust_stats")
        self.validation_log.append("[AUDIT] Bound reconstructed r_calib.robust_stats(scores, min_sigma_floor).")

        # 5. Validate r_skef_core.calibration.thresholds_from_stats
        if not hasattr(self.r_calib, "thresholds_from_stats") or not callable(getattr(self.r_calib, "thresholds_from_stats")):
            available = [n for n, o in inspect.getmembers(self.r_calib, inspect.isfunction)]
            raise RuntimeError(
                f"[RSKEF Integration Failure] r_skef_core.calibration does not export 'thresholds_from_stats'.\n"
                f"Available functions in r_calib: {available}"
            )
        self.fn_thresholds_from_stats = getattr(self.r_calib, "thresholds_from_stats")
        self.validation_log.append("[AUDIT] Bound reconstructed r_calib.thresholds_from_stats(...).")

        # 6. Validate r_skef_core.gmc.GlobalMotionCompensator & estimate_and_warp()
        if not hasattr(self.r_gmc, "GlobalMotionCompensator"):
            raise RuntimeError("[RSKEF Integration Failure] r_skef_core.gmc does not export 'GlobalMotionCompensator'.")
        self.gmc_class = getattr(self.r_gmc, "GlobalMotionCompensator")

        if not hasattr(self.gmc_class, "estimate_and_warp"):
            available_methods = [m for m, _ in inspect.getmembers(self.gmc_class, inspect.isfunction) if not m.startswith("_")]
            raise RuntimeError(
                f"[RSKEF Integration Failure] GlobalMotionCompensator missing required 'estimate_and_warp()' method.\n"
                f"Available methods on GlobalMotionCompensator: {available_methods}"
            )
        self.validation_log.append("[OK] Verified GlobalMotionCompensator.estimate_and_warp(prev_gray, curr_gray) API.")

        # 7. Audit density module
        if hasattr(self.r_density, "compute_density_kinetic_energy"):
            self.validation_log.append(
                "[AUDIT] r_density.compute_density_kinetic_energy() detected. Decoupled from UMN pure motion ablations (CSRNet neural density estimation requires pre-trained PyTorch weights)."
            )

        # 8. Audit detector module
        if hasattr(self.r_detector, "run_rskef_detector"):
            self.fn_run_rskef_detector = getattr(self.r_detector, "run_rskef_detector")
            self.validation_log.append(
                "[AUDIT] r_detector.run_rskef_detector() detected but is not invoked by this custom adapter."
            )

        # 9. Configure RskefConfig and perform startup property pre-flight audit
        self._configure_rskef_config()

    def _configure_rskef_config(self):
        """
        Instantiates RskefConfig and applies proven experimental hyperparameters.
        Audits every property to guarantee that zero UNVERIFIED errors occur at runtime.
        """
        if not hasattr(self.r_config, "RskefConfig"):
            raise RuntimeError("[RSKEF Integration Failure] r_skef_core.config does not export 'RskefConfig'.")

        cfg_cls = getattr(self.r_config, "RskefConfig")
        sig = inspect.signature(cfg_cls.__init__)
        init_params = set(sig.parameters.keys())

        # Extract values for constructor kwargs
        prov_kwargs = {}
        for k, p_data in EXPERIMENTAL_PROVENANCE_REGISTRY.items():
            if k in init_params:
                prov_kwargs[k] = p_data["value"]

        # Ensure all standard default constructor kwargs are set
        prov_kwargs.setdefault("fps", 30.0)
        prov_kwargs.setdefault("mass_representation", "area_mass")
        prov_kwargs.setdefault("csrnet_crowd_floor", 4.0)
        prov_kwargs.setdefault("calibration_strategy", "adapter_fixed_90")
        prov_kwargs.setdefault("calibration_frames", 90)
        prov_kwargs.setdefault("enable_gmc", False)
        prov_kwargs.setdefault("use_directional_disorder", False)
        prov_kwargs.setdefault("persistence_frames", 7)

        try:
            self.cfg_instance = cfg_cls(**prov_kwargs)
        except Exception:
            self.cfg_instance = cfg_cls()
        self.cfg_instance.calibration_strategy = "adapter_fixed_90"
        self.cfg_instance.calibration_frames = 90

        # Explicitly assign public and private backing attributes for all parameters
        for param_name, p_data in EXPERIMENTAL_PROVENANCE_REGISTRY.items():
            val = p_data["value"]
            for target_name in [param_name, f"_{param_name}"]:
                try:
                    setattr(self.cfg_instance, target_name, val)
                except Exception:
                    pass

        # Dedicated config instance for C6 with GMC enabled
        self.cfg_c6 = copy.deepcopy(self.cfg_instance)
        try:
            setattr(self.cfg_c6, "enable_gmc", True)
            setattr(self.cfg_c6, "_enable_gmc", True)
            setattr(self.cfg_c6, "calibration_strategy", "adapter_fixed_90")
            setattr(self.cfg_c6, "calibration_frames", 90)
        except Exception:
            pass

        # Check unresolved_fields() method if exposed by RskefConfig
        if hasattr(self.cfg_instance, "unresolved_fields") and callable(getattr(self.cfg_instance, "unresolved_fields")):
            try:
                unresolved = self.cfg_instance.unresolved_fields()
                if unresolved:
                    for field in unresolved:
                        if field in EXPERIMENTAL_PROVENANCE_REGISTRY:
                            val = EXPERIMENTAL_PROVENANCE_REGISTRY[field]["value"]
                            setattr(self.cfg_instance, field, val)
                            setattr(self.cfg_instance, f"_{field}", val)
                            setattr(self.cfg_c6, field, val)
                            setattr(self.cfg_c6, f"_{field}", val)
            except Exception:
                pass

        # STARTUP PROPERTY PRE-FLIGHT AUDIT
        # Accesses every public property on cfg_instance to detect any lingering UNVERIFIED guards
        failed_properties = []
        for prop in dir(self.cfg_instance):
            if prop.startswith("_"):
                continue
            try:
                val = getattr(self.cfg_instance, prop)
                if not callable(val):
                    self.validation_log.append(f"[Property Verified] cfg.{prop} = {val}")
            except Exception as e:
                failed_properties.append((prop, str(e)))

        if failed_properties:
            err_msg = "[CRITICAL CONFIG ERROR] The following RskefConfig properties are UNRESOLVED:\n"
            for p, err in failed_properties:
                err_msg += f"  - cfg.{p}: {err}\n"
            raise ValueError(err_msg)

        self.validation_log.append(
            "[OK] RskefConfig startup property audit passed. "
            "Runtime calibration_strategy='adapter_fixed_90' is an adapter label only; "
            "BaselineCalibrator is not invoked."
        )

    def instantiate_gmc(self):
        """Instantiates GlobalMotionCompensator using proven parameters."""
        max_feat = EXPERIMENTAL_PROVENANCE_REGISTRY["gmc_max_features"]["value"]
        ransac_th = EXPERIMENTAL_PROVENANCE_REGISTRY["gmc_ransac_threshold"]["value"]
        try:
            return self.gmc_class(max_features=max_feat, ransac_threshold=ransac_th)
        except TypeError:
            try:
                return self.gmc_class(max_feat, ransac_th)
            except Exception:
                return self.gmc_class()


# ======================================================================================
# 3. KINETIC ENERGY EXTRACTION HELPER
# ======================================================================================

def extract_scalar_kinetic_energy(
    feat_dict: Any,
    flow: Optional[np.ndarray] = None
) -> float:
    """
    Extract the explicitly defined area-mass kinetic energy.

    The reconstructed features.py defines e_area_base as the
    intended scalar kinetic-energy feature.

    No arbitrary numeric-key or flow-based fallback is allowed,
    because that could silently change the evaluated metric.
    """
    if not isinstance(feat_dict, dict):
        raise TypeError(
            f"Expected feature dictionary, got {type(feat_dict).__name__}"
        )

    if "e_area_base" not in feat_dict:
        raise KeyError(
            "Required feature 'e_area_base' is missing. "
            "Check r_skef_core/features.py."
        )

    energy = feat_dict["e_area_base"]

    if not isinstance(energy, (int, float, np.number)):
        raise TypeError(
            f"'e_area_base' must be a scalar, got {type(energy).__name__}"
        )

    energy = float(energy)

    if not np.isfinite(energy) or energy < 0:
        raise ValueError(
            f"Invalid kinetic energy value: {energy}"
        )

    return energy
# ======================================================================================
# 4. VERIFIED SYNTHETIC SELF-TEST FOR EVENT-LEVEL CALCULUS
# ======================================================================================

def run_event_metric_self_test():
    """
    Mathematical unit test with known synthetic ground truth, known delay, and known premature alarm.
    Asserts correctness of delay calculation and false-alarm clustering before processing real data.
    """
    total_len = 100
    fps = 30.0
    s_0, e_0 = 0, 100
    calib_w = 20
    s_eval = s_0 + calib_w  # 20
    anom_onset = 60         # Event onset at frame index 60

    # Ground truth: 0 for 0..59, 1 for 60..99
    y_true = np.zeros(total_len, dtype=int)
    y_true[anom_onset:e_0] = 1

    # Synthetic prediction:
    # 1. A normal false alarm at frames 30..32.
    # 2. A premature alarm at frames 58..62 that straddles the event onset.
    # 3. A new event alarm at frame 65, five frames after onset.
    y_pred = np.zeros(total_len, dtype=int)
    y_pred[30:33] = 1
    y_pred[58:63] = 1
    y_pred[65:80] = 1

    # Evaluation
    norm_preds = y_pred[s_eval:anom_onset]
    diffs = np.diff(np.pad(norm_preds, (1, 1), 'constant'))
    fa_clusters = int(np.sum(diffs == 1))
    assert fa_clusters == 2, f"Self-test failed: expected 2 premature clusters, got {fa_clusters}"

    anom_preds = y_pred[anom_onset:e_0]
    previous = y_pred[anom_onset - 1]
    event_starts = np.flatnonzero(
        (anom_preds == 1) & (np.r_[previous, anom_preds[:-1]] == 0)
    )
    assert len(event_starts) > 0, "Self-test failed: expected post-onset event alarm"
    delay_frames = event_starts[0]
    assert delay_frames == 5, f"Self-test failed: expected 5 frames delay, got {delay_frames}"
    delay_ms = round(delay_frames / fps * 1000.0, 1)
    assert delay_ms == 166.7, f"Self-test failed: expected 166.7 ms, got {delay_ms}"
    already_active = np.zeros(total_len, dtype=int)
    already_active[58:] = 1
    active_at_onset = already_active[anom_onset:e_0]
    onset_previous = already_active[anom_onset - 1]
    onset_edges = np.flatnonzero(
        (active_at_onset == 1) & (np.r_[onset_previous, active_at_onset[:-1]] == 0)
    )
    assert len(onset_edges) == 0, "Self-test failed: pre-onset alarm must not count as event detection."

    print("[Self-Test Passed] Event-level metric calculus verified on synthetic ground truth.")


# ======================================================================================
# 5. GENUINE RSKEF MODULAR ADAPTER (ABLATIONS C1 THROUGH C6)
# ======================================================================================

class ModularRSKEFAdapter:
    """
    Drives the local reconstructed r_skef_core feature/calibration modules.
    Differentiates between:
      - Component-level baseline ablations: C1, C2, C3, C4
      - Complete RSKEF detector architecture: C5 (Static camera), C6 (Moving camera + GMC)
    """
    def __init__(self, config_name: str, env: RSKEFEnvironment, calib_window: int = 90):
        self.config_name = config_name
        self.env = env
        self.calib_window = calib_window

        self.buffer: List[float] = []
        self.calibration_scores: List[float] = []
        self.observations_seen = 0
        self.first_evaluation_observation: Optional[int] = None
        self.state = 0
        self.persistence_counter = 0
        from collections import Counter

        self.gmc_attempts = 0
        self.gmc_valid_returned_images = 0
        self.gmc_affine_estimates = 0
        self.gmc_accepted_estimates = 0
        self.gmc_warps_applied = 0
        self.gmc_nonidentity_estimates = 0
        self.gmc_fallbacks = 0
        self.gmc_reported_fallbacks = 0
        self.gmc_missing_diagnostics = 0
        self.gmc_fallback_reasons = Counter()
        self.gmc_diagnostic_reasons = Counter()
        self.last_gmc_diagnostic = None
        self.last_gmc_output_valid = False
        self.feature_extractor_calls = 0

        self.k_min = EXPERIMENTAL_PROVENANCE_REGISTRY["k_min"]["value"]
        self.k_on = EXPERIMENTAL_PROVENANCE_REGISTRY["k_on"]["value"]
        self.k_off = EXPERIMENTAL_PROVENANCE_REGISTRY["k_off"]["value"]
        self.min_sigma_floor = EXPERIMENTAL_PROVENANCE_REGISTRY["min_sigma_floor"]["value"]
        self.min_threshold_floor = EXPERIMENTAL_PROVENANCE_REGISTRY["min_threshold_floor"]["value"]

        # GMC Instance for C6
        self.gmc_instance = None
        if config_name == "C6_RSKEF_GMC":
            self.gmc_instance = self.env.instantiate_gmc()

    def reset_sequence(self):
        self.buffer = []
        self.calibration_scores = []
        self.observations_seen = 0
        self.first_evaluation_observation = None
        self.state = 0
        self.persistence_counter = 0
        self.last_gmc_diagnostic = None
        self.last_gmc_output_valid = False
        if self.gmc_instance is not None and hasattr(self.gmc_instance, "reset"):
            self.gmc_instance.reset()

    def compute_energy(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """
        Extract kinetic energy through extract_motion_features().
        That function internally computes Farneback flow exactly once.
        C6 performs GMC before feature extraction.
        """
        self.last_gmc_diagnostic = None
        self.last_gmc_output_valid = False
        if self.config_name == "C6_RSKEF_GMC" and self.gmc_instance is not None:
            warp_out = self.gmc_instance.estimate_and_warp(
                prev_gray, curr_gray
            )

            diag = {}
            if isinstance(warp_out, tuple):
                warped_prev, diag = warp_out
            else:
                warped_prev = warp_out
            self.last_gmc_diagnostic = diag if isinstance(diag, dict) else None

            valid_warp = (
                isinstance(warped_prev, np.ndarray)
                and warped_prev.shape == prev_gray.shape
            )
            self.last_gmc_output_valid = valid_warp

            # Record diagnostics without resetting totals between clips.
            self.gmc_attempts += 1

            if valid_warp:
                self.gmc_valid_returned_images += 1
            else:
                self.gmc_fallback_reasons["invalid_warp_output"] += 1
                warped_prev = prev_gray

            if isinstance(diag, dict):
                if diag.get("affine_estimated", False):
                    self.gmc_affine_estimates += 1
                if diag.get("affine_estimate_accepted", False):
                    self.gmc_accepted_estimates += 1
                if diag.get("warp_applied", False):
                    self.gmc_warps_applied += 1
                if diag.get("estimated_non_identity_transform", False):
                    self.gmc_nonidentity_estimates += 1
                reason = diag.get("reason") or diag.get("fallback_reason")
                if reason:
                    self.gmc_diagnostic_reasons[str(reason)] += 1

                if diag.get("fallback", False) or not valid_warp:
                    self.gmc_fallbacks += 1
                    fallback_reason = str(reason or "invalid_warp_output")
                    self.gmc_fallback_reasons[fallback_reason] += 1
                if diag.get("fallback", False):
                    self.gmc_reported_fallbacks += 1
            else:
                self.gmc_missing_diagnostics += 1
                self.gmc_fallbacks += int(not valid_warp)

            self.feature_extractor_calls += 1
            feat_dict = self.env.fn_extract_motion_features(
                warped_prev, curr_gray, self.env.cfg_c6
            )

        else:
            self.feature_extractor_calls += 1
            feat_dict = self.env.fn_extract_motion_features(
                prev_gray, curr_gray, self.env.cfg_instance
            )
        return extract_scalar_kinetic_energy(feat_dict)
    
    def _get_robust_thresholds(self, buf_arr: np.ndarray) -> Tuple[float, float]:
        """Calls local robust_stats and thresholds_from_stats with safe unpacking."""
        stats_res = self.env.fn_robust_stats(buf_arr, self.min_sigma_floor)
        if isinstance(stats_res, tuple):
            if len(stats_res) == 2:
                med, sigma = float(stats_res[0]), float(stats_res[1])
            elif len(stats_res) >= 3:
                med, sigma = float(stats_res[0]), float(stats_res[-1])
            else:
                med = float(stats_res[0])
                sigma = max(float(self.min_sigma_floor), 0.01)
        else:
            med = float(stats_res)
            sigma = max(float(self.min_sigma_floor), 0.01)

        th_res = self.env.fn_thresholds_from_stats(med, sigma, self.k_on, self.k_off, self.min_threshold_floor)
        if isinstance(th_res, tuple):
            tau_on, tau_off = float(th_res[0]), float(th_res[1])
        else:
            tau_on = float(th_res)
            tau_off = float(th_res) * 0.7

        return tau_on, tau_off


    def evaluate_step(self, energy: float) -> Tuple[float, float, float, int]:
        """
        Causal calibration and evaluation.

        First 90 energy observations are used exclusively for calibration.
        No configuration evaluates a frame using a threshold calculated
        with that same frame's energy.
        """

        energy = float(energy)
        self.observations_seen += 1

        # ---------------------------------------------------------
        # PHASE 1: Initial calibration (first 90 observations)
        # ---------------------------------------------------------
        if len(self.calibration_scores) < self.calib_window:
            self.calibration_scores.append(energy)
            self.buffer.append(energy)
            return energy, 5.0, 3.0, 0

        if self.first_evaluation_observation is None:
            self.first_evaluation_observation = self.observations_seen

        # ---------------------------------------------------------
        # C1: Fixed mean-based threshold
        # ---------------------------------------------------------
        if self.config_name == "C1_Baseline_Flow":
            threshold = float(np.mean(self.calibration_scores) * 2.5)
            alarm = int(energy > threshold)
            return energy, threshold, threshold, alarm

        # ---------------------------------------------------------
        # C2: Five-observation moving average
        # Threshold remains fixed using initial calibration data.
        # ---------------------------------------------------------
        if self.config_name == "C2_Flow_TemporalFilter":
            history = self.buffer[-4:] + [energy]
            smooth_score = float(np.mean(history))

            threshold = float(
                np.mean(self.calibration_scores) * 2.5
            )

            self.buffer = history[-4:]

            alarm = int(smooth_score > threshold)
            return smooth_score, threshold, threshold, alarm

        # ---------------------------------------------------------
        # C3: Rolling robust MAD, without hysteresis
        # Threshold is calculated BEFORE current energy is added.
        # ---------------------------------------------------------
        if self.config_name == "C3_Flow_RobustMAD":
            tau_on, _ = self._get_robust_thresholds(
                np.asarray(self.buffer, dtype=float)
            )

            alarm = int(energy > tau_on)

            self.buffer.append(energy)
            if len(self.buffer) > 150:
                self.buffer.pop(0)

            return energy, tau_on, tau_on, alarm

        # ---------------------------------------------------------
        # C4: Rolling MAD + hysteresis + persistence
        # Buffer updates after the current decision.
        # ---------------------------------------------------------
        if self.config_name == "C4_Flow_MAD_Hysteresis":
            tau_on, tau_off = self._get_robust_thresholds(
                np.asarray(self.buffer, dtype=float)
            )

            if self.state == 0:
                if energy > tau_on:
                    self.persistence_counter += 1

                    if self.persistence_counter >= self.k_min:
                        self.state = 1
                else:
                    self.persistence_counter = 0

            else:
                if energy < tau_off:
                    self.state = 0
                    self.persistence_counter = 0

            # Ungated rolling update: update after decision,
            # including during an active anomaly.
            self.buffer.append(energy)
            if len(self.buffer) > 150:
                self.buffer.pop(0)

            return energy, tau_on, tau_off, self.state

        # ---------------------------------------------------------
        # C5 and C6: Complete RSKEF, state-gated rolling MAD
        # ---------------------------------------------------------
        tau_on, tau_off = self._get_robust_thresholds(
            np.asarray(self.buffer, dtype=float)
        )

        if self.state == 0:
            if energy > tau_on:
                self.persistence_counter += 1

                if self.persistence_counter >= self.k_min:
                    self.state = 1
            else:
                self.persistence_counter = 0

                self.buffer.append(energy)

        else:
            if energy < tau_off:
                self.state = 0
                self.persistence_counter = 0
                self.buffer.append(energy)

        # Keep rolling buffer bounded.
        if len(self.buffer) > 150:
            self.buffer.pop(0)

        return energy, tau_on, tau_off, self.state


# ======================================================================================
# 6. METRIC EVALUATOR WITH EXACT BOUNDARY GUARDS
# ======================================================================================

def evaluate_metrics_exact(
    y_true_full: np.ndarray,
    y_pred_full: np.ndarray,
    y_scores_full: np.ndarray,
    eval_mask: np.ndarray,
    sequence_meta: List[Dict[str, Any]],
    fps: float = 30.0
) -> Dict[str, Any]:
    """
    Computes frame-level metrics strictly over eval_mask.
    Computes event-level metrics using full sequence coordinates with evaluation boundaries.
    """
    y_t = y_true_full[eval_mask]
    y_p = y_pred_full[eval_mask]
    y_s = y_scores_full[eval_mask]

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    total = tp + tn + fp + fn
    if total != len(y_t):
        raise AssertionError("Confusion-matrix cells do not sum to evaluated-frame count.")

    acc = (tp + tn) / total * 100.0 if total > 0 else 0.0
    prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 100.0
    fpr = fp / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0

    if len(np.unique(y_t)) > 1:
        fpr_a, tpr_a, _ = roc_curve(y_t, y_s)
        auroc = float(auc(fpr_a, tpr_a))
        pr_a, rc_a, _ = precision_recall_curve(y_t, y_s)
        auprc = float(auc(rc_a, pr_a))
    else:
        auroc, auprc = float("nan"), float("nan")

    delays_ms = []
    premature_alarms = 0
    premature_alarm_sequences = 0
    false_alarm_clusters = 0
    event_hits = 0
    events_with_preexisting_alarm = 0

    for seq in sequence_meta:
        s_eval = seq["eval_start"]
        e_eval = seq["eval_end"]
        anom_onset = seq["anom_start"]

        # A. Normal Evaluation Window: [s_eval, anom_onset)
        if anom_onset > s_eval:
            norm_preds = y_pred_full[s_eval:anom_onset]
            diffs = np.diff(np.pad(norm_preds, (1, 1), 'constant'))
            fa_count = int(np.sum(diffs == 1))
            false_alarm_clusters += fa_count
            if fa_count > 0:
                premature_alarms += fa_count
                premature_alarm_sequences += 1

        # A pre-onset alarm is premature, not a zero-delay event detection.
        anom_preds = y_pred_full[anom_onset:e_eval]
        previous = y_pred_full[anom_onset - 1] if anom_onset > 0 else 0
        if previous == 1:
            events_with_preexisting_alarm += 1
        starts = (
            np.flatnonzero((anom_preds == 1) & (np.r_[previous, anom_preds[:-1]] == 0))
            if len(anom_preds) else np.array([], dtype=int)
        )
        if len(starts) > 0:
            event_hits += 1
            delays_ms.append(float(starts[0] / fps * 1000.0))

    hit_rate = (event_hits / len(sequence_meta)) * 100.0 if sequence_meta else 0.0
    mean_delay = float(np.mean(delays_ms)) if delays_ms else float("nan")

    total_normal_frames = int(np.sum(y_t == 0))
    normal_exposure_seconds = total_normal_frames / fps
    total_normal_hours = normal_exposure_seconds / 3600.0
    fa_per_hour = (false_alarm_clusters / total_normal_hours) if total_normal_hours > 0 else 0.0

    return {
        "Accuracy": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1_Score": round(f1, 2),
        "Specificity": round(spec, 2),
        "FPR": round(fpr, 2),
        "AUROC": round(auroc, 4) if not math.isnan(auroc) else "N/A",
        "AUPRC": round(auprc, 4) if not math.isnan(auprc) else "N/A",
        "Event_Hit_Rate": round(hit_rate, 1),
        "Event_Misses": len(sequence_meta) - event_hits,
        "Events_With_Preexisting_Alarm": events_with_preexisting_alarm,
        "Mean_Delay_ms": round(mean_delay, 1) if not math.isnan(mean_delay) else "N/A",
        "Premature_Alarms": premature_alarms,
        "Premature_Alarm_Sequences": premature_alarm_sequences,
        "False_Alarm_Clusters": false_alarm_clusters,
        "Evaluated_Frames": int(len(y_t)),
        "Evaluated_Normal_Frames": total_normal_frames,
        "Normal_Exposure_Seconds": round(normal_exposure_seconds, 3),
        "Normal_Exposure_Hours": round(total_normal_hours, 8),
        "FA_Per_Hour": round(fa_per_hour, 2),
        "Score_Metric_Type": "continuous anomaly scores; C2 uses its five-observation moving average",
        "AUPRC_Method": "trapezoidal area under precision-recall curve",
        "Confusion_Matrix_TN_FP_FN_TP": json.dumps([[tn, fp], [fn, tp]]),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }


# ======================================================================================
# 7. MAIN EXPERIMENTAL RUNNER
# ======================================================================================

def run_umn_ablations(video_path: str, candidate_gt_path: str, output_base: str, smoke_test: bool = False):
    results_root = os.path.normcase(os.path.abspath(RESULTS_ROOT))
    output_base_abs = os.path.normcase(os.path.abspath(output_base))
    if os.path.commonpath([results_root, output_base_abs]) != results_root:
        raise ValueError(f"All outputs must stay under {RESULTS_ROOT}: {output_base}")

    print("=" * 80)
    print("RSKEF UMN CONTROLLED ABLATIONS (FINAL VALIDATION SUITE)")
    print(f"Target Video:            {video_path}")
    print(f"Candidate Ground Truth:  {candidate_gt_path}")
    print(f"Output Base Directory:   {output_base}")
    print(f"Execution Mode:          {'SMOKE TEST (>=650 frames verification)' if smoke_test else 'FULL EVALUATION RUN (7,739 frames)'}")
    print("=" * 80)

    # 1. Environment & Pre-flight Validation
    print("\n[Stage 1/5] Validating Genuine RSKEF APIs & Dynamic Bindings...")
    env = RSKEFEnvironment()
    for log_line in env.validation_log:
        print(f"  {log_line}")

    # 2. Metric Calculus Self-Test
    print("\n[Stage 2/5] Running Metric Calculus Self-Test...")
    run_event_metric_self_test()

    if not os.path.exists(candidate_gt_path):
        raise FileNotFoundError(
            f"Candidate ground truth missing: {candidate_gt_path}.\n"
            f"Run 'python audit_and_verify_umn_gt.py' first."
        )

    df_gt = pd.read_csv(candidate_gt_path)
    gt_records = df_gt.to_dict(orient="records")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open UMN video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    # Smoke test requires >= 650 frames to cross the provisional anomaly onset at frame 486
    max_decode = 650 if smoke_test else 10000
    frames_gray = []
    print(f"\n[Stage 3/5] Decoding video frames into memory (Target={max_decode}, FPS={fps})...")
    while len(frames_gray) < max_decode:
        ret, frame = cap.read()
        if not ret:
            break
        frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    n_frames = len(frames_gray)
    print(f"  Decoded {n_frames} frames successfully.")
    if smoke_test:
        assert n_frames >= 650, f"Smoke test failed: decoded only {n_frames} frames, expected >= 650"
        print(f"  [Smoke Test Scope Verified] Frame count ({n_frames}) extends past Clip 1 anomaly onset (frame 486).")

    # 3. Strict Calibration and Evaluation Mask Alignment
    CALIB_WINDOW = EXPERIMENTAL_PROVENANCE_REGISTRY["calib_window"]["value"]
    y_true_full = np.zeros(n_frames, dtype=int)
    annotated_clip_mask = np.zeros(n_frames, dtype=bool)
    flow_initialization_mask = np.zeros(n_frames, dtype=bool)
    calibration_mask = np.zeros(n_frames, dtype=bool)
    eval_mask = np.zeros(n_frames, dtype=bool)
    clip_processing_metadata = []
    sequence_meta = []

    for r in gt_records:
        s0 = int(r["Start_Frame_1idx"]) - 1
        e0 = min(int(r["End_Frame_1idx"]), n_frames)
        a0 = int(r["Candidate_Anomaly_Onset_1idx"]) - 1

        if s0 >= n_frames or e0 <= s0:
            continue

        if np.any(annotated_clip_mask[s0:e0]):
            raise ValueError(f"Overlapping annotated clip ranges at Clip_ID={r['Clip_ID']}")
        annotated_clip_mask[s0:e0] = True
        if s0 < e0:
            flow_initialization_mask[s0] = True

        energy_observations = max(0, e0 - s0 - 1)
        calibration_observations = min(CALIB_WINDOW, energy_observations)
        calibration_start = s0 + 1
        calibration_end = calibration_start + calibration_observations
        calibration_mask[calibration_start:calibration_end] = True
        first_eval_observation = CALIB_WINDOW + 1 if energy_observations > CALIB_WINDOW else None
        s_eval = calibration_end
        eval_observations = max(0, e0 - s_eval)

        gt_start = max(a0, s0)
        gt_end = min(e0, n_frames)
        if gt_start < gt_end:
            y_true_full[gt_start:gt_end] = 1

        clip_processing_metadata.append({
            "clip_id": int(r["Clip_ID"]),
            "scene_id": int(r["Scene_ID"]),
            "annotated_start_frame_1idx": s0 + 1,
            "decoded_end_frame_1idx": e0,
            "energy_observations_available": energy_observations,
            "calibration_observations": calibration_observations,
            "calibration_target_observations": CALIB_WINDOW,
            "calibration_complete": calibration_observations == CALIB_WINDOW,
            "first_evaluated_observation": first_eval_observation,
            "first_evaluated_frame_1idx": s_eval + 1 if eval_observations > 0 else None,
            "evaluated_observations": eval_observations,
            "candidate_anomaly_onset_frame_1idx": a0 + 1,
            "ground_truth_status": r.get("Ground_Truth_Status", "UNSPECIFIED"),
        })

        if eval_observations > 0:
            if a0 < s_eval:
                raise ValueError(
                    f"Clip {r['Clip_ID']} candidate anomaly onset precedes its first evaluated frame; "
                    "event metrics would include unevaluated onset frames."
                )
            eval_mask[s_eval:e0] = True
            sequence_meta.append({
                "clip_id": r["Clip_ID"],
                "scene_id": r["Scene_ID"],
                "eval_start": s_eval,
                "eval_end": e0,
                "anom_start": a0,
            })

    eval_count = int(np.sum(eval_mask))
    outside_clip_mask = ~annotated_clip_mask
    other_excluded_mask = annotated_clip_mask & ~(
        flow_initialization_mask | calibration_mask | eval_mask
    )
    exclusion_masks = (
        flow_initialization_mask,
        calibration_mask,
        outside_clip_mask,
        other_excluded_mask,
        eval_mask,
    )
    exclusion_membership = np.sum(np.stack(exclusion_masks), axis=0)
    if not np.all(exclusion_membership == 1):
        raise AssertionError("Frame exclusion categories are overlapping or incomplete.")
    if np.any(eval_mask & ~annotated_clip_mask):
        raise AssertionError("Evaluation mask includes frames outside annotated clip ranges.")
    eval_indices = np.flatnonzero(eval_mask)
    if not np.array_equal(eval_indices, np.flatnonzero(eval_mask & annotated_clip_mask)):
        raise AssertionError("Evaluation and annotated-clip indices do not align.")

    calibration_count = int(np.sum(calibration_mask))
    flow_init_count = int(np.sum(flow_initialization_mask))
    outside_clip_count = int(np.sum(outside_clip_mask))
    other_excluded_count = int(np.sum(other_excluded_mask))
    if calibration_count + flow_init_count + outside_clip_count + other_excluded_count + eval_count != n_frames:
        raise AssertionError("Frame exclusion accounting does not equal decoded frame count.")
    print(
        "  Frame accounting: "
        f"decoded={n_frames} = calibration_observations={calibration_count} + "
        f"flow_initialization={flow_init_count} + outside_annotated_clips={outside_clip_count} + "
        f"other_excluded={other_excluded_count} + evaluated={eval_count}."
    )
    print(
        f"  Calibration protocol: {CALIB_WINDOW} initial energy observations per complete clip; "
        f"first evaluated observation={CALIB_WINDOW + 1}."
    )
    expected_energy_observations = sum(
        item["energy_observations_available"] for item in clip_processing_metadata
    )

    # 4. Strict Directory Isolation
    mode_dir = "smoke_test" if smoke_test else "full_run"
    target_dir = os.path.join(output_base, "UMN", mode_dir)
    if os.path.exists(target_dir):
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {target_dir}. "
            "Choose a new --output directory under Results."
        )
    canonical_summary = os.path.join(output_base, "UMN", "ablation_results.csv")
    if not smoke_test and os.path.exists(canonical_summary):
        raise FileExistsError(
            f"Refusing to overwrite existing experiment results: {canonical_summary}"
        )
    out_frames_dir = os.path.join(target_dir, "frame_predictions")
    out_plots_dir = os.path.join(target_dir, "plots")
    for d in [target_dir, out_frames_dir, out_plots_dir]:
        os.makedirs(d, exist_ok=True)

    # Save Provenance and Environment Reports
    prov_file = os.path.join(target_dir, "rskef_runtime_parameter_provenance.json")
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(EXPERIMENTAL_PROVENANCE_REGISTRY, f, indent=2)

    preflight_file = os.path.join(target_dir, "preflight_validation_log.json")
    with open(preflight_file, "w", encoding="utf-8") as f:
        json.dump({"logs": env.validation_log, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)

    configs = [
        ("C1_Baseline_Flow", "Area-mass flow energy + fixed mean-based threshold."),
        ("C2_Flow_TemporalFilter", "C1 threshold + five-observation moving-average score."),
        ("C3_Flow_RobustMAD", "Flow energy + rolling robust threshold; no hysteresis."),
        ("C4_Flow_MAD_Hysteresis", "Rolling robust threshold + hysteresis + persistence; ungated baseline."),
        ("C5_Complete_RSKEF", "Adapter subset: state-gated rolling robust threshold + hysteresis + persistence; no GMC, jerk gate, or EMA."),
        ("C6_RSKEF_GMC", "C5 adapter subset + reconstructed GMC; no independent camera-motion accuracy validation.")
    ]

    print(f"\n[Stage 4/5] Executing 6 Controlled Configurations...")
    summary_rows = []
    roc_entries = []

    for cname, cdesc in configs:
        t0 = time.time()
        print(f"\n  >> Running {cname} ...")
        print(f"     Description: {cdesc}")
        adapter = ModularRSKEFAdapter(cname, env, calib_window=CALIB_WINDOW)

        scores = np.zeros(n_frames, dtype=float)
        preds = np.zeros(n_frames, dtype=int)
        tau_high_arr = np.zeros(n_frames, dtype=float)
        tau_low_arr = np.zeros(n_frames, dtype=float)
        gmc_frame_fields = {
            "GMC_Attempted": np.zeros(n_frames, dtype=bool),
            "GMC_Diagnostic_Present": np.zeros(n_frames, dtype=bool),
            "GMC_Output_Valid": np.zeros(n_frames, dtype=bool),
            "GMC_Diagnostic_Output_Valid": np.zeros(n_frames, dtype=bool),
            "GMC_Success": np.zeros(n_frames, dtype=bool),
            "GMC_Fallback": np.zeros(n_frames, dtype=bool),
            "GMC_Reported_Fallback": np.zeros(n_frames, dtype=bool),
            "GMC_Affine_Estimated": np.zeros(n_frames, dtype=bool),
            "GMC_Affine_Estimate_Accepted": np.zeros(n_frames, dtype=bool),
            "GMC_Warp_Applied": np.zeros(n_frames, dtype=bool),
            "GMC_NonIdentity_Transform_Estimate": np.zeros(n_frames, dtype=bool),
            "GMC_Reason": np.full(n_frames, "", dtype=object),
            "GMC_Features": np.full(n_frames, np.nan),
            "GMC_Matches": np.full(n_frames, np.nan),
            "GMC_Inliers": np.full(n_frames, np.nan),
            "GMC_Inlier_Ratio": np.full(n_frames, np.nan),
            "GMC_Tx": np.full(n_frames, np.nan),
            "GMC_Ty": np.full(n_frames, np.nan),
            "GMC_Rotation_Deg": np.full(n_frames, np.nan),
            "GMC_Scale": np.full(n_frames, np.nan),
        }

        for r in gt_records:
            adapter.reset_sequence()
            s0 = int(r["Start_Frame_1idx"]) - 1
            e0 = min(int(r["End_Frame_1idx"]), n_frames)
            if s0 >= n_frames or e0 <= s0:
                continue

            for f in range(s0 + 1, e0):
                e_val = adapter.compute_energy(frames_gray[f - 1], frames_gray[f])
                score, th, tl, alarm = adapter.evaluate_step(e_val)
                scores[f] = score
                tau_high_arr[f] = th
                tau_low_arr[f] = tl
                preds[f] = alarm
                diag = adapter.last_gmc_diagnostic
                if cname == "C6_RSKEF_GMC":
                    gmc_frame_fields["GMC_Attempted"][f] = True
                    if isinstance(diag, dict):
                        gmc_frame_fields["GMC_Diagnostic_Present"][f] = True
                        for field, key in (
                            ("GMC_Success", "success"),
                            ("GMC_Diagnostic_Output_Valid", "warp_output_valid"),
                            ("GMC_Reported_Fallback", "fallback"),
                            ("GMC_Affine_Estimated", "affine_estimated"),
                            ("GMC_Affine_Estimate_Accepted", "affine_estimate_accepted"),
                            ("GMC_Warp_Applied", "warp_applied"),
                            ("GMC_NonIdentity_Transform_Estimate", "estimated_non_identity_transform"),
                        ):
                            gmc_frame_fields[field][f] = bool(diag.get(key, False))
                        gmc_frame_fields["GMC_Output_Valid"][f] = adapter.last_gmc_output_valid
                        gmc_frame_fields["GMC_Fallback"][f] = (
                            bool(diag.get("fallback", False))
                            or not adapter.last_gmc_output_valid
                        )
                        gmc_frame_fields["GMC_Reason"][f] = str(diag.get("reason", ""))
                        for field, key in (
                            ("GMC_Features", "n_features"),
                            ("GMC_Matches", "n_matches"),
                            ("GMC_Inliers", "n_inliers"),
                            ("GMC_Inlier_Ratio", "inlier_ratio"),
                            ("GMC_Tx", "tx"),
                            ("GMC_Ty", "ty"),
                            ("GMC_Rotation_Deg", "rotation_deg"),
                            ("GMC_Scale", "scale"),
                        ):
                            if diag.get(key) is not None:
                                gmc_frame_fields[field][f] = float(diag[key])
                    else:
                        gmc_frame_fields["GMC_Output_Valid"][f] = adapter.last_gmc_output_valid
                        gmc_frame_fields["GMC_Fallback"][f] = not adapter.last_gmc_output_valid

            expected_observations = max(0, e0 - s0 - 1)
            if adapter.observations_seen != expected_observations:
                raise AssertionError(
                    f"Clip {r['Clip_ID']} processed {adapter.observations_seen} energy observations; "
                    f"expected {expected_observations}."
                )
            clip_meta = next(
                item for item in clip_processing_metadata if str(item["clip_id"]) == str(r["Clip_ID"])
            )
            if adapter.first_evaluation_observation != clip_meta["first_evaluated_observation"]:
                raise AssertionError(
                    f"Clip {r['Clip_ID']} first evaluation observation disagrees with the frame mask."
                )
            clip_meta.setdefault("verified_by_configuration", {})[cname] = {
                "observations_processed": adapter.observations_seen,
                "calibration_observations_seen": len(adapter.calibration_scores),
                "first_evaluation_observation": adapter.first_evaluation_observation,
            }

        if adapter.feature_extractor_calls != expected_energy_observations:
            raise AssertionError(
                f"{cname} called the feature extractor {adapter.feature_extractor_calls} times; "
                f"expected once for each of {expected_energy_observations} energy observations."
            )
        if cname == "C6_RSKEF_GMC" and adapter.gmc_attempts != expected_energy_observations:
            raise AssertionError(
                f"C6 GMC attempts={adapter.gmc_attempts}, expected {expected_energy_observations}."
            )

        elapsed = time.time() - t0
        fps_real = n_frames / elapsed if elapsed > 0 else 0.0

        # Save predictions
        pred_csv = os.path.join(out_frames_dir, f"{cname}_predictions.csv")
        prediction_rows = {
            "Frame_1idx": np.arange(1, n_frames + 1),
            "Ground_Truth": y_true_full,
            "Annotated_Clip_Mask": annotated_clip_mask.astype(int),
            "Evaluated_Mask": eval_mask.astype(int),
            "Calibration_Observation_Mask": calibration_mask.astype(int),
            "Flow_Initialization_Mask": flow_initialization_mask.astype(int),
            "Outside_Annotated_Clip_Mask": outside_clip_mask.astype(int),
            "Other_Excluded_Mask": other_excluded_mask.astype(int),
            "Energy_Score": scores,
            "Tau_High": tau_high_arr,
            "Tau_Low": tau_low_arr,
            "Alarm_Prediction": preds
        }
        prediction_rows.update(gmc_frame_fields)
        pd.DataFrame(prediction_rows).to_csv(pred_csv, index=False)

        metrics = evaluate_metrics_exact(
            y_true_full=y_true_full,
            y_pred_full=preds,
            y_scores_full=scores,
            eval_mask=eval_mask,
            sequence_meta=sequence_meta,
            fps=fps
        )
        metrics["Configuration"] = cname
        metrics["GMC_Attempts"] = adapter.gmc_attempts
        metrics["GMC_Valid_Returned_Images"] = adapter.gmc_valid_returned_images
        metrics["GMC_Affine_Estimates"] = adapter.gmc_affine_estimates
        metrics["GMC_Accepted_Affine_Estimates"] = adapter.gmc_accepted_estimates
        metrics["GMC_Warps_Applied"] = adapter.gmc_warps_applied
        metrics["GMC_NonIdentity_Transform_Estimates"] = adapter.gmc_nonidentity_estimates
        metrics["GMC_Fallbacks"] = adapter.gmc_fallbacks
        metrics["GMC_Reported_Fallbacks"] = adapter.gmc_reported_fallbacks
        metrics["GMC_Missing_Diagnostics"] = adapter.gmc_missing_diagnostics
        metrics["Feature_Extractor_Calls"] = adapter.feature_extractor_calls
        metrics["GMC_Fallback_Reasons"] = json.dumps(
            dict(adapter.gmc_fallback_reasons)
        )
        metrics["GMC_Diagnostic_Reasons"] = json.dumps(
            dict(adapter.gmc_diagnostic_reasons)
        )
        metrics["Description"] = cdesc
        metrics["Throughput_FPS"] = round(fps_real, 1)
        metrics["Execution_Scope"] = "SMOKE_TEST_NON_BENCHMARK" if smoke_test else "FULL_BENCHMARK_EVALUATION"
        summary_rows.append(metrics)

        if len(np.unique(y_true_full[eval_mask])) > 1:
            fprs, tprs, _ = roc_curve(y_true_full[eval_mask], scores[eval_mask])
            roc_entries.append((cname, fprs, tprs, metrics["AUROC"]))

        print(
            "     Confusion matrix (from final metric dictionary): "
            f"TP={metrics['TP']} TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']} "
            f"(n={metrics['Evaluated_Frames']})"
        )
        print(
            f"     Acc: {metrics['Accuracy']}% | Rec: {metrics['Recall']}% | "
            f"Prec: {metrics['Precision']}% | F1: {metrics['F1_Score']}% | "
            f"AUROC: {metrics['AUROC']} | AUPRC: {metrics['AUPRC']} | "
            f"Delay: {metrics['Mean_Delay_ms']}ms | FA/Hr: {metrics['FA_Per_Hour']} "
            f"(normal exposure={metrics['Normal_Exposure_Seconds']}s)"
        )
        print(f"     Final metric row: {json.dumps(metrics, sort_keys=True)}")

    # 5. Export Summary Results
    print(f"\n[Stage 5/5] Exporting Results and Diagnostic Plots to: {target_dir}")
    df_summary = pd.DataFrame(summary_rows)
    cols = [
        "Configuration", "GMC_Attempts", "GMC_Valid_Returned_Images",
        "GMC_Affine_Estimates", "GMC_Accepted_Affine_Estimates", "GMC_Warps_Applied",
        "GMC_NonIdentity_Transform_Estimates", "GMC_Fallbacks", "GMC_Reported_Fallbacks",
        "GMC_Missing_Diagnostics", "GMC_Fallback_Reasons", "GMC_Diagnostic_Reasons",
        "Feature_Extractor_Calls", "Description", "Accuracy", "Precision", "Recall",
        "F1_Score", "Specificity", "FPR", "AUROC", "AUPRC", "Score_Metric_Type",
        "AUPRC_Method", "Event_Hit_Rate", "Event_Misses",
        "Events_With_Preexisting_Alarm", "Mean_Delay_ms", "Premature_Alarms",
        "Premature_Alarm_Sequences", "False_Alarm_Clusters", "Evaluated_Frames",
        "Evaluated_Normal_Frames", "Normal_Exposure_Seconds", "Normal_Exposure_Hours",
        "FA_Per_Hour", "Throughput_FPS", "Execution_Scope",
        "Confusion_Matrix_TN_FP_FN_TP", "TP", "TN", "FP", "FN"
    ]
    df_summary = df_summary[cols]

    summary_path = os.path.join(target_dir, "ablation_results.csv")
    df_summary.to_csv(summary_path, index=False)
    persisted_summary = pd.read_csv(summary_path, keep_default_na=False)
    if len(persisted_summary) != len(summary_rows):
        raise AssertionError("Persisted summary row count differs from the in-memory metrics.")
    for row_index, metrics in enumerate(summary_rows):
        persisted = persisted_summary.iloc[row_index]
        for key, value in metrics.items():
            saved_value = persisted[key]
            if isinstance(value, (int, float)):
                agrees = math.isclose(float(saved_value), float(value), rel_tol=0.0, abs_tol=1e-9)
            else:
                agrees = str(saved_value) == str(value)
            if not agrees:
                raise AssertionError(
                    f"Persisted summary disagrees with console metric {key} "
                    f"for {metrics['Configuration']}: {saved_value} != {value}"
                )

    # Full run canonical update
    if not smoke_test:
        df_summary.to_csv(canonical_summary, index=False)

    def file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    source_paths = {
        "runner": __file__,
        "features": env.r_features.__file__,
        "gmc": env.r_gmc.__file__,
        "calibration": env.r_calib.__file__,
        "config": env.r_config.__file__,
    }
    config_values_by_configuration = {}
    buffer_protocol_by_configuration = {}
    for cname, _ in configs:
        cfg = env.cfg_c6 if cname == "C6_RSKEF_GMC" else env.cfg_instance
        config_values_by_configuration[cname] = {
            key: value for key, value in vars(cfg).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        buffer_protocol_by_configuration[cname] = {
            "calibration_sample_buffer_max": CALIB_WINDOW,
            "post_calibration_rolling_buffer_max": (
                4 if cname == "C2_Flow_TemporalFilter"
                else 0 if cname == "C1_Baseline_Flow"
                else 150
            ),
            "buffer_policy": (
                "fixed initial calibration sample; five-observation score uses the last four prior energies"
                if cname == "C2_Flow_TemporalFilter"
                else "fixed initial calibration sample"
                if cname == "C1_Baseline_Flow"
                else "rolling robust-score sample, capped at 150"
            ),
        }

    feature_module = env.r_features
    gmc_module = env.r_gmc
    calibration_module = env.r_calib
    meta_path = os.path.join(target_dir, "experiment_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_path": video_path,
            "dataset_scope": {
                "video": os.path.basename(video_path),
                "decoded_frames": n_frames,
                "decoded_frame_range_1idx": [1, n_frames],
                "smoke_test_decode_limit": max_decode if smoke_test else None,
                "processed_clips": clip_processing_metadata,
                "clip_ids_not_decoded": [
                    int(r["Clip_ID"]) for r in gt_records
                    if int(r["Start_Frame_1idx"]) - 1 >= n_frames
                ],
            },
            "frame_accounting": {
                "decoded_frames": n_frames,
                "calibration_energy_observations": calibration_count,
                "first_frame_optical_flow_initializations": flow_init_count,
                "frames_outside_annotated_clip_ranges": outside_clip_count,
                "other_excluded_frames": other_excluded_count,
                "evaluated_frames": eval_count,
                "categories_are_disjoint_and_complete": True,
            },
            "calibration": {
                "observations_per_complete_clip": CALIB_WINDOW,
                "first_evaluated_observation": CALIB_WINDOW + 1,
                "implementation": "ModularRSKEFAdapter.evaluate_step() fixed initial-sample buffer",
                "baseline_calibrator_invoked": False,
                "runtime_config_strategy_label": "adapter_fixed_90",
                "config_calibration_frames": 90,
                "partial_clip_policy": "Use all available observations for calibration; evaluate none until observation 91.",
                "rolling_buffer_maximum_by_configuration": buffer_protocol_by_configuration,
            },
            "feature_extractor": {
                "status": env.feature_extractor_status,
                "function": "r_skef_core.features.extract_motion_features",
                "flow_function": "r_skef_core.features.compute_flow",
                "farneback_parameters": feature_module.FARNEBACK_PARAMS,
                "active_magnitude_threshold": feature_module.ACTIVE_MAG_THRESHOLD,
                "velocity_cap_percentile": feature_module.VELOCITY_CAP_PERCENTILE,
                "adapter_calls_feature_extractor_once_per_energy_observation": True,
                "expected_feature_extractor_calls_per_configuration": expected_energy_observations,
                "expected_dense_farneback_calls_per_configuration": expected_energy_observations,
                "note": "extract_motion_features calls compute_flow once; the adapter does not separately call compute_flow. C6 GMC's sparse Lucas-Kanade tracking is a separate estimator, not a second dense Farneback pass. The adapter reads only e_area_base; divergence/disorder multipliers, detector jerk gate, EMA, and BaselineCalibrator are not used in adapter scoring.",
            },
            "gmc": {
                "status": "reconstructed estimator; not verified historical RSKEF",
                "enabled_only_for": ["C6_RSKEF_GMC"],
                "diagnostic_meaning": {
                    "warp_output_valid": "A returned array is present and shape-compatible; may be the unwarped fallback.",
                    "affine_estimated": "RANSAC produced an affine candidate; does not imply it passed acceptance guards.",
                    "affine_estimate_accepted": "Candidate passed this implementation's inlier and plausibility guards.",
                    "warp_applied": "The accepted affine candidate was passed to warpAffine.",
                    "estimated_non_identity_transform": "Estimated transform differs from identity beyond 1e-6; not proof of true camera motion.",
                    "fallback": "Original previous frame was returned without applying an accepted affine warp.",
                    "reason_ok": "Affine candidate accepted and warp returned; does not establish accuracy or real camera motion.",
                },
                "actual_camera_motion_accuracy_independently_verified": False,
                "settings": {
                    "max_features": EXPERIMENTAL_PROVENANCE_REGISTRY["gmc_max_features"],
                    "ransac_threshold": EXPERIMENTAL_PROVENANCE_REGISTRY["gmc_ransac_threshold"],
                    "min_matches": gmc_module._MIN_MATCHES,
                    "min_inliers": gmc_module._MIN_INLIERS,
                    "min_inlier_ratio": gmc_module._MIN_INLIER_RATIO,
                    "min_scale": gmc_module._MIN_SCALE,
                    "max_scale": gmc_module._MAX_SCALE,
                    "max_rotation_degrees": gmc_module._MAX_ROTATION_DEG,
                    "max_translation_ratio": gmc_module._MAX_TRANSLATION_RATIO,
                },
            },
            "ground_truth": {
                "source": candidate_gt_path,
                "sha256_version": file_sha256(candidate_gt_path),
                "status": sorted({str(r.get("Ground_Truth_Status", "UNSPECIFIED")) for r in gt_records}),
                "independent_validation_status": "provisional candidate; not verified benchmark ground truth",
                "evaluation_mask_and_ground_truth_indexing": "All frame metrics select y_true, score, and prediction with the identical eval_mask.",
                "ground_truth_evaluation_mask_identical_to_prediction_mask": True,
            },
            "metrics": {
                "thresholded_metrics": ["Accuracy", "Precision", "Recall", "F1_Score", "Specificity", "FPR", "TP", "TN", "FP", "FN"],
                "ranking_metrics": {
                    "scores": "Continuous adapter anomaly score; C2 uses a five-observation moving average; no thresholded labels are used for AUROC/AUPRC.",
                    "AUROC": "sklearn roc_curve followed by trapezoidal auc",
                    "AUPRC": "sklearn precision_recall_curve followed by trapezoidal auc(recall, precision), not average precision",
                },
                "event_detection": "Delay is to the first rising-edge alarm on or after candidate onset; an alarm already active on the preceding frame is counted in Events_With_Preexisting_Alarm and is not treated as zero-delay event detection. Event_Misses is reported.",
                "premature_alarm_count": "Number of rising-edge alarm clusters during evaluated normal exposure; sequence count is separately reported.",
                "false_alarm_rate_denominator": "Only evaluated ground-truth-normal frames; Normal_Exposure_Seconds and Normal_Exposure_Hours are exported.",
            },
            "configuration_parameters": {
                "provenance_registry": EXPERIMENTAL_PROVENANCE_REGISTRY,
                "resolved_rskef_config_by_configuration": config_values_by_configuration,
                "adapter_parameters": {
                    "initial_calibration_observations": {
                        "value": CALIB_WINDOW,
                        "provenance": "Explicit validation protocol, not a recovered historical parameter.",
                    },
                    "fixed_threshold_multiplier": {
                        "value": 2.5,
                        "configurations": ["C1_Baseline_Flow", "C2_Flow_TemporalFilter"],
                        "provenance": "Current adapter implementation.",
                    },
                    "temporal_filter_window_observations": {
                        "value": 5,
                        "configurations": ["C2_Flow_TemporalFilter"],
                        "provenance": "Current adapter implementation.",
                    },
                    "persistence_observations": {
                        "value": EXPERIMENTAL_PROVENANCE_REGISTRY["k_min"]["value"],
                        "configurations": ["C4_Flow_MAD_Hysteresis", "C5_Complete_RSKEF", "C6_RSKEF_GMC"],
                        "provenance": "Explicit validation parameter; not verified historical RSKEF.",
                    },
                    "robust_threshold_constants": {
                        "mad_to_sigma": calibration_module.MAD_TO_SIGMA,
                        "t_off_floor_factor": calibration_module.T_OFF_FLOOR_FACTOR,
                        "provenance": "Current reconstructed calibration.py implementation.",
                    },
                },
                "configuration_design": {
                    cname: description for cname, description in configs
                },
                "differences_by_design": "C1/C2 fixed threshold; C2 adds score smoothing; C3 robust threshold without hysteresis; C4 adds hysteresis/persistence without state-gated updates; C5 adds state-gated updates; C6 adds reconstructed GMC only.",
            },
            "source_versions": {
                name: {"path": path, "sha256": file_sha256(path)}
                for name, path in source_paths.items()
            },
            "evaluated_frames": eval_count,
            "smoke_test": smoke_test,
            "valid_benchmark_metrics": False,
            "disclaimer": "Smoke-test results use provisional candidate ground truth and reconstructed feature/GMC code; they are for pipeline verification and must not be cited as benchmark accuracy.",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, f, indent=2)

    if roc_entries:
        plt.figure(figsize=(8, 6), dpi=300)
        for cname, fprs, tprs, auroc_val in roc_entries:
            plt.plot(fprs, tprs, lw=2, label=f"{cname} (AUC = {auroc_val})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
        plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
        plt.ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
        plt.title(f"UMN Controlled Ablations ROC ({mode_dir.upper()})", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        roc_plot_path = os.path.join(out_plots_dir, "umn_controlled_roc_curves.png")
        plt.savefig(roc_plot_path)
        plt.close()
        print(f"  [Plot Saved] ROC Analysis: {roc_plot_path}")

    print(f"\n[EXECUTION COMPLETE]")
    print(f"  Summary CSV:      {summary_path}")
    print(f"  Parameter Logs:   {prov_file}")
    if smoke_test:
        print("  NOTICE: Smoke test verified code execution. These partial metrics are NOT benchmark numbers.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSKEF UMN Controlled Ablations Runner")
    parser.add_argument("--video", default=r"C:\crowd_project\Research_Validation\UMN\Crowd-Activity-All.avi", help="Path to UMN AVI")
    parser.add_argument("--gt", default=r"C:\crowd_project\Research_Validation\Results\UMN\candidate_umn_ground_truth.csv", help="Candidate GT CSV")
    parser.add_argument("--output", default=r"C:\crowd_project\Research_Validation\Results", help="Base output directory")
    parser.add_argument("--smoke_test", action="store_true", help="Quick verification run on >=650 frames")
    args = parser.parse_args()

    run_umn_ablations(args.video, args.gt, args.output, smoke_test=args.smoke_test)
