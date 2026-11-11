"""
Unified detector engine orchestrating R-SKEF execution across configurations.
"""

from collections import deque
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from r_skef_core.config import RskefConfig
from r_skef_core.gmc import GlobalMotionCompensator
from r_skef_core.features import extract_motion_features
from r_skef_core.density import estimate_density_map, compute_density_kinetic_energy
from r_skef_core.calibration import BaselineCalibrator
from r_skef_core.metrics import evaluate_frame_metrics, evaluate_event_metrics, evaluate_operational_metrics

def run_rskef_detector(
    video_path: str,
    gt_ranges: List[Tuple[int, int]],
    cfg: RskefConfig,
    reference_prior: Optional[Dict[str, float]] = None,
    csrnet_model: Optional[object] = None,
) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = cfg.fps

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read first frame of {video_path}")

    prev_frame = cv2.resize(prev_frame, (cfg.resize_width, cfg.resize_height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    gmc = GlobalMotionCompensator(cfg.gmc_max_features, cfg.gmc_ransac_threshold) if cfg.enable_gmc else None
    calibrator = BaselineCalibrator(cfg, reference_prior)

    energy_history = deque(maxlen=cfg.jerk_delta_frames + 2)
    gate_history = deque(maxlen=cfg.surge_hold_frames)

    frame_idx = 0
    ema_score: Optional[float] = None
    consecutive_exceed = 0
    alarm_active = False
    alarm_duration = 0

    records: List[Dict] = []
    gmc_diagnostics_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame = cv2.resize(frame, (cfg.resize_width, cfg.resize_height))
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Global Motion Compensation (if enabled)
        if gmc is not None:
            aligned_prev, gmc_diag = gmc.estimate_and_warp(prev_gray, curr_gray)
            gmc_diagnostics_history.append(gmc_diag)
            motion_prev = aligned_prev
        else:
            motion_prev = prev_gray

        # 2. Extract motion features
        feat = extract_motion_features(motion_prev, curr_gray, cfg)

        # 3. Crowd Mass & Kinetic Energy Representation
        if cfg.mass_representation == "area_mass":
            e_base = feat["e_area_base"]
            crowd_count = feat["num_active"]
        elif cfg.mass_representation in ["csrnet_global", "csrnet_active_mass"]:
            density_map = estimate_density_map(frame, csrnet_model)
            e_base, _, crowd_count = compute_density_kinetic_energy(
                density_map, feat["mag"], feat["active_mask"],
                normalization_mode=cfg.mass_representation,
                crowd_floor=cfg.csrnet_crowd_floor,
            )
        else:
            raise ValueError(f"Unknown mass representation: {cfg.mass_representation}")

        # 4. Multiplicative Temporal Jerk Surge Gate
        energy_history.append(e_base)
        if len(energy_history) >= cfg.jerk_delta_frames:
            e_past = energy_history[0]
            delta_e = max(0.0, (e_base - e_past) / cfg.jerk_delta_frames)
            r_jerk = delta_e / (e_past + cfg.min_threshold_floor)
        else:
            r_jerk = 0.0

        r_sq = r_jerk ** 2
        theta_sq = cfg.theta_jerk ** 2
        g_inst = cfg.g_0 + (1.0 - cfg.g_0) * (r_sq / (r_sq + theta_sq))
        gate_history.append(g_inst)
        g_active = float(max(gate_history))

        # Composite Gated Anomaly Energy
        composite_energy = e_base * g_active * feat["f_div"] * feat["f_disorder"]

        # Causal Temporal EMA Smoothing
        if ema_score is None:
            ema_score = composite_energy
        else:
            ema_score = cfg.ema_alpha * composite_energy + (1.0 - cfg.ema_alpha) * ema_score

        # Ground truth flag for current frame
        gt_label = int(any(start <= frame_idx <= end for start, end in gt_ranges))

        # 5. Dual-Threshold Hysteresis FSM State Update
        t_on = calibrator.t_on
        t_off = calibrator.t_off

        if not alarm_active:
            alarm_duration = 0
            if ema_score > t_on:
                consecutive_exceed += 1
                if consecutive_exceed >= cfg.persistence_frames:
                    alarm_active = True
                    pred_label = 1
                else:
                    pred_label = 0
            else:
                consecutive_exceed = max(0, consecutive_exceed - 1)
                pred_label = 0
        else:
            alarm_duration += 1
            if ema_score < t_off:
                alarm_active = False
                consecutive_exceed = 0
                pred_label = 0
            elif alarm_duration > cfg.max_alarm_unanchored_frames and g_active <= (cfg.g_0 + 0.15):
                alarm_active = False
                consecutive_exceed = 0
                pred_label = 0
            else:
                pred_label = 1

        # 6. Baseline Calibration Step
        current_t_on, current_t_off, should_evaluate = calibrator.step(
            ema_score, pred_label, consecutive_exceed, g_active
        )

        if should_evaluate:
            records.append({
                "frame": frame_idx,
                "global_magnitude": round(feat["global_mag"], 5),
                "e_base": round(e_base, 5),
                "r_jerk": round(r_jerk, 5),
                "g_active": round(g_active, 5),
                "f_div": round(feat["f_div"], 5),
                "f_disorder": round(feat["f_disorder"], 5),
                "anomaly_score": round(ema_score, 5),
                "threshold_on": round(current_t_on, 5),
                "threshold_off": round(current_t_off, 5),
                "predicted_label": pred_label,
                "ground_truth_label": gt_label,
                "crowd_count": round(crowd_count, 2),
            })

        prev_gray = curr_gray

    cap.release()

    if not records:
        return {"error": "No frames evaluated"}

    y_true = [r["ground_truth_label"] for r in records]
    y_pred = [r["predicted_label"] for r in records]
    y_score = [r["anomaly_score"] for r in records]

    frame_m = evaluate_frame_metrics(y_true, y_pred, y_score)
    event_m = evaluate_event_metrics(records, gt_ranges, fps=fps)
    op_m = evaluate_operational_metrics(records, gt_ranges, fps=fps)

    startup_discarded = cfg.calibration_frames if cfg.calibration_strategy == "fixed_150" else 0
    result = {
        "video": video_path,
        "evaluated_frames": len(records),
        "startup_frames_discarded": startup_discarded,
        "total_video_frames": frame_idx,
        "fps": fps,
        **frame_m,
        **event_m,
        **op_m,
        "records": records,
    }
    return result
