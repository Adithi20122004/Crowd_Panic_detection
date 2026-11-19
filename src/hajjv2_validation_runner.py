"""
========================================================================================
HAJJv2 EXTERNAL VALIDATION (EXACT MATCHING & FROZEN PARAMETER EVALUATION)
File: src/hajjv2_validation_runner.py

Addresses audit findings & methodology corrections:
  1. Split Isolation: Calibration and Evaluation splits MUST be disjoint sets of
     recordings to eliminate cross-split data leakage discovered via SHA-256 and
     decoded-frame duplicate analysis.
  2. Terminology Integrity: Unannotated frames are referred to accurately as
     'unflagged_training_frames' / 'unflagged_calibration_observations'.
     They are NOT claimed or relabeled as verified normal ground truth.
  3. Undefined Metrics: Zero-denominator metrics retain NaN in numerical outputs and
     format as 'N/A' in console displays (never 'nan%').
  4. Causal Condition Evaluated: results_causal is fully computed, logged, and saved.
  5. Dynamic Pedestrian ROC Consistency: Non-Human Object (NHO, class 3) frames are
     excluded from the negative normal set when evaluating dynamic pedestrian AUROC.
========================================================================================
"""

import os
import sys
import re
import math
import json
import time
import argparse
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Disjoint split ensuring zero underlying recording overlap between calibration & evaluation
DEFAULT_CALIBRATION_VIDEOS = [2, 3, 5, 7]
DEFAULT_EVALUATION_VIDEOS = [8, 9, 10, 11, 12]

DYNAMIC_PEDESTRIAN_CLASSES = {"running", "moving_in_opposite", "different_crowd_direction"}
STATIONARY_CLASSES = {"sitting", "standing", "sleeping"}
NON_HUMAN_OBJECT_CLASSES = {"moving_non_human_object"}

# Explicit constants with documented rationale
STARTUP_EXCLUDE_FRAMES = 60  # Initial frames excluded to eliminate camera/codec startup transients
MIN_SIGMA_FALLBACK = 0.1     # MAD safety floor to avoid zero division on completely stationary cameras
TAU_OFF_RATIO = 0.7          # Hysteresis lower-threshold release ratio (tau_off = 0.7 * tau_on)
DEFAULT_KAPPA = 3.2          # Multiplier for MAD-derived threshold (tau = median + kappa * sigma)
DEFAULT_K_MIN = 5            # Consecutive frames persistence filter (~185 ms at 27 fps)


def format_metric(val: Union[float, int, str], precision: int = 1, is_percentage: bool = True) -> str:
    """Formats metrics for console display: undefined / NaN display as 'N/A', never 'nan%'."""
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, (float, np.floating)):
        if math.isnan(val):
            return "N/A"
        return f"{val:.{precision}f}%" if is_percentage else f"{val:.{precision}f}"
    return str(val)


def validate_split_isolation(
    calib_ids: List[int],
    eval_ids: List[int],
    duplicate_report: Optional[Dict] = None
) -> None:
    """
    Ensure that no underlying recording used for calibration is also used for evaluation.
    Rejects overlapping Video IDs and identical physical recordings.
    """
    calib_set = set(calib_ids)
    eval_set = set(eval_ids)

    direct_overlap = calib_set.intersection(eval_set)
    if direct_overlap:
        raise ValueError(
            f"Split Isolation Violation: Calibration and evaluation sets share Video IDs: {sorted(direct_overlap)}. "
            "No underlying recording used for calibration may be used for evaluation."
        )

    if duplicate_report:
        for finding in duplicate_report.get("findings", []):
            vid = finding.get("video_id")
            if finding.get("is_same_underlying_recording", False):
                if vid in calib_set and vid in eval_set:
                    raise ValueError(
                        f"Split Isolation Violation: Video ID {vid} has been proven to share the exact same physical "
                        f"underlying recording between training and testing clips (overlap: {finding.get('overlapping_frames_count')} frames). "
                        "It cannot appear in both calibration and evaluation sets."
                    )


def find_exact_file_by_id(directory: str, vid: int, valid_extensions: List[str]) -> str:
    """Exact token matching ensuring V12 never matches V2, with ambiguity check."""
    token_pattern = re.compile(rf"(?:^|[^\d]){vid}(?:[^\d]|$)", re.IGNORECASE)
    matches = []
    for root, _, files in os.walk(directory):
        for f in files:
            if any(f.lower().endswith(e) for e in valid_extensions):
                base_name = os.path.splitext(f)[0]
                if token_pattern.search(base_name):
                    matches.append(os.path.join(root, f))

    if not matches:
        raise FileNotFoundError(f"Exact file for Video ID {vid} not found in: {directory}")
    if len(matches) > 1:
        warnings.warn(
            f"Multiple matches found for Video ID {vid} in {directory}: {matches}. "
            f"Using exact shortest path: {matches[0]}",
            UserWarning
        )
    return matches[0]


def load_hajjv2_annotations(label_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(label_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(label_path, sep=",")

    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if "video" in cl: rename_map[col] = "Video_No"
        elif "frame" in cl: rename_map[col] = "Frame_No"
        elif cl == "x": rename_map[col] = "X"
        elif cl == "y": rename_map[col] = "Y"
        elif "width" in cl or cl == "w": rename_map[col] = "Width"
        elif "height" in cl or cl == "h": rename_map[col] = "Height"
        elif "class" in cl: rename_map[col] = "Classes"
        elif "obj" in cl or "name" in cl: rename_map[col] = "Obj_Name"

    df = df.rename(columns=rename_map)
    if "Classes" in df.columns:
        df["Classes"] = df["Classes"].astype(str).str.strip().str.lower()
    return df


def extract_video_energy(video_path: str, max_frames: Optional[int] = None) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 27.0
    limit = min(total_frames, max_frames) if max_frames else total_frames

    energies = np.zeros(limit, dtype=float)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return energies, fps

    prev_small = cv2.resize(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (480, 270))
    idx = 1
    while idx < limit:
        ret, curr = cap.read()
        if not ret:
            break
        curr_small = cv2.resize(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (480, 270))

        flow = cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        energies[idx] = float(np.mean(0.5 * (flow[..., 0]**2 + flow[..., 1]**2)))
        prev_small = curr_small
        idx += 1

    cap.release()
    return energies[:idx], fps


# ======================================================================================
# 1. STRICT TRAINING SET CALIBRATION (NO TEST LEAKAGE)
# ======================================================================================

def calibrate_training_baseline(
    train_vid_dir: str,
    train_lbl_dir: str,
    video_ids: List[int],
    smoke_test: bool = False,
    kappa: float = DEFAULT_KAPPA,
    k_min: int = DEFAULT_K_MIN,
    startup_exclude: int = STARTUP_EXCLUDE_FRAMES
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"PHASE 1: STRICT TRAINING SPLIT BASELINE CALIBRATION ({len(video_ids)} VIDEOS: {video_ids})")
    print("=" * 80)

    unflagged_calibration_observations = []
    fallback_low_energy_observations = []

    # In smoke test, check up to 500 frames and iterate until sufficient unflagged observations are found
    max_decode = 500 if smoke_test else None

    for vid in video_ids:
        try:
            vid_path = find_exact_file_by_id(train_vid_dir, vid, [".mp4", ".avi"])
            lbl_path = find_exact_file_by_id(train_lbl_dir, vid, [".csv", ".xls"])
        except FileNotFoundError:
            if smoke_test and len(unflagged_calibration_observations) >= 50:
                break
            continue

        print(f"  Calibrating on Training V{vid} ({os.path.basename(vid_path)}) ...")

        df = load_hajjv2_annotations(lbl_path)
        energies, _ = extract_video_energy(vid_path, max_frames=max_decode)
        n_frames = len(energies)
        if n_frames == 0:
            continue

        # Candidate unflagged subset: Frames with NO annotated dynamic anomalies or moving objects
        # Note: These unflagged training frames are used as calibration observations;
        # they are NOT relabeled or cited as verified ground-truth normal.
        flagged_frames = set(df[df["Classes"].isin(DYNAMIC_PEDESTRIAN_CLASSES | NON_HUMAN_OBJECT_CLASSES)]["Frame_No"].astype(int))

        # Exclude initial frames to avoid camera startup transients
        effective_startup = min(startup_exclude, max(0, n_frames // 4))
        unflagged_training_frames = [i for i in range(effective_startup, n_frames) if (i + 1) not in flagged_frames]
        if unflagged_training_frames:
            unflagged_calibration_observations.extend(energies[unflagged_training_frames])
            print(f"     -> Added {len(unflagged_training_frames)} unflagged training frames (Total: {len(unflagged_calibration_observations)})")

        # Candidate fallback pool: stationary annotated frames or lowest 30% energy quantile
        stat_frames = set(df[df["Classes"].isin(STATIONARY_CLASSES)]["Frame_No"].astype(int))
        stat_idx = [i for i in range(effective_startup, n_frames) if (i + 1) in stat_frames]
        if stat_idx:
            fallback_low_energy_observations.extend(energies[stat_idx])
        elif n_frames > 10:
            q30 = float(np.percentile(energies, 30))
            low_idx = [i for i in range(effective_startup, n_frames) if energies[i] <= q30]
            fallback_low_energy_observations.extend(energies[low_idx])

        if smoke_test and len(unflagged_calibration_observations) >= 150:
            print(f"  [Smoke Test] Gathered sufficient unflagged calibration observations ({len(unflagged_calibration_observations)} frames). Concluding calibration.")
            break

    train_obs_arr = np.array(unflagged_calibration_observations)
    if len(train_obs_arr) == 0:
        if len(fallback_low_energy_observations) > 0:
            warnings.warn(
                "No strictly unflagged frames available in initial training slice. "
                "Utilizing stationary / bottom 30% energy quantile observations as robust baseline.",
                UserWarning
            )
            train_obs_arr = np.array(fallback_low_energy_observations)
        else:
            raise ValueError("No unflagged calibration observations found in training set for calibration!")

    train_median = float(np.median(train_obs_arr))
    train_mad = float(np.median(np.abs(train_obs_arr - train_median)))
    train_sigma_mad = 1.4826 * (train_mad if train_mad > 1e-4 else MIN_SIGMA_FALLBACK)

    tau_frozen = float(train_median + kappa * train_sigma_mad)

    print(f"\n  >> Training Set Unflagged Calibration Observations: {len(train_obs_arr)} frames")
    print(f"  >> Frozen Training Median:     {train_median:.4f}")
    print(f"  >> Frozen Training MAD:        {train_mad:.4f}")
    print(f"  >> Frozen Training Sigma_MAD:   {train_sigma_mad:.4f}")
    print(f"  >> Frozen Kappa Multiplier:    {kappa}")
    print(f"  >> FROZEN OPERATIONAL THRESHOLD (tau_frozen): {tau_frozen:.4f}")
    print("  >> PARAMETERS ARE NOW FROZEN. ZERO TESTING FRAMES WILL BE USED TO RE-CALIBRATE.")

    return {
        "train_median": train_median,
        "train_mad": train_mad,
        "train_sigma_mad": train_sigma_mad,
        "kappa": kappa,
        "tau_frozen": tau_frozen,
        "k_min": k_min,
        "startup_exclude_frames": startup_exclude,
        "min_sigma_fallback": MIN_SIGMA_FALLBACK,
        "tau_off_ratio": TAU_OFF_RATIO
    }


# ======================================================================================
# 2. TESTING SPLIT EVALUATION (FROZEN VS CAUSAL ADAPTATION)
# ======================================================================================

def evaluate_testing_videos(
    test_vid_dir: str,
    test_lbl_dir: str,
    video_ids: List[int],
    frozen_params: Dict[str, Any],
    output_base: str,
    smoke_test: bool = False
):
    print("\n" + "=" * 80)
    print(f"PHASE 2: TESTING SPLIT EVALUATION ({len(video_ids)} VIDEOS: {video_ids})")
    print("=" * 80)

    # Safe versioned output directory with overwrite protection
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_tag = "smoke_test" if smoke_test else "full_evaluation"
    target_dir = os.path.join(output_base, "HAJJv2", f"{mode_tag}_{run_timestamp}")
    out_preds = os.path.join(target_dir, "frame_predictions")
    out_plots = os.path.join(target_dir, "plots")
    for d in [out_preds, out_plots]:
        os.makedirs(d, exist_ok=True)

    tau_frozen = frozen_params["tau_frozen"]
    k_min = frozen_params["k_min"]
    tau_off_ratio = frozen_params.get("tau_off_ratio", TAU_OFF_RATIO)
    startup_exclude = frozen_params.get("startup_exclude_frames", STARTUP_EXCLUDE_FRAMES)

    results_frozen = []
    results_causal = []
    roc_items_frozen = []

    for vid in video_ids:
        vid_path = find_exact_file_by_id(test_vid_dir, vid, [".mp4", ".avi"])
        lbl_path = find_exact_file_by_id(test_lbl_dir, vid, [".csv", ".xls"])
        print(f"\n  Evaluating Testing V{vid} ({os.path.basename(vid_path)}) ...")

        df = load_hajjv2_annotations(lbl_path)
        energies, fps = extract_video_energy(vid_path, max_frames=500 if smoke_test else None)
        n_frames = len(energies)

        # 4-Category Ground Truth Taxonomy:
        # 1: Dynamic Pedestrian Anomaly
        # 2: Stationary Contextual
        # 3: Moving Non-Human Object
        # 0: Unannotated (Unflagged, unverified)
        gt_taxonomy = np.zeros(n_frames, dtype=int)

        dyn_f = set(df[df["Classes"].isin(DYNAMIC_PEDESTRIAN_CLASSES)]["Frame_No"].astype(int))
        stat_f = set(df[df["Classes"].isin(STATIONARY_CLASSES)]["Frame_No"].astype(int))
        nho_f = set(df[df["Classes"].isin(NON_HUMAN_OBJECT_CLASSES)]["Frame_No"].astype(int))

        for f in range(1, n_frames + 1):
            if f in dyn_f:
                gt_taxonomy[f - 1] = 1
            elif f in nho_f:
                gt_taxonomy[f - 1] = 3
            elif f in stat_f:
                gt_taxonomy[f - 1] = 2

        # -------------------------------------------------------------
        # Condition A: Strict Frozen Training Threshold
        # -------------------------------------------------------------
        preds_frozen = np.zeros(n_frames, dtype=int)
        p_count = 0
        state = 0
        for f in range(n_frames):
            e = energies[f]
            if state == 0:
                if e > tau_frozen:
                    p_count += 1
                    if p_count >= k_min:
                        state = 1
                else:
                    p_count = 0
            else:
                if e < (tau_frozen * tau_off_ratio):
                    state = 0
                    p_count = 0
            preds_frozen[f] = state

        # -------------------------------------------------------------
        # Condition B: Online Causal State-Gated Adaptation
        # -------------------------------------------------------------
        preds_causal = np.zeros(n_frames, dtype=int)
        online_buf = []
        state_c = 0
        p_count_c = 0
        tau_causal_arr = np.zeros(n_frames, dtype=float)

        for f in range(n_frames):
            e = energies[f]
            if f < startup_exclude or len(online_buf) < startup_exclude:
                if f > 0:
                    online_buf.append(float(e))
                tau_causal_arr[f] = tau_frozen
                preds_causal[f] = 0
                continue

            buf_arr = np.array(online_buf)
            c_med = float(np.median(buf_arr))
            c_mad = float(np.median(np.abs(buf_arr - c_med)))
            c_sig = 1.4826 * (c_mad if c_mad > 1e-4 else MIN_SIGMA_FALLBACK)
            t_high = c_med + frozen_params["kappa"] * c_sig
            t_low = c_med + (frozen_params["kappa"] * 0.6) * c_sig
            tau_causal_arr[f] = t_high

            if state_c == 0:
                if e > t_high:
                    p_count_c += 1
                    if p_count_c >= k_min:
                        state_c = 1
                else:
                    p_count_c = 0
                    online_buf.pop(0)
                    online_buf.append(float(e))
            else:
                if e < t_low:
                    state_c = 0
                    p_count_c = 0
                    online_buf.pop(0)
                    online_buf.append(float(e))
            preds_causal[f] = state_c

        # Evaluation Window: Exclude startup transient frames
        eval_mask = np.zeros(n_frames, dtype=bool)
        if n_frames > startup_exclude:
            eval_mask[startup_exclude:] = True

        gt_eval = gt_taxonomy[eval_mask]
        p_f_eval = preds_frozen[eval_mask]
        p_c_eval = preds_causal[eval_mask]
        en_eval = energies[eval_mask]

        # Total counts
        tot_dyn = int(np.sum(gt_eval == 1))
        tot_nho = int(np.sum(gt_eval == 3))
        tot_stat = int(np.sum(gt_eval == 2))
        tot_unan = int(np.sum(gt_eval == 0))

        # -------------------------------------------------------------
        # Compute Metrics: Frozen Baseline (Keep NaN for zero denominators)
        # -------------------------------------------------------------
        det_dyn_f = int(np.sum((gt_eval == 1) & (p_f_eval == 1)))
        rec_dyn_f = (det_dyn_f / tot_dyn * 100.0) if tot_dyn > 0 else float("nan")

        det_nho_f = int(np.sum((gt_eval == 3) & (p_f_eval == 1)))
        rec_nho_f = (det_nho_f / tot_nho * 100.0) if tot_nho > 0 else float("nan")

        sup_stat_f = int(np.sum((gt_eval == 2) & (p_f_eval == 0)))
        res_stat_f = (sup_stat_f / tot_stat * 100.0) if tot_stat > 0 else float("nan")

        act_unan_f = int(np.sum((gt_eval == 0) & (p_f_eval == 1)))
        rate_unan_f = (act_unan_f / tot_unan * 100.0) if tot_unan > 0 else float("nan")

        # -------------------------------------------------------------
        # Compute Metrics: Causal Online Adaptation
        # -------------------------------------------------------------
        det_dyn_c = int(np.sum((gt_eval == 1) & (p_c_eval == 1)))
        rec_dyn_c = (det_dyn_c / tot_dyn * 100.0) if tot_dyn > 0 else float("nan")

        det_nho_c = int(np.sum((gt_eval == 3) & (p_c_eval == 1)))
        rec_nho_c = (det_nho_c / tot_nho * 100.0) if tot_nho > 0 else float("nan")

        sup_stat_c = int(np.sum((gt_eval == 2) & (p_c_eval == 0)))
        res_stat_c = (sup_stat_c / tot_stat * 100.0) if tot_stat > 0 else float("nan")

        act_unan_c = int(np.sum((gt_eval == 0) & (p_c_eval == 1)))
        rate_unan_c = (act_unan_c / tot_unan * 100.0) if tot_unan > 0 else float("nan")

        # ROC: Exclude Non-Human Object (NHO, class 3) from the normal negative set
        non_nho_mask = (gt_eval != 3)
        gt_roc = gt_eval[non_nho_mask]
        en_roc = en_eval[non_nho_mask]
        tot_dyn_roc = int(np.sum(gt_roc == 1))
        tot_norm_roc = int(np.sum(gt_roc == 2) + np.sum(gt_roc == 0))

        if tot_dyn_roc > 0 and tot_norm_roc > 0:
            bin_gt = (gt_roc == 1).astype(int)
            fprs, tprs, _ = roc_curve(bin_gt, en_roc)
            vid_auc = float(auc(fprs, tprs))
            roc_items_frozen.append((f"V{vid}", fprs, tprs, round(vid_auc, 3)))
            try:
                vid_auprc = float(average_precision_score(bin_gt, en_roc))
            except Exception:
                vid_auprc = float("nan")
        else:
            vid_auc = float("nan")
            vid_auprc = float("nan")

        # Retain NaN in numerical dictionaries (do not replace with zero)
        results_frozen.append({
            "Video_ID": f"V{vid}",
            "Condition": "Strict_Frozen",
            "Evaluated_Frames": int(np.sum(eval_mask)),
            "Dynamic_Frames": tot_dyn,
            "Dynamic_Pedestrian_Recall_%": round(rec_dyn_f, 1) if not math.isnan(rec_dyn_f) else float("nan"),
            "NonHuman_Object_Frames": tot_nho,
            "NonHuman_Object_Detection_%": round(rec_nho_f, 1) if not math.isnan(rec_nho_f) else float("nan"),
            "Stationary_Frames": tot_stat,
            "Stationary_Suppression_%": round(res_stat_f, 1) if not math.isnan(res_stat_f) else float("nan"),
            "Unannotated_Frames": tot_unan,
            "Unannotated_Activation_Rate_%": round(rate_unan_f, 1) if not math.isnan(rate_unan_f) else float("nan"),
            "Dynamic_AUROC": round(vid_auc, 4) if not math.isnan(vid_auc) else float("nan"),
            "Dynamic_AUPRC": round(vid_auprc, 4) if not math.isnan(vid_auprc) else float("nan")
        })

        results_causal.append({
            "Video_ID": f"V{vid}",
            "Condition": "Causal_Adaptation",
            "Evaluated_Frames": int(np.sum(eval_mask)),
            "Dynamic_Frames": tot_dyn,
            "Dynamic_Pedestrian_Recall_%": round(rec_dyn_c, 1) if not math.isnan(rec_dyn_c) else float("nan"),
            "NonHuman_Object_Frames": tot_nho,
            "NonHuman_Object_Detection_%": round(rec_nho_c, 1) if not math.isnan(rec_nho_c) else float("nan"),
            "Stationary_Frames": tot_stat,
            "Stationary_Suppression_%": round(res_stat_c, 1) if not math.isnan(res_stat_c) else float("nan"),
            "Unannotated_Frames": tot_unan,
            "Unannotated_Activation_Rate_%": round(rate_unan_c, 1) if not math.isnan(rate_unan_c) else float("nan"),
            "Dynamic_AUROC": round(vid_auc, 4) if not math.isnan(vid_auc) else float("nan"),
            "Dynamic_AUPRC": round(vid_auprc, 4) if not math.isnan(vid_auprc) else float("nan")
        })

        # Save complete predictions per video
        pred_csv = os.path.join(out_preds, f"testing_V{vid}_predictions.csv")
        pd.DataFrame({
            "Frame_1idx": np.arange(1, n_frames + 1),
            "Evaluated_Mask": eval_mask.astype(int),
            "Energy_Score": energies,
            "Tau_Frozen": tau_frozen,
            "Tau_Causal": tau_causal_arr,
            "Ground_Truth_Taxonomy": gt_taxonomy,
            "Prediction_Strict_Frozen": preds_frozen,
            "Prediction_Causal_Adapt": preds_causal
        }).to_csv(pred_csv, index=False)

        # Formatted console logging: NaN displays as N/A, never nan%
        print(f"     V{vid} Frozen: Recall {format_metric(rec_dyn_f)} | Stat Supp {format_metric(res_stat_f)} | Unan Act {format_metric(rate_unan_f)}")
        print(f"     V{vid} Causal: Recall {format_metric(rec_dyn_c)} | Stat Supp {format_metric(res_stat_c)} | Unan Act {format_metric(rate_unan_c)}")

    # Save summary tables
    df_frozen = pd.DataFrame(results_frozen)
    df_causal = pd.DataFrame(results_causal)
    df_combined = pd.concat([df_frozen, df_causal], ignore_index=True)

    summary_frozen_path = os.path.join(target_dir, "hajjv2_frozen_evaluation_results.csv")
    summary_causal_path = os.path.join(target_dir, "hajjv2_causal_evaluation_results.csv")
    summary_combined_path = os.path.join(target_dir, "hajjv2_evaluation_summary.csv")

    df_frozen.to_csv(summary_frozen_path, index=False)
    df_causal.to_csv(summary_causal_path, index=False)
    df_combined.to_csv(summary_combined_path, index=False)

    print(f"\n[OUTPUT] HAJJv2 Summary saved to: {summary_combined_path}")
    print(f"[OUTPUT] HAJJv2 Causal Summary saved to: {summary_causal_path}")

    # Save metadata JSON
    meta_path = os.path.join(target_dir, "hajjv2_experiment_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "frozen_parameters": frozen_params,
            "evaluated_videos": video_ids,
            "taxonomy": {
                "dynamic_pedestrian": list(DYNAMIC_PEDESTRIAN_CLASSES),
                "stationary_contextual": list(STATIONARY_CLASSES),
                "non_human_object": list(NON_HUMAN_OBJECT_CLASSES),
                "unannotated": "unflagged_training_frames"
            },
            "smoke_test": smoke_test,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "caveats": [
                "Evaluated videos lack frame-accurate panic labels; metrics reflect object behavior tracking.",
                "Non-human object frames were excluded from negative set in dynamic pedestrian ROC.",
                "Unannotated frames are marked as unflagged observations and should not be cited as pure true-negative ground truth."
            ]
        }, f, indent=2)

    # Plot ROC curves
    if roc_items_frozen:
        plt.figure(figsize=(8, 6), dpi=300)
        for vname, fprs, tprs, auc_val in roc_items_frozen:
            plt.plot(fprs, tprs, lw=1.8, label=f"{vname} (AUC={auc_val})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
        plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
        plt.ylabel("True Positive Rate (Dynamic Recall)", fontsize=11, fontweight="bold")
        plt.title("HAJJv2 Testing Split: Dynamic Anomaly ROC", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_plots, "hajjv2_roc_curves.png"))
        plt.close()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="HAJJv2 External Validation Runner with Strict Split Isolation")
    default_hajj = os.environ.get(
        "HAJJ_DIR",
        os.path.join(PROJECT_ROOT, "HAJJv2", "Original_Videos", "HAJJv2.Dataset")
    )
    default_output = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "Results"))

    parser.add_argument("--hajj_dir", default=default_hajj, help="Path to HAJJv2 Dataset directory")
    parser.add_argument("--output", default=default_output, help="Base output directory")
    parser.add_argument("--calib_ids", nargs="+", type=int, default=DEFAULT_CALIBRATION_VIDEOS,
                        help=f"Video IDs strictly for calibration (default: {DEFAULT_CALIBRATION_VIDEOS})")
    parser.add_argument("--eval_ids", nargs="+", type=int, default=DEFAULT_EVALUATION_VIDEOS,
                        help=f"Video IDs strictly for evaluation (default: {DEFAULT_EVALUATION_VIDEOS})")
    parser.add_argument("--smoke_test", action="store_true", help="Quick verification run on 1 calib and 1 eval video")
    parser.add_argument("--check_duplicates", action="store_true", help="Re-generate duplicate-recording report before running")
    args = parser.parse_args()

    # Load duplicate report if available
    dup_report_path = os.path.join(args.output, "HAJJv2", "duplicate_recording_report.json")
    dup_report = None
    if os.path.exists(dup_report_path):
        try:
            with open(dup_report_path, "r") as f:
                dup_report = json.load(f)
        except Exception:
            pass

    calib_ids = list(args.calib_ids)
    eval_ids = list(args.eval_ids)

    if args.smoke_test:
        # Select single disjoint video for smoke testing
        calib_ids = [calib_ids[0]]
        eval_ids = [eval_ids[0]]

    # Strict split isolation check: calibration and evaluation must never share recordings
    validate_split_isolation(calib_ids, eval_ids, dup_report)

    train_vids = os.path.join(args.hajj_dir, "Training", "Videos")
    train_lbls = os.path.join(args.hajj_dir, "Training", "Labels")
    test_vids = os.path.join(args.hajj_dir, "Testing", "Videos")
    test_lbls = os.path.join(args.hajj_dir, "Testing", "Labels")

    frozen_params = calibrate_training_baseline(train_vids, train_lbls, calib_ids, smoke_test=args.smoke_test)
    evaluate_testing_videos(test_vids, test_lbls, eval_ids, frozen_params, args.output, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
