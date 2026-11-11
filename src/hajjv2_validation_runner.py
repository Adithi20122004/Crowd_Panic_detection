"""
========================================================================================
HAJJv2 EXTERNAL VALIDATION (EXACT MATCHING & FROZEN PARAMETER EVALUATION)
File: hajjv2_validation_runner.py
Directory: C:\\crowd_project\\Research_Validation\\hajjv2_validation_runner.py

Purpose:
  1. Strict exact token matching for video IDs (V2, V3, V5, V7, V8, V9, V10, V11, V12).
  2. Four-category explicit taxonomy:
     - State 1: Dynamic Pedestrian Anomalies (running, moving_in_opposite, different_crowd_direction)
     - State 2: Stationary Contextual Behaviors (sitting, standing, sleeping)
     - State 3: Moving Non-Human Objects (moving_non_human_object)
     - State 0: Unannotated Frames (Explicitly documented as unverified)
  3. Strict Training Calibration: Baseline Median and MAD are computed from verified normal
     training frames only, and FROZEN.
  4. Testing Split Evaluation: Evaluates test videos using the FROZEN training threshold.
     Zero testing frames are used for baseline estimation.
  5. Evaluates both Strict Frozen Baseline and Causal Adaptation as separate conditions.
  6. Supports smoke-test mode via --smoke_test flag.
  7. Outputs to C:\\crowd_project\\Research_Validation\\Results\\HAJJv2.
========================================================================================
"""

import os
import sys
import re
import math
import json
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VIDEO_IDS = [2, 3, 5, 7, 8, 9, 10, 11, 12]

DYNAMIC_PEDESTRIAN_CLASSES = {"running", "moving_in_opposite", "different_crowd_direction"}
STATIONARY_CLASSES = {"sitting", "standing", "sleeping"}
NON_HUMAN_OBJECT_CLASSES = {"moving_non_human_object"}


def find_exact_file_by_id(directory: str, vid: int, valid_extensions: List[str]) -> str:
    """Exact token matching ensuring V12 never matches V2."""
    token_pattern = re.compile(rf"(?:^|[^\d]){vid}(?:[^\d]|$)", re.IGNORECASE)
    matches = []
    for root, _, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if any(f.lower().endswith(e) for e in valid_extensions):
                base_name = os.path.splitext(f)[0]
                if token_pattern.search(base_name):
                    matches.append(os.path.join(root, f))

    if not matches:
        raise FileNotFoundError(f"Exact file for Video ID {vid} not found in: {directory}")
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
    smoke_test: bool = False
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PHASE 1: STRICT TRAINING SPLIT BASELINE CALIBRATION (9 VIDEOS)")
    print("=" * 80)

    normal_training_observations = []
    video_list = video_ids[:2] if smoke_test else video_ids

    for vid in video_list:
        vid_path = find_exact_file_by_id(train_vid_dir, vid, [".mp4", ".avi"])
        lbl_path = find_exact_file_by_id(train_lbl_dir, vid, [".csv", ".xls"])
        print(f"  Calibrating on Training V{vid} ({os.path.basename(vid_path)}) ...")

        df = load_hajjv2_annotations(lbl_path)
        energies, _ = extract_video_energy(vid_path, max_frames=200 if smoke_test else None)
        n_frames = len(energies)

        # Defensible calibration subset: Frames with NO annotated dynamic anomalies or non-human objects
        flagged_frames = set(df[df["Classes"].isin(DYNAMIC_PEDESTRIAN_CLASSES | NON_HUMAN_OBJECT_CLASSES)]["Frame_No"].astype(int))
        
        # Exclude initial 60 frames to avoid camera startup transients
        verified_normal_idx = [i for i in range(60, n_frames) if (i + 1) not in flagged_frames]
        if verified_normal_idx:
            normal_training_observations.extend(energies[verified_normal_idx])

    train_obs_arr = np.array(normal_training_observations)
    if len(train_obs_arr) == 0:
        raise ValueError("No normal observations found in training set for calibration!")

    train_median = float(np.median(train_obs_arr))
    train_mad = float(np.median(np.abs(train_obs_arr - train_median)))
    train_sigma_mad = 1.4826 * (train_mad if train_mad > 1e-4 else 0.1)

    kappa = 3.2
    tau_frozen = float(train_median + kappa * train_sigma_mad)

    print(f"\n  >> Training Set Normal Observations: {len(train_obs_arr)} frames")
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
        "k_min": 5
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
    print("PHASE 2: TESTING SPLIT EVALUATION (FROZEN PARAMETERS, EXACT MATCHING)")
    print("=" * 80)

    out_preds = os.path.join(output_base, "HAJJv2", "frame_predictions")
    out_plots = os.path.join(output_base, "HAJJv2", "plots")
    for d in [out_preds, out_plots]:
        os.makedirs(d, exist_ok=True)

    tau_frozen = frozen_params["tau_frozen"]
    k_min = frozen_params["k_min"]

    video_list = video_ids[:2] if smoke_test else video_ids
    results_frozen = []
    results_causal = []
    roc_items = []

    for vid in video_list:
        vid_path = find_exact_file_by_id(test_vid_dir, vid, [".mp4", ".avi"])
        lbl_path = find_exact_file_by_id(test_lbl_dir, vid, [".csv", ".xls"])
        print(f"\n  Evaluating Testing V{vid} ({os.path.basename(vid_path)}) ...")

        df = load_hajjv2_annotations(lbl_path)
        energies, fps = extract_video_energy(vid_path, max_frames=200 if smoke_test else None)
        n_frames = len(energies)

        # 4-Category Ground Truth Taxonomy
        # 1: Dynamic Pedestrian Anomaly
        # 2: Stationary Contextual
        # 3: Moving Non-Human Object
        # 0: Unannotated (Documented as Unverified)
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
        # Condition A: Strict Frozen Training Threshold (Zero Test Adaptation)
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
                if e < (tau_frozen * 0.7):
                    state = 0
                    p_count = 0
            preds_frozen[f] = state

        # -------------------------------------------------------------
        # Condition B: Online Causal State-Gated Adaptation (Evaluated Separately)
        # -------------------------------------------------------------
        preds_causal = np.zeros(n_frames, dtype=int)
        online_buf = []
        state_c = 0
        p_count_c = 0
        tau_causal_arr = np.zeros(n_frames, dtype=float)

        for f in range(n_frames):
            e = energies[f]
            if len(online_buf) < 60:
                online_buf.append(float(e))
                tau_causal_arr[f] = tau_frozen
                preds_causal[f] = 0
                continue

            buf_arr = np.array(online_buf)
            c_med = float(np.median(buf_arr))
            c_mad = float(np.median(np.abs(buf_arr - c_med)))
            c_sig = 1.4826 * (c_mad if c_mad > 1e-4 else 0.1)
            t_high = c_med + 3.2 * c_sig
            t_low = c_med + 2.0 * c_sig
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

        # Evaluation Window: Exclude first 60 frames
        eval_mask = np.zeros(n_frames, dtype=bool)
        if n_frames > 60:
            eval_mask[60:] = True

        gt_eval = gt_taxonomy[eval_mask]
        p_f_eval = preds_frozen[eval_mask]
        p_c_eval = preds_causal[eval_mask]
        en_eval = energies[eval_mask]

        # Calculate metrics for Frozen Condition
        # 1. Dynamic Pedestrian Recall
        tot_dyn = int(np.sum(gt_eval == 1))
        det_dyn_f = int(np.sum((gt_eval == 1) & (p_f_eval == 1)))
        rec_dyn_f = (det_dyn_f / tot_dyn * 100.0) if tot_dyn > 0 else float("nan")

        # 2. Non-Human Object Detection Rate
        tot_nho = int(np.sum(gt_eval == 3))
        det_nho_f = int(np.sum((gt_eval == 3) & (p_f_eval == 1)))
        rec_nho_f = (det_nho_f / tot_nho * 100.0) if tot_nho > 0 else float("nan")

        # 3. Stationary Contextual Resistance (Suppression)
        tot_stat = int(np.sum(gt_eval == 2))
        sup_stat_f = int(np.sum((gt_eval == 2) & (p_f_eval == 0)))
        res_stat_f = (sup_stat_f / tot_stat * 100.0) if tot_stat > 0 else float("nan")

        # 4. Unannotated Activation Rate (Unverified frames)
        tot_unan = int(np.sum(gt_eval == 0))
        act_unan_f = int(np.sum((gt_eval == 0) & (p_f_eval == 1)))
        rate_unan_f = (act_unan_f / tot_unan * 100.0) if tot_unan > 0 else float("nan")

        # ROC against Dynamic Ground Truth
        if tot_dyn > 0 and (tot_stat + tot_unan) > 0:
            bin_gt = (gt_eval == 1).astype(int)
            fprs, tprs, _ = roc_curve(bin_gt, en_eval)
            vid_auc = float(auc(fprs, tprs))
            roc_items.append((f"V{vid}", fprs, tprs, round(vid_auc, 3)))
        else:
            vid_auc = float("nan")

        results_frozen.append({
            "Video_ID": f"V{vid}",
            "Evaluated_Frames": int(np.sum(eval_mask)),
            "Dynamic_Frames": tot_dyn,
            "Dynamic_Pedestrian_Recall_%": round(rec_dyn_f, 1) if not math.isnan(rec_dyn_f) else "N/A",
            "NonHuman_Object_Frames": tot_nho,
            "NonHuman_Object_Detection_%": round(rec_nho_f, 1) if not math.isnan(rec_nho_f) else "N/A",
            "Stationary_Frames": tot_stat,
            "Stationary_Suppression_%": round(res_stat_f, 1) if not math.isnan(res_stat_f) else "N/A",
            "Unannotated_Frames": tot_unan,
            "Unannotated_Activation_Rate_%": round(rate_unan_f, 1) if not math.isnan(rate_unan_f) else "N/A",
            "Dynamic_AUROC": round(vid_auc, 4) if not math.isnan(vid_auc) else "N/A"
        })

        # Save complete predictions
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

        print(f"     V{vid} Dynamic Recall: {rec_dyn_f}% | Stationary Suppression: {res_stat_f}% | Unannotated Activation: {rate_unan_f}%")

    # Save summary table
    df_eval = pd.DataFrame(results_frozen)
    summary_path = os.path.join(output_base, "HAJJv2", "evaluation_results.csv")
    df_eval.to_csv(summary_path, index=False)
    print(f"\n[OUTPUT] HAJJv2 Strict Frozen Summary saved to: {summary_path}")

    # Save metadata JSON
    with open(os.path.join(output_base, "HAJJv2", "hajjv2_experiment_metadata.json"), "w") as f:
        json.dump({
            "frozen_parameters": frozen_params,
            "evaluated_videos": video_list,
            "taxonomy": {
                "dynamic_pedestrian": list(DYNAMIC_PEDESTRIAN_CLASSES),
                "stationary_contextual": list(STATIONARY_CLASSES),
                "non_human_object": list(NON_HUMAN_OBJECT_CLASSES),
                "unannotated": "unverified_frames"
            },
            "smoke_test": smoke_test,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, f, indent=2)

    # Plot ROC curves
    if roc_items:
        plt.figure(figsize=(8, 6), dpi=300)
        for vname, fprs, tprs, auc_val in roc_items:
            plt.plot(fprs, tprs, lw=1.8, label=f"{vname} (AUC={auc_val})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
        plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
        plt.ylabel("True Positive Rate (Dynamic Recall)", fontsize=11, fontweight="bold")
        plt.title("HAJJv2 Testing Split: Dynamic Anomaly ROC (Strict Frozen Calibration)", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_plots, "hajjv2_strict_frozen_roc.png"))
        plt.close()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hajj_dir", default=r"C:\crowd_project\Research_Validation\HAJJv2\Original_Videos\HAJJv2.Dataset")
    parser.add_argument("--output", default=r"C:\crowd_project\Research_Validation\Results")
    parser.add_argument("--smoke_test", action="store_true", help="Quick verification on first 2 videos")
    args = parser.parse_args()

    train_vids = os.path.join(args.hajj_dir, "Training", "Videos")
    train_lbls = os.path.join(args.hajj_dir, "Training", "Labels")
    test_vids = os.path.join(args.hajj_dir, "Testing", "Videos")
    test_lbls = os.path.join(args.hajj_dir, "Testing", "Labels")

    frozen_params = calibrate_training_baseline(train_vids, train_lbls, VIDEO_IDS, smoke_test=args.smoke_test)
    evaluate_testing_videos(test_vids, test_lbls, VIDEO_IDS, frozen_params, args.output, smoke_test=args.smoke_test)

if __name__ == "__main__":
    main()
