"""
========================================================================================
UMN GROUND TRUTH DIAGNOSTIC AUDIT
File: audit_and_verify_umn_gt.py
Directory: C:\\crowd_project\\Research_Validation\\audit_and_verify_umn_gt.py

Purpose:
  1. Reads Crowd-Activity-All.avi (7,739 frames).
  2. Computes both Farneback Optical-Flow Kinetic Energy and Pixel Difference as diagnostics.
  3. Audits candidate boundaries and anomaly intervals without falsely claiming independent verification.
  4. Keeps candidate annotations strictly labeled as 'PROVISIONAL_CANDIDATE'.
  5. Exports diagnostic data to:
     Results\\UMN\\candidate_umn_ground_truth.csv
     Results\\UMN\\plots\\umn_energy_profile_diagnostic.png
========================================================================================
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VIDEO_PATH = r"C:\crowd_project\Research_Validation\UMN\Crowd-Activity-All.avi"
RESULTS_DIR = r"C:\crowd_project\Research_Validation\Results\UMN"
OUTPUT_CSV = os.path.join(RESULTS_DIR, "candidate_umn_ground_truth.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Published literature candidate sequences (1-indexed)
CANDIDATE_SEQUENCES = [
    {"clip_id": 1,  "scene_id": 1, "scene_name": "Lawn",   "start": 1,    "end": 625,  "cand_anom": 486},
    {"clip_id": 2,  "scene_id": 1, "scene_name": "Lawn",   "start": 626,  "end": 1450, "cand_anom": 1011},
    {"clip_id": 3,  "scene_id": 2, "scene_name": "Indoor", "start": 1451, "end": 2140, "cand_anom": 1741},
    {"clip_id": 4,  "scene_id": 2, "scene_name": "Indoor", "start": 2141, "end": 2720, "cand_anom": 2411},
    {"clip_id": 5,  "scene_id": 2, "scene_name": "Indoor", "start": 2721, "end": 3430, "cand_anom": 3011},
    {"clip_id": 6,  "scene_id": 2, "scene_name": "Indoor", "start": 3431, "end": 4040, "cand_anom": 3711},
    {"clip_id": 7,  "scene_id": 2, "scene_name": "Indoor", "start": 4041, "end": 4740, "cand_anom": 4411},
    {"clip_id": 8,  "scene_id": 2, "scene_name": "Indoor", "start": 4741, "end": 5865, "cand_anom": 5411},
    {"clip_id": 9,  "scene_id": 3, "scene_name": "Plaza",  "start": 5866, "end": 6515, "cand_anom": 6171},
    {"clip_id": 10, "scene_id": 3, "scene_name": "Plaza",  "start": 6516, "end": 7155, "cand_anom": 6821},
    {"clip_id": 11, "scene_id": 3, "scene_name": "Plaza",  "start": 7156, "end": 7739, "cand_anom": 7451},
]

def run_diagnostic_audit():
    print("=" * 80)
    print("STEP 2: UMN GROUND TRUTH DIAGNOSTIC AUDIT")
    print(f"Target Video: {VIDEO_PATH}")
    print("=" * 80)

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video missing: {VIDEO_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    ret, prev = cap.read()
    if not ret:
        raise IOError("Failed to read first frame.")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    optical_flow_energies = [0.0]
    pixel_diff_energies = [0.0]

    print("Computing Farneback optical-flow kinetic energy and pixel difference...")
    frame_idx = 1
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # 1. Pixel Difference Diagnostic
        diff = cv2.absdiff(curr_gray, prev_gray)
        pixel_diff_energies.append(float(np.mean(diff)))

        # 2. Dense Optical Flow Kinetic Energy Diagnostic
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        v_sq = flow[..., 0]**2 + flow[..., 1]**2
        optical_flow_energies.append(float(np.mean(0.5 * v_sq)))

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    total_decoded = len(optical_flow_energies)
    flow_arr = np.array(optical_flow_energies)
    diff_arr = np.array(pixel_diff_energies)

    print(f"Container metadata frames: {meta_frames}")
    print(f"Total frames successfully decoded: {total_decoded}")

    records = []
    print("\nAuditing candidate intervals with optical flow motion energy:")
    print(f"{'Clip':<6} {'Scene':<8} {'Range (1-idx)':<16} {'Cand Onset':<12} {'Pre-Anom Flow':<15} {'Post-Anom Flow':<15} {'Flow Ratio':<12} {'Status'}")
    print("-" * 95)

    for seq in CANDIDATE_SEQUENCES:
        cid = seq["clip_id"]
        s_0 = seq["start"] - 1
        e_0 = min(seq["end"], total_decoded)
        a_0 = seq["cand_anom"] - 1

        pre_flow = flow_arr[s_0:a_0]
        post_flow = flow_arr[a_0:e_0]

        mean_pre = float(np.mean(pre_flow)) if len(pre_flow) > 0 else 0.0
        mean_post = float(np.mean(post_flow)) if len(post_flow) > 0 else 0.0
        ratio = (mean_post / mean_pre) if mean_pre > 0 else 0.0

        status = "PROVISIONAL_CANDIDATE"

        print(f"Clip {cid:<2} {seq['scene_name']:<8} [{s_0+1}..{e_0}]        {a_0+1:<12} {mean_pre:<15.3f} {mean_post:<15.3f} {ratio:<12.2f}x {status}")

        records.append({
            "Clip_ID": cid,
            "Scene_ID": seq["scene_id"],
            "Scene_Name": seq["scene_name"],
            "Start_Frame_1idx": s_0 + 1,
            "End_Frame_1idx": e_0,
            "Candidate_Anomaly_Onset_1idx": a_0 + 1,
            "Pre_Anomaly_Frames": a_0 - s_0,
            "Post_Anomaly_Frames": e_0 - a_0,
            "Pre_Anomaly_Mean_Flow_Energy": round(mean_pre, 4),
            "Post_Anomaly_Mean_Flow_Energy": round(mean_post, 4),
            "Optical_Flow_Energy_Ratio": round(ratio, 2),
            "Pre_Anomaly_Mean_Pixel_Diff": round(float(np.mean(diff_arr[s_0:a_0])), 4),
            "Post_Anomaly_Mean_Pixel_Diff": round(float(np.mean(diff_arr[a_0:e_0])), 4),
            "Ground_Truth_Status": status
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OUTPUT] Candidate ground-truth audit saved to: {OUTPUT_CSV}")

    # Generate diagnostic plot
    plt.figure(figsize=(14, 5), dpi=300)
    plt.plot(flow_arr, label="Farneback Optical Flow Kinetic Energy", color="#6366f1", lw=1.2)
    for r in records:
        plt.axvline(x=r["Candidate_Anomaly_Onset_1idx"], color="red", linestyle="--", alpha=0.6, lw=1.0)
        plt.axvline(x=r["Start_Frame_1idx"], color="gray", linestyle=":", alpha=0.5, lw=0.8)
    plt.title("UMN Crowd-Activity-All.avi: Complete Diagnostic Motion Energy Profile", fontsize=12, fontweight="bold")
    plt.xlabel("Frame Index (1 to 7,739)", fontsize=10)
    plt.ylabel("Mean Kinetic Energy (0.5 * v^2)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    diag_plot_path = os.path.join(PLOTS_DIR, "umn_energy_profile_diagnostic.png")
    plt.savefig(diag_plot_path)
    plt.close()
    print(f"[OUTPUT] Diagnostic motion plot saved to: {diag_plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostic_audit()
