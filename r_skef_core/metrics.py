"""
Comprehensive evaluation metrics engine:
  - Frame-Level: Accuracy, Precision, Recall, F1, Specificity, AUROC
  - Event-Level: Event Hit Rate, Detection Delay (mean & median), Missed Events, Pre-Event Alarms
  - Operational: False Alarms per Video-Hour, False Alarm Duration (sec)
  - Segregation: Separates anomaly-bearing videos from normal-only videos
"""

import math
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

def evaluate_frame_metrics(y_true: List[int], y_pred: List[int], y_score: List[float]) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    has_both_classes = (len(set(y_true)) > 1)
    if has_both_classes:
        try:
            fpr_c, tpr_c, _ = roc_curve(y_true, y_score)
            roc_auc = round(float(auc(fpr_c, tpr_c)) * 100, 2)
        except Exception:
            roc_auc = "N/A"
    else:
        # AUROC is mathematically undefined for single-class (normal-only) ground truth
        roc_auc = "N/A"

    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "specificity": round(spec * 100, 2),
        "fpr": round(fpr * 100, 2),
        "fnr": round(fnr * 100, 2),
        "auc": roc_auc,
    }

def evaluate_event_metrics(
    records: List[Dict],
    gt_ranges: List[Tuple[int, int]],
    fps: float = 30.0,
    tolerance_sec: float = 1.0,
) -> Dict[str, Union[float, int, str]]:
    total_events = len(gt_ranges)
    if total_events == 0:
        return {
            "total_events": 0,
            "detected_events": 0,
            "missed_events": 0,
            "event_hit_rate": "N/A",
            "mean_delay_sec": "N/A",
            "median_delay_sec": "N/A",
            "pre_event_alarms": "N/A",
        }

    frame_to_pred = {r["frame"]: r["predicted_label"] for r in records}
    tolerance_frames = int(tolerance_sec * fps)

    detected = 0
    missed = 0
    pre_event_alarms = 0
    delays_sec = []

    for start, end in gt_ranges:
        # Check pre-event false alarm immediately preceding onset
        if start - 1 in frame_to_pred and frame_to_pred[start - 1] == 1:
            pre_event_alarms += 1

        # Search for first alarm within [start, end + tolerance_frames]
        first_alarm = None
        for f in range(start, end + tolerance_frames + 1):
            if f in frame_to_pred and frame_to_pred[f] == 1:
                first_alarm = f
                break

        if first_alarm is not None:
            detected += 1
            delay = (first_alarm - start) / fps
            delays_sec.append(delay)
        else:
            missed += 1

    event_hit_rate = (detected / total_events) * 100.0
    mean_delay = float(np.mean(delays_sec)) if delays_sec else 0.0
    median_delay = float(np.median(delays_sec)) if delays_sec else 0.0

    return {
        "total_events": total_events,
        "detected_events": detected,
        "missed_events": missed,
        "event_hit_rate": round(event_hit_rate, 2),
        "mean_delay_sec": round(mean_delay, 3),
        "median_delay_sec": round(median_delay, 3),
        "pre_event_alarms": pre_event_alarms,
    }

def evaluate_operational_metrics(
    records: List[Dict],
    gt_ranges: List[Tuple[int, int]],
    fps: float = 30.0,
) -> Dict[str, float]:
    total_frames = len(records)
    if total_frames == 0:
        return {"far_per_hour": 0.0, "total_fa_duration_sec": 0.0, "false_alarm_bursts": 0}

    # Identify normal frames
    def is_normal(frame_idx: int) -> bool:
        return not any(start <= frame_idx <= end for start, end in gt_ranges)

    normal_frames = [r for r in records if is_normal(r["frame"])]
    total_normal_frames = len(normal_frames)
    normal_hours = (total_normal_frames / fps) / 3600.0 if total_normal_frames > 0 else (total_frames / fps) / 3600.0

    # Count contiguous false alarm bursts during normal frames
    fa_bursts = 0
    fa_frame_count = 0
    in_burst = False

    for r in records:
        f = r["frame"]
        pred = r["predicted_label"]
        if is_normal(f):
            if pred == 1:
                fa_frame_count += 1
                if not in_burst:
                    fa_bursts += 1
                    in_burst = True
            else:
                in_burst = False
        else:
            in_burst = False

    far_per_hour = (fa_bursts / normal_hours) if normal_hours > 0 else 0.0
    fa_duration_sec = fa_frame_count / fps

    return {
        "false_alarm_bursts": int(fa_bursts),
        "total_fa_duration_sec": round(fa_duration_sec, 2),
        "far_per_hour": round(far_per_hour, 2),
    }
