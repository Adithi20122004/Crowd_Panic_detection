"""
r_skef_core/metrics.py -- Comprehensive evaluation metrics engine.

METHODOLOGICAL PRINCIPLES:
  1. Undefined Metric Denominators:
     - When denominators are zero (e.g. 0 positive frames, 0 negative frames, 0 events),
       metrics retain NaN and format as 'N/A' on console/output tables.
     - Never replace undefined values with zero.
  2. UMN Event Metrics:
     - Partitions events into three mutually exclusive categories:
       a) Events with an alarm already active immediately before onset (pre-existing alarm).
       b) Clean post-onset rising-edge detections (rising edge 0 -> 1 on/after onset with no pre-existing alarm).
       c) Events with no valid detection (neither pre-existing alarm nor post-onset detection).
     - An event with an alarm active immediately before onset is NEVER counted as a clean post-onset
       detection, even if the alarm clears and rises again.
     - Clean detection latency (mean, median, individual delays) is reported separately for clean hits.
  3. Frame-Level Metrics:
     - Preserves full confusion matrix (TP, FP, TN, FN) and derived frame-level statistics.
"""

import math
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    average_precision_score,
)


def format_metric(val: Union[float, int, str], precision: int = 1, is_percentage: bool = True) -> str:
    """Format numerical metrics for console display; undefined / NaN metrics display as 'N/A', never 'nan%'."""
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, (float, np.floating)):
        if math.isnan(val):
            return "N/A"
        return f"{val:.{precision}f}%" if is_percentage else f"{val:.{precision}f}"
    return str(val)


def evaluate_frame_metrics(y_true: List[int], y_pred: List[int], y_score: List[float]) -> Dict[str, Union[float, int, str]]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    # Denominators: keep NaN when denominator is zero, never replace with 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    if not math.isnan(prec) and not math.isnan(rec) and (prec + rec) > 0:
        f1 = float(2 * (prec * rec) / (prec + rec))
    else:
        f1 = float("nan")

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan")

    has_both_classes = (len(set(y_true)) > 1)
    if has_both_classes:
        try:
            fpr_c, tpr_c, _ = roc_curve(y_true, y_score)
            roc_auc = round(float(auc(fpr_c, tpr_c)) * 100, 2)
        except Exception:
            roc_auc = "N/A"
        try:
            ap_score = round(float(average_precision_score(y_true, y_score)) * 100, 2)
        except Exception:
            ap_score = "N/A"
    else:
        # AUROC and AUPRC are mathematically undefined for single-class datasets
        roc_auc = "N/A"
        ap_score = "N/A"

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2) if not math.isnan(prec) else "N/A",
        "recall": round(rec * 100, 2) if not math.isnan(rec) else "N/A",
        "f1": round(f1 * 100, 2) if not math.isnan(f1) else "N/A",
        "specificity": round(spec * 100, 2) if not math.isnan(spec) else "N/A",
        "fpr": round(fpr * 100, 2) if not math.isnan(fpr) else "N/A",
        "fnr": round(fnr * 100, 2) if not math.isnan(fnr) else "N/A",
        "precision_raw": prec,
        "recall_raw": rec,
        "specificity_raw": spec,
        "fpr_raw": fpr,
        "fnr_raw": fnr,
        "auc": roc_auc,
        "auprc": ap_score,
    }


def evaluate_event_metrics(
    records: List[Dict],
    gt_ranges: List[Tuple[int, int]],
    fps: float = 30.0,
    tolerance_sec: float = 1.0,
    max_latency_sec: Optional[float] = None,
) -> Dict[str, Union[float, int, str, List[float]]]:
    total_events = len(gt_ranges)
    if total_events == 0:
        return {
            "total_events": 0,
            "events_with_preexisting_alarm": 0,
            "clean_post_onset_detections": 0,
            "events_no_valid_detection": 0,
            "clean_event_hit_rate": "N/A",
            "clean_mean_delay_sec": "N/A",
            "clean_median_delay_sec": "N/A",
            "clean_delays_sec": [],
            # Backwards compatibility aliases
            "detected_events": 0,
            "missed_events": 0,
            "event_hit_rate": "N/A",
            "mean_delay_sec": "N/A",
            "median_delay_sec": "N/A",
            "all_delays_sec": [],
            "pre_event_alarms": 0,
        }

    frame_to_pred = {r["frame"]: r["predicted_label"] for r in records}
    tolerance_frames = int(tolerance_sec * fps)

    events_with_preexisting_alarm = 0
    clean_post_onset_detections = 0
    events_no_valid_detection = 0
    clean_delays_sec = []

    for start, end in gt_ranges:
        # Check pre-event false alarm immediately preceding onset
        was_active_at_onset = bool(start - 1 in frame_to_pred and frame_to_pred[start - 1] == 1)
        if was_active_at_onset:
            events_with_preexisting_alarm += 1
            # An event with an alarm already active immediately before onset must
            # not be counted as a clean post-onset detection, even if the alarm
            # later clears and rises again.
            continue

        # Clean search: alarm was inactive immediately prior to onset.
        # Find first rising edge (0 -> 1) at or after start
        max_search_frame = end + tolerance_frames
        if max_latency_sec is not None:
            max_search_frame = min(max_search_frame, start + int(max_latency_sec * fps))

        first_clean_rising_edge = None
        prev_val = frame_to_pred.get(start - 1, 0)

        for f in range(start, max_search_frame + 1):
            curr_val = frame_to_pred.get(f, 0)
            if curr_val == 1 and prev_val == 0:
                first_clean_rising_edge = f
                break
            prev_val = curr_val

        if first_clean_rising_edge is not None:
            clean_post_onset_detections += 1
            delay = (first_clean_rising_edge - start) / fps
            clean_delays_sec.append(round(delay, 3))
        else:
            events_no_valid_detection += 1

    clean_hit_rate = (clean_post_onset_detections / total_events) * 100.0 if total_events > 0 else float("nan")
    clean_mean_delay = round(float(np.mean(clean_delays_sec)), 3) if clean_delays_sec else "N/A"
    clean_median_delay = round(float(np.median(clean_delays_sec)), 3) if clean_delays_sec else "N/A"

    return {
        "total_events": total_events,
        "events_with_preexisting_alarm": events_with_preexisting_alarm,
        "clean_post_onset_detections": clean_post_onset_detections,
        "events_no_valid_detection": events_no_valid_detection,
        "clean_event_hit_rate": round(clean_hit_rate, 2) if not math.isnan(clean_hit_rate) else "N/A",
        "clean_mean_delay_sec": clean_mean_delay,
        "clean_median_delay_sec": clean_median_delay,
        "clean_delays_sec": clean_delays_sec,
        # Backwards compatibility aliases
        "detected_events": clean_post_onset_detections,
        "missed_events": events_no_valid_detection,
        "event_hit_rate": round(clean_hit_rate, 2) if not math.isnan(clean_hit_rate) else "N/A",
        "mean_delay_sec": clean_mean_delay,
        "median_delay_sec": clean_median_delay,
        "all_delays_sec": clean_delays_sec,
        "pre_event_alarms": events_with_preexisting_alarm,
    }


def evaluate_operational_metrics(
    records: List[Dict],
    gt_ranges: List[Tuple[int, int]],
    fps: float = 30.0,
) -> Dict[str, Union[float, int]]:
    total_frames = len(records)
    if total_frames == 0:
        return {
            "false_alarm_bursts": 0,
            "total_normal_frames": 0,
            "normal_exposure_sec": 0.0,
            "normal_exposure_hours": 0.0,
            "total_fa_duration_sec": 0.0,
            "far_per_hour": 0.0,
        }

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

    far_per_hour = (fa_bursts / normal_hours) if normal_hours > 0 else float("nan")
    fa_duration_sec = fa_frame_count / fps

    return {
        "false_alarm_bursts": int(fa_bursts),
        "total_normal_frames": int(total_normal_frames),
        "normal_exposure_sec": round(normal_hours * 3600.0, 2),
        "normal_exposure_hours": round(normal_hours, 5),
        "total_fa_duration_sec": round(fa_duration_sec, 2),
        "far_per_hour": round(far_per_hour, 2) if not math.isnan(far_per_hour) else "N/A",
    }
