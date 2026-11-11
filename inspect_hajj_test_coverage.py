import os
import glob
import re
import cv2
import pandas as pd

BASE = r"C:\crowd_project\Research_Validation\HAJJv2\Original_Videos\HAJJv2.Dataset"

VIDEO_DIR = os.path.join(BASE, "Testing", "Videos")
LABEL_DIR = os.path.join(BASE, "Testing", "Labels")

DYNAMIC_CLASSES = {
    "running",
    "moving_in_opposite",
    "different_crowd_direction",
    "moving_non_human_object"
}

print("\nHAJJv2 TEST ANNOTATION COVERAGE")
print("=" * 110)

for label_path in sorted(glob.glob(os.path.join(LABEL_DIR, "*"))):

    filename = os.path.basename(label_path)

    match = re.search(r"V(\d+)_Test", filename, re.IGNORECASE)

    if not match:
        continue

    video_id = match.group(1)
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

    if not os.path.exists(video_path):
        print(f"Missing video: {video_id}")
        continue

    df = pd.read_csv(label_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video: {video_id}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    annotated_frames = set(df["Frame_No"].unique())

    dynamic_frames = set(
        df[df["Classes"].isin(DYNAMIC_CLASSES)]["Frame_No"].unique()
    )

    stationary_classes = {"sitting", "standing", "sleeping"}

    stationary_frames = set(
        df[df["Classes"].isin(stationary_classes)]["Frame_No"].unique()
    )

    empty_annotation_frames = total_frames - len(annotated_frames)

    print(f"\nVIDEO V{video_id}")
    print("-" * 55)
    print(f"Total video frames:              {total_frames}")
    print(f"Frames with annotations:         {len(annotated_frames)}")
    print(f"Frames without annotations:      {empty_annotation_frames}")
    print(f"Frames with dynamic labels:      {len(dynamic_frames)}")
    print(f"Frames with stationary labels:   {len(stationary_frames)}")
    print(f"Dynamic annotation coverage:     {len(dynamic_frames)/total_frames*100:.2f}%")

print("\n" + "=" * 110)
print("Coverage inspection completed.")