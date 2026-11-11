import os
import glob
import cv2
import pandas as pd

root = r"Research_Validation\HAJJv2\Original_Videos\HAJJv2.Dataset"

for split, suffix in [("Training", "Train"), ("Testing", "Test")]:
    print("\n" + "=" * 65)
    print(split.upper())
    print("=" * 65)

    video_dir = os.path.join(root, split, "Videos")
    label_dir = os.path.join(root, split, "Labels")

    for video_path in sorted(glob.glob(os.path.join(video_dir, "*.mp4"))):
        video_no = os.path.splitext(os.path.basename(video_path))[0]

        cap = cv2.VideoCapture(video_path)
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        pattern = os.path.join(
            label_dir, f"HajjDataset_V{video_no}_{suffix}.*"
        )
        matches = glob.glob(pattern)

        print(f"\nVideo {video_no}: {actual_frames} frames | {fps:.2f} FPS")

        if len(matches) != 1:
            print("Annotation files found:", len(matches))
            continue

        df = pd.read_csv(matches[0])

        annotated_max = int(df["Frame_No"].max())
        annotated_min = int(df["Frame_No"].min())

        print("Annotation file:", os.path.basename(matches[0]))
        print("Annotation range:", annotated_min, "-", annotated_max)
        print("Annotation rows:", len(df))
        print("Fits video:", annotated_max <= actual_frames)