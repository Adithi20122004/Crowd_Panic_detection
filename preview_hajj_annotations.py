import os
import cv2
import pandas as pd

BASE = (
    r"C:\crowd_project\Research_Validation"
    r"\HAJJv2\Original_Videos\HAJJv2.Dataset"
)

OUTPUT = (
    r"C:\crowd_project\Research_Validation"
    r"\Results\HAJJv2_Visual_Inspection"
)

os.makedirs(OUTPUT, exist_ok=True)

VIDEOS_TO_INSPECT = ["12", "10"]

COLORS = {
    "running": (0, 0, 255),
    "moving_in_opposite": (0, 165, 255),
    "different_crowd_direction": (255, 0, 255),
    "moving_non_human_object": (255, 255, 0),
    "sitting": (0, 255, 0),
    "standing": (255, 255, 0),
    "sleeping": (255, 0, 0)
}

for video_id in VIDEOS_TO_INSPECT:

    video_path = os.path.join(
        BASE, "Testing", "Videos", f"{video_id}.mp4"
    )

    label_path = os.path.join(
        BASE, "Testing", "Labels",
        f"HajjDataset_V{video_id}_Test.csv"
    )

    # Some files have .csv.xls extensions.
    if not os.path.exists(label_path):
        label_path = label_path + ".xls"

    df = pd.read_csv(label_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open:", video_path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(
        OUTPUT, f"V{video_id}_annotated.mp4"
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    annotations = {
        frame: group
        for frame, group in df.groupby("Frame_No")
    }

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        if frame_number in annotations:

            rows = annotations[frame_number]

            for _, row in rows.iterrows():

                x = int(row["X"])
                y = int(row["Y"])
                w = int(row["Width"])
                h = int(row["Height"])

                label = str(row["Classes"])

                color = COLORS.get(label, (255, 255, 255))

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

        cv2.putText(
            frame,
            f"Video V{video_id} | Frame {frame_number}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(frame)

    cap.release()
    writer.release()

    print("Created:", output_path)

print("\nVisual inspection videos generated.")