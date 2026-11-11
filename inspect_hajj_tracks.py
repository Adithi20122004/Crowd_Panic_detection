import os
import glob
import pandas as pd

root = r"Research_Validation\HAJJv2\Original_Videos\HAJJv2.Dataset"

for split in ["Training", "Testing"]:
    print("\n" + "=" * 65)
    print(split.upper())
    print("=" * 65)

    label_dir = os.path.join(root, split, "Labels")

    for path in sorted(glob.glob(os.path.join(label_dir, "*"))):
        df = pd.read_csv(path)

        df = df.sort_values(["Obj_Name", "Frame_No"])

        gaps = []
        track_lengths = []

        for obj, group in df.groupby("Obj_Name"):
            frames = sorted(group["Frame_No"].unique())
            track_lengths.append(len(frames))

            if len(frames) > 1:
                gaps.extend([
                    b - a for a, b in zip(frames[:-1], frames[1:])
                ])

        print("\n", os.path.basename(path))
        print("Tracked objects:", len(track_lengths))
        print("Mean frames per object:", round(sum(track_lengths) / len(track_lengths), 2))
        print("Median frames per object:", pd.Series(track_lengths).median())
        print("Consecutive frame transitions:",
              sum(g == 1 for g in gaps))
        print("Non-consecutive transitions:",
              sum(g > 1 for g in gaps))
