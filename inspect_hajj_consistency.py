import os
import glob
import pandas as pd

root = r"Research_Validation\HAJJv2\Original_Videos\HAJJv2.Dataset"

for split, suffix in [("Training", "Train"), ("Testing", "Test")]:
    print("\n" + "=" * 65)
    print(split.upper())
    print("=" * 65)

    label_dir = os.path.join(root, split, "Labels")

    for path in sorted(glob.glob(os.path.join(label_dir, "*"))):
        df = pd.read_csv(path)

        video = os.path.basename(path)

        frame_counts = df.groupby("Frame_No").size()
        class_counts = df["Classes"].value_counts()

        print(f"\n{video}")
        print("Unique annotated frames:", df["Frame_No"].nunique())
        print("Objects per frame - mean:", round(frame_counts.mean(), 2))
        print("Objects per frame - min:", frame_counts.min())
        print("Objects per frame - max:", frame_counts.max())
        print("Unique object IDs:", df["Obj_Name"].nunique())
        print("Duplicate object-frame annotations:",
              df.duplicated(["Frame_No", "Obj_Name"]).sum())
        print("Behaviour categories:", class_counts.to_dict())
