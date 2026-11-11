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

        transitions = []

        for obj, group in df.groupby("Obj_Name"):
            group = group.sort_values("Frame_No")
            classes = group["Classes"].drop_duplicates().tolist()

            if len(classes) > 1:
                transitions.append((obj, classes))

        print("\n", os.path.basename(path))
        print("Objects changing behaviour:", len(transitions))

        for obj, classes in transitions[:10]:
            print(" ", obj, ":", " -> ".join(classes))
