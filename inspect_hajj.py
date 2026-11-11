import os
import glob
import pandas as pd

root = r"Research_Validation\HAJJv2\Original_Videos\HAJJv2.Dataset"

for split in ["Training", "Testing"]:
    print("\n" + "=" * 65)
    print(split.upper())
    print("=" * 65)

    files = glob.glob(os.path.join(root, split, "Labels", "*"))

    for path in sorted(files):
        print(f"\nFILE: {os.path.basename(path)}")

        try:
            df = pd.read_csv(path)

            print("Columns:", list(df.columns))
            print("Rows:", len(df))

            if "Classes" in df.columns:
                print("Classes:")
                print(df["Classes"].value_counts().to_dict())

            if "Video_No" in df.columns and "Frame_No" in df.columns:
                print("Video numbers:", sorted(df["Video_No"].dropna().unique()))
                print("Frame range:", df["Frame_No"].min(), "-", df["Frame_No"].max())

        except Exception as e:
            print("Could not read as CSV:", str(e))
