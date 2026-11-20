import pandas as pd
import os

base = r".\data\sep28k\audio"

for name in ["train", "val", "test"]:
    src = rf".\data\sep28k\{name}.csv"
    dst = rf".\data\sep28k\{name}_audio.csv"

    df = pd.read_csv(src)

    available = []

    for _, row in df.iterrows():
        wav = os.path.join(
            base,
            str(row["Show"]).strip(),
            str(row["EpId"]).strip() + ".wav"
        )
        available.append(os.path.exists(wav))

    df = df[available].copy()
    df.to_csv(dst, index=False)

    print(f"{name}: {len(df)} clips -> {dst}")
    print("Class distribution:")
    print(df["majority_label"].value_counts().to_string())
    print()
