import pandas as pd
import os

base = r".\data\sep28k\audio"

for name in ["train", "val", "test"]:
    df = pd.read_csv(rf".\data\sep28k\{name}.csv")

    missing = 0

    for _, row in df.iterrows():
        wav = os.path.join(
            base,
            str(row["Show"]).strip(),
            str(row["EpId"]).strip() + ".wav"
        )

        if not os.path.exists(wav):
            missing += 1

    print(f"{name}: clips={len(df)}, missing_audio={missing}")
