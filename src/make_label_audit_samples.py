import os
import random
import pandas as pd
import soundfile as sf

BASE = r".\data\sep28k"
OUT = os.path.join(BASE, "audit_samples")
os.makedirs(OUT, exist_ok=True)

LABELS = [
    "Block",
    "Interjection",
    "NoStutteredWords",
    "Prolongation",
    "SoundRep",
    "WordRep",
]

random.seed(42)

manifests = {
    "train": os.path.join(BASE, "train_audio.csv"),
    "val": os.path.join(BASE, "val_audio.csv"),
    "test": os.path.join(BASE, "test_audio.csv"),
}

frames = []
for split, path in manifests.items():
    df = pd.read_csv(path)
    df["split"] = split
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

# Select one clip per episode first, so samples are spread across episodes.
selected = []

for label in LABELS:
    candidates = df[df["majority_label"] == label].copy()

    episode_cols = ["Show", "EpId"]
    episodes = candidates[episode_cols].drop_duplicates()

    if len(episodes) < 5:
        raise RuntimeError(f"Not enough unique episodes for {label}")

    chosen_episodes = episodes.sample(n=5, random_state=42).to_dict("records")

    for ep in chosen_episodes:
        ep_clips = candidates[
            (candidates["Show"] == ep["Show"]) &
            (candidates["EpId"] == ep["EpId"])
        ]

        row = ep_clips.sample(n=1, random_state=42).iloc[0]
        selected.append(row)

rows = []

for i, row in enumerate(selected, start=1):
    sample_id = f"sample_{i:02d}"
    out_path = os.path.join(OUT, f"{sample_id}.wav")

    audio_path = os.path.join(
        BASE,
        "audio",
        str(row["Show"]).strip(),
        f"{int(row['EpId'])}.wav",
    )

    start = int(row["Start"])
    stop = int(row["Stop"])

    audio, sr = sf.read(audio_path)

    clip = audio[start:stop]

    # Ensure mono.
    if clip.ndim > 1:
        clip = clip.mean(axis=1)

    # Pad/truncate to exactly 3 seconds.
    target_len = 48000

    if len(clip) < target_len:
        padded = [0.0] * target_len
        padded[:len(clip)] = clip.tolist()
        clip = padded
    else:
        clip = clip[:target_len]

    sf.write(out_path, clip, 16000)

    rows.append({
        "sample_id": sample_id,
        "split": row["split"],
        "show": row["Show"],
        "epid": int(row["EpId"]),
        "clipid": int(row["ClipId"]),
        "label": row["majority_label"],
        "vote_count": int(row["vote_count"]),
        "start": start,
        "stop": stop,
        "wav_path": out_path,
    })

index = pd.DataFrame(rows)

# Sort by class for easier inspection.
index["label_order"] = index["label"].map(
    {label: i for i, label in enumerate(LABELS)}
)
index = index.sort_values(["label_order", "sample_id"]).drop(columns=["label_order"])

index.to_csv(os.path.join(OUT, "index.csv"), index=False)

print()
print("Created 30 audit samples.")
print(f"Output folder: {OUT}")
print()
print(index[["sample_id", "split", "show", "epid", "label", "vote_count"]].to_string(index=False))
print()
print("Index:", os.path.join(OUT, "index.csv"))
