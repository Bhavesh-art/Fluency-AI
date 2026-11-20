import pandas as pd
import os
from scipy.io import wavfile

df = pd.read_csv(r".\data\sep28k\train_audio.csv")
rows = df.sample(20, random_state=42)

base = r".\data\sep28k\audio"

rates = []
channels = []
durations = []

for _, row in rows.iterrows():
    wav = os.path.join(
        base,
        str(row["Show"]).strip(),
        str(row["EpId"]).strip() + ".wav"
    )

    sr, audio = wavfile.read(wav)

    rates.append(sr)

    if len(audio.shape) == 1:
        channels.append(1)
    else:
        channels.append(audio.shape[1])

    durations.append(len(audio) / sr)

print("Sampled clips:", len(rows))
print("Sample rates:", sorted(set(rates)))
print("Channels:", sorted(set(channels)))
print(f"Duration range: {min(durations):.3f} - {max(durations):.3f} sec")
