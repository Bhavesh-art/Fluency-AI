import os
import time
import pandas as pd
import torch
import torchaudio
import torch.nn as nn

DATA_DIR = r".\data\sep28k"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

df = pd.read_csv(os.path.join(DATA_DIR, "train_audio.csv"))

print("Training clips:", len(df))
print(
    "Unique episodes:",
    df[["Show", "EpId"]].drop_duplicates().shape[0],
)


class CachedLoader:
    def __init__(self, max_cached_episodes=3):
        self.max_cached_episodes = max_cached_episodes
        self.cache = {}

    def load_episode(self, show, epid):
        key = (show, epid)

        if key in self.cache:
            waveform, sr = self.cache.pop(key)
            self.cache[key] = (waveform, sr)
            return waveform, sr

        wav_path = os.path.join(
            AUDIO_DIR,
            show,
            epid + ".wav",
        )

        waveform, sr = torchaudio.load(wav_path)

        self.cache[key] = (waveform, sr)

        while len(self.cache) > self.max_cached_episodes:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        return waveform, sr


def extract_clip(loader, i):
    row = df.iloc[i]

    show = str(row["Show"]).strip()
    epid = str(row["EpId"]).strip()

    waveform, sr = loader.load_episode(show, epid)

    start = int(row["Start"])
    stop = int(row["Stop"])

    clip = waveform[:, start:stop]

    if clip.shape[1] < 48000:
        clip = nn.functional.pad(
            clip,
            (0, 48000 - clip.shape[1])
        )
    elif clip.shape[1] > 48000:
        clip = clip[:, :48000]

    return clip


# Test 1: current approach — reload episode every time
start = time.time()

for i in range(100):
    row = df.iloc[i]

    show = str(row["Show"]).strip()
    epid = str(row["EpId"]).strip()

    wav_path = os.path.join(
        AUDIO_DIR,
        show,
        epid + ".wav",
    )

    waveform, sr = torchaudio.load(wav_path)

    start_sample = int(row["Start"])
    stop_sample = int(row["Stop"])

    clip = waveform[:, start_sample:stop_sample]

current_time = time.time() - start

print()
print(f"Current approach, 100 clips: {current_time:.2f} sec")


# Test 2: cached approach
loader = CachedLoader(max_cached_episodes=3)

start = time.time()

for i in range(100):
    clip = extract_clip(loader, i)

cached_time = time.time() - start

print(f"Cached approach, 100 clips: {cached_time:.2f} sec")

print()
print(f"Speedup: {current_time / cached_time:.2f}x")