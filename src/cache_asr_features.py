from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import torchaudio

from src.asr import ASR
from src.asr_features import extract_asr_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sep28k"


FEATURE_COLUMNS = [
    "word_count",
    "filler_count",
    "adjacent_repeat_count",
    "char_repeat_count",
    "repeated_word_rate",
    "filler_rate",
    "avg_word_duration",
    "short_word_rate",
]


def load_clip(row):
    wav_path = (
        DATA_DIR
        / "audio"
        / str(row["Show"]).strip()
        / f"{int(row['EpId'])}.wav"
    )

    start = int(row["Start"])
    stop = int(row["Stop"])

    audio, sr = torchaudio.load(
        str(wav_path),
        frame_offset=start,
        num_frames=stop - start,
    )

    audio = audio.mean(dim=0)

    return audio, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val"],
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    manifest_path = DATA_DIR / f"{args.split}_audio.csv"
    output_path = DATA_DIR / f"asr_features_{args.split}.csv"

    df = pd.read_csv(manifest_path)

    if args.max_clips is not None:
        df = df.iloc[:args.max_clips].copy()

    if output_path.exists():
        cached = pd.read_csv(output_path)

        completed = set(
            zip(
                cached["Show"].astype(str),
                cached["EpId"].astype(int),
                cached["Start"].astype(int),
                cached["Stop"].astype(int),
            )
        )
    else:
        completed = set()

    print(f"Split: {args.split}")
    print(f"Manifest rows: {len(df)}")
    print(f"Already cached: {len(completed)}")
    print(f"Remaining: {len(df) - len(completed)}")

    asr = ASR()

    new_rows = []
    start_time = time.time()

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        key = (
            str(row["Show"]).strip(),
            int(row["EpId"]),
            int(row["Start"]),
            int(row["Stop"]),
        )

        if key in completed:
            continue

        try:
            audio, sr = load_clip(row)

            tmp_path = DATA_DIR / "_tmp_asr_clip.wav"

            torchaudio.save(
                str(tmp_path),
                audio.unsqueeze(0),
                sr,
            )

            result = asr.transcribe(tmp_path)

            features, details = extract_asr_features(result)

            record = {
                "Show": str(row["Show"]).strip(),
                "EpId": int(row["EpId"]),
                "Start": int(row["Start"]),
                "Stop": int(row["Stop"]),
                "majority_label": str(row["majority_label"]),
                "transcript": details["text"],
            }

            for name, value in zip(FEATURE_COLUMNS, features):
                record[name] = float(value)

            new_rows.append(record)

        except Exception as exc:
            print(
                f"[ERROR] {i}/{len(df)} "
                f"{row['Show']} EpId={int(row['EpId'])} "
                f"Start={int(row['Start'])}: {exc}",
                flush=True,
            )
            continue

        if len(new_rows) >= 25:
            new_df = pd.DataFrame(new_rows)

            if output_path.exists():
                existing = pd.read_csv(output_path)
                combined = pd.concat(
                    [existing, new_df],
                    ignore_index=True,
                )
            else:
                combined = new_df

            combined.to_csv(output_path, index=False)

            completed.update(
                zip(
                    new_df["Show"].astype(str),
                    new_df["EpId"].astype(int),
                    new_df["Start"].astype(int),
                    new_df["Stop"].astype(int),
                )
            )

            new_rows = []

        processed = i
        elapsed = time.time() - start_time
        rate = processed / max(elapsed, 1.0)
        remaining = len(df) - processed
        eta_min = remaining / max(rate, 1e-6) / 60

        if processed % 10 == 0:
            print(
                f"[{processed}/{len(df)}] "
                f"rate={rate:.2f} clips/s "
                f"ETA={eta_min:.1f} min",
                flush=True,
            )

    if new_rows:
        new_df = pd.DataFrame(new_rows)

        if output_path.exists():
            existing = pd.read_csv(output_path)
            combined = pd.concat(
                [existing, new_df],
                ignore_index=True,
            )
        else:
            combined = new_df

        combined.to_csv(output_path, index=False)

    print("\nFinished.")
    print(f"Output: {output_path}")

    if output_path.exists():
        final = pd.read_csv(output_path)
        print(f"Cached clips: {len(final)}")


if __name__ == "__main__":
    main()
