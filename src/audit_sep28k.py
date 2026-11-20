import os
import pandas as pd
import soundfile as sf


DATA_DIR = r".\data\sep28k"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

FILES = {
    "train": os.path.join(DATA_DIR, "train.csv"),
    "val": os.path.join(DATA_DIR, "val.csv"),
    "test": os.path.join(DATA_DIR, "test.csv"),
    "train_audio": os.path.join(DATA_DIR, "train_audio.csv"),
    "val_audio": os.path.join(DATA_DIR, "val_audio.csv"),
    "test_audio": os.path.join(DATA_DIR, "test_audio.csv"),
}

LABELS = [
    "Block",
    "Interjection",
    "NoStutteredWords",
    "Prolongation",
    "SoundRep",
    "WordRep",
]


def load_csv(path):
    df = pd.read_csv(path)
    print(f"Loaded {path}: {len(df)} rows")
    return df


def check_basic_structure(df, name):
    print(f"\n--- {name} structure ---")
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    required = ["Show", "EpId", "Start", "Stop", "majority_label", "vote_count"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print("MISSING REQUIRED COLUMNS:", missing)
    else:
        print("Required columns: OK")

    if "majority_label" in df.columns:
        print("\nLabel counts:")
        print(df["majority_label"].value_counts().to_string())

    if "vote_count" in df.columns:
        print("\nVote count:")
        print(df["vote_count"].value_counts().sort_index().to_string())


def check_splits(train, val, test):
    print("\n==============================")
    print("SPLIT OVERLAP CHECK")
    print("==============================")

    def episode_keys(df):
        return set(
            zip(
                df["Show"].astype(str).str.strip(),
                df["EpId"].astype(str).str.strip(),
            )
        )

    train_ep = episode_keys(train)
    val_ep = episode_keys(val)
    test_ep = episode_keys(test)

    print("Train episodes:", len(train_ep))
    print("Validation episodes:", len(val_ep))
    print("Test episodes:", len(test_ep))

    print("\nEpisode overlap:")
    print("Train ∩ Val :", len(train_ep & val_ep))
    print("Train ∩ Test:", len(train_ep & test_ep))
    print("Val   ∩ Test:", len(val_ep & test_ep))

    if not (train_ep & val_ep or train_ep & test_ep or val_ep & test_ep):
        print("Episode-level split: PASS")
    else:
        print("WARNING: episode overlap detected")


def check_class_distribution(train, val, test):
    print("\n==============================")
    print("CLASS DISTRIBUTION")
    print("==============================")

    for name, df in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        counts = df["majority_label"].value_counts()

        print(f"\n{name}: {len(df)} clips")

        for label in LABELS:
            count = int(counts.get(label, 0))
            pct = 100 * count / len(df)
            print(f"{label:20s} {count:6d}  ({pct:6.2f}%)")


def check_invalid_ranges(df, name):
    print(f"\n--- {name}: clip range checks ---")

    bad_start = (df["Start"] < 0).sum()
    bad_stop = (df["Stop"] <= 0).sum()
    bad_order = (df["Stop"] <= df["Start"]).sum()

    durations = df["Stop"] - df["Start"]

    print("Negative Start:", int(bad_start))
    print("Non-positive Stop:", int(bad_stop))
    print("Stop <= Start:", int(bad_order))

    print("\nClip length in samples:")
    print("Min:", int(durations.min()))
    print("Max:", int(durations.max()))
    print("Mean:", round(float(durations.mean()), 2))
    print("Median:", round(float(durations.median()), 2))

    expected = 48000

    exact = (durations == expected).sum()
    print(f"Exactly 48000 samples (3 sec @ 16 kHz): {exact}/{len(df)}")

    unusual = ((durations < 40000) | (durations > 56000)).sum()
    print(f"Outside 2.5-3.5 sec range: {unusual}")


def check_missing_audio(audio_df):
    print("\n==============================")
    print("AUDIO COVERAGE")
    print("==============================")

    missing = []
    found = []

    for _, row in audio_df.iterrows():
        show = str(row["Show"]).strip()
        epid = str(row["EpId"]).strip()

        wav = os.path.join(AUDIO_DIR, show, f"{epid}.wav")

        if os.path.exists(wav):
            found.append((show, epid))
        else:
            missing.append((show, epid))

    total = len(audio_df)

    print("Manifest clips:", total)
    print("Audio available:", len(found))
    print("Audio missing:", len(missing))
    print("Coverage:", round(100 * len(found) / total, 2), "%")

    print("\nMissing audio by source episode:")

    missing_counts = {}

    for show, epid in missing:
        key = show
        missing_counts[key] = missing_counts.get(key, 0) + 1

    for show, count in sorted(
        missing_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{show:25s} {count:6d}")

    return missing


def check_missing_audio_by_class(audio_df):
    print("\n==============================")
    print("MISSING AUDIO BY CLASS")
    print("==============================")

    audio_df = audio_df.copy()

    def exists(row):
        show = str(row["Show"]).strip()
        epid = str(row["EpId"]).strip()
        wav = os.path.join(AUDIO_DIR, show, f"{epid}.wav")
        return os.path.exists(wav)

    audio_df["audio_exists"] = audio_df.apply(exists, axis=1)

    table = pd.crosstab(
        audio_df["majority_label"],
        audio_df["audio_exists"],
    )

    print(table.to_string())

    print("\nCoverage by class:")

    for label in LABELS:
        subset = audio_df[audio_df["majority_label"] == label]

        total = len(subset)
        available = int(subset["audio_exists"].sum())

        pct = 100 * available / total if total else 0

        print(
            f"{label:20s} "
            f"{available:5d}/{total:<5d} "
            f"({pct:6.2f}%)"
        )


def check_duplicate_clips(df, name):
    print(f"\n--- {name}: duplicate check ---")

    key_cols = [
        "Show",
        "EpId",
        "Start",
        "Stop",
    ]

    dup_count = df.duplicated(subset=key_cols).sum()

    print("Duplicate clip records:", int(dup_count))

    if dup_count == 0:
        print("Duplicate check: PASS")
    else:
        print("WARNING: duplicate clip records detected")


def inspect_audio_files(audio_df, max_files=200):
    print("\n==============================")
    print("AUDIO FILE SANITY CHECK")
    print("==============================")

    checked = 0
    bad = []

    for _, row in audio_df.iterrows():

        if checked >= max_files:
            break

        show = str(row["Show"]).strip()
        epid = str(row["EpId"]).strip()

        wav = os.path.join(AUDIO_DIR, show, f"{epid}.wav")

        if not os.path.exists(wav):
            continue

        try:
            info = sf.info(wav)

            if info.samplerate != 16000:
                bad.append(
                    (show, epid, "sample_rate", info.samplerate)
                )

            if info.channels != 1:
                bad.append(
                    (show, epid, "channels", info.channels)
                )

            if info.frames <= 0:
                bad.append(
                    (show, epid, "zero_frames", info.frames)
                )

        except Exception as e:
            bad.append(
                (show, epid, "read_error", str(e))
            )

        checked += 1

    print("Audio files checked:", checked)

    if bad:
        print("\nProblems:")
        for item in bad[:20]:
            print(item)

        if len(bad) > 20:
            print(f"... and {len(bad) - 20} more")
    else:
        print("Sampled audio files: OK")


def main():
    print("==========================================")
    print("SEP-28k DATABASE AUDIT")
    print("==========================================")

    train = load_csv(FILES["train"])
    val = load_csv(FILES["val"])
    test = load_csv(FILES["test"])

    train_audio = load_csv(FILES["train_audio"])
    val_audio = load_csv(FILES["val_audio"])
    test_audio = load_csv(FILES["test_audio"])

    check_basic_structure(train, "TRAIN")
    check_basic_structure(val, "VALIDATION")
    check_basic_structure(test, "TEST")

    check_splits(train, val, test)

    check_class_distribution(train, val, test)

    check_invalid_ranges(train, "TRAIN")
    check_invalid_ranges(val, "VALIDATION")
    check_invalid_ranges(test, "TEST")

    check_duplicate_clips(train, "TRAIN")
    check_duplicate_clips(val, "VALIDATION")
    check_duplicate_clips(test, "TEST")

    all_audio = pd.concat(
        [train_audio, val_audio, test_audio],
        ignore_index=True,
    )

    check_missing_audio(all_audio)

    check_missing_audio_by_class(all_audio)

    inspect_audio_files(all_audio)

    print("\n==========================================")
    print("AUDIT COMPLETE")
    print("==========================================")


if __name__ == "__main__":
    main()