import os
import pandas as pd
import soundfile as sf


DATA_DIR = r".\data\sep28k"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

FILES = [
    os.path.join(DATA_DIR, "train_audio.csv"),
    os.path.join(DATA_DIR, "val_audio.csv"),
    os.path.join(DATA_DIR, "test_audio.csv"),
]


def main():
    print("==========================================")
    print("SEP-28k AUDIO / LABEL ALIGNMENT AUDIT")
    print("==========================================")

    total = 0
    checked = 0
    missing = 0
    invalid = 0
    errors = 0

    invalid_examples = []

    for csv_path in FILES:
        df = pd.read_csv(csv_path)

        print(f"\nChecking: {csv_path}")
        print(f"Clips: {len(df)}")

        for _, row in df.iterrows():
            total += 1

            show = str(row["Show"]).strip()
            epid = str(row["EpId"]).strip()

            start = int(row["Start"])
            stop = int(row["Stop"])

            wav_path = os.path.join(
                AUDIO_DIR,
                show,
                f"{epid}.wav"
            )

            if not os.path.exists(wav_path):
                missing += 1

                if len(invalid_examples) < 20:
                    invalid_examples.append(
                        (
                            show,
                            epid,
                            start,
                            stop,
                            "MISSING_AUDIO"
                        )
                    )

                continue

            try:
                info = sf.info(wav_path)

                actual_frames = info.frames

                checked += 1

                problems = []

                if start < 0:
                    problems.append("negative_start")

                if stop <= start:
                    problems.append("stop_before_start")

                if stop > actual_frames:
                    problems.append(
                        f"stop_beyond_audio "
                        f"(stop={stop}, frames={actual_frames})"
                    )

                if start >= actual_frames:
                    problems.append(
                        f"start_beyond_audio "
                        f"(start={start}, frames={actual_frames})"
                    )

                if problems:
                    invalid += 1

                    if len(invalid_examples) < 20:
                        invalid_examples.append(
                            (
                                show,
                                epid,
                                start,
                                stop,
                                "; ".join(problems)
                            )
                        )

            except Exception as e:
                errors += 1

                if len(invalid_examples) < 20:
                    invalid_examples.append(
                        (
                            show,
                            epid,
                            start,
                            stop,
                            f"READ_ERROR: {e}"
                        )
                    )

    print("\n==========================================")
    print("RESULTS")
    print("==========================================")

    print(f"Total manifest clips:       {total}")
    print(f"Audio files checked:        {checked}")
    print(f"Missing audio:              {missing}")
    print(f"Invalid clip ranges:        {invalid}")
    print(f"Audio read errors:          {errors}")

    valid = checked - invalid - errors

    if checked > 0:
        print(
            f"Valid aligned clips:        "
            f"{valid}/{checked} "
            f"({100 * valid / checked:.2f}%)"
        )

    print("\n==========================================")
    print("VERDICT")
    print("==========================================")

    if missing == 0 and invalid == 0 and errors == 0:
        print("PASS: Every manifest clip fits inside its WAV file.")
        print("Audio/label coordinate alignment looks correct.")
    else:
        print("WARNING: Alignment problems were found.")

    if invalid_examples:
        print("\nExamples of problems:")

        for item in invalid_examples:
            print(
                f"Show={item[0]}, "
                f"EpId={item[1]}, "
                f"Start={item[2]}, "
                f"Stop={item[3]}, "
                f"Problem={item[4]}"
            )


if __name__ == "__main__":
    main()