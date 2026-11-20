from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEP28K_DIR = PROJECT_ROOT / "sep28k"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sep28k"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LABEL_COLUMNS = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection",
    "NoStutteredWords",
]


def load_data():
    labels_path = SEP28K_DIR / "SEP-28k_labels.csv"
    episodes_path = SEP28K_DIR / "SEP-28k_episodes.csv"

    labels = pd.read_csv(labels_path)

    episodes = pd.read_csv(
        episodes_path,
        header=None,
        names=[
            "ShowName",
            "EpisodeName",
            "AudioURL",
            "Dataset",
            "EpId",
        ],
        skipinitialspace=True,
    )

    return labels, episodes


def create_manifest(labels, episodes):
    merged = labels.merge(
        episodes[["Dataset", "EpId", "AudioURL"]],
        left_on=["Show", "EpId"],
        right_on=["Dataset", "EpId"],
        how="left",
        validate="many_to_one",
    )

    if merged["AudioURL"].isna().any():
        raise RuntimeError("Some clips could not be mapped to an audio URL.")

    merged["majority_label"] = merged[LABEL_COLUMNS].idxmax(axis=1)
    merged["vote_count"] = merged[LABEL_COLUMNS].max(axis=1)

    # Keep only clips with at least 2 of 3 annotators agreeing.
    manifest = merged[merged["vote_count"] >= 2].copy()

    # Unique source recording used as the grouping unit.
    manifest["group"] = (
        manifest["Show"].astype(str)
        + "_"
        + manifest["EpId"].astype(str)
    )

    return manifest


def split_by_episode(manifest):
    groups = manifest["group"]

    train_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, temp_indices = next(
        train_splitter.split(manifest, groups=groups)
    )

    train = manifest.iloc[train_indices].copy()
    temp = manifest.iloc[temp_indices].copy()

    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    val_indices, test_indices = next(
        val_splitter.split(temp, groups=temp["group"])
    )

    val = temp.iloc[val_indices].copy()
    test = temp.iloc[test_indices].copy()

    return train, val, test


def main():
    print("Loading SEP-28k metadata...")
    labels, episodes = load_data()

    print(f"Total labeled clips: {len(labels)}")

    manifest = create_manifest(labels, episodes)

    print(f"Usable clips (>=2/3 agreement): {len(manifest)}")
    print("\nClass distribution:")
    print(manifest["majority_label"].value_counts())

    train, val, test = split_by_episode(manifest)

    print("\nSplit sizes:")
    print(f"Train: {len(train)} clips")
    print(f"Validation: {len(val)} clips")
    print(f"Test: {len(test)} clips")

    print("\nUnique source episodes:")
    print(f"Train: {train['group'].nunique()}")
    print(f"Validation: {val['group'].nunique()}")
    print(f"Test: {test['group'].nunique()}")

    # Verify that no source episode appears in multiple splits.
    train_groups = set(train["group"])
    val_groups = set(val["group"])
    test_groups = set(test["group"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    columns = [
        "Show",
        "EpId",
        "ClipId",
        "Start",
        "Stop",
        "AudioURL",
        "majority_label",
        "vote_count",
    ]

    train[columns].to_csv(OUTPUT_DIR / "train.csv", index=False)
    val[columns].to_csv(OUTPUT_DIR / "val.csv", index=False)
    test[columns].to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nManifest files created:")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "val.csv")
    print(OUTPUT_DIR / "test.csv")


if __name__ == "__main__":
    main()