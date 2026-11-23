import os
import pandas as pd
import torch
import torch.nn as nn
import torchaudio

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from transformers import HubertModel


DATA_DIR = r".\data\sep28k"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
CHECKPOINT = os.path.join(DATA_DIR, "hubert_baseline.pt")
TEST_CSV = os.path.join(DATA_DIR, "test_audio.csv")

LABELS = [
    "Block",
    "Interjection",
    "NoStutteredWords",
    "Prolongation",
    "SoundRep",
    "WordRep",
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
CLIP_SAMPLES = 48000


class SEP28KDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, max_cached_episodes=3):
        self.df = pd.read_csv(csv_path)
        self.max_cached_episodes = max_cached_episodes
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def _load_episode(self, show, epid):
        key = (show, int(epid))

        if key in self.cache:
            return self.cache[key]

        wav_path = os.path.join(
            AUDIO_DIR,
            str(show),
            f"{int(epid)}.wav"
        )

        waveform, sr = torchaudio.load(wav_path)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sr, SAMPLE_RATE
            )

        waveform = waveform.mean(dim=0)

        if len(self.cache) >= self.max_cached_episodes:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = waveform
        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        show = str(row["Show"])
        epid = int(row["EpId"])

        waveform = self._load_episode(show, epid)

        start = int(row["Start"])
        stop = int(row["Stop"])

        clip = waveform[start:stop]

        if clip.numel() < CLIP_SAMPLES:
            clip = nn.functional.pad(
                clip,
                (0, CLIP_SAMPLES - clip.numel())
            )
        else:
            clip = clip[:CLIP_SAMPLES]

        label = LABEL_TO_ID[row["majority_label"]]

        return clip, torch.tensor(label, dtype=torch.long)


class HubertClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.encoder = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960"
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            outputs = self.encoder(x)

        features = outputs.last_hidden_state.mean(dim=1)

        return self.classifier(features)


def main():
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = SEP28KDataset(TEST_CSV)

    print(f"Test clips: {len(dataset)}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = HubertClassifier().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Checkpoint validation accuracy: {checkpoint['val_accuracy']:.4f}")

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader, start=1):
            x = x.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=(DEVICE.type == "cuda")
            ):
                logits = model(x)

            predictions = logits.argmax(dim=1).cpu()

            all_targets.extend(y.tolist())
            all_predictions.extend(predictions.tolist())

            if batch_idx % 50 == 0:
                print(
                    f"Evaluated batch {batch_idx}/{len(loader)}"
                )

    accuracy = sum(
        p == t for p, t in zip(
            all_predictions,
            all_targets
        )
    ) / len(all_targets)

    print("\n=== HuBERT Test Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            all_targets,
            all_predictions,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            all_targets,
            all_predictions,
            labels=list(range(len(LABELS))),
        )
    )


if __name__ == "__main__":
    main()

