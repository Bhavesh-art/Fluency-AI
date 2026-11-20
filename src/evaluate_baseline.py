import os
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = r".\data\sep28k"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
CHECKPOINT = os.path.join(DATA_DIR, "baseline_cnn.pt")

LABELS = [
    "Block",
    "Interjection",
    "NoStutteredWords",
    "Prolongation",
    "SoundRep",
    "WordRep",
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=64,
)


class SEP28KDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        wav_path = os.path.join(
            AUDIO_DIR,
            str(row["Show"]).strip(),
            str(row["EpId"]).strip() + ".wav",
        )

        waveform, sr = torchaudio.load(wav_path)

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

        feature = mel(clip).log1p()

        label = LABEL_TO_ID[row["majority_label"]]

        return feature, label


class SmallCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model = SmallCNN(num_classes=len(LABELS)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SEP28KDataset(
        os.path.join(DATA_DIR, "val_audio.csv")
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)

            logits = model(features)
            predictions = logits.argmax(dim=1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    print()
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print("Rows = actual, columns = predicted")
    print(pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
        ),
        index=LABELS,
        columns=LABELS,
    ).to_string())


if __name__ == "__main__":
    main()
