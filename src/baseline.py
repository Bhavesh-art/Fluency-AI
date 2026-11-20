import os
import pandas as pd
import torch
import torch.nn as nn
import torchaudio

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
    def __init__(self, csv_path, max_cached_episodes=3):
        self.df = pd.read_csv(csv_path)

        # Keep only a few recently used episodes in RAM.
        self.max_cached_episodes = max_cached_episodes
        self.episode_cache = {}

    def __len__(self):
        return len(self.df)

    def _load_episode(self, show, epid):
        key = (show, epid)

        if key in self.episode_cache:
            waveform, sr = self.episode_cache.pop(key)
            self.episode_cache[key] = (waveform, sr)
            return waveform, sr

        wav_path = os.path.join(
            AUDIO_DIR,
            show,
            epid + ".wav",
        )

        waveform, sr = torchaudio.load(wav_path)

        if sr != 16000:
            raise RuntimeError(
                f"Expected 16000 Hz, got {sr} Hz: {wav_path}"
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        self.episode_cache[key] = (waveform, sr)

        while len(self.episode_cache) > self.max_cached_episodes:
            oldest_key = next(iter(self.episode_cache))
            del self.episode_cache[oldest_key]

        return waveform, sr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        show = str(row["Show"]).strip()
        epid = str(row["EpId"]).strip()

        waveform, sr = self._load_episode(show, epid)

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


def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    train_ds = SEP28KDataset(
        os.path.join(DATA_DIR, "train_audio.csv")
    )

    val_ds = SEP28KDataset(
        os.path.join(DATA_DIR, "val_audio.csv")
    )

    print("Train clips:", len(train_ds))
    print("Validation clips:", len(val_ds))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = SmallCNN().to(device)

    class_weights = torch.tensor(
    [1.5622, 0.9859, 0.3607, 1.8790, 2.4624, 1.5750],
    dtype=torch.float32,
    device=device,
)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    EPOCHS = 5

    print(f"Starting {EPOCHS}-epoch weighted training...")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_seen += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"Batch {batch_idx + 1}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        train_loss = running_loss / total_seen
        train_acc = total_correct / total_seen

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} complete | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "sample_rate": 16000,
            "n_mels": 64,
        },
        CHECKPOINT,
    )

    print()
    print("Training complete.")
    print(f"Checkpoint: {CHECKPOINT}")

    running_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_seen += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"loss={loss.item():.4f}"
            )

    train_loss = running_loss / total_seen
    train_acc = total_correct / total_seen

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        device,
        criterion,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "sample_rate": 16000,
            "n_mels": 64,
        },
        CHECKPOINT,
    )

    print()
    print("Training complete.")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
