import os
import pandas as pd
import torch
import torch.nn as nn
import torchaudio

from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel


DATA_DIR = r".\data\sep28k"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
CHECKPOINT = os.path.join(DATA_DIR, "hubert_weighted.pt")

TRAIN_CSV = os.path.join(DATA_DIR, "train_audio.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_audio.csv")

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


class SEP28KDataset(Dataset):
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


class_weights = torch.tensor([1.5622, 0.9859, 0.3607, 1.8790, 2.4624, 1.5750], dtype=torch.float32)


def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def main():
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = SEP28KDataset(TRAIN_CSV)
    val_ds = SEP28KDataset(VAL_CSV)

    print(f"Train clips: {len(train_ds)}")
    print(f"Validation clips: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = HubertClassifier().to(DEVICE)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    best_val_loss = float("inf")

    for epoch in range(1, 6):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=(DEVICE.type == "cuda")
            ):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

            if batch_idx % 100 == 0:
                print(
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader)

        print(
            f"Epoch {epoch}/5: "
            f"Train loss {train_loss:.4f}, "
            f"train acc {train_acc:.4f}, "
            f"val loss {val_loss:.4f}, "
            f"val acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": LABELS,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                CHECKPOINT,
            )

            print(f"Saved best checkpoint: {CHECKPOINT}")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()





