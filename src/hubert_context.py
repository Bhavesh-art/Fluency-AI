import os
import pandas as pd
import torch
import torch.nn as nn
import torchaudio

from torch.utils.data import DataLoader, Dataset
from transformers import HubertModel


DATA_DIR = r".\data\sep28k"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

TRAIN_CSV = os.path.join(DATA_DIR, "train_audio.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_audio.csv")
CHECKPOINT = os.path.join(DATA_DIR, "hubert_context.pt")

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

# SEP-28k labeled event = exactly 3 seconds.
TARGET_SAMPLES = 3 * SAMPLE_RATE

# Additional surrounding context.
CONTEXT_SAMPLES = 5 * SAMPLE_RATE

# Total input:
# 5 sec before + 3 sec event + 5 sec after = 13 sec.
TOTAL_SAMPLES = TARGET_SAMPLES + (2 * CONTEXT_SAMPLES)
TOTAL_SECONDS = TOTAL_SAMPLES / SAMPLE_RATE

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


class SEP28KContextDataset(Dataset):
    def __init__(self, csv_path, max_cached_episodes=2):
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
            show,
            f"{int(epid)}.wav",
        )

        if not os.path.exists(wav_path):
            raise FileNotFoundError(
                f"Audio file not found: {wav_path}"
            )

        waveform, sr = torchaudio.load(wav_path)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                SAMPLE_RATE,
            )

        # Convert to mono.
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        if len(self.cache) >= self.max_cached_episodes:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = waveform

        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        show = str(row["Show"]).strip()
        epid = int(row["EpId"])

        start = int(row["Start"])
        stop = int(row["Stop"])

        waveform = self._load_episode(show, epid)

        # The labeled region itself.
        target_length = stop - start

        # SEP-28k should contain a 3-second region.
        # If a row differs slightly, preserve its exact region
        # and then normalize the final target section to 3 seconds.
        target = waveform[start:stop]

        if target.numel() > TARGET_SAMPLES:
            target = target[:TARGET_SAMPLES]

        elif target.numel() < TARGET_SAMPLES:
            target = nn.functional.pad(
                target,
                (0, TARGET_SAMPLES - target.numel()),
            )

        # Prefer exactly 5 seconds before and after the labeled event.
        desired_start = start - CONTEXT_SAMPLES
        desired_stop = stop + CONTEXT_SAMPLES

        # Normally this gives:
        # [start - 5 sec, start + 3 sec + 5 sec]
        #
        # Near an episode boundary, shift the complete window
        # so that we still obtain as much real context as possible.
        if desired_start < 0:
            desired_start = 0

        if desired_stop > waveform.numel():
            desired_stop = waveform.numel()

        clip = waveform[desired_start:desired_stop]

        # If boundary clipping made the complete context shorter,
        # pad rather than dropping the sample.
        if clip.numel() < TOTAL_SAMPLES:
            clip = nn.functional.pad(
                clip,
                (0, TOTAL_SAMPLES - clip.numel()),
            )

        elif clip.numel() > TOTAL_SAMPLES:
            clip = clip[:TOTAL_SAMPLES]

        label = LABEL_TO_ID[row["majority_label"]]

        return clip, torch.tensor(label, dtype=torch.long)


class HubertContextClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.encoder = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960"
        )

        # Frozen pretrained HuBERT.
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
        # HuBERT remains frozen.
        with torch.no_grad():
            outputs = self.encoder(x)

        # Mean-pool the complete 13-second representation.
        features = outputs.last_hidden_state.mean(dim=1)

        return self.classifier(features)


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == y).sum().item()
            total_samples += x.size(0)

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def main():
    print("=" * 60)
    print("HuBERT 13-SECOND CONTEXT EXPERIMENT")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print()
    print(f"Target event:     3 seconds")
    print(f"Context before:   5 seconds")
    print(f"Context after:    5 seconds")
    print(f"Total input:      {TOTAL_SECONDS:.0f} seconds")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Epochs:           {EPOCHS}")

    train_dataset = SEP28KContextDataset(TRAIN_CSV)
    val_dataset = SEP28KContextDataset(VAL_CSV)

    print()
    print(f"Train clips:      {len(train_dataset)}")
    print(f"Validation clips: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print()
    print("Loading facebook/hubert-base-ls960...")

    model = HubertContextClassifier().to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for batch_idx, (x, y) in enumerate(
            train_loader,
            start=1,
        ):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)

            predictions = logits.argmax(dim=1)
            running_correct += (
                (predictions == y).sum().item()
            )

            running_samples += x.size(0)

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
        )

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}: "
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
                    "val_loss": val_loss,
                    "epoch": epoch + 1,
                    "model_name": "facebook/hubert-base-ls960",
                    "sample_rate": SAMPLE_RATE,
                    "target_seconds": 3,
                    "context_before_seconds": 5,
                    "context_after_seconds": 5,
                    "total_seconds": TOTAL_SECONDS,
                    "labels": LABELS,
                },
                CHECKPOINT,
            )

            print("Saved best checkpoint.")

    print()
    print("=" * 60)
    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
