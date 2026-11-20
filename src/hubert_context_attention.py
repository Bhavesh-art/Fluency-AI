from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sep28k"

TRAIN_CSV = DATA_DIR / "train_audio.csv"
VAL_CSV = DATA_DIR / "val_audio.csv"

MODEL_NAME = "facebook/hubert-base-ls960"
CHECKPOINT = DATA_DIR / "hubert_context_attention.pt"

SR = 16000

TARGET_SECONDS = 3
CONTEXT_BEFORE = 5
CONTEXT_AFTER = 5

TARGET_SAMPLES = TARGET_SECONDS * SR
BEFORE_SAMPLES = CONTEXT_BEFORE * SR
AFTER_SAMPLES = CONTEXT_AFTER * SR
TOTAL_SAMPLES = (
    BEFORE_SAMPLES
    + TARGET_SAMPLES
    + AFTER_SAMPLES
)

BATCH_SIZE = 2
EPOCHS = 5

LR = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 42


LABELS = [
    "Block",
    "Interjection",
    "NoStutteredWords",
    "Prolongation",
    "SoundRep",
    "WordRep",
]

LABEL_TO_ID = {
    label: i
    for i, label in enumerate(LABELS)
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class SEP28KContextDataset(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        show = str(row["Show"]).strip()
        epid = int(row["EpId"])

        start = int(row["Start"])
        stop = int(row["Stop"])

        wav_path = (
            DATA_DIR
            / "audio"
            / show
            / f"{epid}.wav"
        )

        if not wav_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {wav_path}"
            )

        # -------------------------------------------------
        # Read ONLY the required 13-second region.
        #
        # The labeled event is [start, stop].
        # We want:
        #
        # start - 5 sec
        #        |
        #        v
        # [-----][---------][-----]
        #   5 s     3 s        5 s
        #
        # This avoids loading the entire 20-90 minute
        # episode WAV into RAM.
        # -------------------------------------------------

        read_start = max(
            0,
            start - BEFORE_SAMPLES,
        )

        read_end = (
            stop + AFTER_SAMPLES
        )

        num_frames = (
            read_end - read_start
        )

        waveform, sr = torchaudio.load(
            wav_path,
            frame_offset=read_start,
            num_frames=num_frames,
        )

        if sr != SR:
            raise RuntimeError(
                f"Unexpected sample rate {sr} "
                f"for {wav_path}. "
                f"Expected {SR} Hz."
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0
            )
        else:
            waveform = waveform.squeeze(0)

        # -------------------------------------------------
        # Boundary handling
        #
        # If the event is near the beginning/end of an
        # episode, the requested 13 sec may not exist.
        # Pad with zeros to maintain a fixed 13-sec input.
        # -------------------------------------------------

        expected_length = TOTAL_SAMPLES

        if waveform.shape[0] < expected_length:

            waveform = nn.functional.pad(
                waveform,
                (
                    0,
                    expected_length
                    - waveform.shape[0],
                ),
            )

        elif waveform.shape[0] > expected_length:

            waveform = waveform[
                :expected_length
            ]

        label = LABEL_TO_ID[
            row["majority_label"]
        ]

        return (
            waveform,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


class AttentionPooling(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(
                hidden_size,
                128,
            ),
            nn.Tanh(),
            nn.Linear(
                128,
                1,
            ),
        )

    def forward(self, x):

        # x:
        # [batch, time, hidden]

        scores = self.attention(
            x
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            x
            * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled


class HuBERTContextAttention(
    nn.Module
):

    def __init__(self):
        super().__init__()

        self.hubert = (
            HubertModel.from_pretrained(
                MODEL_NAME
            )
        )

        # Disable HuBERT training-time masking because the encoder is frozen.
        self.hubert.config.mask_time_prob = 0.0
        self.hubert.config.mask_feature_prob = 0.0

        # Freeze pretrained HuBERT.
        for param in (
            self.hubert.parameters()
        ):
            param.requires_grad = False

        hidden_size = (
            self.hubert.config.hidden_size
        )

        self.pooling = (
            AttentionPooling(
                hidden_size
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                256,
                len(LABELS),
            ),
        )

    def forward(self, waveform):

        # HuBERT is frozen, so no gradients
        # need to be stored for this part.
        with torch.no_grad():

            outputs = self.hubert(
                input_values=waveform
            )

        hidden = (
            outputs.last_hidden_state
        )

        pooled = (
            self.pooling(hidden)
        )

        logits = (
            self.classifier(
                pooled
            )
        )

        return logits


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
):

    training = (
        optimizer is not None
    )

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (waveforms, labels) in enumerate(loader, start=1):

        if batch_idx == 1 or batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(loader)}', flush=True)

        waveforms = waveforms.to(DEVICE)

        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(
            training
        ):

            logits = model(
                waveforms
            )

            loss = criterion(
                logits,
                labels,
            )

            if training:

                loss.backward()

                optimizer.step()

        batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_correct += (
            (
                logits.argmax(
                    dim=1
                )
                == labels
            )
            .sum()
            .item()
        )

        total_samples += (
            batch_size
        )

    return (
        total_loss
        / total_samples,
        total_correct
        / total_samples,
    )


def main():

    print("=" * 70)
    print(
        "13-SECOND HuBERT + "
        "ATTENTION POOLING"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Target event: "
        f"{TARGET_SECONDS} sec"
    )

    print(
        f"Context before: "
        f"{CONTEXT_BEFORE} sec"
    )

    print(
        f"Context after: "
        f"{CONTEXT_AFTER} sec"
    )

    print(
        f"Total input: "
        f"{TOTAL_SAMPLES / SR:.1f} sec"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs: "
        f"{EPOCHS}"
    )

    train_dataset = (
        SEP28KContextDataset(
            TRAIN_CSV
        )
    )

    val_dataset = (
        SEP28KContextDataset(
            VAL_CSV
        )
    )

    print(
        f"Train clips: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation clips: "
        f"{len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    print("\nLoading HuBERT...")

    model = (
        HuBERTContextAttention()
        .to(DEVICE)
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    model.pooling.parameters(),
                "lr": LR,
            },
            {
                "params":
                    model.classifier.parameters(),
                "lr": LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        start_time = time.time()

        train_loss, train_acc = (
            run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        val_loss, val_acc = (
            run_epoch(
                model,
                val_loader,
                criterion,
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"\nEpoch "
            f"{epoch}/{EPOCHS}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.4f} "
            f"acc: "
            f"{train_acc:.4f}"
        )

        print(
            f"Val loss: "
            f"{val_loss:.4f} "
            f"acc: "
            f"{val_acc:.4f}"
        )

        print(
            f"Time: "
            f"{elapsed / 60:.2f} min"
        )

        if (
            val_loss
            < best_val_loss
        ):

            best_val_loss = (
                val_loss
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "model_name":
                        MODEL_NAME,

                    "target_seconds":
                        TARGET_SECONDS,

                    "context_before":
                        CONTEXT_BEFORE,

                    "context_after":
                        CONTEXT_AFTER,

                    "pooling":
                        "attention",

                    "labels":
                        LABELS,

                    "best_val_loss":
                        best_val_loss,

                    "epoch":
                        epoch,
                },
                CHECKPOINT,
            )

            print(
                "Saved best "
                "checkpoint: "
                f"{CHECKPOINT}"
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        "Checkpoint: "
        f"{CHECKPOINT}"
    )


if __name__ == "__main__":
    main()





