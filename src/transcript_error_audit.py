import os
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from transformers import HubertModel
from asr import ASR


DATA_DIR = r".\data\sep28k"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
TEST_CSV = os.path.join(DATA_DIR, "test_audio.csv")
CHECKPOINT = os.path.join(DATA_DIR, "hubert_weighted.pt")
OUTPUT_CSV = os.path.join(DATA_DIR, "transcript_error_audit.csv")

LABELS = [
    "Block",
    "Interjection",
    "NoStutteredWords",
    "Prolongation",
    "SoundRep",
    "WordRep",
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
CLIP_SAMPLES = 48000


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


def load_clip(row):
    wav_path = os.path.join(
        AUDIO_DIR,
        str(row["Show"]),
        f'{int(row["EpId"])}.wav'
    )

    waveform, sr = torchaudio.load(wav_path)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sr, SAMPLE_RATE
        )

    waveform = waveform.mean(dim=0)

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

    return clip


def main():
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(TEST_CSV)

    print(f"Test clips available: {len(df)}")

    model = HubertClassifier().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT}")

    asr = ASR()

    rows = []

    with torch.no_grad():
        for class_name in LABELS:
            class_df = df[df["majority_label"] == class_name]

            sample_n = min(10, len(class_df))

            sample = class_df.sample(
                n=sample_n,
                random_state=42
            )

            for _, row in sample.iterrows():
                clip = load_clip(row)

                x = clip.unsqueeze(0).to(DEVICE)

                with torch.amp.autocast(
                    "cuda",
                    enabled=(DEVICE.type == "cuda")
                ):
                    logits = model(x)

                probabilities = torch.softmax(logits, dim=1)[0]
                prediction_id = int(torch.argmax(probabilities))
                confidence = float(probabilities[prediction_id])

                wav_path = os.path.join(
                    AUDIO_DIR,
                    str(row["Show"]),
                    f'{int(row["EpId"])}.wav'
                )

                # Create a temporary 3-second WAV for Whisper.
                temp_path = (
                    Path(DATA_DIR) / "_audit_temp_clip.wav"
                )

                torchaudio.save(
                    str(temp_path),
                    clip.unsqueeze(0).cpu(),
                    SAMPLE_RATE,
                )

                result = asr.transcribe(temp_path)

                transcript = result.get("text", "").strip()

                rows.append({
                    "Show": row["Show"],
                    "EpId": int(row["EpId"]),
                    "Start": int(row["Start"]),
                    "Stop": int(row["Stop"]),
                    "true_label": class_name,
                    "predicted_label": ID_TO_LABEL[prediction_id],
                    "confidence": confidence,
                    "correct_prediction": (
                        prediction_id == LABEL_TO_ID[class_name]
                    ),
                    "transcript": transcript,
                    "audio_path": wav_path,
                })

                print(
                    f"{len(rows):02d}/60 | "
                    f"True={class_name:<18} "
                    f"Pred={ID_TO_LABEL[prediction_id]:<18} "
                    f"Conf={confidence:.3f} | "
                    f"{transcript}"
                )

    if Path(DATA_DIR, "_audit_temp_clip.wav").exists():
        Path(DATA_DIR, "_audit_temp_clip.wav").unlink()

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    print()
    print("Audit complete.")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Rows: {len(result_df)}")


if __name__ == "__main__":
    main()
