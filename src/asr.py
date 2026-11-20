from pathlib import Path

import soundfile as sf
from transformers import pipeline


MODEL_NAME = "openai/whisper-small"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO = (
    PROJECT_ROOT
    / "seed-vc"
    / "examples"
    / "source"
    / "source_s1.wav"
)


class ASR:
    def __init__(self):
        self.model = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            device=0,
            model_kwargs={"attn_implementation": "eager"},
)

    def transcribe(self, audio_path: str | Path):
        audio_path = Path(audio_path)

        audio, sample_rate = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        result = self.model(
            {
                "raw": audio,
                "sampling_rate": sample_rate,
            },
            return_timestamps="word",
            generate_kwargs={
                "task": "transcribe",
                "language": "en",
                "temperature": 0.0,
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                },
        )

        return result


if __name__ == "__main__":
    asr = ASR()
    result = asr.transcribe(DEFAULT_AUDIO)

    print("\nTRANSCRIPT:")
    print(result["text"])

    print("\nWORD TIMESTAMPS:")
    for word in result["chunks"]:
        print(word)