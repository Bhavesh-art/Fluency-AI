from pathlib import Path

from asr import ASR
from disfluency import detect_disfluencies


PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FILE = PROJECT_ROOT / "seb-szeles-english.mp3"


def main():
    print("Loading ASR model...")

    asr = ASR()

    print(f"\nProcessing: {AUDIO_FILE}")

    result = asr.transcribe(AUDIO_FILE)

    print("\nTRANSCRIPT:")
    print(result["text"])

    words = result["chunks"]
    print("\nWORD TIMESTAMPS:\n")
    for word in words:
        print(word)

    print("\nDISFLUENCY ANALYSIS:")

    events = detect_disfluencies(words)

    if not events:
        print("No obvious disfluencies detected.")
    else:
        for event in events:
            print(event)


if __name__ == "__main__":
    main()