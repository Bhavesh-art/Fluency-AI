import re
from typing import Any


FILLERS = {
    "um",
    "uh",
    "er",
    "erm",
    "uhh",
    "umm",
}


def normalize_word(word: str) -> str:
    """Normalize a word for comparison."""
    return re.sub(r"[^a-zA-Z0-9']", "", word).lower()


def detect_word_repetitions(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Detect immediate repetitions of the same word.

    Example:
        I I I think

    becomes one detected event containing:
        I, I, I
    """

    events = []
    i = 0

    while i < len(words):
        current = normalize_word(words[i]["text"])

        if not current:
            i += 1
            continue

        j = i + 1

        while j < len(words):
            next_word = normalize_word(words[j]["text"])

            if next_word == current:
                j += 1
            else:
                break

        repetition_count = j - i

        if repetition_count >= 2:
            events.append(
                {
                    "type": "word_repetition",
                    "words": [words[k]["text"].strip() for k in range(i, j)],
                    "count": repetition_count,
                    "start": words[i]["timestamp"][0],
                    "end": words[j - 1]["timestamp"][1],
                }
            )

        i = j

    return events


def detect_phrase_repetitions(
    words: list[dict[str, Any]],
    max_phrase_length: int = 4,
) -> list[dict[str, Any]]:
    """
    Detect immediate repetition of short phrases.

    Example:
        I think I think we should go

    detects:
        I think
        I think
    """

    events = []

    normalized = [
        normalize_word(word["text"])
        for word in words
    ]

    n = len(words)

    for phrase_length in range(2, max_phrase_length + 1):
        i = 0

        while i + 2 * phrase_length <= n:
            first = normalized[i:i + phrase_length]
            second = normalized[i + phrase_length:i + 2 * phrase_length]

            if (
                all(first)
                and first == second
            ):
                events.append(
                    {
                        "type": "phrase_repetition",
                        "words": [
                            word["text"].strip()
                            for word in words[i:i + 2 * phrase_length]
                        ],
                        "count": 2,
                        "start": words[i]["timestamp"][0],
                        "end": words[i + 2 * phrase_length - 1]["timestamp"][1],
                    }
                )

                i += phrase_length * 2
            else:
                i += 1

    return events


def detect_fillers(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect common filler words."""

    events = []

    for word in words:
        normalized = normalize_word(word["text"])

        if normalized in FILLERS:
            events.append(
                {
                    "type": "filler",
                    "words": [word["text"].strip()],
                    "count": 1,
                    "start": word["timestamp"][0],
                    "end": word["timestamp"][1],
                }
            )

    return events


def detect_disfluencies(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Run all baseline disfluency detectors.
    """

    events = []

    events.extend(detect_word_repetitions(words))
    events.extend(detect_phrase_repetitions(words))
    events.extend(detect_fillers(words))

    events.sort(key=lambda event: event["start"])

    return events


if __name__ == "__main__":
    # Small artificial test.
    # We use this before connecting the detector to Whisper.
    test_words = [
        {"text": "I", "timestamp": (0.0, 0.2)},
        {"text": "I", "timestamp": (0.2, 0.4)},
        {"text": "I", "timestamp": (0.4, 0.6)},
        {"text": "think", "timestamp": (0.6, 0.9)},
        {"text": "um", "timestamp": (0.9, 1.2)},
        {"text": "I", "timestamp": (1.2, 1.4)},
        {"text": "think", "timestamp": (1.4, 1.7)},
        {"text": "I", "timestamp": (1.7, 1.9)},
        {"text": "think", "timestamp": (1.9, 2.2)},
    ]

    results = detect_disfluencies(test_words)

    print("\nDETECTED DISFLUENCIES:\n")

    for event in results:
        print(event)