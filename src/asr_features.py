from pathlib import Path
import re
import numpy as np


FILLERS = {
    "um", "umm", "uh", "uhh",
    "er", "erm", "hmm", "hm"
}


def extract_asr_features(result):
    text = str(result.get("text", "")).strip()

    chunks = result.get("chunks", [])

    words = []
    durations = []

    for chunk in chunks:
        word = str(chunk.get("text", "")).strip()
        timestamp = chunk.get("timestamp")

        if not word:
            continue

        words.append(word)

        if (
            timestamp is not None
            and timestamp[0] is not None
            and timestamp[1] is not None
        ):
            start, end = timestamp
            if end >= start:
                durations.append(end - start)

    normalized = [
        re.sub(r"[^a-zA-Z']", "", w).lower()
        for w in words
    ]

    normalized = [
        w for w in normalized if w
    ]

    word_count = len(normalized)

    filler_count = sum(
        w in FILLERS
        for w in normalized
    )

    adjacent_repeat_count = sum(
        normalized[i] == normalized[i - 1]
        for i in range(1, len(normalized))
    )

    char_repeat_count = len(
        re.findall(
            r"([a-zA-Z])\1{2,}",
            text
        )
    )

    repeated_word_rate = (
        adjacent_repeat_count / max(word_count - 1, 1)
    )

    filler_rate = (
        filler_count / max(word_count, 1)
    )

    avg_word_duration = (
        float(np.mean(durations))
        if durations
        else 0.0
    )

    short_word_rate = (
        sum(len(w) <= 2 for w in normalized)
        / max(word_count, 1)
    )

    features = np.array(
        [
            word_count,
            filler_count,
            adjacent_repeat_count,
            char_repeat_count,
            repeated_word_rate,
            filler_rate,
            avg_word_duration,
            short_word_rate,
        ],
        dtype=np.float32,
    )

    return features, {
        "text": text,
        "words": normalized,
        "word_count": word_count,
        "filler_count": filler_count,
        "adjacent_repeat_count": adjacent_repeat_count,
        "char_repeat_count": char_repeat_count,
        "repeated_word_rate": repeated_word_rate,
        "filler_rate": filler_rate,
        "avg_word_duration": avg_word_duration,
        "short_word_rate": short_word_rate,
    }
