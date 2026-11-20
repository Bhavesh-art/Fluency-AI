import pandas as pd
import re

PATH = r".\data\sep28k\transcript_error_audit.csv"

df = pd.read_csv(PATH)
df["transcript"] = df["transcript"].fillna("").astype(str)

FILLERS = {
    "um", "uh", "er", "erm", "umm", "uhh", "hmm", "hm"
}

def words(text):
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())

def has_adjacent_word_repeat(text):
    w = words(text)
    return any(w[i] == w[i + 1] for i in range(len(w) - 1))

def count_adjacent_word_repeats(text):
    w = words(text)
    return sum(w[i] == w[i + 1] for i in range(len(w) - 1))

def has_filler(text):
    w = words(text)
    return any(x in FILLERS for x in w)

def has_char_repetition(text):
    t = text.lower().replace(" ", "")
    return bool(re.search(r"(.)\1{2,}", t))

def has_hyphenated_repeat(text):
    return bool(re.search(r"\b([a-z]+)-\s*\1\b", text.lower()))

def transcript_word_count(text):
    return len(words(text))

df["word_count"] = df["transcript"].apply(transcript_word_count)
df["adjacent_word_repeats"] = df["transcript"].apply(count_adjacent_word_repeats)
df["has_word_repeat"] = df["transcript"].apply(has_adjacent_word_repeat)
df["has_filler"] = df["transcript"].apply(has_filler)
df["has_char_repetition"] = df["transcript"].apply(has_char_repetition)
df["has_hyphenated_repeat"] = df["transcript"].apply(has_hyphenated_repeat)

print("\n=== OVERALL TRANSCRIPT ANALYSIS ===")
print(f"Total clips: {len(df)}")
print(f"Contains adjacent word repetition: {df['has_word_repeat'].sum()}")
print(f"Contains filler: {df['has_filler'].sum()}")
print(f"Contains character repetition: {df['has_char_repetition'].sum()}")
print(f"Contains hyphenated repetition: {df['has_hyphenated_repeat'].sum()}")

print("\n=== BY TRUE LABEL ===")

summary = df.groupby("true_label").agg(
    clips=("true_label", "size"),
    word_repeat_rate=("has_word_repeat", "mean"),
    filler_rate=("has_filler", "mean"),
    char_repeat_rate=("has_char_repetition", "mean"),
    hyphenated_repeat_rate=("has_hyphenated_repeat", "mean"),
    avg_word_count=("word_count", "mean"),
    avg_hubert_confidence=("confidence", "mean"),
    hubert_accuracy=("correct_prediction", "mean"),
)

print(summary.round(3).to_string())

print("\n=== TRANSCRIPTS WITH WORD REPETITION ===")

repeated = df[df["has_word_repeat"]][
    ["true_label", "predicted_label", "confidence", "transcript"]
]

if len(repeated):
    print(repeated.to_string(index=False))
else:
    print("None")

print("\n=== TRANSCRIPTS WITH FILLERS ===")

fillers = df[df["has_filler"]][
    ["true_label", "predicted_label", "confidence", "transcript"]
]

if len(fillers):
    print(fillers.to_string(index=False))
else:
    print("None")

print("\n=== TRANSCRIPTS WITH CHARACTER REPETITION ===")

chars = df[df["has_char_repetition"]][
    ["true_label", "predicted_label", "confidence", "transcript"]
]

if len(chars):
    print(chars.to_string(index=False))
else:
    print("None")

OUTPUT = r".\data\sep28k\transcript_features_60.csv"
df.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)
