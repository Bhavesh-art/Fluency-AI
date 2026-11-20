# Fluency AI

**Real-Time Disfluent Speech Correction and Speaker Identity-Preserving Speech Reconstruction**

Fluency AI is a BTech major project for detecting speech disfluencies, correcting the spoken content while preserving meaning, and reconstructing corrected speech while preserving the original speaker identity.

## Current Status

The project has completed SEP-28k dataset preparation, episode-level train/validation/test splitting, audio coverage auditing, CNN, Wav2Vec2, and HuBERT experiments, temporal-context experiments, Whisper transcript auditing, and ASR feature extraction development.

## Current Best Acoustic Model

The current strongest acoustic model is a frozen acebook/hubert-base-ls960 encoder with mean pooling and a lightweight classifier.

- Accuracy: **63.89%**
- Macro Precision: **55.16%**
- Macro Recall: **40.87%**
- Macro F1: **44.11%**
- Weighted F1: **59.67%**

## Dataset

The primary dataset is SEP-28k. Clips with at least 2/3 annotator agreement were retained, giving 26,441 usable clips.

An episode-level group split was used to reduce leakage between train, validation, and test sets.

- Train: 21,500 clips / 308 episodes
- Validation: 2,500 clips / 38 episodes
- Test: 2,441 clips / 39 episodes

19,616 clips currently have recovered audio. The unavailable audio comes from podcast episodes whose original media URLs are no longer accessible.

## Models Evaluated

- Small CNN mel-spectrogram baseline
- Wav2Vec2 baseline
- Wav2Vec2 multi-layer model
- Weighted Wav2Vec2 multi-layer model
- HuBERT baseline
- Weighted HuBERT
- 13-second HuBERT context model
- 13-second HuBERT temporal-attention model

## ASR Investigation

Whisper openai/whisper-small is being investigated as a complementary modality. Transcript analysis showed that acoustic disfluencies are not always represented reliably in ASR text.

Current ASR features include:

- Word count
- Filler count
- Adjacent repeated-word count
- Character repetition count
- Repeated-word rate
- Filler rate
- Average word duration
- Short-word rate

The next major experiment is **HuBERT + ASR multimodal fusion**.

## Seed-VC

Seed-VC has been tested for speaker-identity-preserving speech reconstruction. Initial experiments demonstrated preservation of linguistic content and strong same-speaker similarity.

Seed-VC repositories, model weights, audio, datasets, and virtual environments are intentionally excluded from this repository.

## Repository Structure

`	ext
Fluency-AI/
├── src/
│   ├── ASR and feature extraction
│   ├── SEP-28k preparation and auditing
│   ├── CNN baseline
│   ├── Wav2Vec2 experiments
│   ├── HuBERT experiments
│   ├── Context experiments
│   └── Transcript analysis
├── check_audio_coverage.py
├── inspect_audio.py
├── make_audio_manifests.py
├── .gitignore
└── README.md
`

## Important Files

- src/sep28k_prepare.py — dataset preparation and group-aware splitting
- src/hubert_baseline.py — main HuBERT baseline
- src/hubert_weighted.py — weighted HuBERT experiment
- src/hubert_context.py — 13-second context experiment
- src/hubert_context_attention.py — temporal-attention experiment
- src/asr.py — Whisper ASR
- src/asr_features.py — ASR feature extraction
- src/cache_asr_features.py — resumable ASR feature caching
- src/transcript_error_audit.py — transcript/disfluency audit

## Planned Work

1. Complete efficient ASR feature extraction.
2. Train HuBERT + ASR multimodal fusion.
3. Compare multimodal performance against the 63.89% HuBERT baseline.
4. Implement disfluency-aware text correction.
5. Integrate TTS.
6. Integrate Seed-VC reconstruction.
7. Evaluate the complete pipeline using WER/CER, semantic preservation, disfluency removal, speaker similarity, speech quality, and latency.

## Repository Policy

Large datasets, audio files, downloaded model weights, checkpoints, virtual environments, and generated files are excluded using .gitignore.
