# HuBERT Baseline Results

## Model
- Model: facebook/hubert-base-ls960
- Encoder: Frozen HuBERT
- Classifier: 768 -> 256 -> 6
- Input: 3-second, 16 kHz audio
- Test clips: 1,750
- Device: NVIDIA GeForce RTX 4060 Laptop GPU

## Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 63.89% |
| Macro Precision | 55.16% |
| Macro Recall | 40.87% |
| Macro F1 | 44.11% |
| Weighted F1 | 59.67% |

## Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Block | 46.67% | 20.59% | 28.57% |
| Interjection | 58.80% | 56.61% | 57.68% |
| NoStutteredWords | 68.68% | 91.82% | 78.58% |
| Prolongation | 65.45% | 34.62% | 45.28% |
| SoundRep | 52.94% | 14.52% | 22.78% |
| WordRep | 38.39% | 27.04% | 31.73% |

## Confusion Matrix

Rows = true class, columns = predicted class.

```text
[[ 42  19 114   3  11  15]
 [  2 137  78   1   3  21]
 [ 14  37 842   4   1  19]
 [  7  12  49  36   0   0]
 [ 23   9  53   7  18  14]
 [  2  19  90   4   1  43]]
```