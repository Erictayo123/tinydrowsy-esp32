# Dataset Preparation

## Sources

| Source | Images | Purpose |
|--------|--------|---------|
| MRL+CEW | ~30,000 | Primary training |
| DDD | ~4,000 | Secondary training |
| Selfies | ~500 | Validation/Test |

## Preprocessing

1. Convert to grayscale
2. Resize to 64×64
3. Save as JPEG (quality 95)

## Split Strategy

| Split | Ratio | Sources |
|-------|-------|---------|
| Train | 70% | MRL+CEW, DDD |
| Validation | 15% | Mixed |
| Test | 15% | Mixed + Selfies |

## Augmentation

- Random rotation: ±8°
- Random translation: ±5%
- Color jitter: brightness/contrast ±20%
- Gaussian blur: kernel_size=3
- Normalization: [-1, 1]