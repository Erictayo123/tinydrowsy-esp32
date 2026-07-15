"""
augmentations.py
----------------
Training and validation transforms for TinyDrowsy.

Training:
    - Resize to 64×64
    - Random rotation (±8°)
    - Random affine translation (±5%)
    - Color jitter (brightness/contrast ±20%)
    - Gaussian blur (kernel_size=3)
    - Convert to tensor
    - Normalize to [-1, 1]

Validation/Test:
    - Resize to 64×64
    - Convert to tensor
    - Normalize to [-1, 1]

Normalization:
    MEAN = 0.5, STD = 0.5
    This maps pixel values from [0, 1] to [-1, 1]:
    normalized = (pixel - 0.5) / 0.5 = pixel * 2 - 1
"""

from torchvision import transforms


IMAGE_SIZE = 64
MEAN = [0.5]
STD = [0.5]


def get_train_transform():
    """Full augmentation pipeline for training."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.20, contrast=0.20),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


def get_validation_transform():
    """Deterministic preprocessing for validation/test."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])