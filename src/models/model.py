"""
model.py
--------
TinyEyeNet V2 – a lightweight depthwise‑separable CNN for ESP32-S3.
Supports any number of classes (automatically derived from dataset).
"""

import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class TinyEyeNetV2(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        super().__init__()
        # Initial conv (no depthwise yet)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Depthwise separable blocks
        self.block1 = DepthwiseSeparableBlock(16, 32, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = DepthwiseSeparableBlock(32, 64, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = DepthwiseSeparableBlock(64, 96, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global average pooling and classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = TinyEyeNetV2(num_classes=2)
    dummy = torch.randn(1, 1, 64, 64)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")