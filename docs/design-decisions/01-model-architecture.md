# Decision: Model Architecture

**Date:** 2026-07-15
**Status:** Accepted

## Context

The goal was to create a lightweight CNN for eye-state classification that could run on an ESP32-S3 with limited memory and compute resources, while keeping accuracy high enough for a real-time driver-monitoring use case.

## Decision

**TinyEyeNetV2** — a depthwise-separable CNN with:
- 10,690 parameters
- 64×64 grayscale input
- 2-class output (open / closed)
- No softmax in the model itself (applied in firmware, post-inference)

## Rationale

1. **Depthwise-separable convolutions** reduce parameter count by roughly 10× compared to standard convolutions, which is essential for fitting the model into ESP32-S3 flash/RAM budgets.
2. **64×64 input** balances accuracy against compute cost — large enough to preserve eye-region detail, small enough to keep inference latency low.
3. **No softmax in-graph** allows direct logit comparison on the embedded device, which is cheaper than computing an exponential/normalization step on a microcontroller.
4. **10,690 parameters** comfortably fits into 27.4 KB after INT8 quantization, leaving headroom in the 16MB flash / 8MB PSRAM budget for camera buffers and firmware.

## Alternatives Considered

| Model | Parameters | Accuracy | Size | Verdict |
|---|---|---|---|---|
| MobileNetV2 | 2.2M | 94% | 14 MB | Too large for ESP32-S3 flash/RAM |
| ResNet-18 | 11M | 95% | 45 MB | Too large, would require external storage |
| **TinyEyeNetV2** | **10.6K** | **99.87%** | **27.4 KB** | **✅ Selected** |

## Consequences

- ✅ Fits in 27.4 KB after INT8 quantization
- ✅ 99.87% accuracy exceeds requirements
- ✅ Easy to deploy on ESP32-S3 with ESP-DL
- ❌ Requires software downscaling from the camera's native resolution (160×120) to 64×64, adding a small preprocessing cost per frame
