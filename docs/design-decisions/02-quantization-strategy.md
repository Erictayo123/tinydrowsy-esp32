# Decision: Quantization Strategy

**Date:** 2026-07-15
**Status:** Accepted

## Context

TinyEyeNetV2 is trained in FP32 with PyTorch, but the ESP32-S3 does not have a floating-point-optimized AI accelerator — its AI instruction extensions are built around 8-bit integer math. Running FP32 inference directly on-device would be slower and would use far more flash/RAM than necessary.

## Decision

Quantize the trained model to **INT8** using **ESP-PPQ** (Espressif's post-training quantization toolchain), producing a `.espdl` file for deployment via ESP-DL.

## Rationale

1. **ESP32-S3 AI instructions are INT8-native.** Espressif's SIMD/AI extensions on the S3 are designed around 8-bit integer operations, so INT8 models get hardware-accelerated inference; FP32 models do not.
2. **Size reduction.** Quantizing from FP32 to INT8 cuts model size by roughly 4×, which is what takes TinyEyeNetV2 from a ~100+ KB FP32 export down to 27.4 KB.
3. **Post-training quantization (PTQ) over quantization-aware training (QAT).** Given the model's small size and the accuracy margin available (99.87% in FP32), PTQ with a representative calibration set was sufficient to preserve accuracy without the added training complexity of QAT.
4. **Per-tensor exponent scaling.** ESP-DL represents quantized tensors with an integer exponent (power-of-two scale) rather than an arbitrary floating-point scale, which keeps the dequantization step on-device cheap (a bit-shift rather than a multiply).

## Alternatives Considered

| Approach | Model Size | On-device Speed | Accuracy Impact | Verdict |
|---|---|---|---|---|
| FP32 (no quantization) | ~100+ KB | Slow (no HW accel) | None | Rejected — too slow, too large |
| INT8 PTQ (ESP-PPQ) | 27.4 KB | Fast (HW accel) | Negligible (accuracy retained) | ✅ Selected |
| INT8 QAT | ~27 KB | Fast (HW accel) | Marginal gain, higher training cost | Not needed given PTQ results |

## Consequences

- ✅ 27.4 KB final model size, well within flash/RAM budget
- ✅ Hardware-accelerated INT8 inference on ESP32-S3's AI instructions
- ✅ Fast dequantization via exponent-based scaling in firmware
- ❌ Requires careful handling of exponent/scale bookkeeping in `model.cpp` (see [ESP-DL Integration Notes](../deployment/esp-dl-integration.md))
- ❌ Some domain-shift sensitivity observed when live camera conditions (lighting, angle) diverge from the calibration/training distribution — noted as an area for future refinement
