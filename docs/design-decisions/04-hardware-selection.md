# Decision: Hardware Selection

**Date:** 2026-07-15
**Status:** Accepted

## Context

TinyDrowsy needed a low-cost, camera-capable microcontroller platform able to run a quantized CNN in real time, without relying on external cloud inference (important for an in-vehicle safety application where latency and connectivity can't be guaranteed).

## Decision

**ESP32-S3-WROOM-1-N16R8** (16MB Flash, 8MB PSRAM) paired with an **OV2640 camera module**.

## Rationale

1. **Built-in AI acceleration.** The S3 variant of ESP32 includes vector/AI instruction extensions that meaningfully speed up INT8 CNN inference compared to the base ESP32.
2. **Sufficient memory headroom.** 16MB flash comfortably holds firmware, the 27.4 KB model, and partition overhead; 8MB PSRAM gives room for camera frame buffers without starving other tasks.
3. **ESP-DL support.** Espressif maintains ESP-DL specifically for the S3's AI instructions, which meant the quantization and deployment pipeline (ESP-PPQ → `.espdl` → ESP-DL runtime) was a supported, documented path rather than a custom port.
4. **Cost.** ESP32-S3 boards are inexpensive relative to alternatives like a Jetson Nano or Raspberry Pi + accelerator, which matters for a project aimed at affordable, scalable deployment (echoing the broader goal of low-cost embedded solutions).
5. **Camera ecosystem.** OV2640 is a well-supported, low-cost camera module with mature ESP-IDF driver support, avoiding the need to write a custom camera driver from scratch.

## Alternatives Considered

| Platform | AI Accel. | Cost | Camera Support | Verdict |
|---|---|---|---|---|
| ESP32 (base, non-S3) | None | Lowest | Good | Rejected — no AI instruction extensions, slower inference |
| Raspberry Pi Zero 2 W | CPU-only (no dedicated accel) | Moderate | Good | Rejected — higher power draw, overkill for this model size |
| Jetson Nano | Strong (GPU) | High | Good | Rejected — cost and power draw not justified for a 27.4 KB model |
| **ESP32-S3-WROOM-1-N16R8 + OV2640** | **Yes (INT8)** | **Low** | **Good (ESP-IDF driver)** | **✅ Selected** |

## Consequences

- ✅ Real-time inference (~8.8 FPS) achieved without external compute
- ✅ Low bill-of-materials cost, suitable for eventual affordable deployment at scale
- ✅ Direct support from Espressif's ESP-DL toolchain
- ❌ Camera resolution/optics are modest compared to higher-end modules, which contributes to the domain-shift and preprocessing considerations noted in the [dataset strategy](03-dataset-strategy.md) and [model architecture](01-model-architecture.md) docs
