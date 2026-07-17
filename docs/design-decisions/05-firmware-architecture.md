# Decision: Firmware Architecture

**Date:** 2026-07-15
**Status:** Accepted

## Context

The firmware needs to run a real-time loop that captures a camera frame, preprocesses it, runs INT8 inference, and reports a result — repeatedly, without memory leaks or stack overflows, on a resource-constrained microcontroller.

## Decision

A modular ESP-IDF firmware structure with clearly separated concerns:

```
firmware/main/
├── main.cpp      # Real-time inference loop (capture → preprocess → infer → report)
├── model.cpp     # ESP-DL model loading and inference
├── model.hpp     # Model interface
├── camera.cpp    # OV2640 camera driver
└── camera.hpp    # Camera interface
```

Plus a custom 16MB `partitions.csv` and tuned `sdkconfig.defaults` (notably a larger main task stack).

## Rationale

1. **Separation of camera and model logic.** Keeping `camera.cpp`/`camera.hpp` independent from `model.cpp`/`model.hpp` means either can be modified or swapped (e.g., a different camera module, or a future model revision) without touching the other.
2. **ESP-DL model loading from flash.** The model is embedded directly in flash (`fbs::MODEL_LOCATION_IN_FLASH_RODATA`) rather than loaded from an SD card or over the network, avoiding external storage dependencies and keeping boot-to-inference time short.
3. **Static/heap allocation for large buffers.** Frame buffers and model-internal tensors are kept off the stack (static or PSRAM-backed heap allocation) to avoid stack overflows — a common failure mode on ESP32 given its limited default stack size.
4. **Custom partition table.** The default ESP-IDF partition layout doesn't reserve enough contiguous flash for a 16MB board with room for OTA and model storage, so a custom `partitions.csv` was needed.
5. **Increased main task stack size.** `CONFIG_ESP_MAIN_TASK_STACK_SIZE` is raised (to 16KB) to accommodate ESP-DL's internal call depth during inference, which exceeds the ESP-IDF default.

## Alternatives Considered

| Approach | Modularity | Stability | Verdict |
|---|---|---|---|
| Single monolithic `main.cpp` | Low | Harder to debug memory issues | Rejected |
| Model loaded from SD card | Moderate | Adds external dependency, slower boot | Rejected — flash embedding is simpler and faster |
| **Modular main/model/camera split, model in flash** | **High** | **Stable (verified over sustained runs)** | **✅ Selected** |

## Consequences

- ✅ Clean separation of concerns made debugging the domain-shift and inference-timing issues (see project status) more tractable
- ✅ Stable memory usage across sustained real-time inference runs
- ✅ Fast boot-to-inference, no external storage dependency
- ❌ Any camera hardware change still requires updating `camera.cpp`'s GPIO pin configuration and possibly XCLK frequency (see [Troubleshooting](../../README.md#troubleshooting) in the README)

See also: [ESP-DL Integration Notes](../deployment/esp-dl-integration.md) for the low-level model loading, tensor access, and quantize/dequantize code used inside `model.cpp`.
