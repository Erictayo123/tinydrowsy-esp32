# ESP-DL v3.3.5 Integration Notes

## Model Loading

```cpp
model = new dl::Model(
    (const char *)tinydrowsy_int8_espdl_start,
    fbs::MODEL_LOCATION_IN_FLASH_RODATA,
    64 * 1024,          // max_internal_size
    dl::MEMORY_MANAGER_GREEDY,
    nullptr,
    true                // param_copy
);
```

## Memory Management

- **`max_internal_size = 64KB`** — Internal RAM reserved for critical tensors.
- **`param_copy = true`** — Copies parameters to PSRAM for faster access during inference.
- **`fbs::MODEL_LOCATION_IN_FLASH_RODATA`** — Model is embedded directly in flash, avoiding external storage.

## Tensor Access

```cpp
// Input: shape [1, 64, 64, 1], exponent -6
auto inputs = model->get_inputs();
dl::TensorBase *input_tensor = inputs.begin()->second;
int input_exponent = input_tensor->exponent;

// Output: shape [1, 2], exponent -3
auto outputs = model->get_outputs();
dl::TensorBase *output_tensor = outputs.begin()->second;
int output_exponent = output_tensor->exponent;
```

## Quantization / Dequantization

```cpp
// Quantize input (uint8 → [-1,1] → int8)
float normalized = (pixel / 255.0f - 0.5f) / 0.5f;
input_ptr[i] = dl::quantize<int8_t>(normalized, DL_RESCALE(input_exponent));

// Run inference
model->run();

// Dequantize output (int8 → float)
float output = dl::dequantize(output_ptr[i], DL_SCALE(output_exponent));
```

## Common Pitfalls

1. **Model must be 16-byte aligned** — use `__attribute__((aligned(16)))`.
2. **Main task stack must be ≥ 16KB** — set `CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384`.
3. **Large arrays should be static or heap-allocated** — avoid stack allocation for frame buffers or tensors.
4. **PSRAM is essential** for camera frame buffers — without it, expect allocation failures under load.

## Related Docs

- [Model Architecture](../design-decisions/01-model-architecture.md)
- [Quantization Strategy](../design-decisions/02-quantization-strategy.md)
- [Firmware Architecture](../design-decisions/05-firmware-architecture.md)
