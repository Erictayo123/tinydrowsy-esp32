# \# ESP-DL v3.3.5 Integration Notes

# 

# \## Model Loading

# 

# ```cpp

# model = new dl::Model(

# &#x20;   (const char \*)tinydrowsy\_int8\_espdl\_start,

# &#x20;   fbs::MODEL\_LOCATION\_IN\_FLASH\_RODATA,

# &#x20;   64 \* 1024,          // max\_internal\_size

# &#x20;   dl::MEMORY\_MANAGER\_GREEDY,

# &#x20;   nullptr,

# &#x20;   true                // param\_copy

# );



# Memory Management



# max\_internal\_size = 64KB: Internal RAM for critical tensors

# 

# param\_copy = true: Copy parameters to PSRAM for faster access

# 

# fbs::MODEL\_LOCATION\_IN\_FLASH\_RODATA: Model embedded in flash

# 

# Tensor Access

# cpp

# // Input: shape \[1, 64, 64, 1], exponent -6

# auto inputs = model->get\_inputs();

# dl::TensorBase \*input\_tensor = inputs.begin()->second;

# int input\_exponent = input\_tensor->exponent;

# 

# // Output: shape \[1, 2], exponent -3

# auto outputs = model->get\_outputs();

# dl::TensorBase \*output\_tensor = outputs.begin()->second;

# int output\_exponent = output\_tensor->exponent;



# Quantization/Dequantization

# cpp

# // Quantize input (uint8 → \[-1,1] → int8)

# float normalized = (pixel / 255.0f - 0.5f) / 0.5f;

# input\_ptr\[i] = dl::quantize<int8\_t>(normalized, DL\_RESCALE(input\_exponent));

# 

# // Run inference

# model->run();

# 

# // Dequantize output (int8 → float)

# float output = dl::dequantize(output\_ptr\[i], DL\_SCALE(output\_exponent));



# Common Pitfalls

# Model must be 16-byte aligned - Use \_\_attribute\_\_((aligned(16)))

# 

# Main task stack must be >= 16KB - Set CONFIG\_ESP\_MAIN\_TASK\_STACK\_SIZE=16384

# 

# Large arrays should be static or heap-allocated - Avoid stack allocation

# 

# PSRAM is essential for camera frame buffers

