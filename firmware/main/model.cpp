#include "model.hpp"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "esp_log.h"
#include <map>
#include <string>
#include <cmath>

static const char *TAG = "MODEL";

// Embedded model binary with 16-byte alignment (CRITICAL)
extern const uint8_t tinydrowsy_int8_espdl_start[] asm("_binary_tinydrowsy_int8_espdl_start") __attribute__((aligned(16)));
extern const uint8_t tinydrowsy_int8_espdl_end[] asm("_binary_tinydrowsy_int8_espdl_end") __attribute__((aligned(16)));

static dl::Model *model = nullptr;
static dl::TensorBase *input_tensor = nullptr;
static dl::TensorBase *output_tensor = nullptr;

static std::string input_name;
static std::string output_name;
static int input_exponent = 0;
static int output_exponent = 0;

int init_model(void) 
{
    if (model) {
        ESP_LOGW(TAG, "Model already initialized");
        return 0;
    }

    size_t model_size = tinydrowsy_int8_espdl_end - tinydrowsy_int8_espdl_start;
    ESP_LOGI(TAG, "Loading quantized model: %zu bytes (%.2f KB)", 
             model_size, model_size / 1024.0);

    // Verify alignment
    uintptr_t addr = (uintptr_t)tinydrowsy_int8_espdl_start;
    ESP_LOGI(TAG, "Model address: 0x%08x, aligned: %s", addr, ((addr & 0xF) == 0) ? "YES" : "NO");
    
    if (addr % 16 != 0) {
        ESP_LOGE(TAG, "Model is not 16-byte aligned! This will cause crashes.");
        return -1;
    }

    // CORRECT constructor - 6 arguments (no input_shapes)
    model = new dl::Model(
        (const char *)tinydrowsy_int8_espdl_start,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        64 * 1024,                          // max_internal_size
        dl::MEMORY_MANAGER_GREEDY,          // mm_type
        nullptr,                            // key (no encryption)
        true                                // param_copy
    );

    if (!model) {
        ESP_LOGE(TAG, "Failed to create model!");
        return -1;
    }

    // Get input tensor info
    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    if (model_inputs.empty()) {
        ESP_LOGE(TAG, "Model has no inputs!");
        delete model;
        model = nullptr;
        return -1;
    }

    auto input_iter = model_inputs.begin();
    input_name = input_iter->first;
    input_tensor = input_iter->second;
    input_exponent = input_tensor->exponent;

    // Get output tensor info
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    if (model_outputs.empty()) {
        ESP_LOGE(TAG, "Model has no outputs!");
        delete model;
        model = nullptr;
        return -1;
    }

    auto output_iter = model_outputs.begin();
    output_name = output_iter->first;
    output_tensor = output_iter->second;
    output_exponent = output_tensor->exponent;

    // Print model info
    ESP_LOGI(TAG, "✅ Model loaded successfully!");
    ESP_LOGI(TAG, "Input:  name='%s', shape=[%zu,%zu,%zu,%zu], exponent=%d", 
             input_name.c_str(),
             input_tensor->shape[0],
             input_tensor->shape[1],
             input_tensor->shape[2],
             input_tensor->shape[3],
             input_exponent);
    ESP_LOGI(TAG, "Output: name='%s', shape=[%zu,%zu], exponent=%d",
             output_name.c_str(),
             output_tensor->shape[0],
             output_tensor->shape[1],
             output_exponent);

    return 0;
}

void print_model_info(void)
{
    if (!model) {
        ESP_LOGW(TAG, "Model not initialized. Call init_model() first.");
        return;
    }

    ESP_LOGI(TAG, "=== Model Info ===");
    ESP_LOGI(TAG, "Input:  name='%s', shape=[%zu,%zu,%zu,%zu], exponent=%d", 
             input_name.c_str(),
             input_tensor->shape[0],
             input_tensor->shape[1],
             input_tensor->shape[2],
             input_tensor->shape[3],
             input_exponent);
    ESP_LOGI(TAG, "Output: name='%s', shape=[%zu,%zu], exponent=%d",
             output_name.c_str(),
             output_tensor->shape[0],
             output_tensor->shape[1],
             output_exponent);
    ESP_LOGI(TAG, "=================");
}

int predict_eye(const uint8_t *image_data, float *logit_open, float *logit_closed)
{
    if (!model || !input_tensor || !output_tensor) {
        ESP_LOGE(TAG, "Model not initialized!");
        return -1;
    }

    if (!image_data) {
        ESP_LOGE(TAG, "Null image data!");
        return -1;
    }

    // Get pointer to input tensor data
    int8_t *input_ptr = (int8_t *)input_tensor->data;
    
    // Step 1: Normalize and quantize the input
    // Training preprocessing: (pixel/255.0 - 0.5) / 0.5 = pixel * 2 - 1
    // Input range: uint8 0-255 → float [-1, 1] → quantized int8
    size_t total_pixels = 64 * 64;
    for (size_t i = 0; i < total_pixels; i++) {
        // Convert uint8 to float in [-1, 1]
        float normalized = (image_data[i] / 255.0f - 0.5f) / 0.5f;
        // Quantize using the input tensor's exponent
        // Note: dl::quantize uses inverse scale, so we use DL_RESCALE
        input_ptr[i] = dl::quantize<int8_t>(normalized, DL_RESCALE(input_exponent));
    }

    // Step 2: Run inference
    model->run();

    // Step 3: Dequantize and read output
    int8_t *output_ptr = (int8_t *)output_tensor->data;
    
    // Dequantize both logits
    // CORRECT mapping: output_ptr[0] = closed, output_ptr[1] = open
    *logit_closed = dl::dequantize(output_ptr[0], DL_SCALE(output_exponent));
    *logit_open = dl::dequantize(output_ptr[1], DL_SCALE(output_exponent));

    return 0;
}