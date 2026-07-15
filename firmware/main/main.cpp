#include <stdio.h>
#include <cmath>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "model.hpp"
#include "camera.hpp"

static const char *TAG = "MAIN";

void softmax(float *logits, float *probs, size_t n)
{
    float max_val = logits[0];
    for (size_t i = 1; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    
    for (size_t i = 0; i < n; i++) {
        probs[i] /= sum;
    }
}

void generate_dummy_frame(uint8_t *image_data)
{
    static int counter = 0;
    for (int i = 0; i < 64 * 64; i++) {
        int x = i % 64;
        int y = i / 64;
        image_data[i] = (x * y + counter) % 256;
    }
    counter++;
}

static uint8_t image_data[64 * 64];

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  TinyDrowsy ESP32-S3 Firmware v1.0");
    ESP_LOGI(TAG, "  Real-time Eye-State Classification");
    ESP_LOGI(TAG, "========================================");

    ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());

    if (init_model() != 0) {
        ESP_LOGE(TAG, "Failed to initialize model!");
        return;
    }
    print_model_info();

    bool camera_available = (init_camera() == 0);
    if (camera_available) {
        ESP_LOGI(TAG, "✅ Camera ready!");
    } else {
        ESP_LOGI(TAG, "Continuing with dummy frames...");
    }

    ESP_LOGI(TAG, "Free heap after init: %lu bytes", esp_get_free_heap_size());

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Starting real-time inference...");
    ESP_LOGI(TAG, "========================================");

    float logit_open = 0.0f;
    float logit_closed = 0.0f;
    float probs[2] = {0.0f, 0.0f};
    
    int frame_count = 0;
    uint32_t start_time = esp_log_timestamp();
    float fps = 0.0f;

    while (1) {
        int ret = capture_grayscale_frame(image_data);
        
        if (ret != 0) {
            generate_dummy_frame(image_data);
        }
        
        // predict_eye returns: logit_open = open score, logit_closed = closed score
        if (predict_eye(image_data, &logit_open, &logit_closed) == 0) {
            // Build logits array: [closed, open]
            float logits[2] = {logit_closed, logit_open};
            softmax(logits, probs, 2);
            
            // probs[0] = closed probability, probs[1] = open probability
            bool eye_open = (probs[1] > probs[0]);
            const char *prediction = eye_open ? "OPEN" : "CLOSED";
            float confidence = eye_open ? probs[1] : probs[0];
            
            if (frame_count % 10 == 0) {
                ESP_LOGI(TAG, "Frame %d | Open: %.2f%% Closed: %.2f%% | %s (%.2f%%)",
                    frame_count,
                    probs[1] * 100,  // open probability
                    probs[0] * 100,  // closed probability
                    prediction,
                    confidence * 100);
            }
        } else {
            ESP_LOGE(TAG, "Inference failed!");
        }
        
        frame_count++;
        
        if (frame_count % 100 == 0) {
            uint32_t elapsed = esp_log_timestamp() - start_time;
            fps = 100.0f / (elapsed / 1000.0f);
            ESP_LOGI(TAG, "📊 FPS: %.1f | Heap: %lu bytes", fps, esp_get_free_heap_size());
            start_time = esp_log_timestamp();
        }
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}