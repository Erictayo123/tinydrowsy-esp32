#include <stdio.h>
#include <cmath>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "model.hpp"
#include "camera.hpp"
#include "uart_transmitter.h"   // ← ADD

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

    // ============================================================
    // ADD #1: Initialize UART transmitter
    // ============================================================
    uart_transmitter_config_t tx_cfg = uart_transmitter_default_config();
    if (uart_transmitter_init(&tx_cfg) != 0) {
        ESP_LOGE(TAG, "Failed to initialize UART transmitter!");
    } else {
        ESP_LOGI(TAG, "✅ UART transmitter ready!");
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

        if (predict_eye(image_data, &logit_open, &logit_closed) == 0) {
            float logits[2] = {logit_closed, logit_open};
            softmax(logits, probs, 2);

            bool eye_open = (probs[1] > probs[0]);
            const char *prediction = eye_open ? "OPEN" : "CLOSED";
            float confidence = eye_open ? probs[1] : probs[0];

            if (frame_count % 10 == 0) {
                ESP_LOGI(TAG, "Frame %d | Open: %.2f%% Closed: %.2f%% | %s (%.2f%%)",
                    frame_count,
                    probs[1] * 100,
                    probs[0] * 100,
                    prediction,
                    confidence * 100);
            }

            // ============================================================
            // ADD #2: Publish frame to UART transmitter
            // ============================================================
            // Note: for now we send the raw grayscale 64×64 buffer.
            // probs[1] = open, probs[0] = closed
            // perclos and alert_state are placeholders — we'll wire
            // them from your existing PERCLOS logic in a later milestone.
            uart_transmitter_publish(
                image_data,
                64 * 64,
                probs[1],       // open_prob
                probs[0],       // closed_prob
                0.0f,           // perclos (TODO)
                0,              // alert_state (TODO)
                fps             // transmitter FPS
            );
        } else {
            ESP_LOGE(TAG, "Inference failed!");
        }

        frame_count++;

        if (frame_count % 100 == 0) {
            uint32_t elapsed = esp_log_timestamp() - start_time;
            fps = 100.0f / (elapsed / 1000.0f);
            ESP_LOGI(TAG, "📊 FPS: %.1f | Heap: %lu bytes", fps, esp_get_free_heap_size());
            start_time = esp_log_timestamp();

            // ============================================================
            // ADD #3: Print UART transmitter stats (every 100 frames)
            // ============================================================
            uart_transmitter_stats_t tx_stats;
            uart_transmitter_get_stats(&tx_stats);
            ESP_LOGI(TAG, "📡 TX: sent=%lu dropped=%lu bytes=%lu err=%lu fps=%.1f",
                     tx_stats.frames_sent,
                     tx_stats.frames_dropped,
                     tx_stats.bytes_sent,
                     tx_stats.send_errors,
                     tx_stats.current_fps);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}