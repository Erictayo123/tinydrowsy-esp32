#include "camera.hpp"
#include "esp_log.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include <cstring>

static const char *TAG = "CAMERA";

// Camera pins for ESP32-S3 CAM (OV2640)
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5
#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13

static camera_config_t camera_config = {
    .pin_pwdn  = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,

    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,

    // XCLK frequency
    .xclk_freq_hz = 20000000,  // 20MHz

    // LED flash pin
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    // Pixel format - use GRAYSCALE for direct grayscale output
    .pixel_format = PIXFORMAT_GRAYSCALE,

    // Frame size - use QQVGA (160x120), then software resize to 64x64
    .frame_size = FRAMESIZE_QQVGA,  // 160x120 - smallest supported

    // JPEG quality (only used for JPEG format)
    .jpeg_quality = 12,
    .fb_count = 1,  // Single frame buffer (memory efficient)
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

static camera_fb_t *fb = nullptr;
static bool camera_initialized = false;

// Software resize function: downscale image to 64x64
static void downscale_to_64x64(uint8_t *src, uint8_t *dst, int src_w, int src_h) {
    // Simple nearest-neighbor downscaling
    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            int src_x = (x * src_w) / 64;
            int src_y = (y * src_h) / 64;
            dst[y * 64 + x] = src[src_y * src_w + src_x];
        }
    }
}

int init_camera(void)
{
    if (camera_initialized) {
        ESP_LOGW(TAG, "Camera already initialized");
        return 0;
    }

    ESP_LOGI(TAG, "Initializing OV2640 camera...");

    // Initialize camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        return -1;
    }

    camera_initialized = true;
    ESP_LOGI(TAG, "✅ Camera initialized successfully!");

    // Get sensor info - FIXED: uncommented and properly declared
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        ESP_LOGI(TAG, "  Sensor: OV2640");
        ESP_LOGI(TAG, "  Pixel Format: GRAYSCALE");
        ESP_LOGI(TAG, "  Frame Size: 160x120 (QQVGA)");
    }

    // Apply settings for better performance - FIXED: s is now declared
    if (s) {
        s->set_framesize(s, FRAMESIZE_QQVGA);
        s->set_pixformat(s, PIXFORMAT_GRAYSCALE);
        s->set_quality(s, 10);
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
        s->set_special_effect(s, 0);
        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);
        s->set_wb_mode(s, 0);
        s->set_exposure_ctrl(s, 1);
        s->set_aec2(s, 1);
        s->set_ae_level(s, 0);
        s->set_aec_value(s, 300);
        s->set_gain_ctrl(s, 1);
        s->set_agc_gain(s, 0);
        s->set_gainceiling(s, GAINCEILING_2X);
        s->set_bpc(s, 0);
        s->set_wpc(s, 1);
        s->set_raw_gma(s, 1);
        s->set_lenc(s, 1);
        s->set_dcw(s, 1);
        s->set_colorbar(s, 0);
    }

    ESP_LOGI(TAG, "✅ Camera settings applied");
    print_camera_info();

    return 0;
}

int capture_grayscale_frame(uint8_t *image_data)
{
    if (!camera_initialized) {
        ESP_LOGE(TAG, "Camera not initialized!");
        return -1;
    }

    if (!image_data) {
        ESP_LOGE(TAG, "Null image buffer!");
        return -1;
    }

    // Capture frame
    fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Failed to capture frame!");
        return -1;
    }

    // Check frame format and size
    if (fb->format != PIXFORMAT_GRAYSCALE) {
        ESP_LOGW(TAG, "Frame format is %d, expected GRAYSCALE", fb->format);
        esp_camera_fb_return(fb);
        fb = nullptr;
        return -1;
    }

    // If frame is already 64x64, copy directly
    if (fb->width == 64 && fb->height == 64 && fb->len == 4096) {
        memcpy(image_data, fb->buf, 64 * 64);
    }
    // Otherwise, downscale from QQVGA (160x120) to 64x64
    else if (fb->width >= 64 && fb->height >= 64) {
        downscale_to_64x64(fb->buf, image_data, fb->width, fb->height);
    } else {
        ESP_LOGE(TAG, "Frame size too small: %dx%d", fb->width, fb->height);
        esp_camera_fb_return(fb);
        fb = nullptr;
        return -1;
    }

    // Return frame buffer to driver
    esp_camera_fb_return(fb);
    fb = nullptr;

    return 0;
}

void print_camera_info(void)
{
    if (!camera_initialized) {
        ESP_LOGW(TAG, "Camera not initialized");
        return;
    }

    sensor_t *s = esp_camera_sensor_get();
    ESP_LOGI(TAG, "=== Camera Info ===");
    if (s) {
        ESP_LOGI(TAG, "  Sensor: OV2640");
    } else {
        ESP_LOGI(TAG, "  Sensor: Unknown");
    }
    ESP_LOGI(TAG, "  Frame Size: 160x120 (QQVGA)");
    ESP_LOGI(TAG, "  Pixel Format: Grayscale");
    ESP_LOGI(TAG, "  Output Size: 64x64 (software resize)");
    ESP_LOGI(TAG, "==================");
}