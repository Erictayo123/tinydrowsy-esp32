#include "esp_log.h"

static const char *TAG = "RECEIVER";

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  TinyDrowsy Receiver Firmware v0.1");
    ESP_LOGI(TAG, "  HMI display for drowsiness detection");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Receiver boot OK — awaiting Phase 6 M2");
}
