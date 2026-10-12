#include "uart_transmitter.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include <string.h>

/* ============================================================
 * Constants
 * ============================================================ */

static const char *TAG = "UART_TX";

#define PACKET_SYNC_0        0xAA
#define PACKET_SYNC_1        0x55
#define PACKET_HEADER_SIZE   31   /* see uart_packet_t */
#define MAX_FRAME_SIZE       4096 /* 64x64 grayscale */
#define UART_TX_BUFFER_SIZE  8192
#define UART_RX_BUFFER_SIZE  1024

/* ============================================================
 * Internal Message (pointer-based, small)
 *
 * IMPORTANT: This struct is intentionally tiny (~32 bytes).
 * The 4 KB frame lives in PSRAM and is freed by the TX task
 * after the packet is sent. This keeps the task stack small.
 * ============================================================ */

typedef struct {
    uint8_t *frame_data;      /* PSRAM buffer, owned by TX task */
    uint16_t frame_len;
    float    open_prob;
    float    closed_prob;
    float    perclos;
    uint8_t  alert_state;
    float    fps;
    uint32_t timestamp_ms;
} tx_message_t;

/* ============================================================
 * Module State (fully encapsulated)
 * ============================================================ */

static struct {
    uart_transmitter_config_t config;

    QueueHandle_t   queue;
    TaskHandle_t    task;
    bool            initialized;
    bool            running;

    /* Statistics */
    uint32_t frames_sent;
    uint32_t frames_dropped;
    uint32_t bytes_sent;
    uint32_t send_errors;
    float    current_fps;

    /* Internal FPS accounting */
    uint32_t fps_window_start_ms;
    uint32_t fps_window_frames;
} tx = { 0 };

/* ============================================================
 * Packet Building
 * ============================================================ */

static uint16_t compute_checksum(const uint8_t *data, size_t len)
{
    uint16_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum ^= (uint16_t)data[i];
    }
    return sum;
}

/**
 * Build a complete UART packet.
 *
 *   Offset  Size  Field
 *   ------  ----  -----
 *   0       1     Sync 0xAA
 *   1       1     Sync 0x55
 *   2       2     Sequence
 *   4       4     Timestamp (ms)
 *   8       2     Frame length
 *   10      2     Reserved (padding)
 *   12      4     Open prob (float)
 *   16      4     Closed prob (float)
 *   20      4     PERCLOS (float)
 *   24      1     Alert state
 *   25      4     FPS (float)
 *   29      2     Checksum (XOR of bytes 2..28)
 *   ---
 *   31      N     Frame data
 */
static size_t build_packet(const tx_message_t *msg,
                           uint8_t            *buf,
                           size_t              buf_size,
                           uint16_t            sequence)
{
    if (buf_size < PACKET_HEADER_SIZE + msg->frame_len) {
        return 0;
    }

    size_t o = 0;

    buf[o++] = PACKET_SYNC_0;
    buf[o++] = PACKET_SYNC_1;

    memcpy(&buf[o], &sequence,         2); o += 2;
    memcpy(&buf[o], &msg->timestamp_ms, 4); o += 4;
    memcpy(&buf[o], &msg->frame_len,   2); o += 2;

    uint16_t reserved = 0;
    memcpy(&buf[o], &reserved,         2); o += 2;

    memcpy(&buf[o], &msg->open_prob,    4); o += 4;
    memcpy(&buf[o], &msg->closed_prob,  4); o += 4;
    memcpy(&buf[o], &msg->perclos,      4); o += 4;

    buf[o++] = msg->alert_state;

    memcpy(&buf[o], &msg->fps,          4); o += 4;

    /* Checksum covers bytes 2..(PACKET_HEADER_SIZE-2) = 27 bytes */
    uint16_t checksum = compute_checksum(&buf[2], PACKET_HEADER_SIZE - 4);
    memcpy(&buf[o], &checksum,          2); o += 2;

    memcpy(&buf[o], msg->frame_data, msg->frame_len);
    o += msg->frame_len;

    return o;
}

/* ============================================================
 * Transmitter Task
 *
 * Stack frame is small: `packet_buf` is static (.bss) and `msg`
 * holds only a pointer. The 4 KB frame lives in PSRAM and is
 * freed after each send.
 * ============================================================ */

static void uart_transmitter_task(void *arg)
{
    ESP_LOGI(TAG, "UART transmitter task started");
    ESP_LOGI(TAG, "  UART%d  TX=%d  RX=%d  baud=%d",
             tx.config.uart_num, tx.config.tx_pin,
             tx.config.rx_pin, tx.config.baud_rate);
    ESP_LOGI(TAG, "  Max FPS: %d  Queue: %d",
             tx.config.max_fps, tx.config.queue_len);

    /* Static packet buffer — 4127 bytes in .bss, NOT on stack */
    static uint8_t packet_buf[PACKET_HEADER_SIZE + MAX_FRAME_SIZE];

    /* Small local message (holds pointer, not the frame) */
    tx_message_t msg;

    uint16_t sequence = 0;
    uint32_t last_send_ms = 0;
    uint32_t min_interval_ms = 1000 / tx.config.max_fps;

    tx.fps_window_start_ms = esp_log_timestamp();
    tx.fps_window_frames   = 0;

    while (tx.running) {
        /* Block until a frame is available */
        if (xQueueReceive(tx.queue, &msg, pdMS_TO_TICKS(100)) != pdTRUE) {
            continue;
        }

        /* --- Rate limiting --- */
        uint32_t now = esp_log_timestamp();
        uint32_t elapsed = now - last_send_ms;
        if (elapsed < min_interval_ms) {
            vTaskDelay(pdMS_TO_TICKS(min_interval_ms - elapsed));
            now = esp_log_timestamp();
        }

        /* --- Build packet --- */
        size_t pkt_len = build_packet(&msg, packet_buf,
                                       sizeof(packet_buf), sequence);
        if (pkt_len == 0) {
            ESP_LOGW(TAG, "Packet build failed (frame_len=%u)", msg.frame_len);
            tx.send_errors++;
            if (msg.frame_data) {
                heap_caps_free(msg.frame_data);
                msg.frame_data = NULL;
            }
            continue;
        }

        /* --- Send via UART --- */
        int written = uart_write_bytes(tx.config.uart_num,
                                        packet_buf, pkt_len);
        if (written < 0 || (size_t)written != pkt_len) {
            ESP_LOGW(TAG, "UART write failed: %d (expected %u)",
                     written, (unsigned)pkt_len);
            tx.send_errors++;
        } else {
            tx.frames_sent++;
            tx.bytes_sent += pkt_len;
            sequence++;
            tx.fps_window_frames++;
        }

        /* --- Free the frame buffer (we own it now) --- */
        if (msg.frame_data) {
            heap_caps_free(msg.frame_data);
            msg.frame_data = NULL;
        }

        last_send_ms = now;

        /* --- FPS accounting --- */
        if (now - tx.fps_window_start_ms >= 1000) {
            tx.current_fps = (float)tx.fps_window_frames * 1000.0f
                             / (float)(now - tx.fps_window_start_ms);
            tx.fps_window_frames   = 0;
            tx.fps_window_start_ms = now;
        }
    }

    ESP_LOGI(TAG, "UART transmitter task exiting");
    vTaskDelete(NULL);
}

/* ============================================================
 * Public API
 * ============================================================ */

int uart_transmitter_init(const uart_transmitter_config_t *config)
{
    if (tx.initialized) {
        ESP_LOGW(TAG, "Already initialized");
        return 0;
    }
    if (!config) {
        ESP_LOGE(TAG, "config is NULL");
        return -1;
    }

    tx.config = *config;

    /* --- Configure UART driver --- */
    uart_config_t uart_cfg = {
        .baud_rate  = tx.config.baud_rate,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    esp_err_t err;

    err = uart_driver_install(tx.config.uart_num,
                              UART_RX_BUFFER_SIZE,
                              UART_TX_BUFFER_SIZE,
                              0, NULL, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "uart_driver_install failed: %d", err);
        return -1;
    }

    err = uart_param_config(tx.config.uart_num, &uart_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "uart_param_config failed: %d", err);
        uart_driver_delete(tx.config.uart_num);
        return -1;
    }

    err = uart_set_pin(tx.config.uart_num,
                       tx.config.tx_pin,
                       tx.config.rx_pin,
                       UART_PIN_NO_CHANGE,
                       UART_PIN_NO_CHANGE);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "uart_set_pin failed: %d", err);
        uart_driver_delete(tx.config.uart_num);
        return -1;
    }

    /* --- Create queue (small items — only pointers) --- */
    tx.queue = xQueueCreate(tx.config.queue_len, sizeof(tx_message_t));
    if (!tx.queue) {
        ESP_LOGE(TAG, "Queue creation failed");
        uart_driver_delete(tx.config.uart_num);
        return -1;
    }

    /* --- Start task with 6 KB stack (plenty now that msg is a pointer) --- */
    tx.running = true;
    BaseType_t ok = xTaskCreatePinnedToCore(
        uart_transmitter_task,
        "uart_tx",
        32768,           /* 6 KB stack — msg is now 32 bytes, not 4 KB */
        NULL,
        3,              /* low priority (inference runs at 5) */
        &tx.task,
        0               /* Core 0 */
    );
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "Task creation failed");
        vQueueDelete(tx.queue);
        uart_driver_delete(tx.config.uart_num);
        tx.queue = NULL;
        tx.running = false;
        return -1;
    }

    tx.initialized = true;
    ESP_LOGI(TAG, "UART transmitter initialized");
    return 0;
}

int uart_transmitter_publish(const uint8_t *frame_data,
                              size_t         frame_len,
                              float          open_prob,
                              float          closed_prob,
                              float          perclos,
                              int            alert_state,
                              float          fps)
{
    if (!tx.initialized || !tx.running) {
        return -1;
    }
    if (!frame_data || frame_len == 0 || frame_len > MAX_FRAME_SIZE) {
        return -1;
    }

    /* Allocate a copy of the frame in PSRAM.
       The TX task will free it after the packet is sent. */
    uint8_t *copy = (uint8_t *)heap_caps_malloc(
        frame_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!copy) {
        ESP_LOGW(TAG, "Frame alloc failed (len=%u)", (unsigned)frame_len);
        tx.frames_dropped++;
        return -1;
    }
    memcpy(copy, frame_data, frame_len);

    tx_message_t msg = {
        .frame_data   = copy,
        .frame_len    = (uint16_t)frame_len,
        .open_prob    = open_prob,
        .closed_prob  = closed_prob,
        .perclos      = perclos,
        .alert_state  = (uint8_t)alert_state,
        .fps          = fps,
        .timestamp_ms = esp_log_timestamp(),
    };

    if (xQueueSend(tx.queue, &msg, 0) != pdTRUE) {
        heap_caps_free(copy);
        tx.frames_dropped++;
        return -1;
    }
    return 0;
}

void uart_transmitter_get_stats(uart_transmitter_stats_t *stats)
{
    if (!stats) return;
    stats->frames_sent    = tx.frames_sent;
    stats->frames_dropped = tx.frames_dropped;
    stats->bytes_sent     = tx.bytes_sent;
    stats->send_errors    = tx.send_errors;
    stats->current_fps    = tx.current_fps;
}

void uart_transmitter_stop(void)
{
    if (!tx.initialized) return;

    tx.running = false;
    vTaskDelay(pdMS_TO_TICKS(200));   /* let task exit */

    if (tx.queue) {
        vQueueDelete(tx.queue);
        tx.queue = NULL;
    }
    uart_driver_delete(tx.config.uart_num);
    tx.initialized = false;
    ESP_LOGI(TAG, "UART transmitter stopped");
}
