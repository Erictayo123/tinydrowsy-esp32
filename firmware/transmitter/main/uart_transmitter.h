#ifndef UART_TRANSMITTER_H
#define UART_TRANSMITTER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * UART Transmitter Configuration
 * ============================================================ */

typedef struct {
    int      uart_num;       /* UART_NUM_1 or UART_NUM_2 */
    int      tx_pin;         /* TX GPIO */
    int      rx_pin;         /* RX GPIO (unused but required by driver) */
    int      baud_rate;      /* 921600 recommended */
    uint16_t queue_len;      /* Frame queue depth (3 default) */
    uint8_t  max_fps;        /* Rate limit (15 default) */
} uart_transmitter_config_t;

/* ============================================================
 * Transmission Statistics
 * ============================================================ */

typedef struct {
    uint32_t frames_sent;      /* Successfully sent frames */
    uint32_t frames_dropped;   /* Dropped (queue full) */
    uint32_t bytes_sent;       /* Total bytes transmitted */
    uint32_t send_errors;      /* UART write errors */
    float    current_fps;      /* Actual transmission FPS */
} uart_transmitter_stats_t;

/* ============================================================
 * Public API
 * ============================================================ */

/**
 * @brief Initialize the UART transmitter
 * @param config Configuration (must not be NULL)
 * @return 0 on success, -1 on failure
 */
int uart_transmitter_init(const uart_transmitter_config_t *config);

/**
 * @brief Publish a frame with metadata (non-blocking)
 *
 * Copies frame data to internal queue. Caller can immediately
 * reuse their buffer. Actual UART transmission happens in a
 * low-priority background task.
 *
 * @param frame_data   Pointer to 64x64 grayscale frame
 * @param frame_len    Length of frame data (expects 4096)
 * @param open_prob    Open probability (0.0 - 1.0)
 * @param closed_prob  Closed probability (0.0 - 1.0)
 * @param perclos      PERCLOS value (0.0 - 1.0)
 * @param alert_state  0=Awake, 1=Soft, 2=Strong, 3=Critical
 * @param fps          Transmitter FPS (for receiver display)
 * @return 0 on success, -1 if queue full (frame dropped)
 */
int uart_transmitter_publish(const uint8_t *frame_data,
                              size_t         frame_len,
                              float          open_prob,
                              float          closed_prob,
                              float          perclos,
                              int            alert_state,
                              float          fps);

/**
 * @brief Get current transmission statistics
 * @param stats Output structure
 */
void uart_transmitter_get_stats(uart_transmitter_stats_t *stats);

/**
 * @brief Stop the transmitter and free resources
 */
void uart_transmitter_stop(void);

/* ============================================================
 * Default Configuration Helper
 * ============================================================ */

#define UART_TX_DEFAULT_UART_NUM    1
#define UART_TX_DEFAULT_TX_PIN      17
#define UART_TX_DEFAULT_RX_PIN      18
#define UART_TX_DEFAULT_BAUD_RATE   921600
#define UART_TX_DEFAULT_QUEUE_LEN   3
#define UART_TX_DEFAULT_MAX_FPS     15

static inline uart_transmitter_config_t uart_transmitter_default_config(void)
{
    uart_transmitter_config_t cfg = {
        .uart_num  = UART_TX_DEFAULT_UART_NUM,
        .tx_pin    = UART_TX_DEFAULT_TX_PIN,
        .rx_pin    = UART_TX_DEFAULT_RX_PIN,
        .baud_rate = UART_TX_DEFAULT_BAUD_RATE,
        .queue_len = UART_TX_DEFAULT_QUEUE_LEN,
        .max_fps   = UART_TX_DEFAULT_MAX_FPS,
    };
    return cfg;
}

#ifdef __cplusplus
}
#endif

#endif /* UART_TRANSMITTER_H */