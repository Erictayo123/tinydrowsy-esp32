#ifndef MODEL_HPP
#define MODEL_HPP

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the ESP-DL model
 * @return 0 on success, -1 on failure
 */
int init_model(void);

/**
 * @brief Print model input/output tensor information
 */
void print_model_info(void);

/**
 * @brief Run inference on a 64x64 grayscale image
 *
 * @param image_data Pointer to 64*64 uint8 grayscale pixels
 * @param logit_open Output logit for Open class
 * @param logit_closed Output logit for Closed class
 * @return 0 on success, -1 on failure
 */
int predict_eye(const uint8_t *image_data,
                float *logit_open,
                float *logit_closed);

#ifdef __cplusplus
}
#endif

#endif // MODEL_HPP
