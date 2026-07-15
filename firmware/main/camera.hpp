#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the OV2640 camera
 * @return 0 on success, -1 on failure
 */
int init_camera(void);

/**
 * @brief Capture a frame and convert to 64x64 grayscale
 * @param[out] image_data 64x64 grayscale buffer (must be allocated by caller)
 * @return 0 on success, -1 on failure
 */
int capture_grayscale_frame(uint8_t *image_data);

/**
 * @brief Print camera configuration
 */
void print_camera_info(void);

#ifdef __cplusplus
}
#endif

#endif // CAMERA_HPP