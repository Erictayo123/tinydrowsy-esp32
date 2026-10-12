idf_component_register(
    SRCS "main.cpp" "model.cpp" "camera.cpp" "uart_transmitter.c"
    INCLUDE_DIRS "."
    EMBED_FILES "tinydrowsy_int8.espdl"
    REQUIRES esp-dl esp32-camera driver
)

# Custom linker script (model alignment)
target_link_options(${COMPONENT_LIB} PRIVATE
    -T "${CMAKE_CURRENT_LIST_DIR}/linker.ld"
)

# ESP-DL include directories
target_include_directories(${COMPONENT_LIB} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../components/esp-dl
    ${CMAKE_CURRENT_LIST_DIR}/../components/esp-dl/dl
    ${CMAKE_CURRENT_LIST_DIR}/../components/esp-dl/dl/tensor/include
    ${CMAKE_CURRENT_LIST_DIR}/../components/esp-dl/dl/model/include
    ${CMAKE_CURRENT_LIST_DIR}/../components/esp-dl/fbs_loader/include
)