idf_component_register(
    SRCS "main.cpp" "model.cpp" "camera.cpp"
    INCLUDE_DIRS "."
    EMBED_FILES "tinydrowsy_int8.espdl"
    REQUIRES esp-dl esp32-camera
)

# Add custom linker script
target_link_options(${COMPONENT_LIB} PRIVATE
    -T "${CMAKE_CURRENT_LIST_DIR}/linker.ld"
)

# Add ESP-DL include directories
target_include_directories(${COMPONENT_LIB} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../components/esp-dl
    ${CMAKE_CURRENT_LIST_DIR}/../components/esp-dl/dl
    ${CMAKE_CURRENT_LIST_DIR}/../components/esp-dl/dl/tensor/include
    ${CMAKE_CURRENT_LIST_DIR}/../components/esp-dl/dl/model/include
    ${CMAKE_CURRENT_LIST_DIR}/../components/esp-dl/fbs_loader/include
)