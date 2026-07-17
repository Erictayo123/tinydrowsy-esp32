# TinyDrowsy ESP32-S3

**Real-time drowsiness detection on ESP32-S3 using embedded AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ESP-IDF v5.3.3](https://img.shields.io/badge/ESP--IDF-v5.3.3-blue)](https://github.com/espressif/esp-idf)
[![ESP-DL v3.3.5](https://img.shields.io/badge/ESP--DL-v3.3.5-green)](https://github.com/espressif/esp-dl)
[![ESP32-S3](https://img.shields.io/badge/ESP32-S3-red)](https://www.espressif.com/en/products/socs/esp32-s3)

## Overview

TinyDrowsy is an end-to-end embedded AI system for detecting driver drowsiness using eye-state classification. It runs a quantized convolutional neural network (CNN) on an ESP32-S3 with an OV2640 camera, achieving real-time inference with only 27.4 KB of model size.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Training Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│  PyTorch CNN  →  ONNX Export  →  INT8 Quantization  →  .espdl    │
│  (99.87% acc)    (ONNX opset 11)  (ESP-PPQ)            (27.4 KB) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Deployment Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  ESP32-S3  ←  OV2640 Camera  ←  ESP-DL v3.3.5  ←  ESP-IDF        │
│  (8.8 FPS)    (64×64 grayscale) (INT8 inference)   (v5.3.3)      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- 🎯 **99.87% test accuracy** with only 10,690 parameters
- ⚡ **27.4 KB INT8 model** (roughly 100× smaller than MobileNetV2)
- 📸 **Real-time inference** at ~8.8 FPS on ESP32-S3
- 🧠 **ESP-DL acceleration** leveraging ESP32-S3's AI instructions
- 📊 **PERCLOS-based** drowsiness metric (future work)
- 🔋 **Stable memory usage** with 8MB PSRAM

## Repository Structure

```
tinydrowsy-esp32/
├── src/                    # Training and export pipeline
│   ├── data/                # Dataset handling (MRL+CEW, DDD, selfies)
│   ├── models/               # TinyEyeNetV2 model definition
│   ├── training/              # Training scripts
│   └── export/                 # ONNX export and quantization
├── firmware/                 # ESP32-S3 firmware
│   ├── main/
│   │   ├── main.cpp            # Real-time inference loop
│   │   ├── model.cpp             # ESP-DL model loading and inference
│   │   ├── model.hpp              # Model interface
│   │   ├── camera.cpp              # OV2640 camera driver
│   │   └── camera.hpp               # Camera interface
│   ├── partitions.csv         # Custom 16MB partition table
│   └── sdkconfig.defaults      # ESP-IDF configuration
├── dataset/                  # Dataset management
├── docs/                      # Documentation and design decisions
├── hardware/                   # Schematics and BOM
└── tools/                        # Utility scripts
```

## Quick Start

### Prerequisites
- ESP32-S3 development board with OV2640 camera
- ESP-IDF v5.3.3
- Python 3.11 with dependencies

### Setup

```bash
# Clone the repository
git clone https://github.com/Erictayo123/tinydrowsy-esp32.git
cd tinydrowsy-esp32

# Set up Python environment
conda env create -f environment.yml
conda activate tinydrowsy

# Set up ESP-IDF
source ~/esp/esp-idf/export.sh
```

### Training

```bash
# Prepare dataset
python src/data/prepare_dataset.py

# Train model
python src/training/train.py

# Export to ONNX
python src/export/export_onnx.py

# Quantize to INT8
python src/export/quantize_espdl.py
```

### Firmware Deployment

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

## Performance

| Metric | Value |
|---|---|
| Model Parameters | 10,690 |
| Model Size (quantized) | 27.4 KB |
| Test Accuracy | 99.87% |
| Inference Latency | ~113 ms |
| FPS | ~8.8 |
| RAM Usage | ~156 KB |
| Flash Usage | ~1.08 MB |

## Hardware Requirements

- ESP32-S3-WROOM-1-N16R8 (16MB Flash, 8MB PSRAM)
- OV2640 camera module
- USB-C cable for power and serial

## Design Decisions

For detailed explanations of design choices, see the design decisions documentation:

- [Model Architecture](docs/design-decisions/01-model-architecture.md) — Why TinyEyeNetV2?
- [Quantization Strategy](docs/design-decisions/02-quantization-strategy.md) — Why INT8 for ESP32-S3?
- [Dataset Strategy](docs/design-decisions/03-dataset-strategy.md) — Training on multiple sources
- [Hardware Selection](docs/design-decisions/04-hardware-selection.md) — Why ESP32-S3?
- [Firmware Architecture](docs/design-decisions/05-firmware-architecture.md) — ESP-DL integration overview

Full ESP-DL integration reference: [docs/deployment/esp-dl-integration.md](docs/deployment/esp-dl-integration.md)

## Troubleshooting

**Camera Not Detected**
- Check GPIO pin configuration in `camera.cpp`
- Verify power supply (camera needs stable 3.3V)
- Try reducing XCLK frequency

**Stack Overflow**
- Increase `CONFIG_ESP_MAIN_TASK_STACK_SIZE` in `sdkconfig`
- Use static buffers for large arrays

## Acknowledgments

- Espressif for ESP-IDF and ESP-DL
- PyTorch for the deep learning framework
- MRL Eye Dataset and DDD (Driver Drowsiness Dataset)

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built for the embedded AI community.
