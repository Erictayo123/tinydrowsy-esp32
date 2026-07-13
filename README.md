\# TinyDrowsy ESP32-S3



\*\*Real-time drowsiness detection on ESP32-S3 using embedded AI\*\*



\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

\[!\[ESP-IDF v5.3.1](https://img.shields.io/badge/ESP--IDF-v5.3.1-blue)](https://github.com/espressif/esp-idf)

\[!\[ESP-DL v3.3.5](https://img.shields.io/badge/ESP--DL-v3.3.5-green)](https://github.com/espressif/esp-dl)



\## Overview



TinyDrowsy is an end-to-end embedded AI system for detecting driver drowsiness using eye-state classification. It runs a quantized convolutional neural network (CNN) on an ESP32-S3 with an OV2640 camera, achieving real-time inference at low power and memory footprint.



\## Features



\- 🎯 \*\*Custom CNN\*\* optimized for embedded deployment

\- 📸 \*\*Real-time inference\*\* on OV2640 camera frames

\- ⚡ \*\*INT8 quantization\*\* using ESP-PPQ for 4× memory reduction

\- 🧠 \*\*ESP-DL acceleration\*\* leveraging ESP32-S3's AI instructions

\- 🔋 \*\*Low power\*\* (< 100 mA during inference)

\- 📊 \*\*PERCLOS-based\*\* drowsiness metric

\- 🚨 \*\*Alarm output\*\* with LED and buzzer



\## System Architecture



!\[High-level architecture](docs/architecture/high-level-architecture.png)



The pipeline consists of:

1\. PyTorch training → ONNX export

2\. INT8 quantization using ESP-PPQ

3\. Firmware inference on ESP32-S3

4\. Real-time decision logic



\## Performance



| Metric | Value |

|--------|-------|

| Inference latency | \~50 ms |

| Model size (flash) | 24 KB |

| RAM usage | \~150 KB |

| Accuracy (quantized) | 95.2% |



\## Repository Structure

tinydrowsy-esp32/

├── src/ # Training and export pipeline

├── firmware/ # ESP32-S3 firmware

├── docs/ # Documentation and design decisions

├── dataset/ # Dataset management

├── hardware/ # Schematics and BOM

└── tools/ # Utility scripts



text



\## Quick Start



\### Prerequisites

\- ESP32-S3 development board with OV2640 camera

\- ESP-IDF v5.3.1

\- Python 3.11 with dependencies



\### Setup



```bash

\# Clone the repository

git clone https://github.com/yourusername/tinydrowsy-esp32.git

cd tinydrowsy-esp32



\# Set up Python environment

conda env create -f environment.yml

conda activate tinydrowsy



\# Set up ESP-IDF

get\_idf  # alias to source \~/esp/esp-idf/export.sh

Training

bash

\# Prepare dataset

python src/data/dataset.py --prepare



\# Train model

python src/training/train.py --config configs/default.yaml



\# Export to ONNX

python src/export/export\_onnx.py



\# Quantize to INT8

python src/export/quantize\_espdl.py

Firmware

bash

cd firmware

idf.py build

idf.py -p /dev/ttyUSB0 flash monitor

Design Decisions

See docs/design-decisions/ for detailed explanations of:



Model architecture choices



Quantization strategy



Hardware selection



Firmware architecture



License

MIT License - see LICENSE for details.



Acknowledgments

Espressif for ESP-IDF and ESP-DL



PyTorch for deep learning framework



Dataset: MRL Eye Dataset

