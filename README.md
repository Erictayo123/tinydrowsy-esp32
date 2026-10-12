# TinyDrowsy — ESP32-S3 Dual-Board Drowsiness Detection System

An embedded AI system for real-time eye-state classification, running a quantized INT8 CNN on ESP32-S3 hardware. Split across two boards: a **transmitter** that captures and classifies camera frames, and a **receiver** that displays them on a small TFT as a human-machine interface (HMI).

**Status:** Transmitter working at ~8.4 FPS, 0 dropped frames over 500+ tested frames. Receiver in development.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Hardware](#hardware)
3. [Wiring](#wiring)
4. [Software Stack](#software-stack)
5. [Development Environment Setup](#development-environment-setup)
6. [Building and Flashing](#building-and-flashing)
7. [Architecture and Design Decisions](#architecture-and-design-decisions)
8. [UART Packet Format](#uart-packet-format)
9. [Repository Layout](#repository-layout)
10. [Testing and Verification](#testing-and-verification)
11. [Troubleshooting](#troubleshooting)
12. [Git Workflow](#git-workflow)
13. [Project History and Lessons Learned](#project-history-and-lessons-learned)
14. [Roadmap](#roadmap)

---

## System Overview

```
┌─────────────────────────────┐         ┌─────────────────────────────┐
│   TRANSMITTER               │         │   RECEIVER                  │
│   ESP32-S3-CAM (N16R8)      │         │   ESP32-S3-CAM (N16R8)      │
│                             │         │                             │
│   OV2640 camera ──┐         │         │                             │
│                   │         │         │                             │
│   Capture 64×64   │         │  UART   │   Parse packets             │
│   grayscale       │         │ 921600  │   Upscale 64×64 → 96×96     │
│         │         │         │  TX/RX  │   Draw on ST7735            │
│         ▼         │         ├────────►│   Overlay metadata          │
│   INT8 CNN        │         │         │                             │
│   inference       │         │         │   No inference.             │
│         │         │         │         │   Display only.             │
│         ▼         │         │         │                             │
│   Pack frame +    │         │         │                             │
│   metadata        │         │         │                             │
│         │         │         │         │                             │
│         ▼         │         │         │                             │
│   UART TX         │         │         │                             │
└─────────────────────────────┘         └─────────────────────────────┘
         │                                          ▲
         │                                          │
         └────────── GPIO 17 (TX) ──────────────────┤
                    GPIO 18 (RX) ──────────────────┘
                    GND ───────────────────────────
```

**Why two boards?** The original design was a single ESP32-S3 doing capture + inference + a WiFi web dashboard. The web dashboard approach was abandoned because the WiFi stack consumed too much heap on a device already running a CNN and camera DMA, browser latency made the "live" view feel dead, and the dashboard was fragile (browser caching, mDNS issues).

Two wired boards is simpler, faster, and more robust. The receiver has one job: show the user their face and whether the system sees them as drowsy.

---

## Hardware

| Board | Role | Key specs |
|-------|------|-----------|
| ESP32-S3-CAM (N16R8) | **Transmitter** | OV2640 camera, 16 MB flash, 8 MB PSRAM |
| ESP32-S3-CAM (N16R8) | **Receiver** | No camera used, ST7735 128×160 TFT connected |

Both boards are on the same breadboard, connected with ~20 cm jumpers.

### ST7735 TFT (on receiver)

| ST7735 pin | ESP32-S3 GPIO |
|------------|---------------|
| SCK        | 12 |
| SDA (MOSI) | 11 |
| CS         | 10 |
| A0 (DC)    | 9 |
| RST        | 8 |
| LED        | 3.3V |
| VCC        | 3.3V |
| GND        | GND |

---

## Wiring

### Inter-board UART

| Signal | Transmitter | Receiver |
|--------|-------------|----------|
| TX     | GPIO 17     | GPIO 18  |
| RX     | GPIO 18     | GPIO 17  |
| GND    | GND         | GND      |
| Baud   | 921600      | 921600   |

Three wires. Crossed TX/RX. Common ground.

**Why UART and not SPI/I2C?** UART is the simplest bidirectional protocol with a hardware FIFO and DMA on ESP32-S3. It doesn't require clock sync, doesn't share a bus with the camera or display, and the driver handles flow control if needed later. At 921600 baud we have ~92 KB/s effective throughput — plenty for 4 KB frames at 15 FPS (60 KB/s peak).

---

## Software Stack

| Component | Version | Notes |
|-----------|---------|-------|
| ESP-IDF | **v5.3.3** | LTS, stable |
| ESP-DL | **v3.3.5** | Symlinked into project |
| Xtensa toolchain | `esp-13.2.0_20240530` | GCC 13.2.0 |
| Python env | `idf5.3_py3.12_env` | Auto-managed by `idf.py` |
| Model format | INT8 `.espdl` | 27.41 KB, 10,690 params |
| Host OS | WSL2 Ubuntu | On Windows |
| Project path | `~/projects/tinydrowsy-esp32/` | **Must be on ext4, not `/mnt/c/`** |

### Model

- **Trained:** 99.87% accuracy on held-out test set
- **Exported:** ONNX opset 11
- **Quantized:** INT8, 27.41 KB `.espdl`, <1.2% accuracy loss
- **Input:** `[1, 64, 64, 1]` grayscale, exponent -6
- **Output:** `[1, 2]` logits (class 0 = closed, class 1 = open), exponent -3

---

## Development Environment Setup

### One-time setup

```bash
# Clone ESP-IDF v5.3.3
mkdir -p ~/esp && cd ~/esp
git clone -b v5.3.3 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32s3

# Clone ESP-DL v3.3.5
cd ~/esp
git clone -b v3.3.5 --recursive https://github.com/espressif/esp-dl.git

# Symlink ESP-DL into the project
cd ~/projects/tinydrowsy-esp32/firmware
mkdir -p components
ln -s ~/esp/esp-dl/esp-dl components/esp-dl
readlink -f components/esp-dl   # verify it resolves
```

### Per-terminal setup

```bash
source ~/esp/esp-idf/export.sh
cd ~/projects/tinydrowsy-esp32/firmware
```

### Critical environment rules

1. **Never** build from `/mnt/c/`. ESP-IDF's CMake and file watching break on NTFS/DrvFs. Always work in `~/` (ext4 inside WSL2).
2. **Symlinks survive `git clone`** but break across Windows↔WSL file copies. If you copy the project between filesystems, re-create the `components/esp-dl` symlink.
3. **Toolchain corruption can be silent.** If CMake reports "could not identify compiler," check for 0-byte files in `~/.espressif/tools/` and reinstall.

---

## Building and Flashing

### Standard workflow

```bash
source ~/esp/esp-idf/export.sh
cd ~/projects/tinydrowsy-esp32/firmware

idf.py build
idf.py -p /dev/ttyACM0 flash monitor
# Exit monitor: Ctrl+]
```

### When a build behaves as if source changes aren't being picked up

```bash
idf.py fullclean
rm -rf build managed_components sdkconfig
idf.py set-target esp32s3
idf.py build
```

This nukes everything including `sdkconfig` (regenerated from `sdkconfig.defaults`) and re-downloads managed components.

### Diagnostic commands

```bash
# Check which files got recompiled in the last build
ls -la build/esp-idf/main/CMakeFiles/__idf_main.dir/

# Check the stack frame size of a specific function
xtensa-esp32s3-elf-objdump -d build/tinydrowsy_firmware.elf | grep -A1 "<function_name>:"

# Check the ELF hash (appears in each boot log as "ELF file SHA256")
sha256sum build/tinydrowsy_firmware.elf | head -c 16

# Check file line endings (look for ^M$ = CRLF)
cat -A main/somefile.c | grep -c '\^M$'

# Check when a file was last modified
stat main/somefile.c
```

---

## Architecture and Design Decisions

Every decision below was made for a reason, usually discovered the hard way. Read this before changing anything.

### D1: Two boards, not one

**Decision:** Split into transmitter + receiver over wired UART.

**Rejected:** Single board with WiFi web dashboard. Failed because the WiFi stack + HTTP server + camera DMA + CNN inference together exceeded comfortable heap margins and made latency unpredictable.

**Trade-off accepted:** Requires a second board and 3 wires. Worth it for reliability and latency.

### D2: Raw grayscale frames, not JPEG

**Decision:** Send raw 64×64 grayscale (4,096 bytes) per frame.

**Why:**
- Zero CPU cost on both ends (no encode/decode).
- Zero latency added beyond the UART transfer (~45 ms at 921600 baud).
- Zero heap pressure from JPEG buffers.
- 15 FPS × 4,127 bytes/packet = 62 KB/s, under the ~92 KB/s UART ceiling.

**Rejected:** JPEG at quality 30-50 for 64×64 would compress to ~600 bytes, saving bandwidth. But encode/decode cost is real, and bandwidth isn't the bottleneck. Raw is the right call.

### D3: 31-byte header, all metadata inline

**Decision:** Every packet carries sync bytes, sequence, timestamp, frame length, reserved padding, three probabilities, alert state, FPS, and checksum, followed by 4,096 bytes of raw frame.

**Why:** The receiver is stateless with respect to display content. It parses what arrived and renders. Inline metadata means one packet is one complete state update.

**Sync bytes `0xAA 0x55`:** Chosen because they're easy to detect and unlikely to appear by chance. They can appear in frame data, but the receiver only hunts for sync at packet boundaries, after having consumed exactly `frame_len` bytes from the previous packet.

**Checksum:** XOR of header bytes 2..28. Not cryptographic. Just enough to detect a dropped byte. If it fails, the packet is discarded and the previous frame stays on screen.

### D4: Pointer-based queue, not value-based

**Decision:** `tx_message_t` is 32 bytes and contains a `uint8_t *frame_data` pointer to a PSRAM buffer, not the frame itself.

**Why:** The obvious design (queue holds the entire 4 KB frame) means:
- Queue storage is 12 KB (3 items × 4 KB).
- The task that receives from the queue allocates a 4 KB struct **on its stack**. **This was the original bug** — the task frame was 4,176 bytes on a 6 KB stack, which overflowed the moment `uart_write_bytes` needed scratch space.

Pointer-based means:
- Queue item is 32 bytes → 96 bytes total for depth-3 queue.
- Task stack frame is 80 bytes (verified via `objdump`).
- The frame lives in PSRAM, allocated by the publisher, freed by the TX task after send.

**Ownership rule:** Publisher allocates, TX task frees. Never free in the publisher — the TX task might still be reading when the publisher returns.

### D5: Drop on queue full, never block

**Decision:** `xQueueSend(..., 0)` — zero timeout. If the queue is full, drop the newest frame and increment `frames_dropped`.

**Why:** The inference loop runs at ~8.4 FPS. The TX task is rate-limited to 15 FPS max. In practice the queue never fills. But if it does, blocking the inference loop would make latency worse, not better.

**Observed:** `dropped=0` across 500+ frames in testing.

### D6: TX task priority 3, pinned to core 0

**Decision:** `xTaskCreatePinnedToCore(uart_transmitter_task, "uart_tx", 6144, NULL, 3, &tx.task, 0)`.

**Why:**
- Priority 3 is below the inference loop (`app_main`, priority 5). TX should never preempt inference.
- Core 0 is where `app_main` and ESP-IDF system tasks live. Pinning the TX task to core 0 keeps core 1 free for the camera DMA.
- 6144-byte stack. After the refactor, the frame is 80 bytes, so this is generous. Check `objdump` before increasing.

### D7: ST7735 display, 96×96 upscale (receiver, in development)

**Decision:** Upscale 64×64 frames to 96×96 with nearest-neighbor, use the remaining vertical space for a metadata text band.

**Why 96×96, not 128×128:**
- The ST7735 is 128×160. At 128×128 image, only 32 rows remain for text — too tight.
- At 96×96 image, 64 rows remain for text — comfortable for 4-5 lines.
- Nearest-neighbor 1.5× is cheap: 9,216 pixel writes per frame, ~74k writes/s at 8 FPS, well within SPI speed.
- 96×96 is enough resolution to see whether the face is in frame. Higher res adds nothing for the HMI use case.

### D8: Display freezes on last good frame, no blanking

**Decision:** If no packet arrives within a timeout, keep the last successfully decoded frame on screen. Do not blank, do not show a "no signal" splash.

**Why:** The link is wired. Packet loss should be near-zero. If it happens, freezing is less disruptive than flashing. A subtle "stale" indicator (dimming the FPS text, or a small dot) can be added later if useful.

### D9: Receiver does NOT run inference

**Decision:** The receiver is a pure HMI. It parses packets and displays what the transmitter computed.

**Why:** The receiver has no camera. Running a second copy of the CNN is redundant compute with no new information. The transmitter already sends open/closed probabilities, PERCLOS, alert state, and FPS. The receiver just renders.

**Future-proofing:** If UDP mode is ever added and the link becomes untrusted, the receiver could optionally re-verify frames by running inference itself. Not now. YAGNI.

---

## UART Packet Format

**Total packet size:** 31 bytes header + N bytes frame (N = 4096 for 64×64 grayscale).

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 1 | Sync 0 | Always `0xAA` |
| 1 | 1 | Sync 1 | Always `0x55` |
| 2 | 2 | Sequence | `uint16_t`, wraps at 65535 |
| 4 | 4 | Timestamp | `uint32_t` ms since boot (transmitter clock) |
| 8 | 2 | Frame length | `uint16_t`, expected 4096 |
| 10 | 2 | Reserved | Padding, always 0 |
| 12 | 4 | Open prob | `float`, 0.0–1.0 |
| 16 | 4 | Closed prob | `float`, 0.0–1.0 |
| 20 | 4 | PERCLOS | `float`, 0.0–1.0 |
| 24 | 1 | Alert state | 0=Awake, 1=Soft, 2=Strong, 3=Critical |
| 25 | 4 | FPS | `float`, transmitter FPS |
| 29 | 2 | Checksum | XOR of bytes 2..28 |
| 31 | N | Frame data | Raw grayscale, row-major |

**Byte order:** Little-endian (native Xtensa). Both ends are ESP32-S3, so no conversion needed.

**Receiver sync algorithm:**
1. Hunt for `0xAA` followed by `0x55`.
2. Read the next 29 bytes (header remainder).
3. Verify checksum over bytes 2..28.
4. If checksum fails, discard and return to step 1.
5. Read `frame_len` bytes.
6. Dispatch to display task.

**Important:** Frame data can contain `0xAA 0x55` by coincidence. The receiver must not sync-hunt inside a frame — it consumes exactly `frame_len` bytes after a validated header.

---

## Repository Layout

```
tinydrowsy-esp32/
├── README.md                          # This file
├── .gitignore
└── firmware/                          # ESP-IDF project
    ├── CMakeLists.txt
    ├── partitions.csv                 # 16 MB flash layout
    ├── sdkconfig.defaults             # Kconfig defaults
    ├── components/
    │   └── esp-dl -> ~/esp/esp-dl/esp-dl   # Symlink (NOT committed)
    └── main/
        ├── CMakeLists.txt
        ├── idf_component.yml
        ├── linker.ld                  # Model 16-byte alignment
        ├── main.cpp                   # app_main: inference loop
        ├── model.cpp / model.hpp      # ESP-DL wrapper
        ├── camera.cpp / camera.hpp    # OV2640 driver + 64×64 resize
        ├── uart_transmitter.c / .h    # TX task (working)
        ├── tinydrowsy_int8.espdl      # Quantized model (27.41 KB)
        └── (future) uart_receiver.c / .h
```

---

## Testing and Verification

### Transmitter smoke test

Flash, open monitor, watch for:

```
I (xxxx) UART_TX: UART transmitter initialized
I (xxxx) MAIN: Frame 0 | Open: ... | CLOSED (...)
I (xxxx) MAIN: 📊 FPS: 8.4 | Heap: ...
I (xxxx) MAIN: 📡 TX: sent=100 dropped=0 bytes=412700 err=0 fps=8.6
```

**Pass criteria after 500 frames:**
- `sent=500`, `dropped=0`, `err=0`
- `bytes = 500 × 4127 = 2063500`
- `fps` in range 8.0–9.0
- No `***ERROR***` lines
- Heap stable (no leak)

### End-to-end test (when receiver exists)

1. Power both boards.
2. Transmitter sends within ~2 seconds of boot.
3. Receiver shows a live image + metadata overlay within ~1 second.
4. Metadata matches: receiver FPS tracks transmitter FPS.
5. Power-cycle the transmitter — receiver freezes, then resumes.

### PC-side packet decoding (sanity check before writing receiver)

Use a USB-serial adapter and a Python script:

```python
import serial

ser = serial.Serial('/dev/ttyUSB0', 921600, timeout=1)
buf = b''

while True:
    buf += ser.read(4096)
    while b'\xaa\x55' in buf:
        i = buf.index(b'\xaa\x55')
        if len(buf) - i < 31:
            break
        header = buf[i:i+31]
        frame_len = int.from_bytes(header[8:10], 'little')
        if len(buf) - i < 31 + frame_len:
            break
        frame = buf[i+31:i+31+frame_len]
        buf = buf[i+31+frame_len:]
        print(f"seq={int.from_bytes(header[2:4], 'little')} "
              f"len={frame_len}")
```

---

## Troubleshooting

### "Stack overflow in task X"

**Diagnosis:**
```bash
xtensa-esp32s3-elf-objdump -d build/tinydrowsy_firmware.elf | grep -A1 "<task_function>:"
```
The `entry a1, N` line is the stack frame size. If N > ~1/4 of the task stack, that's the problem.

**Fix:** Move large local variables to `static` (`.bss`) or heap-allocate them in PSRAM.

### "Nothing changed in my build"

**Diagnosis:**
```bash
stat main/somefile.c                                                  # modified?
ls -la build/esp-idf/main/CMakeFiles/__idf_main.dir/somefile.c.obj    # recompiled?
sha256sum build/tinydrowsy_firmware.elf                               # changed?
```

**Fixes:**
```bash
idf.py fullclean && idf.py build
# or, nuclear:
rm -rf build managed_components sdkconfig
idf.py set-target esp32s3
idf.py build
```

### "git push rejected: branch is currently checked out"

**Cause:** Pushing into a non-bare repository whose branch is checked out.

**Fix:** Use a bare repository as the remote (`git init --bare` or `git clone --bare`). See [Git Workflow](#git-workflow).

### Kconfig inline comments

**Symptom:** A line like `CONFIG_FOO=y  # explanation` is ignored.

**Fix:** Kconfig requires comments on their own line.

### "unknown kconfig symbol"

**Cause:** A symbol in `sdkconfig.defaults` belongs to a managed component (`esp32-camera`) or was removed.

**Fix:** Check the current component Kconfig for the correct name. Non-fatal, but clutters the build log.

### "FbsLoader: The address of fbs model in flash is not aligned with 16 bytes"

**Cause:** The embedded `.espdl` model's flash address isn't 16-byte aligned.

**Fix:** Adjust `.rodata` alignment in `linker.ld`. Cosmetic for now — the model still loads and inference works.

---

## Git Workflow

### Remotes

- **`origin`** — local bare hub at `~/git-hubs/tinydrowsy-esp32.git` (WSL-native backup)
- **`github`** — `https://github.com/Erictayo123/tinydrowsy-esp32.git` (public)

### Push pattern

```bash
# Push to both
git push origin develop
git push github develop
```

Or set up a push URL that goes to both:

```bash
git remote set-url --add --push origin ~/git-hubs/tinydrowsy-esp32.git
git remote set-url --add --push origin https://github.com/Erictayo123/tinydrowsy-esp32.git
# Now `git push origin develop` pushes to both
```

### Rules

- **Never** push into a non-bare repo. If you need a new hub, use `git init --bare` or `git clone --bare`.
- **Never** work from `/mnt/c/` in WSL. The old Windows-side copy has been retired; delete it once GitHub confirms your commits.
- **Keep LF line endings.** Fix CRLF with `sed -i 's/\r$//' file` if `cat -A file | grep '\^M$'` shows hits.
- **Never commit** `build/`, `managed_components/`, `sdkconfig`, or `*.bak`.

---

## Project History and Lessons Learned

### Timeline

| Phase | Description | Status |
|-------|-------------|--------|
| Training | CNN on eye-state dataset, 99.87% accuracy | ✅ Done |
| ONNX export | opset 11, validated | ✅ Done |
| Quantization | INT8, 27.41 KB `.espdl`, <1.2% loss | ✅ Done |
| Single-board firmware | Inference, camera, PERCLOS, alerts | ✅ Done |
| Web dashboard | WiFi + HTTP server | ❌ Abandoned |
| **Phase 5: UART transmitter** | **Dual-board TX over wired UART** | **✅ Working** |
| Phase 6: UART receiver | Display + metadata overlay on ST7735 | ⏸️ Not started |
| Phase 7: UDP wireless mode | Optional, same packet format | ⏸️ Optional |
| Phase 8: Documentation | This README | ✅ In progress |

### The two-week file staleness bug (Sep 27 – Oct 12, 2026)

**Symptom:** UART transmitter task (`main/uart_transmitter.c`) stack-overflowed at Frame 0. Every fix attempt — increasing task stack from 4 KB to 6 KB to 32 KB, refactoring `tx_message_t` from a 4,131-byte value struct to a 32-byte pointer struct — produced **identical** crash logs and **identical** ELF SHA256 (`f8d75774c`).

**Investigation:**
- Disassembly showed `entry a1, 0x1050` (4,176 bytes) on `uart_transmitter_task`, matching the *pre-refactor* struct size.
- `git diff` showed `uart_transmitter.c` as unmodified relative to the last commit.
- `stat main/uart_transmitter.c` showed mtime **Sep 27 22:11** — the file had not been touched in 15 days.
- `cat -A main/uart_transmitter.c` revealed `^M$` (CRLF) line endings on every line, indicating a Windows-side editor had last written it.

**Root cause:** The pointer-based `tx_message_t` refactor existed only in the chat session. The file on disk was still the original value-based version. Every build since Sep 27 was building the old code. The 32 KB stack test was meaningless — the compiler allocated a 4 KB frame at task entry regardless of stack size, and the canary tripped immediately.

**Fix:** Opened `main/uart_transmitter.c` in a WSL-native editor (`nano`), replaced its contents with the pointer-based version, verified via `grep -n frame_data` (showed `uint8_t *frame_data;` not `uint8_t frame_data[4096];`), rebuilt. `objdump` then showed `entry a1, 80`. ELF SHA256 changed to `83e3adc50`. Fixed.

**Lessons:**

1. **When a fix "doesn't work," verify the fix is in the binary before debugging anything else.** Two commands pay for themselves every time:
   - `xtensa-esp32s3-elf-objdump -d build/...elf | grep -A1 "<function>:"` — shows the actual stack frame size.
   - `stat src/file.c` — shows the actual modification time.

2. **Identical ELF hashes across "different" builds mean the source didn't change.** Trust the hash, not your editor's save indicator.

3. **Never edit ESP-IDF project files from the Windows side through `\\wsl$` or `/mnt/c`.** The CRLF markers are the fingerprint of this mistake. Use `nano`, `vim`, or VS Code's WSL remote extension opening `~/projects/...` directly.

4. **Never build on `/mnt/c/`.** ESP-IDF's CMake and file watching break on NTFS/DrvFs. Always `~/`.

5. **Symlinks survive `git clone`** but break across Windows↔WSL file copies. Re-create the ESP-DL symlink if you copy the project between filesystems.

6. **Toolchain corruption can be silent.** Check for 0-byte files in `~/.espressif/tools/` if CMake can't identify the compiler.

7. **Kconfig doesn't support inline comments.** Comment must be on its own line.

8. **`static` on local variables in a task moves large buffers to `.bss`** instead of the stack. Essential for anything >1 KB.

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 5 | UART transmitter | ✅ Working |
| Phase 6 | UART receiver + ST7735 display | 🔨 Next |
| Phase 7 | UDP wireless mode (optional) | ⏸️ Deferred |
| Phase 8 | Documentation (this README) | ✅ In progress |

### Phase 6 plan (receiver)

**Responsibilities:**
1. UART RX task: hunt sync bytes, parse 31-byte header, validate checksum, read 4096-byte frame, dispatch to display task via queue.
2. Display task: receive frame + metadata, upscale 64×64 → 96×96 with nearest-neighbor, draw to ST7735 via `esp_lcd` SPI, overlay text band.

**Task architecture (must follow the transmitter's pattern):**
- Pointer-based queue between RX and display tasks. Never pass a 4 KB struct by value.
- Frame buffer in PSRAM (double-buffered: one buffer being drawn, one being filled).
- Both tasks with small stacks. Verify with `objdump` after writing.

**Display layout (128×160 ST7735):**
```
┌──────────────────────────────┐
│                              │
│         96×96 image          │   rows 0..95
│       (upscaled 64×64)       │
│                              │
├──────────────────────────────┤
│ Open:    12.3%               │   rows 96..159 (4-5 lines)
│ Closed:  87.7%               │
│ PERCLOS: 41.2%               │
│ Alert:   SOFT                │
│ FPS:     8.4                 │
└──────────────────────────────┘
```

**Timeout behavior:** If no packet for > 500 ms, freeze last frame and dim the FPS line to indicate staleness. Resume on next good packet.

**Estimated phases within Phase 6:**
1. UART RX-only firmware (parse + print to serial, no display).
2. Add ST7735 driver + solid-color fill.
3. Add frame rendering (no metadata).
4. Add metadata text overlay.
5. Polish: staleness indicator, freeze-on-loss.
