# 🔌 Industrial C / C++ Integration for `snn_core`

This directory demonstrates how to integrate the `snn_core` library into an embedded C or C++ project (FreeRTOS, Zephyr RTOS, STM32CubeIDE, ESP-IDF) without writing a single line of Rust in your application firmware.

---

## 🛠️ Step 1: Compile `snn_core` as a Static C Library (`libsnn_core.a`)

### On Host PC (macOS / Linux):
```bash
cargo rustc --release --crate-type staticlib
```
The resulting archive is generated in `target/release/libsnn_core.a`.

### For Bare-Metal ARM Cortex-M4 Targets:
```bash
cargo rustc --release --target thumbv7em-none-eabi --no-default-features --crate-type staticlib
```

---

## 🏗️ Step 2: Include the C Header in Your Project

Include the public header [`include/snn_core.h`](../../include/snn_core.h) in your C source files:
```c
#include "snn_core.h"
```

---

## 🚀 Step 3: Build and Run the C Application

Using CMake:
```bash
mkdir build && cd build
cmake ..
make
./c_embedded_demo
```

Or directly using `gcc` / `clang`:
```bash
gcc -std=c99 main.c -I../../include -L../../target/release -lsnn_core -lm -o c_embedded_demo
./c_embedded_demo
```
