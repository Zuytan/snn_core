# 🧠 snn-core-compiler

**Industrial Ahead-of-Time (AOT) Compiler for Spiking Neural Networks (NIR Standard).**

Compiles models trained in **PyTorch, snnTorch, SpikingJelly, Norse, or Sinabs** directly into pure, zero-allocation `#![no_std]` **Rust** or **C99** firmware ready to flash on ultra-low-power microcontrollers (ARM Cortex-M0+/M4/M7/M33/M55/M85, RISC-V RV32I/P).

---

## ⚡ Installation

```bash
# Direct local installation
pip install -e tools/snn_compiler

# Or with full NIR support
pip install -e "tools/snn_compiler[all]"
```

---

## 🚀 Usage

### Command-Line Interface (`snn-compile`)

```bash
# 1. Compile NIR model to static Rust network (.rodata in Flash ROM)
snn-compile model.nir -o generated_network.rs

# 2. Compile with INT8 Quantization (scale Q16.16) and CSR Sparsity pruning
snn-compile model.nir --quantize-i8 --threshold 0.001 -o generated_network.rs

# 3. Generate pure C99 header and structures for Keil / STM32Cube / Zephyr
snn-compile model.nir --target c -o model_network.h

# 4. Generate with Arena Memory Planner for 44% SRAM reduction
snn-compile model.nir --arena -o generated_network.rs
```

### Python API

```python
import snn_compiler

# Compile NIR model to Rust
snn_compiler.compile_nir_to_rust(
    "model.nir",
    output_path="generated_network.rs",
    quantize_i8=True,
    sparsity_threshold=0.001,
)

# Compile NIR model to C
snn_compiler.compile_nir_to_c(
    "model.nir",
    output_path="model_network.h",
)
```
