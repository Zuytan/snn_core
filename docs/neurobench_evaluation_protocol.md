# 🔬 Standardized NeuroBench & MLCommons Tiny Evaluation Protocol for `snn_core`

> **Status:** protocol and metric definitions only. No result has been submitted to NeuroBench or MLCommons, and the energy figures produced by `metrics.rs` are estimates from fixed constants, not measurements.

This document formalizes the methodology, metrics, and reproduction instructions for public evaluation of **`snn_core`** in compliance with international **NeuroBench** and **MLCommons Tiny (TinyMLPerf)** standards.

---

## 🏛️ 1. Core NeuroBench Principles & Official Metrics

NeuroBench evaluates the algorithmic, temporal, and energy efficiency of neuromorphic systems across 5 key dimensions:

| NeuroBench Metric | Definition in `snn_core` | Rust / C99 Implementation |
| :--- | :--- | :--- |
| **Flash Footprint (.rodata)** | Compressed synaptic weights (CSR Fix32 or Quantized INT8) and static LIF configurations. | `MODEL_DESC` / `SnnModelDescriptor` |
| **SRAM Arena Footprint** | Dynamic runtime memory ($V_{\text{mem}}(t)$, refractory counters, shared accumulators $\max(N_l)$, ping-pong spike buffers). Strict zero heap. | `ExecutionContext` / `SnnExecutionContext` |
| **Spike Sparsity ($S_t$)** | Fraction of silent neurons over time: $S_t = 1 - \frac{\sum s_i(t)}{N}$. | [`snn_core::domain::metrics::SparsityMeter`](../src/domain/metrics.rs) |
| **Synaptic Operations (SynOps)** | Total synaptic accumulations triggered by spike events ($s_i(t) = 1$): pure **AC (Additions)** vs dense **MAC (Multiply-Accumulate)**. | [`snn_core::metrics::SynOpsTracker`](../src/domain/metrics.rs) |
| **CPU energy estimate** | $E = \text{cycles} \times P_{\text{active}} / f_{\text{clock}}$, from cycles counted on the target (DWT) and an active-power figure. An estimate; energy claims need a current probe (PPK2, Joulescope). Per-operation pJ models are deprecated: their constants describe neuromorphic ASICs, not MCUs. | [`snn_core::metrics::EnergyEstimator::cpu_active_nanojoules`](../src/domain/metrics.rs) |
| **Latency-to-Decision** | Number of time-steps ($t_{\text{decision}}$) required to exceed the classification readout confidence threshold. | [`snn_core::domain::metrics::LatencyTracker`](../src/domain/metrics.rs) |

---

## 🎯 2. Standardized Reference Tasks

### A. Spiking Heidelberg Digits (SHD) — Bio-Acoustic / Event-Based Audio
* **Input**: 700 biomimetic cochlear channels (1 kHz audio spike stream).
* **Network Topology**:
  - Layer 1: Dense Feedforward CSR 700 $\to$ 128 LIF ($V_{\text{th}} = 1.0$, $\tau_{\text{mem}} = 20\,\text{ms}$, Soft Reset).
  - Layer 2: Recurrent 128 LIF with feedback synapses $W_{\text{rec}}$ ($128 \times 128$).
  - Readout: Dense Linear 128 $\to$ 20 classes with native CTZ instruction decoding.
* **Run Benchmark**:
  ```bash
  cargo run --release --example 05_synthetic_cochlea_demo
  ```

### B. DVS-Gesture / N-MNIST — 2D Neuromorphic Vision
* **Input**: Dynamic Vision Sensor (DVS) event stream, Address-Event Representation ($x, y, p, t$).
* **Network Topology**:
  - DMA Decoding: [`AerDmaRingBuffer`](../src/domain/aer.rs) into [`SpikeBitset`](../src/domain/bitset.rs).
  - Layer 1: `Conv2dSynapses` (1 in_channel, 8 out_channels, 3x3 kernel, stride 1, padding 1).
  - Layer 2: `MaxPool2dConfig` ($2 \times 2$ bitwise logical reduction, stride $2 \times 2$, zero FPU).
  - Layer 3: Dense CSR $8 \times 16 \times 16 \to 128$ LIF.
  - Layer 4: Readout 128 $\to$ 11 gestures.

---

## ⚙️ 3. AOT Deployment Protocol & Metric Collection

### Step 1: Training & Standard Export (PyTorch / snnTorch / SpikingJelly)
Train your model and export it to standard **ONNX** or **NIR** format:
```python
import torch
import nir

# Standard NIR export
nir_graph = export_to_nir(trained_snn_model)
nir.write("model.nir", nir_graph)
```

### Step 2: PTQ Quantization & AOT Compilation
Compile the model into bare-metal Rust (`#![no_std]`) or C firmware tables (linked with `libsnn_core.a`) with real dataset calibration:
```bash
# Bare-metal Rust compilation with INT8 quantization and KL calibration
snn-compile model.onnx \
    -o src/generated_model.rs \
    --quantize-i8 \
    --calibration kl \
    --calibrate-data calib_dataset.npy \
    --target rust

# Or C tables for projects linking libsnn_core.a (STM32CubeIDE / Keil / IAR):
snn-compile model.onnx \
    -o firmware/generated_model.h \
    --quantize-i8 \
    --calibration kl \
    --calibrate-data calib_dataset.npy \
    --target c
```

### Step 3: Silicon & Algorithmic Metric Collection
Execute the compiled model within the `snn_core` metric evaluator:
```rust
use snn_core::metrics::{EnergyEstimator, SparsityMeter, SynOpsTracker};

let mut synops = SynOpsTracker::new();
let mut sparsity = SparsityMeter::new();

// NeuroBench temporal evaluation loop (T time-steps)
for t in 0..T_STEPS {
    let spikes_emitted = run_one_step(t); // your model step; returns the number of output spikes
    synops.record_synaptic_spikes(fan_out * spikes_emitted);
    sparsity.record_tick(MODEL_OUTPUT_NEURONS, spikes_emitted);
}

println!("Total SynOps: {}", synops.total_synops());
println!("Sparsity: {:.2}%", sparsity.sparsity_percent());
// On the target: cycles from the DWT counter, active power in µW, clock in Hz.
println!("CPU-active energy estimate: {} nJ",
         EnergyEstimator::cpu_active_nanojoules(cycles, active_power_uw, clock_hz));
```

---

## 📊 4. MLCommons Tiny (TinyMLPerf) Certification Guidelines

To prepare `snn_core` benchmarks for the official MLCommons Tiny committee (nothing has been submitted):
1. **Target Hardware**: STM32 Nucleo-L4R5ZI reference board or Nordic nRF5340 / nRF52840.
2. **Physical Measurement**: Connect a calibrated power analyzer (Nordic Power Profiler Kit II - PPK2) to the microcontroller's VDD supply rail.
3. **Target Specifications**:
   - Deep sleep (`WFI`) between events enabled by $> 90\%$ temporal sparsity.
   - Energy per inference compared with CMSIS-NN dense baselines on the same task and accuracy (target to be demonstrated, not assumed).
