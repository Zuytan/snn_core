# 🧠 `snn_core`

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-blue.svg)](LICENSE.md)
[![Crates.io](https://img.shields.io/badge/crates.io-v0.4.0-orange.svg)](https://crates.io/crates/snn_core)
[![Documentation](https://docs.rs/snn_core/badge.svg)](https://docs.rs/snn_core)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](https://docs.rs/snn_core)

**`snn_core`** is a bare-metal neuromorphic engine written in pure Rust, engineered for ultra-efficient Spiking Neural Network (SNN) and Bit-Parallel Temporal Network (**BPTN**) inference on resource-constrained microcontrollers and edge architectures (ARM Cortex-M0+/M3/M4 without FPU `thumbv7em-none-eabi`, RISC-V, ESP32).

---

## ⚡ Technology Showcase: BPTN Validated on Real Silicon (Wio-E5)

> 🏆 **Cycle-Accurate Physical Measurements on Seeed Studio Wio-E5 (STM32WLE5JC @ 48 MHz):**  
> * **256 temporal synapses evaluated in 170.69 µs** (8,193 CPU cycles) over a 64-tick window (1 s of audio).
> * **0.50 CPU cycle / synapse / tick** amortized (with **ZERO hardware multiplications, 0 MACs**).
> * **2.37 nJ / inference** ($\mathbf{2.3\times}$ faster than LIF SNN, $\mathbf{7.1\times}$ faster than Micro-GRU).
> * **Estimated battery life > 10 years** on a standard CR2032 coin cell (idle standby CPU load: $0.017\%$).
> * 📄 **Full Physical Measurement Report:** [`docs/wio_e5_silicon_benchmark_report.md`](docs/wio_e5_silicon_benchmark_report.md).
> * 🧪 **Run the silicon verification suite directly on your connected Wio-E5:**
>   ```bash
>   python3 tools/run_silicon_tests.py
>   ```

---

## 🚀 Core Pillars & Architectural Specifications

* **Strict Fix32 & Fix16 Saturating Arithmetic Without Numerical Bias**:
  - [`Fix32`](src/domain/numeric.rs) (Saturating Q16.16 with unbiased round-to-nearest): High numerical precision avoiding power-hungry software floating-point emulation.
  - [`Fix16`](src/domain/numeric.rs) (Saturating Q8.8): Cuts SRAM memory footprint in half with zero hardware panic risks.
* **Invariant Division Optimization (`FastDivider`)**:
  - [`FastDivider`](src/domain/hardware.rs): Replaces expensive hardware division (`/`) and modulo (`%`) instructions in 1D/2D convolution loops via precomputed invariant reciprocal $\lceil 2^{64}/D \rceil$, eliminating the 35–50 cycle penalties of software `__aeabi_uidiv` routines on Cortex-M0+/RISC-V.
* **Zero-Allocation Memory Planner (`ActivationArena`) & Dense Layouts**:
  - [`ActivationArena`](src/domain/arena.rs): Shared ping-pong alternate memory planner with strict zero dynamic heap allocation. Minimizes deep network SRAM footprint by reusing synaptic accumulators dimensioned to $\max(N_l)$ instead of $\sum N_l$ (saving $33\%$ to $66\%$ on accumulator memory across network depth $L$).
  - [`ArenaLifPopulation`](src/domain/arena.rs): Membrane-only state LIF neurons (**5 bytes / neuron** in `Fix32`, **3 bytes / neuron** in `Fix16`).
  - [`CompactLif16Population`](src/domain/compact_population.rs): **5 bytes per neuron** (`Fix16` membrane $2\text{B} + \text{acc } 2\text{B} + \text{refractory } 1\text{B}$).
  - [`CompactLifPopulation`](src/domain/compact_population.rs): **9 bytes per neuron** in high-precision `Fix32` ($4\text{B} + 4\text{B} + 1\text{B}$), compared to conventional 16–24 byte unoptimized layouts.
  - [`LazyDecayLifPopulation`](src/domain/lazy_population.rs): Event-driven lazy decay engine where membrane potentials are updated ($V(t) = V(t_{\text{last}}) \cdot \text{LUT}[\Delta t] + W$) **only upon incoming spike arrival**. Silent neurons consume **0 clock cycles** and 0 dynamic energy.
* **Silicon Hardware Acceleration (ARM DSP, Helium MVE, RISC-V DSP)**:
  - **ARM Cortex-M DSP** (`feature = "cortex-m-dsp"`): `smlad` (dual 16x16 MAC in 1 cycle), `qadd16` (dual 16-bit saturating add), `ssat_16` (1-cycle hardware saturation).
  - **ARMv8.1-M Helium (MVE)** (`feature = "cortex-m-helium"`): 128-bit vector primitives (`vqadd_4x32`, `vqadd_8x16`) with native MVE hardware acceleration on Cortex-M55 / Cortex-M85 and branchless saturating software fallback.
  - **RISC-V Packed SIMD / DSP** (`feature = "riscv-dsp"`): Hardware `kadd16` and `kmada` instructions for RV32P architectures (with portable software fallback).
* **Industrial RTOS & Embedded Ecosystem Integration**:
  - **Zephyr RTOS** (`zephyr/module.yml`, `zephyr/Kconfig`, `zephyr/CMakeLists.txt`): Native Zephyr module with full application sample.
  - **ESP-IDF** (`components/snn_core/idf_component.yml`): Official managed component for ESP32 / ESP32-S3 / ESP32-C3 / ESP32-C6.
  - **CMSIS-Pack** (`snn_core.pdsc`): ARM standard software pack for Keil MDK, IAR Embedded Workbench, and STM32CubeIDE.
* **Open Architecture via Traits & Macros**:
  - [`NeuronDynamics`](src/traits.rs): Configurable integration dynamics and membrane equations.
  - [`SpikingPopulation`](src/traits.rs): Static and dynamic neuronal population management (with hardware CTZ event-driven spike dispatch [`propagate_spikes_words`](src/traits.rs)).
  - [`SynapseMatrix`](src/traits.rs): Contiguous event-driven propagation in compressed sparse row (CSR) or Flash ROM.
  - [`DelayedSpikingLayer`](src/domain/layers.rs), [`ChainedSpikingNetwork`](src/domain/layers.rs), [`ArenaSpikingNetwork`](src/domain/arena.rs) and macro [`arena_spiking_network!`](src/domain/arena.rs): Zero-overhead layer chaining without `RefCell` indirections.
* **SNN Topologies & Embedded Convolutions**:
  - **Recurrent Reservoirs** ([`GenericStaticNetwork`](src/domain/static_network.rs)): Non-linear temporal dynamics and echo state properties.
  - **1D & 2D Spatial Convolutions** ([`Conv1dSynapses`](src/domain/layers.rs), [`Conv2dSynapses`](src/domain/layers.rs)): Native support for `ConvDataLayout::Chw` and `ConvDataLayout::Hwc` (SIMD/CMSIS-friendly contiguous accumulations without divisions) for time series, audio spectrograms, and Dynamic Vision Sensors (DVS).
* **Integrated Biophysical Neuron Models with Dual Reset Modes**:
  - **LIF** ([`LifDynamics`](src/domain/models/lif.rs)): *Leaky Integrate-and-Fire* with [`ResetMode::Hard`](src/domain/models/lif.rs) (clamp to $V_{\text{reset}}$) and [`ResetMode::Subtract`](src/domain/models/lif.rs) (*Soft Reset* $V - V_{\text{th}}$ preserving temporal residual information under high firing rates).
  - **ALIF** ([`AlifDynamics`](src/domain/models/alif.rs)): *Adaptive LIF* with dynamic threshold accommodation ($V_{\text{base}} + A$) and bias.
  - **CUBA-LIF** ([`CubaLifDynamics`](src/domain/models/cuba_lif.rs)): *Current-Based LIF* with bi-exponential synaptic filtering ($\tau_{\text{syn}}, \tau_{\text{mem}}$), strictly compliant with the NIR / snnTorch standard.
  - **Izhikevich** ([`IzhikevichDynamics`](src/domain/models/izhikevich.rs)): Non-linear 2-variable coupled model ($v, u$) evaluated with dual sub-stepping without Q16.16 overflow.
* **Compact Spike Delay Lines**:
  - [`StaticSpikeWordDelayLine`](src/domain/delay_ring.rs): Circular ring packed into `u32` bit words (8x less SRAM than boolean arrays) for bio-plausible delay-tuned synapses.
* **On-Device Learning & Decoders (Classification & Regression)**:
  - **Causal Event-Driven Zero-Heap STDP** ([`StaticStdpLearner`](src/domain/stdp.rs), [`RewardModulatedStdp`](src/domain/stdp.rs)): Causal Hebbian plasticity with reward modulation without heap allocation (`vec!` eliminated).
  - **Classification Readout** ([`TrainableReadoutLayer`](src/application/readout.rs), [`StaticReadoutLayer`](src/application/readout.rs)).
  - **Online Regression Readout** ([`TrainableRegressionReadout`](src/application/readout.rs)): Delta/LMS and Normalized LMS (NLMS) gradient descent in fixed-point Q16.16 for streaming time series (e.g. NARMA-10).
* **Unified Inference Engine & Dynamic Memory Arena (`ModelDescriptor` & `ExecutionContext`)**:
  - [`ModelDescriptor`](src/domain/engine.rs) & [`ExecutionContext`](src/domain/engine.rs): High-level standardized interface (akin to TFLite Micro / CMSIS-NN) strictly separating immutable static topology/weights in Flash ROM (`.rodata`) from dynamic working memory partitioned in SRAM without heap allocations.
  - Native support for **recurrent synapses $W_{\text{rec}}$** with temporal state retention $s_l(t-1)$ without heap allocation.
  - Full support for **2D Convolutional layers (`new_conv2d`)** and **2D Spatial Max-Pooling (`new_max_pool_2d`)** via bitwise disjunctive reduction.
  - Zero-copy asynchronous DMA / interrupt streaming via [`AerDmaRingBuffer`](src/domain/aer.rs).
* **Industrial Dual NIR & ONNX Interoperability (AOT Compiler)**:
  - **Python Package `snn-core-compiler`** (`tools/snn_compiler`, CLI `snn-compile`): Directly converts standard **ONNX** (`.onnx`) and **NIR** (`.nir`) computation graphs (PyTorch / SpikingJelly / snnTorch) into static Rust firmware (`.rodata`) or **autonomous standalone C99** (`--target c99-standalone`, zero Cargo/Rust dependencies).
  - Post-Training Quantization (**PTQ**) with calibration on real datasets (`--calibrate-data dataset.npy`) via Kullback-Leibler (KL) divergence or symmetric Min-Max.
  - SRAM arena planning through **interval coloring and liveness analysis** (*Greedy Memory Planner*), ensuring support for arbitrary DAG topologies and residual skip connections without memory collisions.
  - Exact **SynOps** accounting (pure AC accumulations vs dense MACs), temporal sparsity, and **silicon energy estimation (pJ / nJ)** via [`HardwareProfile`](src/domain/metrics.rs), compliant with the [NeuroBench](docs/neurobench_evaluation_protocol.md) metrology suite.
* **Optional Silicon-Native BPTN Engine (Bit-Parallel Temporal Neural Network)** (`feature = "bptn"`):
  - Silicon-native research track isolated under a dedicated feature flag: Word-RAM 64-bit vectorization (`AND`, `XOR`, `POPCNT`), bistable R-BPTN latches, and dyadic multi-scale H-BPTN pyramids. Decoupled from the standard production neuromorphic core.

---

## 📦 Quick Start

Add `snn_core` to your `Cargo.toml`:

```toml
[dependencies]
# For bare-metal embedded targets without an allocator (static ROM/RAM):
snn_core = { version = "0.4.0", default-features = false }

# For environments with heap allocator support or desktop simulation:
# snn_core = { version = "0.4.0", features = ["alloc"] }
```

### Minimal Bare-Metal Example (`no_std`, Zero Heap Allocation)

Store synaptic matrices directly in immutable Flash ROM and perform zero-alloc inference:

```rust
use snn_core::domain::compact_population::CompactLifPopulation;
use snn_core::models::lif::{LifConfig, ResetMode};
use snn_core::{Fix32, GenericStaticNetwork, RomCsrSynapses};

// 1. CSR matrices stored directly in immutable Flash ROM
static IN_ROW_PTR: [u16; 2] = [0, 1];
static IN_TARGETS: [u16; 1] = [0];
static IN_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(98304)]; // 1.5 in Q16.16 fixed-point

static INT_ROW_PTR: [u16; 3] = [0, 1, 1];
static INT_TARGETS: [u16; 1] = [1];
static INT_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(65536)]; // 1.0 in Q16.16 fixed-point

let in_synapses = RomCsrSynapses::new(&IN_ROW_PTR, &IN_TARGETS, &IN_WEIGHTS);
let int_synapses = RomCsrSynapses::new(&INT_ROW_PTR, &INT_TARGETS, &INT_WEIGHTS);

// 2. Ultra-dense population of 2 LIF neurons (9 bytes/neuron in Fix32 SRAM, subtractive ResetMode)
let config = LifConfig::new(Fix32::ONE, Fix32::from_bits(58982), Fix32::ZERO, 1)
    .with_reset_mode(ResetMode::Subtract);
let neurons = CompactLifPopulation::<2>::new(config);

// 3. Generic static network (2 neurons, 1 input channel)
let mut network: GenericStaticNetwork<_, _, _, 2, 1> =
    GenericStaticNetwork::new(neurons, int_synapses, in_synapses);

let inputs = [Fix32::ONE];
let spikes = network.tick(&inputs);
assert!(spikes[0]);
```

### ⚡ Unified Industrial Standard (`ModelDescriptor` & `ExecutionContext`)

Eliminate complex generic parameters and ensure 100% allocation-free deployment with the simplicity of TFLite Micro / CMSIS-NN:

```rust
use snn_core::domain::engine::{LayerDescriptor, ModelDescriptor};
use snn_core::domain::models::lif::LifConfig;
use snn_core::domain::rom_csr::RomCsrSynapses;
use snn_core::Fix32;

// 1. Static topology and weight definitions in Flash ROM (.rodata)
static L1_ROW: [u16; 3] = [0, 1, 2];
static L1_TGT: [u16; 2] = [0, 1];
static L1_W: [Fix32; 2] = [Fix32::from_bits(98304), Fix32::from_bits(65536)]; // 1.5 and 1.0 in Fix32
let l1_syn = RomCsrSynapses::new(&L1_ROW, &L1_TGT, &L1_W);
let l1_cfg = LifConfig::new(Fix32::ONE, Fix32::from_bits(58982), Fix32::ZERO, 1);

static LAYERS: [LayerDescriptor<'static>; 1] = [
    LayerDescriptor::new(2, 2, l1_syn, l1_cfg, None)
];
pub static MODEL: ModelDescriptor<'static> = ModelDescriptor::new(&LAYERS);

// 2. Initialize dynamic arena in raw SRAM memory (zero heap allocation)
let mut sram_arena = [0u8; 256];
let mut ctx = MODEL.init_context(&mut sram_arena).expect("SRAM must be sufficient and aligned");

// 3. Real-time intra-tick inference
let inputs = [Fix32::ONE, Fix32::ZERO];
let mut out_spikes = [0u32; 1];
MODEL.step(&mut ctx, &inputs, &mut out_spikes).expect("Inference succeeded");
```

---

## 🛠️ Industrial ONNX & NIR AOT Compiler (`snn-core-compiler`)

Convert models trained with **PyTorch**, **SpikingJelly**, or **snnTorch** exported to **ONNX** (`.onnx`) or **NIR** (`.nir`) directly into ultra-optimized static Rust firmware:

```bash
# Install CLI tool
pip install -e tools/snn_compiler

# Ingest ONNX model with INT8 quantization and DAG scheduling:
snn-compile model.onnx \
    --output src/generated_model.rs \
    --quantize-i8 \
    --reset-mode subtract

# Ingest NIR model:
snn-compile model.nir \
    --output src/generated_model.rs \
    --quantize-i8
```

The generated code directly exposes:
- `pub static MODEL_DESC: ModelDescriptor<'static>`: Ready-to-use Flash ROM descriptor.
- SRAM arena scheduling via interval coloring (*Liveness Analysis*), handling complex DAG graphs and skip connections without memory hazards.
- Native support for bare-metal targets without an allocator (`thumbv7em-none-eabi`, `thumbv6m-none-eabi`, `riscv32imc-unknown-none-elf`).

---

## 🌐 C / RTOS Firmware Integration (`include/snn_core_c99.h` & `include/snn_core.h`)

`snn_core` provides two integration paths for industrial firmware teams:
1. **Header-Only Standalone C99 Runtime (`include/snn_core_c99.h`)**: Zero Rust/Cargo dependencies, directly embeddable into **STM32CubeIDE**, **Keil MDK**, **IAR Embedded Workbench**, **ESP-IDF**, **Zephyr RTOS**, or any `arm-none-eabi-gcc -std=c99` toolchain.
2. **Static C FFI API (`include/snn_core.h`)**: C interface linking against the compiled Rust static library `libsnn_core.a`.

- 📖 [Complete C/C++ Integration Guide](docs/c_integration_guide.md)
- 🔬 [NeuroBench Evaluation Protocol](docs/neurobench_evaluation_protocol.md)
- 📑 [Flash ROM CSR Binary Format Specification](docs/binary_format_spec.md)

```c
#include "snn_core_c99.h"
#include "generated_model.h"

// 1. Initialize raw SRAM arena (zero heap, statically computed memory)
uint8_t arena[MODEL_ARENA_SIZE] __attribute__((aligned(4)));
SnnExecutionContext ctx;
model_init(arena, sizeof(arena), &ctx);

// 2. Real-time multilayer inference (zero allocation, inter-layer ping-pong)
int32_t inputs[MODEL_INPUT_FEATURES] = { snn_fix32_from_float(1.2f) };
uint32_t out_spikes[MODEL_OUTPUT_WORDS] = { 0 };
model_step(&ctx, inputs, MODEL_INPUT_FEATURES, out_spikes, MODEL_OUTPUT_WORDS);

// 3. Classification readout decoding
int32_t logits[NUM_CLASSES];
snn_readout_dense(readout_weights, NULL, out_spikes, MODEL_OUTPUT_NEURONS, NUM_CLASSES, logits);
```

---

## 📊 SNN vs Traditional TinyML: Head-to-Head Comparison

On a sentinel surveillance / anomaly detection benchmark evaluated on an ARM Cortex-M4 without FPU (Seeed Wio-E5 / STM32WLE5JC @ 48 MHz):

| Metric | Traditional Dense TinyML (TFLite Micro / CMSIS-NN) | `snn_core` Event-Driven (Sparse LIF) | Real-World Engineering Trade-offs |
| :--- | :---: | :---: | :---: |
| **Sentinel Detection Accuracy** | 100.0 % | 100.0 % | Iso-accuracy |
| **Synaptic Operations / Inference** | 896 systematic dense MACs | **98 ACs (Sparse SynOps)** | **x9.1 Fewer Operations** |
| **Activation Sparsity** | 0.0 % (All neurons evaluated) | **95.2 % to 100.0 % (Silent)** | **95% to 100% compute spared** |
| **Standby CPU Cycles (Silicon)** | ~45,000 cycles (64-pt FFT + MLP) | **773 cycles (16.1 µs @ 48 MHz)** | **x58.2 faster in standby** |
| **Energy per Inference (CMOS model)** | 0.40 µJ | **0.03 µJ** | **x13.3 Less Energy** |
| **Standby Current (Silicon)** | ~4.2 mA @ 48 MHz | **~18 µA (Sleep WFI 99.8%)** | **99.6% battery power savings** |
| **SRAM Allocated per Neuron** | 1B (INT8) to 8B (Float32/Q31) | **5B ([`CompactLif16Population`](src/domain/compact_population.rs))** / **9B ([`CompactLifPopulation`](src/domain/compact_population.rs))** | **Temporal State Trade-off\*** |

*\*Industrial integrity note on SRAM: A spiking neuron (LIF) must maintain its continuous analog membrane potential and refractory counter (5 bytes in `Fix16`, 9 bytes in `Fix32`), which is more compact than a Float32 dense network (~8 bytes/neuron), but requires more SRAM than a single static INT8 activation byte (1 byte/neuron). This memory trade-off is heavily offset by temporal sparsity (>95%), which completely avoids recalculating 896 MACs on every cycle and allows the microcontroller to remain in deep low-power sleep (`WFI`) 99.8% of the time.*

*Full cycle-accurate physical measurements report: [`docs/wio_e5_silicon_benchmark_report.md`](docs/wio_e5_silicon_benchmark_report.md).*

---

## 📂 Executable Examples

```bash
# Example 1: LIF Reservoir & ASCII raster plot in terminal
cargo run --example 01_lif_reservoir

# Example 2: Spike frequency adaptation (LIF vs ALIF)
cargo run --example 02_alif_adaptation

# Example 3: Dynamics and ASCII voltmeter for Izhikevich model
cargo run --example 03_izhikevich_regimes

# Example 4: 100% static zero-heap embedded inference pipeline (no_std)
cargo run --example 04_static_embedded_pipeline

# Example 5: Bio-acoustic / speech benchmark in NeuroBench format
cargo run --example 05_shd_cochlea_benchmark

# Example 6: Quantitative head-to-head comparison TinyML vs snn_core (same CPU)
cargo run --release --example 06_tinyml_vs_snn_head_to_head

# Example 7: Full AOT deployment from standard NIR to no_std firmware
cargo run --example 07_nir_aot_codegen

# Example 8: Hardware profiling of DWT transform for audio/DVS streams
cargo run --example 08_dwt_hardware_profiling

# Example 9: Micro-architectural head-to-head comparison: Biophysical SNN vs Silicon-Native BPTN
cargo run --release --example 09_bptn_vs_lif_head_to_head

# Example 10: Long-horizon temporal memory and noise robustness stress test
cargo run --release --example 10_bptn_long_horizon_memory

# Example 11: End-to-End BPTN pipeline (Sensors -> Framing -> BPTN -> CTZ Readout)
cargo run --release --example 11_bptn_global_pipeline

# Example 12: Real-time Embedded Cybersecurity Benchmark (Edge IDS on CAN bus at 500 kbps)
cargo run --release --example 12_cyber_can_ids_killer_benchmark --features bptn
```

---

## ⚡ Performance Benchmarks (Criterion)

```bash
# Run entire benchmark suite
cargo bench

# Embedded static reservoir benchmark (100N, 2,500 synapses)
cargo bench -- "StaticNetwork"

# Comparative micro-benchmark of neuron dynamics (LIF / ALIF / Izhikevich)
cargo bench -- "Neuron Dynamics"
```

---

## 📜 License

Copyright (c) 2026 Zuytan. Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md).

- ✅ **Free** for personal use, research, education, hobby projects, and noncommercial organizations.
- ❌ **Commercial use is not permitted** under this license (including integration into commercial products, firmware, or services).
- 💼 **Commercial licensing** is available on request — please open an issue or contact the author via GitHub ([@Zuytan](https://github.com/Zuytan)).
