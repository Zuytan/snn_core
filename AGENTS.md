# 🧠 AGENTS.md — Engineering Directives for `snn_core`

This document defines the architectural standards, performance mandates, and software quality criteria for all agents and contributors working on `snn_core`.

---

## 🏛️ Non-Negotiable Core Principles

1. **Strict `#![no_std]` & Zero Heap Allocation in Inference**
   - Zero dynamic heap allocation is permitted on the critical inference path.
   - All runtime inference data structures must reside either in Flash ROM (`&'static [T]`) or on the stack / static memory (`[T; N]`).
   - Heap allocator support (`feature = "alloc"`) is strictly reserved for dynamic initialization, unconstrained online learning, and offline conversion tooling.

2. **Pure Fixed-Point Arithmetic (`Fix32` / `Fix16`) Without Mandatory FPU**
   - The primary execution targets are ultra-low-power microcontrollers (ARM Cortex-M0+, Cortex-M4 without FPU, RISC-V RV32I).
   - All membrane, adaptive, and synaptic calculations must be executed using scaled signed integers (Q16.16 for `Fix32`, Q8.8 for `Fix16`).
   - Any potential arithmetic overflow must be handled through saturating operations (`saturating_add`, `saturating_sub`).

3. **Compact Memory Footprint (Minimal SRAM Footprint)**
   - For `Fix16`: 5 bytes per neuron ($V_{\text{mem}}$ 2B, input accumulator 2B, refractory 1B).
   - For `Fix32`: 9 bytes per neuron ($V_{\text{mem}}$ 4B, input accumulator 4B, refractory 1B).
   - Spike trains must be transmitted via 32-bit words (`SpikeBitset`) decoded with native hardware `CTZ` (`trailing_zeros`), ensuring 1 bit per neuron instead of a boolean byte.

4. **Embedded Silicon Hardware Acceleration (ARM DSP / SIMD / Helium)**
   - Critical operations must leverage hardware intrinsics when acceleration feature flags are active:
     - `smlad`: Dual Multiply-Accumulate 16-bit in 1 clock cycle (`cortex-m-dsp`).
     - `ssat_16`: Immediate hardware saturation without conditional branching (`cortex-m-dsp`).
     - `vqadd_4x32` / `vqadd_8x16`: 128-bit SIMD vector saturation (`cortex-m-helium`).
     - `fast_decay_m0`: Binary shift division approach for Cortex-M0+ microcontrollers.

5. **Event-Driven Synaptic Plasticity (STDP)**
   - LTP (Long-Term Potentiation) must execute in $\mathcal{O}(\text{spikes} \times \text{in-degree})$ using `StaticCscIndex` or `CscIndex`, strictly avoiding any $\mathcal{O}(E)$ scan over the synaptic matrix.

6. **Industrial Standards Interoperability**
   - Compliance with the **NIR (Neuromorphic Intermediate Representation)** standard and ONNX via AOT compilers (`tools/nir_to_snn_core.py` and `tools/snn_compiler`).
   - Evaluation metrics compliant with **NeuroBench** and **MLCommons / TinyML** benchmarks (AC SynOps vs dense MACs, Sparsity, Physical CMOS energy, Latency-to-First-Spike, Time-to-Decision).

---

## 📂 Codebase Layout

```text
snn_core/
├── Cargo.toml                  # Dependencies & strict pedantic Clippy configuration
├── src/
│   ├── lib.rs                  # Library entry point (#![no_std])
│   ├── traits.rs               # Abstraction traits (SpikingPopulation, SynapseMatrix, SpikingNetwork)
│   ├── application/
│   │   ├── can_parser.rs       # Deterministic automotive CAN frame stream parser
│   │   ├── ffi.rs              # C-compatible FFI bindings for dynamic/static embedding
│   │   ├── nir_import.rs       # AOT import and Padé rational approximation for NIR models
│   │   ├── readout.rs          # Dense decoders and trainable classification/regression readouts
│   │   └── simulator.rs        # Discrete-time simulation orchestrator
│   └── domain/
│       ├── aer.rs              # Address-Event Representation (AER) DMA ring buffer
│       ├── arena.rs            # ActivationArena: zero-alloc ping-pong memory planner
│       ├── bitset.rs           # SpikeBitset: 1 bit/neuron vectorized spike bus
│       ├── bptn.rs             # Bit-Parallel Temporal Networks (R-BPTN, H-BPTN, bistable latches)
│       ├── compact_population.rs # Ultra-compact neuron populations (5B/9B per neuron)
│       ├── delay_ring.rs       # Allocation-free circular spike delay lines
│       ├── encoding.rs         # Temporal encoders (Rate, Latency, Delta-Modulation)
│       ├── engine.rs           # Unified ModelDescriptor & ExecutionContext engine
│       ├── hardware.rs         # DSP intrinsics (smlad, ssat, Helium, FastDivider)
│       ├── layers.rs           # Modular layers (Conv1d, Conv2d, MaxPool2d, Recurrent, Residual)
│       ├── lazy_population.rs  # Lazy decay engine (O(1) per spike event)
│       ├── metrics.rs          # NeuroBench metrics (SynOps, Sparsity, CMOS energy)
│       ├── models/             # Biophysical models (LIF, Cuba-LIF, ALIF, Izhikevich)
│       ├── numeric.rs          # Fix32 (Q16.16) and Fix16 (Q8.8) saturating types
│       ├── rom_csr.rs          # Flash ROM CSR sparse synapses (Fix32, quantized INT8)
│       └── stdp.rs             # STDP and R-STDP plasticity with CSC indexing
├── tests/                      # Integration, conformance, fuzzing, and adversarial test suites
├── benches/                    # Criterion performance benchmarks
├── examples/                   # 12 executable engineering demonstrator benchmarks
├── include/                    # snn_core_c99.h standalone header and C FFI headers
├── tools/                      # CLI compilers (snn-compile, NIR/ONNX converters, silicon runners)
└── docs/                       # Formal specs, whitepapers, and hardware reports
```

---

## 🧪 Mandatory Validation Commands

Before committing or releasing any code:
```bash
# 1. Strict Clippy verification across all targets and features
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --all-targets --no-default-features -- -D warnings

# 2. Complete unit and integration test suite execution
cargo test

# 3. Bare-metal cross-compilation without std (ARM Cortex-M4 without FPU)
cargo check --target thumbv7em-none-eabi --no-default-features
```
