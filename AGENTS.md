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
snn_core/                       # workspace root = the runtime package `snn_core`
├── Cargo.toml                  # runtime manifest + [workspace] + shared pedantic Clippy lints
├── src/                        # RUNTIME (no_std, no dependencies, no heap on the inference path)
│   ├── lib.rs, traits.rs
│   ├── application/            # nir_import, readout, simulator, time
│   └── domain/                 # numeric, hardware, bitset, models/, rom_csr, static_csr, delay_ring,
│                               # layers, arena, engine (ModelDescriptor/ExecutionContext),
│                               # compact/lazy populations, static networks, stdp, metrics, aer, encoding
├── tests/                      # runtime integration tests (aot, nir conformance, traits)
├── crates/
│   ├── snn_core_bptn/          # experimental BPTN engine + adversarial tests (workspace member)
│   └── snn_core_c/             # C ABI library libsnn_core.a / .so + FFI fuzz tests (standalone package)
├── demos/                      # examples, Criterion benches, CAN parser, NARMA-10 (workspace member)
├── examples/wio_e5_benchmark/  # on-board firmware (standalone package, ARM target)
├── include/                    # snn_core.h, the C header of the FFI library (kept in sync with Rust by a test)
├── tools/                      # snn-compile (NIR/ONNX), codegen, silicon runners
└── docs/                       # specs and reports (the CLAIMS.md register is in the repo root)
```

Rule: anything not needed to run inference on a microcontroller does not go in the runtime package.

---

## 🧪 Mandatory Validation Commands

Before committing or releasing any code:
```bash
# 1. Strict Clippy verification across all targets and features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy -p snn_core --all-targets --no-default-features -- -D warnings
(cd crates/snn_core_c && cargo clippy --all-targets -- -D warnings)

# 2. Complete unit and integration test suite execution
cargo test --workspace
(cd crates/snn_core_c && cargo test)

# 3. Bare-metal cross-compilation without std (ARM Cortex-M4 without FPU)
cargo check -p snn_core --target thumbv7em-none-eabi --no-default-features
```
