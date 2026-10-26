# Changelog

## Unreleased

### License

- Relicensed from PolyForm Noncommercial 1.0.0 to MIT (`LICENSE`, `Cargo.toml` of all crates, `pyproject.toml`, CMSIS-Pack descriptor, README).

### Restructuring (breaking: paths changed)

- Cargo workspace: `snn_core` (runtime, `no_std`), `crates/snn_core_bptn`, `crates/snn_core_c` (C FFI, standalone), `demos` (examples, benches, `narma`, `can_parser`).
- `snn_core::bptn` moved to the `snn_core_bptn` crate (the `bptn` feature is gone); `snn_core::ffi` moved to `snn_core_c`; `narma` and `can_parser` moved to `snn_core_demos`.
- The C library now exports the BPTN symbols declared in `snn_core.h` (the `bptn` feature is on by default in `snn_core_c`).
- `snn_core_c` is now `no_std` with a spinning panic handler on bare metal and `panic = "abort"` (not yet verified on a bare-metal target).
- Runtime dev-dependencies (criterion, nalgebra, fastrand) moved out of the runtime crate.
- CI runs `--workspace` and the no-default-features runtime build.
- Added `// SAFETY:` justifications for inline assembly and buffer casts, a safety contract for the FFI module, and a `# Panics` section on `ArenaLayerChain`.

### C interface: Rust is the single source of truth

- Removed the hand-written C99 reimplementation (`include/snn_core_c99.h`, `include/snn_core_runtime.h`, `tests/test_c99_standalone.c`) and the `c99-standalone` compiler target. C projects link `libsnn_core.a` and include `include/snn_core.h`; the compiler's `--target c` emits tables that use it.
- Fixed `include/snn_core.h`, which had drifted from the Rust ABI: `SnnLayerDesc` lacked `conv2d`, `pool2d` and `recurrent_synapses`, `SnnDeepNetworkState` lacked `recurrent_spikes` and `total_recurrent_words`, and the Conv2d / MaxPool2d layer types were missing. A C caller using the old header would have passed wrong-sized structs.
- New Rust exports replacing C-only helpers: `snn_fix32_mul`, `snn_saturating_add_i32`, `snn_saturating_sub_i32`, `snn_pade_exp_neg` (Q16.16 argument), `snn_bptn_erode_64`, `snn_bptn_open_filter_64`, `snn_bptn_majority_pool_8x8`, `snn_bptn_parallel_smith_step_u64`, `snn_bptn_hysteresis_step`, `snn_bptn_robust_toggle_step`, plus declarations of the `snn_model_*` functions.
- The old C99 `snn_fix32_mul` truncated while the Rust one rounds half up, so the two were not bit-exact. Only the Rust behaviour remains.
- `crates/snn_core_c/tests/header_parity.rs` fails if an export or a `#[repr(C)]` struct differs from the header; CI compiles `crates/snn_core_c/tests/c/test_ffi.c` (strict C99) against the built library.

### Versioned model container

- New `.snnb` binary format 1.0 (magic, major/minor version, flags, length, CRC-32, 4-byte aligned tables read in place): `snn_core::domain::blob` (`ModelBlob::parse`, `load_layers`, `encode_model`), C function `snn_blob_load`, compiler option `--target blob`, specification in `docs/binary_format_spec.md` section 4.
- Unknown major versions, unknown flags, truncation, CRC errors, misalignment and CSR invariant violations are refused as a whole; a mutation test with repaired CRCs checks that parsing never panics.
- A golden file (`tests/data/model_v1.snnb`) must be reproduced byte for byte by the Rust and Python encoders. `Fix32` is now `repr(transparent)`; added `LayerDescriptor::EMPTY`.

### Inference API

- `ModelDescriptor` + `ExecutionContext` is the documented device inference API (crate docs and README lead with it, with a "Which API to use" table and a memory-model section).
- Deprecated (since 0.5.0, still working): `ChainedSpikingNetwork`, `ArenaSpikingNetwork`, `arena_spiking_network!`. `GenericStaticNetwork` is kept and documented as the host simulation / online-learning network.
- Added `AlignedBuffer<N>` (SRAM buffer with the required 4-byte alignment) and `ModelDescriptor::sram_bytes()`.

### Corrections (accuracy of published claims)

- Fixed the energy unit of the Wio-E5 benchmark: 2.37 µJ, not 2.37 nJ. Energy is now labelled as an estimate (CPU active, no current probe).
- Removed unmeasured comparison figures (LIF SNN and Micro-GRU cycles, energy, SRAM, MACs) from the firmware, README, report, whitepaper and demo page.
- Removed the "more than 10 years on a CR2032" claim.
- README rewritten: states what is not established, shows where BPTN is slower than event-driven LIF, uses `git` dependency instead of an unpublished crates.io version, and its Rust blocks are now compiled as doctests.
- Fixed the `ModelDescriptor` README example (it did not compile) and documented the 4-byte alignment requirement of the SRAM buffer.
- Documented that `Fix32` multiplication rounds half up (it is not unbiased).
- Renamed examples: `05_shd_cochlea_benchmark` to `05_synthetic_cochlea_demo`, `12_cyber_can_ids_killer_benchmark` to `12_can_ids_benchmark`.
- Rewrote `docs/mlperf_tiny_submission_specification.md`: no submission exists.
- Added `CLAIMS.md`, the register of every published number and its status.
- Rewrote example 09: LIF, lazy LIF and BPTN now solve the same synthetic task with the same output and
  report accuracy. The former 14× to 27× (block) and 2.7× to 4.9× (streaming) ratios are withdrawn; on the
  shared task BPTN is about 1.25× faster than dense LIF once per window and slightly slower when evaluated
  at every tick.
- Removed the per-operation energy rows (40 nm ASIC and assumed Cortex-M4 constants) from examples 05, 06
  and 12. Deprecated `HardwareProfile`, `energy_constants` and `EnergyEstimator::estimate_picojoules*` /
  `estimate_nanojoules_fix32*`; added `EnergyEstimator::cpu_active_nanojoules` (cycles × active power / clock).
- `silicon_test_runner` suite 4 is now labelled as a functional test on hand-written bit patterns, not a
  keyword-spotting result on real audio.
- Wio-E5 firmware: the board ran at its 4 MHz reset clock, not 48 MHz. The firmware now sets MSI to 48 MHz
  with 2 Flash wait states (`board.rs`) and prints the clock read back from RCC; all times use it.
  Measured again: BPTN layer 11,972 cycles (249.42 µs, about 3.46 µJ estimated), instead of 8,193 cycles,
  170.69 µs and 2.37 µJ.
- Wio-E5 benchmarks pass inputs and results through `black_box`: several figures (BPTN 4 cycles per window,
  operators in 6 cycles, most `silicon_test_runner` checks) were computed by the compiler, not the core.
- Removed unmeasured figures from the sentinel and main firmware (TinyML cycles, currents, battery life,
  speedup conclusions).
