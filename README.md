# snn_core

[![Status: Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)](#status)
![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)

> **Work in progress / research preview.** APIs, binary formats and architecture may change without notice.
> The crate is not published on crates.io. See [Status and limits](#status) and [`CLAIMS.md`](CLAIMS.md)
> for what is measured and what is not.

`snn_core` is a spiking neural network (SNN) inference engine in Rust for microcontrollers
(ARM Cortex-M0+/M3/M4 without FPU, RISC-V). The runtime is `no_std`, has no dependencies, and
performs no heap allocation during inference. It also contains an experimental bit-parallel
temporal network engine called BPTN, in its own crate (`snn_core_bptn`).

## Status

What exists and is tested (`cargo test`, `clippy -D warnings`, a C smoke test linked against the Rust library):

- Saturating fixed-point types `Fix32` (Q16.16) and `Fix16` (Q8.8).
- LIF, CUBA-LIF, ALIF and Izhikevich neuron models, event-driven spike propagation over bitsets.
- Synapse matrices in compressed sparse row (CSR) layout placed in Flash, with INT8 variants.
- `ModelDescriptor` / `ExecutionContext`: immutable model in Flash, working memory in a caller-provided SRAM buffer.
- An ahead-of-time compiler (`tools/snn_compiler`) from NIR / ONNX graphs to Rust or C (tables linked with `libsnn_core.a`).
- Optional ARM DSP, Helium and RISC-V P-extension intrinsics behind feature flags.

What is **not** established yet:

- No accuracy result on a public dataset. The example "cochlea" data is synthetic.
- No measured comparison with TFLite Micro, CMSIS-NN or other runtimes on the same task and board.
- No energy measurement with a current probe. Energy figures are estimates.
- NIR import covers `Affine`, `Linear`, `LIF`, `CubaLIF` and `IF` only, and reduces per-neuron parameters to scalars.
- No release, no stable API, no CMSIS-Pack, ESP-IDF registry or Zephyr module release.

## BPTN: what is measured

BPTN (bit-parallel temporal network) evaluates temporal kernels with AND, shift and popcount on 64-bit words.

On a Seeed Wio-E5 (STM32WLE5JC, Cortex-M4 at 48 MHz with 2 Flash wait states, no FPU), one BPTN layer of 256
temporal synapses over a 64-tick window takes **11,972 CPU cycles (249.42 µs)** by the DWT cycle counter
([report](docs/wio_e5_silicon_benchmark_report.md)); masks and inputs are synthetic. The CPU-active energy
estimate for that is about **3.46 µJ** (249.42 µs × 13.86 mW, from an assumed run current, not probed;
sleep, sensors and radio excluded). Earlier versions said 8,193 cycles and 170.69 µs: the board was
actually running at its 4 MHz reset clock, see [CLAIMS.md](CLAIMS.md).

BPTN does not beat the event-driven LIF everywhere. The host benchmark
([example 09](demos/examples/09_bptn_vs_lif_head_to_head.rs)) gives both models the same synthetic task
(detect a weak burst on 4 of 16 channels in the last 16 ticks of a 64-tick window, 20,000 windows) and
the same output (one decision per window), with each model's parameters picked by a small grid search:

| Model | Accuracy | Host time per decision vs dense LIF | Flash / SRAM |
| :--- | :--- | :--- | :--- |
| Dense LIF (Fix32) | 90.8 % | 1.00× | 16 B / 9 B |
| BPTN, evaluated once per window | 93.4 % | about 1.25× faster | 260 B / 128 B |
| BPTN, evaluated at every tick | 93.4 % | about 0.95× (slightly slower) | 260 B / 128 B |

One run on an Apple M1 Pro; synthetic data, hand-set models, a single detector neuron. This shows how to
compare the two on equal terms, not an advantage on a real task. No accuracy is reported for BPTN on any
real dataset.

## Quick start

The crate is not on crates.io; use the repository:

```toml
[dependencies]
snn_core = { git = "https://github.com/Zuytan/snn_core", default-features = false }
# optional, experimental:
# snn_core_bptn = { git = "https://github.com/Zuytan/snn_core" }
```

Zero-allocation inference with CSR matrices stored in Flash:

```rust
use snn_core::domain::compact_population::CompactLifPopulation;
use snn_core::models::lif::{LifConfig, ResetMode};
use snn_core::{Fix32, GenericStaticNetwork, RomCsrSynapses};

// CSR matrices in immutable Flash (.rodata)
static IN_ROW_PTR: [u16; 2] = [0, 1];
static IN_TARGETS: [u16; 1] = [0];
static IN_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(98304)]; // 1.5 in Q16.16

static INT_ROW_PTR: [u16; 3] = [0, 1, 1];
static INT_TARGETS: [u16; 1] = [1];
static INT_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(65536)]; // 1.0 in Q16.16

let in_synapses = RomCsrSynapses::new(&IN_ROW_PTR, &IN_TARGETS, &IN_WEIGHTS);
let int_synapses = RomCsrSynapses::new(&INT_ROW_PTR, &INT_TARGETS, &INT_WEIGHTS);

// 2 LIF neurons, 9 bytes of SRAM per neuron in Fix32, subtractive reset
let config = LifConfig::new(Fix32::ONE, Fix32::from_bits(58982), Fix32::ZERO, 1)
    .with_reset_mode(ResetMode::Subtract);
let neurons = CompactLifPopulation::<2>::new(config);

let mut network: GenericStaticNetwork<_, _, _, 2, 1> =
    GenericStaticNetwork::new(neurons, int_synapses, in_synapses);

let inputs = [Fix32::ONE];
let spikes = network.tick(&inputs);
assert!(spikes[0]);
```

Model description in Flash, working memory in one caller-provided SRAM buffer. This is the inference API
for devices; `GenericStaticNetwork` above is for host simulation and online learning. `AlignedBuffer`
provides the 4-byte alignment that `init_context` requires (otherwise `EngineError::MisalignedBuffer`);
`MODEL.sram_bytes()` is the minimum size.

```rust
use snn_core::{AlignedBuffer, Fix32, LayerDescriptor, LifConfig, ModelDescriptor, RomCsrSynapses};

static L1_ROW: [u16; 3] = [0, 1, 2];
static L1_TGT: [u16; 2] = [0, 1];
static L1_W: [Fix32; 2] = [Fix32::from_bits(98304), Fix32::from_bits(65536)]; // 1.5 and 1.0

static L1_SYN: RomCsrSynapses<'static> = RomCsrSynapses::new(&L1_ROW, &L1_TGT, &L1_W);
static LAYERS: [LayerDescriptor<'static>; 1] = [LayerDescriptor::new(
    2,
    2,
    L1_SYN,
    LifConfig::new(Fix32::ONE, Fix32::from_bits(58982), Fix32::ZERO, 1),
    None,
)];
static MODEL: ModelDescriptor<'static> = ModelDescriptor::new(&LAYERS);

let mut sram = AlignedBuffer::<256>::new();
let mut ctx = MODEL.init_context(sram.as_mut_bytes()).expect("SRAM large enough and aligned");

let inputs = [Fix32::ONE, Fix32::ZERO];
let mut out_spikes = [0u32; 1];
MODEL.step(&mut ctx, &inputs, &mut out_spikes).expect("step succeeds");
```

`ChainedSpikingNetwork`, `ArenaSpikingNetwork` and `arena_spiking_network!` are deprecated in favour of
`ModelDescriptor`.

## Shipping a model as data

A model can also be stored as a versioned binary file (`.snnb`: magic, version, length, CRC-32), for
OTA updates or external Flash. `ModelBlob::parse` refuses anything it does not fully understand, then
`load_layers` borrows the tables in place:

```rust
use snn_core::{AlignedBuffer, BlobError, ModelBlob};

fn check(bytes: &[u8]) -> Result<usize, BlobError> {
    // `bytes` must start on a 4-byte boundary (copy into an `AlignedBuffer` if it comes from a stream).
    let blob = ModelBlob::parse(bytes)?;
    Ok(blob.num_layers())
}
# let mut buf = AlignedBuffer::<8>::new();
# assert!(check(buf.as_mut_bytes()).is_err());
```

`snn-compile model.nir --target blob -o model.snnb` writes one (feed-forward Linear/LIF graphs only);
C code loads it with `snn_blob_load`. Format and compatibility rules:
[binary format specification](docs/binary_format_spec.md), section 4.

## Compiling a trained model (NIR / ONNX)

```bash
pip install -e tools/snn_compiler
snn-compile model.nir --output src/generated_model.rs --quantize-i8 --reset-mode subtract
snn-compile model.nir --target c --output generated/model.h
```

The generated code exposes a `MODEL_DESC: ModelDescriptor<'static>` and an SRAM arena size computed by
liveness analysis (interval colouring). Supported NIR nodes are listed under [Status](#status); the
compiler's own tests use a synthetic graph, and graphs exported from snnTorch, Norse or SpikingJelly are
not yet part of CI.

## C integration

The Rust crates are the only implementation of the engine. C and C++ projects link the static library
`libsnn_core.a` built from `crates/snn_core_c` and include `include/snn_core.h`. There is no separate C
reimplementation: a test (`crates/snn_core_c/tests/header_parity.rs`) fails if the header drifts from the
Rust FFI, and CI compiles `crates/snn_core_c/tests/c/test_ffi.c` (strict C99) against the library.
Using it needs a Rust toolchain at build time.

See the [C/C++ integration guide](docs/c_integration_guide.md) and the
[binary format specification](docs/binary_format_spec.md).

## Memory layouts

| Type | Per neuron (SRAM) |
| :--- | :--- |
| `CompactLif16Population` | 5 bytes (Fix16 membrane 2 B, accumulator 2 B, refractory 1 B) |
| `CompactLifPopulation` | 9 bytes (Fix32 membrane 4 B, accumulator 4 B, refractory 1 B) |
| `LazyDecayLifPopulation` | membrane updated only on incoming spikes; silent neurons cost no cycles |

A spiking neuron needs persistent state, so a LIF population uses more SRAM per neuron than the
activations of a dense INT8 network (1 byte). Whether event-driven sparsity pays that back depends on the
workload and has to be measured on the task of interest.

## Repository layout

| Path | Package | Role |
| :--- | :--- | :--- |
| `.` | `snn_core` | The runtime: `no_std`, no dependencies, no heap on the inference path (`alloc` feature for host tools). |
| `crates/snn_core_bptn` | `snn_core_bptn` | Experimental BPTN engine, built on the runtime. |
| `crates/snn_core_c` | `snn_core_c` | C ABI library (`libsnn_core.a`, `libsnn_core.so`) over the runtime and BPTN. Standalone package so the library name stays `snn_core`. |
| `demos` | `snn_core_demos` | Examples, Criterion benchmarks, CAN parser and NARMA-10 helpers. Not part of the runtime. |
| `examples/wio_e5_benchmark` | `wio_e5_benchmark` | On-board benchmark firmware for the Wio-E5 (standalone, ARM target). |
| `tools/` | | NIR/ONNX compiler (`snn_compiler`), Rust codegen, BPTN training demos. |
| `include/` | | C header `snn_core.h` for the FFI library. |

## Examples

```bash
cargo run -p snn_core_demos --example 01_lif_reservoir
cargo run -p snn_core_demos --example 04_static_embedded_pipeline
cargo run -p snn_core_demos --example 05_synthetic_cochlea_demo        # synthetic data, not SHD
cargo run -p snn_core_demos --release --example 06_tinyml_vs_snn_head_to_head   # model estimates, not measurements
cargo run -p snn_core_demos --example 07_nir_aot_codegen
cargo run -p snn_core_demos --release --example 09_bptn_vs_lif_head_to_head
cargo run -p snn_core_demos --release --example 10_bptn_long_horizon_memory
cargo run -p snn_core_demos --release --example 12_can_ids_benchmark
```

The on-board benchmark for the Wio-E5 is in `examples/wio_e5_benchmark`.

## Development

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy -p snn_core --all-targets --no-default-features -- -D warnings
cargo test --workspace
cargo check -p snn_core --target thumbv7em-none-eabi --no-default-features
(cd crates/snn_core_c && cargo test)      # C ABI crate, standalone package
```

## License

Copyright (c) 2026 Zuytan. Licensed under the [MIT License](LICENSE).

Versions published before this change were distributed under the PolyForm Noncommercial License 1.0.0;
copies obtained under that license keep its terms.
