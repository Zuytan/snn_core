# 🌐 Industrial C / C++ Integration Guide for `snn_core`

This guide describes the official procedure for integrating the `snn_core` neuromorphic engine into industrial embedded C and C++ applications (FreeRTOS, Zephyr RTOS, STM32CubeIDE, ESP-IDF, MCUXpresso, Bare-Metal ARM Cortex-M, and RISC-V).

---

## 🏛️ Zero-Allocation Memory Architecture

`snn_core` strictly complies with critical embedded system constraints (MISRA C, ISO 26262, DO-178C):
1. **Flash ROM (`.rodata`)**: All descriptors (`SnnDeepNetworkDesc`), CSR sparse topologies (`SnnCsrMatrix`), convolution kernels, and biases reside in immutable non-volatile memory.
2. **SRAM (`ActivationArena`)**: Dynamic neuronal state ($V_{\text{mem}}$, refractory counters, shared accumulators, and inter-layer ping-pong spike buffers) is stored in a contiguous byte buffer allocated statically or on the stack by the host C firmware.
3. **Zero `malloc`**: Absolutely zero dynamic heap allocation is performed on the inference path.

---

## 📦 1. Integration model

The Rust crates are the single implementation of the engine. C and C++ projects link the static library built from `crates/snn_core_c` and include [`include/snn_core.h`](../include/snn_core.h). A Rust toolchain is needed at build time (not at run time). `crates/snn_core_c/tests/header_parity.rs` fails if the header and the Rust exports diverge.

### Building the static library (`libsnn_core.a`)
In the subfolder `crates/snn_core_c` (a standalone Cargo package), build the static archive for your target microcontroller. The BPTN functions declared in `snn_core.h` are included by the default `bptn` feature; on bare-metal targets the crate is `no_std` and installs a minimal panic handler (release profile uses `panic = "abort"`). Run these commands from that folder:

```bash
# For ARM Cortex-M4 / Cortex-M7 without hardware FPU
cargo build --target thumbv7em-none-eabi --release

# For ARM Cortex-M4F / Cortex-M7F with hard-float FPU and DSP extensions
cargo build --target thumbv7em-none-eabihf --features cortex-m-dsp --release

# For ARM Cortex-M0 / Cortex-M0+ (ultra-low-power)
cargo build --target thumbv6m-none-eabi --features cortex-m0 --release

# For RISC-V RV32I / RV32IMAC
cargo build --target riscv32imac-unknown-none-elf --release
```

The generated archive is located at:
`crates/snn_core_c/target/<TARGET>/release/libsnn_core.a`

---

## 🔌 2. Project Build Configuration

Add the following to your build system (Makefile, CMake, Keil µVision, STM32CubeIDE):
1. Public include path: `-I<snn_core>/include`
2. Link with `-L<path_to_libsnn_core> -lsnn_core` and include:
   ```c
   #include "snn_core.h"
   ```

---

## 🚀 3. Complete Example: Deep Multilayer Spiking Network (Deep SNN)

```c
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "snn_core.h"

// 1. Static Flash ROM topology definition (.rodata)
static const uint16_t L0_ROW_PTRS[] = { 0, 2, 4 };
static const uint16_t L0_TARGETS[]  = { 0, 1, 0, 1 };
static const int32_t  L0_WEIGHTS[]  = { 65536, 32768, 32768, 65536 }; // Fix32 Q16.16

static const SnnLayerDesc LAYERS[] = {
    {
        .layer_type = SNN_LAYER_FEEDFORWARD,
        .in_features = 2,
        .num_neurons = 2,
        .config = {
            .threshold = 65536,         // 1.0 in Q16.16
            .decay_factor = 58982,      // 0.9 in Q16.16
            .reset_potential = 0,
            .refractory_period = 2,
            .bias = 0,
            .reset_mode = 0             // Hard reset
        },
        .synapses = {
            .row_ptrs = L0_ROW_PTRS,
            .targets = L0_TARGETS,
            .weights = L0_WEIGHTS,
            .in_features = 2,
            .out_features = 2,
            .total_synapses = 4
        },
        .bias = NULL,
        .quant_scale = 0
    }
};

static const SnnDeepNetworkDesc MODEL_DESC = {
    .layers = LAYERS,
    .num_layers = 1,
    .in_features = 2
};

int main(void) {
    // 2. Compute exact SRAM arena requirement
    size_t arena_size = snn_deep_network_state_size(&MODEL_DESC);
    
    // Static stack allocation (or global SRAM BSS buffer) aligned to 4 bytes
    uint8_t arena[arena_size] __attribute__((aligned(4)));
    SnnDeepNetworkState state;

    // 3. Initialize state in arena
    int ret = snn_deep_network_init_state(&MODEL_DESC, arena, arena_size, &state);
    if (ret != 0) {
        printf("Arena initialization error: %d\n", ret);
        return ret;
    }

    // 4. Real-time inference (control loop / RTOS task)
    int32_t inputs[2] = {
        snn_fix32_from_float(1.2f),
        snn_fix32_from_float(0.5f)
    };
    uint32_t out_spikes[1] = { 0 };

    for (int tick = 0; tick < 10; ++tick) {
        snn_deep_network_step(
            &MODEL_DESC,
            &state,
            inputs, 2,
            out_spikes, 1
        );

        // Decode spike train (1 bit per neuron)
        bool n0_spiked = snn_bitset_get_spike(out_spikes, 1, 0);
        bool n1_spiked = snn_bitset_get_spike(out_spikes, 1, 1);
        printf("Tick %d: N0=%d, N1=%d\n", tick, n0_spiked, n1_spiked);
    }

    return 0;
}
```

---

## ⚡ 4. Direct Ingestion of Neuromorphic Events (AER / DVS)

Process asynchronous event streams emitted by Dynamic Vision Sensors (Prophesee, iniVation) or neuromorphic cochleas without CPU overhead:

```c
// DMA buffer receiving incoming DVS packets
AerSpike2d dvs_events[128];
size_t event_count = get_dvs_packet_dma(dvs_events, 128);

// Instantaneous 2D bipolar spatial decoding to SpikeBitset (128x128 pixels)
uint32_t spike_words[1024]; // 128 * 128 * 2 polarities / 32 bits = 1024 words
snn_aer_decode_2d_bipolar(dvs_events, event_count, 128, 128, spike_words, 1024);

// Evaluate global activity using single-cycle hardware POPCNT instruction
uint32_t active_spikes = snn_bitset_count_spikes(spike_words, 1024);
```

---

## 🛡️ Error Codes and Fault Robustness

All C FFI functions return deterministic negative integer error codes on invalid configurations, eliminating runtime aborts:
- `0`: Success
- `-1`: Null pointer or missing mandatory parameter
- `-2`: Incompatible network dimensions (e.g. `num_neurons == 0` or `num_layers == 0`)
- `-3`: Insufficient output buffer capacity or SRAM arena size
- `-4`: Misaligned SRAM arena buffer (must be a multiple of 4 bytes)
