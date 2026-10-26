# BPTN Silicon Profiling Report: Seeed Studio Wio-E5 (STM32WLE5JC)

This document records physical measurements of performance, latency, and power consumption for the **BPTN (Bit-Parallel Temporal Neural Network)** engine running bare-metal directly on target silicon.

---

## 1. Hardware Experimental Setup

* **Target Microcontroller:** STMicroelectronics **STM32WLE5JC** (Seeed Studio Wio-E5).
* **Compute Core:** ARM Cortex-M4 without FPU (`ARMv7E-M` instruction set, Rust target `thumbv7em-none-eabi`).
* **CPU clock:** 48 MHz (MSI range 11), **2 Flash wait states**, instruction and data caches on. The
  firmware sets this clock at startup (`examples/wio_e5_benchmark/src/board.rs`) and prints the
  frequency read back from the RCC registers.
* **Assumed power for the energy estimate:** 3.3 V × 4.2 mA = 13.86 mW. This run current was not
  measured on the board; no current probe was used.
* **Cycle-Accurate Instrumentation:** ARM DWT `CYCCNT` (sampling overhead: 1 cycle at 48 MHz).
* **Telemetry Console:** SEGGER RTT over an ST-Link V2 SWD probe, `probe-rs` 0.32.0.
* **Software Runtime:** Rust `#![no_std]`, `#![no_main]`, no heap allocation during inference.

> **Correction (2026-10-25).** Earlier versions of this report said 48 MHz, but the firmware never set
> the clock: the board ran from the reset clock, MSI at 4 MHz with 0 Flash wait states (checked by
> reading RCC_CFGR, RCC_CR and FLASH_ACR on the board). The cycle counts were real but measured at
> 0 wait states, and the times derived from 48 MHz (170.69 µs, 2.37 µJ) were wrong. Some inputs were
> also constant, so the compiler moved part of the work out of the measured loops. All figures below
> were measured again at 48 MHz with inputs and results passed through `core::hint::black_box`.

---

## 2. Micro-Architectural Measurements of BPTN Primitives

Every operator was run 1,000 times; its input and result go through `core::hint::black_box`. For
operators this small the figure is dominated by the measurement harness itself (the same operators
take 20 cycles at 4 MHz with 0 wait states), so read it as an upper bound.

| BPTN Primitive | Algebraic Formula | Cycles (DWT), 48 MHz, 2 WS | Time |
| :--- | :--- | :---: | :---: |
| **1D Morphological Dilation** | $\mathcal{D}_1(x) = x \mid (x \ll 1) \mid (x \gg 1)$ | 20.0 | 417 ns |
| **1D Morphological Erosion** | $\mathcal{E}_1(x) = x \ \& \ (x \ll 1) \ \& \ (x \gg 1)$ | 20.0 | 417 ns |
| **Full Temporal Opening** | $\mathcal{O}_1(x) = \mathcal{D}_1(\mathcal{E}_1(x))$ | 28.0 | 584 ns |
| **2-Bit Hysteresis Latch (HS-BPTN)** | 4-state saturating Smith automaton | 22.0 | 458 ns |
| **Anti-Glitch Parity Latch** | Conditional toggle with 2-tick confirmation | 15.0 | 313 ns |

*Earlier versions reported 6 cycles for the first three operators; those runs let the compiler fold
the constant input.*

---

## 3. BPTN layer, 16 channels × 16 neurons (synthetic masks)

The benchmark evaluates one feedforward BPTN layer:
* **Inputs:** 16 channels, each a 64-bit window (64 time steps).
* **Neurons:** 16 `CompactBptnNeuron<16>`, i.e. $16 \times 16 = 256$ 64-bit temporal synapses.
* **Masks and inputs are synthetic.** This is not a trained keyword-spotting model and no accuracy is
  attached; with this input all 16 neurons fire (mask `0xFFFF`).

### Measurements on STM32WLE5JC (2026-10-25, two runs, identical results)

| Configuration | Cycles per evaluation | Time |
| :--- | :---: | :---: |
| **48 MHz, 2 wait states, inputs through `black_box`** (current firmware) | **11,972** | **249.42 µs** |
| 4 MHz, 0 wait states, inputs through `black_box` (one control run) | 9,081 | 2.27 ms |
| 4 MHz, 0 wait states, constant inputs (earlier firmware) | 8,079 (8,193 in the first report) | 2.02 ms |

* **Unit cost:** 11,972 / (256 × 64) ≈ 0.73 cycle per synapse per time step.
* **Energy estimate, CPU active only:** ≈ **3.46 µJ** (249.42 µs × 13.86 mW). Estimate from an assumed
  run current, not a measurement; excludes sleep current, radio, sensors and audio acquisition.
* The 2 Flash wait states needed at 48 MHz add about 32 % cycles to this layer (9,081 → 11,972).

---

## 4. Comparison with baselines: not measured

Earlier versions of this report listed cycle counts, energy and SRAM figures for a LIF SNN
(~18,500 cycles) and a Micro-GRU (~58,000 cycles). Those numbers were typed into the firmware
as constants; they were **never measured on this board** and have been removed.

What is established on silicon (see sections 2 and 3): the BPTN layer cost in cycles.
What is not established: any speed, energy or memory advantage over LIF SNN, TFLite Micro or
CMSIS-NN on this board.

Host-side measurements on a shared synthetic task are in
[`comparative_benchmark_analysis.md`](comparative_benchmark_analysis.md): BPTN is slightly more accurate
and about 1.25× faster than a LIF neuron when one decision per window is enough, slower when it must
answer at every tick, and it needs more Flash and SRAM.

The protocol for a fair silicon comparison (same task, same accuracy, current measured with a
PPK2 or Joulescope, sleep current included) is tracked in the project plan.

---

## 5. Battery lifetime: not claimed

A previous version claimed more than 10 years on a CR2032. That figure counted only the energy
of the BPTN computation. A real node is dominated by sleep current, wake-ups, sensor and
microphone power, and any radio activity, none of which were measured. No lifetime is claimed
until the full duty cycle is measured.

---

## 6. Independent Replication Instructions

To reproduce these measurements on any Seeed Studio Wio-E5 board:
```bash
# 1. Connect ST-Link V2 probe (SWD: 3V3, GND, SWDIO, SWCLK)
# 2. Build, then flash and monitor real-time RTT telemetry:
(cd examples/wio_e5_benchmark && cargo build --release --bins)
probe-rs run --chip STM32WLE5JC examples/wio_e5_benchmark/target/thumbv7em-none-eabi/release/bptn_benchmark
```

---

## 7. Complete On-Target Hardware Test Suite (`silicon_test_runner`)

To ensure rigorous validation on target hardware, a dedicated test binary executes directly on the physical STM32WLE5JC CPU across 5 validation suites:

1. **Suite 1: Arithmetic & ARM Cortex-M4 DSP Intrinsics** (`Fix32`, `Fix16`, `ssat_16`, `smlad`, `qadd16`).
2. **Suite 2: Bit Vectorization & Hardware CTZ** (`SpikeBitset`, `CLZ`/`RBIT` instructions).
3. **Suite 3: BPTN Primitives & Morphological Filtering** ($\mathcal{D}_1$, $\mathcal{E}_1$, $\mathcal{O}_1$, Smith automaton `BptnHysteresisLatch`, `BptnRobustToggleCell`).
4. **Suite 4: Real Multi-Class KWS Inference** (bit-exact classification of "YES", "NO", "STOP" and total noise rejection).
5. **Suite 5: Compact SNN Populations & INT8 CSR Matrices** (`CompactLifPopulation` 5B/neuron, `QuantizedRomCsrSynapses`).

### Automated Single-Command Runner:
```bash
python3 tools/run_silicon_tests.py
```
*Physical Outcome:* **18/18 tests passed successfully on bare silicon (0 failures).**
