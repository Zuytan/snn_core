# 🔬 Wio-E5 Silicon Profiling Testbench (STM32WLE5JC / ARM Cortex-M4)

This subproject provides an instrumented cycle-accurate bare-metal (`no_std`) testbench for the **STMicroelectronics STM32WLE5JC** microcontroller (ARM Cortex-M4 core @ 48 MHz without FPU) powering the **Seeed Studio Wio-E5**.

It leverages the ARM hardware cycle counter `DWT->CYCCNT` and streams real-time metrics to the host terminal via **RTT** (*Real-Time Transfer*).

---

## 🔌 Hardware Wiring (ST-Link V2 <-> Wio-E5)

Connect the SWD pins from your ST-Link probe to the corresponding pins on the Wio-E5:

| ST-Link V2 Pin | Wio-E5 Pin (Mini / Grove / Module) | Description |
| :---: | :---: | :--- |
| **3.3V** | **3V3 / VDD** | Target logic power supply (mandatory) |
| **GND** | **GND** | Common ground |
| **SWDIO** | **DIO / PA13** | SWD data line |
| **SWCLK** | **CLK / PA14** | SWD clock line |
| *(Optional) RST* | **RST / NRST** | Hardware reset line |

---

## 🚀 Flashing and Execution via `probe-rs`

Once wiring and 3.3V power are verified:

```bash
# 1. Navigate to benchmark directory
cd examples/wio_e5_benchmark

# 2. Flash and stream real-time RTT telemetry
cargo run --release
```

Or directly using the `probe-rs` CLI:

```bash
probe-rs run --chip STM32WLE5JC --protocol swd target/thumbv7em-none-eabi/release/wio_e5_benchmark
```

---

## 📊 Metrics Evaluated by Firmware

The firmware profiles and reports:
1. **Neuron Dynamics**: Exact CPU cycle costs per time-step for `LIF` (Hard & Subtract Reset), `ALIF`, and `Izhikevich` models.
2. **CSR Synapse Matrices**: Cycles per propagated synapse in `Fix32` vs `QuantizedRomCsr` (INT8 without multiplication, measuring direct hardware speedup).
3. **SoA Populations**: SRAM footprint and throughput comparison between `Fix32` (9 bytes/neuron) and `Fix16` (5 bytes/neuron, saving 44% SRAM).
4. **ActivationArena**: Inter-layer *Ping-Pong* double-buffering verification without persistent accumulators.
5. **Lazy Decay Engine**: Dynamic measurements under varying sparsity regimes ($99\%$, $90\%$, $0\%$) identifying the operational break-even threshold.
6. **BPTN Engine**: 64-bit Word-RAM temporal window processing (`XOR`, `AND`, `POPCNT`).
