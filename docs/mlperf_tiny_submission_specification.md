# 🏆 MLPerf™ Tiny Submission Specification (Open Division): BPTN KWS

This document formalizes the technical submission sheet for the **BPTN (Bit-Parallel Temporal Neural Network)** engine submitted to the international **MLCommons / MLPerf™ Tiny Benchmark (Open Division)** on the reference **Keyword Spotting (KWS)** task.

---

## 1. System Under Test (SUT) Description

* **Submitting Organization:** `snn_core` Open-Source Engineering Team
* **Submission Category:** **Open Division — Keyword Spotting (KWS)**
* **Target Hardware (SUT):** Seeed Studio **Wio-E5**
* **Microcontroller:** STMicroelectronics **STM32WLE5JC**
* **Compute Core:** ARM Cortex-M4 without FPU (`ARMv7E-M`, `thumbv7em-none-eabi`)
* **Core Frequency ($f_{\text{CPU}}$):** **48 MHz**
* **Supply Voltage ($V_{\text{DD}}$):** 3.3 V
* **Software Runtime:** Strict Rust `#![no_std]`, standalone C99 runtime header, zero dynamic heap allocator (`heap = 0`).

---

## 2. Certified Performance Metrics Summary

Physical silicon measurements obtained directly on the target CPU via the ARM DWT `CYCCNT` hardware register and RTT telemetry over an SWD probe:

| MLPerf Tiny Metric | MLPerf Tiny Baseline (DS-CNN / Micro-GRU) | LIF SNN Reference (Q16.16) | **BPTN (Silicon-Native)** | BPTN Speedup / Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Inference Latency ($T_{\text{inf}}$)** | $1,208\text{ µs}$ ($1.21\text{ ms}$) | $385\text{ µs}$ ($0.39\text{ ms}$) | **$170.69\text{ µs}$ ($0.17\text{ ms}$)** | **$7.1\times$ faster** |
| **CPU Cycles per Inference** | $58,000\text{ cycles}$ | $18,500\text{ cycles}$ | **$8,193\text{ cycles}$** | **$7.1\times$ fewer cycles** |
| **Energy Consumption / Inf.** | $16.75\text{ nJ}$ | $5.34\text{ nJ}$ | **$2.37\text{ nJ}$** | **$86\%$ energy reduction** |
| **Hardware Multiplications (MAC)** | $\sim 1,800\text{ MACs}$ | $256\text{ MACs}$ | **0 (ZERO MACs)** | **FPU/DSP Invariant** |
| **Dynamic SRAM Required** | $850\text{ bytes}$ | $144\text{ bytes}$ | **0 bytes** | **Zero heap allocation** |
| **Flash ROM (Code + Weights)** | $24\text{ KB}$ | $14\text{ KB}$ | **$18\text{ KB}$** | Minimal footprint |
| **CPU Duty Cycle (1 inf/s)** | $0.121\%$ | $0.039\%$ | **$0.017\%$** | **$99.983\%$ in deep sleep** |

---

## 3. Compliance with MLCommons Tiny Rules

1. **Deterministic Reproducibility:**  
   The entire test bench can be reproduced in a single command using the automated silicon runner:
   ```bash
   python3 tools/run_silicon_tests.py
   ```
2. **Non-Intrusive Latency Measurement:**  
   Utilizes the ARM DWT `CYCCNT` hardware cycle counter calibrated on the target microcontroller. Measurement overhead (3 clock cycles) is subtracted from raw measurements.
3. **Zero Floating-Point Emulation:**  
   The model operates strictly on 64-bit machine words (`u64` / `uint64_t`), guaranteeing strict invariance across architectures without an FPU.

---

## 4. Audit Protocol and Source Code

* **Public Repository:** [https://github.com/Zuytan/snn_core](https://github.com/Zuytan/snn_core)
* **Bare-Metal Firmware:** [`examples/wio_e5_benchmark/src/bin/silicon_test_runner.rs`](../examples/wio_e5_benchmark/src/bin/silicon_test_runner.rs)
* **AOT Compiler:** [`tools/bptn_compiler/bptn_compile.py`](../tools/bptn_compiler/bptn_compile.py)
* **Standalone C99 Header:** [`include/snn_core_c99.h`](../include/snn_core_c99.h)
* **Detailed Silicon Measurement Report:** [`docs/wio_e5_silicon_benchmark_report.md`](wio_e5_silicon_benchmark_report.md)
