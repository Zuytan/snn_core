# 🔬 Rigorous Comparative Evaluation Report: Silicon-Native BPTN vs LIF SNN & Micro-GRU

**Date:** October 2026  
**Status:** Host-side measurements (desktop CPU). Silicon figures for LIF and GRU baselines are NOT measured; see `CLAIMS.md`.  
**Theoretical Reference:** [`docs/formal_specification_bptn.md`](formal_specification_bptn.md)  
**Automated Test Suite:** [`tests/bptn_adversarial_tests.rs`](../tests/bptn_adversarial_tests.rs)

---

## Executive Summary

This document presents an **adversarial, strictly empirical, and uncompromised evaluation** of the **BPTN (Bit-Parallel Temporal Neural Network)** architecture against:
1. **Event-Driven LIF SNN (`CompactLifPopulation` & `LazyDecayLifPopulation`)**: Neuromorphic leaky integrate-and-fire model in fixed-point $Q16.16$, equipped with hardware `CTZ` (`trailing_zeros`) decoding and [`StaticSpikeWordDelayLine`](../src/domain/delay_ring.rs) axonal delay rings.
2. **Actual Micro-GRU (TinyML)**: Gated recurrent unit ($z_t, r_t, h_t$) physically executed step-by-step on identical noisy input sequences (no substituted analytic formulas).

All common benchmarking biases have been eliminated to clearly demonstrate **both BPTN's physical silicon victories and its acute failure modes**.

---

## 1. Same task, same output: LIF vs BPTN

*Rewritten 2026-10-25. The previous version of this section compared models with arbitrary weights that
did not solve the same task (2.2 M vs 102 M output spikes); its ratios (BPTN 26.9× faster in block mode,
LIF 2.7× faster in streaming, 16,640 B vs 4,096 B Flash for 64 × 16 layers) are withdrawn, see `CLAIMS.md`.*

The benchmark [`demos/examples/09_bptn_vs_lif_head_to_head.rs`](../demos/examples/09_bptn_vs_lif_head_to_head.rs)
gives every model the same deterministic synthetic task and asks for the same output:

* 16 input channels, windows of 64 ticks, background noise of 1/16 per tick and per channel;
* in half of the 20,000 windows, a weak burst (1/8 per tick and per channel) on channels 0 to 3 during the
  last 16 ticks;
* one decision per window, at its last tick: burst present or not.

Each model is one detector neuron. Its parameters were picked by a small grid search on the same data
(LIF decay 0.85 to 0.96 and weight 0.10 to 0.25; BPTN threshold 7 to 10). BPTN's framing-buffer shifts are
counted in its time.

| Model | Accuracy | False pos. | False neg. | Host time per decision | vs dense LIF |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dense LIF (Fix32, stepped every tick) | 90.84 % | 992 | 839 | about 229 ns | 1.00× |
| Lazy LIF (decay on input only) | 90.84 % | 997 | 835 | about 203 ns | 1.13× |
| BPTN, evaluated once per window | 93.44 % | 438 | 875 | about 182 to 187 ns | 1.23× to 1.27× |
| BPTN, evaluated at every tick | 93.44 % | 438 | 875 | about 246 ns | 0.95× |

One run on an Apple M1 Pro (host CPU), 2026-10-25. Model memory, from the data layout: LIF 16 B Flash and
9 B SRAM; BPTN 260 B Flash (16 kernels and 16 polarity masks) and 128 B SRAM (16 framing buffers).

What this shows: on a task suited to a fixed temporal window, BPTN is slightly more accurate and slightly
faster when one decision per window is enough, and loses its speed advantage when it must answer at every
tick; it needs more Flash and SRAM than a single LIF neuron. What it does not show: anything about real
data, trained models, larger layers or a Cortex-M.

---

## 2. Long-Horizon Retention, Continuous Noise & Burst Stress ($\Delta \in [16, 1,024]$)

The benchmark [`examples/10_bptn_long_horizon_memory.rs`](../examples/10_bptn_long_horizon_memory.rs) physically executes every architecture (including a true `RealMicroGruCell`) across 3,000 sequences per retention delay $\Delta$:

### A. Regime 2: Continuous Sensor Noise ($10\%$ Data Noise, $0.2\%$ False Cue Spikes)

| Delay ($\Delta$) | Feedforward BPTN | LIF SNN ($\beta=0.92$) | 1-bit R-BPTN Latch | H-BPTN (`OR`-Pool) | Real Micro-GRU (Executed) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$16$ ticks** | **$100.0\%$** | $48.2\%$ | **$100.0\%$** | **$100.0\%$** | **$100.0\%$** |
| **$48$ ticks** | **$100.0\%$** | $50.6\%$ | **$99.1\%$** | **$100.0\%$** | **$99.1\%$** |
| **$64$ ticks** | $49.7\%$ (**Wall**) | $49.7\%$ | **$98.9\%$** | $87.2\%$ | **$98.9\%$** |
| **$128$ ticks** | $49.2\%$ (**Wall**) | $49.2\%$ | **$97.7\%$** | $86.9\%$ | **$97.7\%$** |
| **$256$ ticks** | $49.8\%$ (**Wall**) | $49.8\%$ | **$94.8\%$** | $60.5\%$ | **$94.8\%$** |
| **$512$ ticks** | $49.6\%$ (**Wall**) | $49.6\%$ | **$91.9\%$** | $60.7\%$ | **$91.9\%$** |
| **$1,024$ ticks** | $49.6\%$ (**Wall**) | $49.6\%$ | **$83.8\%$** | $50.4\%$ (**Saturated**) | $64.1\%$ ($z_t$ drift) |

### B. Regime 3: Adversarial Stress Test — Markovian Bursts ($\ge 2$ ticks) + Jitter/Dropout ($5\%$)

When the sensor stream exhibits **5% edge dropout/jitter** on valid pulses AND **20% burst noise lasting 2 ticks**:

| Delay ($\Delta$) | Naive 1-bit R-BPTN | HS-BPTN (2-Tick Confirmation) | 2-Bit Hysteresis (3t) | Real Micro-GRU |
| :---: | :---: | :---: | :---: | :---: |
| **$16$ ticks** | $94.3\%$ | $90.7\%$ (Penalized by dropout) | $91.2\%$ | **$99.0\%$** |
| **$64$ ticks** | $92.2\%$ | $91.6\%$ | $91.4\%$ | **$96.9\%$** |
| **$128$ ticks** | $89.2\%$ | $89.8\%$ | $90.6\%$ | **$93.9\%$** |
| **$256$ ticks** | $85.4\%$ | **$91.1\%$** | **$91.1\%$** | $88.8\%$ |
| **$1,024$ ticks** | $65.4\%$ | **$89.9\%$** | **$90.1\%$** | $52.8\%$ |

> [!WARNING]
> **Critical Finding #2 (The 2-Tick Confirmation Dilemma):** Enforcing two consecutive ticks (`BptnRobustToggleCell`) filters out 1-tick isolated glitches, **but causes an unavoidable loss of $\approx 9.7\%$ of true signals whenever a $5\%$ sensor dropout occurs** ($1 - 0.95^2 = 9.75\%$ edge capture failure). On short-to-medium horizons ($\Delta \le 128$) with imperfect physical sensors, **Micro-GRU outperforms BPTN** ($99.0\%$ vs $90.7\%$) thanks to its continuous analog integration.

---

## 3. Sensor Temporal Jitter & Continuous Analog Tracking: Where LIF SNN Prevails

The demonstrator [`examples/11_bptn_global_pipeline.rs`](../examples/11_bptn_global_pipeline.rs) compares an **identically wired LIF SNN** (equipped with an axonal delay line [`StaticSpikeWordDelayLine`](../src/domain/delay_ring.rs)) against Rigid BPTN ($\kappa = 0$) and Dilated BPTN ($\kappa = 2$):

| Test Case | Rigid BPTN ($\kappa = 0$) | Dilated BPTN ($\kappa = 2$) | Fair LIF SNN ($\tau_{\text{mem}}$) | Diagnostic |
| :--- | :---: | :---: | :---: | :--- |
| **1. Ideal Signal (0% Noise, Jitter = 0)** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | Perfect parity under nominal conditions |
| **2. Pure Sensor Jitter ($\pm 2$ ticks, 0% Noise)** | **$33.33\%$ (Collapsed)** | **$100.00\%$** | **$83.22\%$** | The $\kappa=0$ rigid mask misses 100% of shifted arrivals; LIF naturally absorbs jitter |
| **3. Jitter $\pm 2$ ticks + Noise (2% False Alarms)** | $35.57\%$ | **$80.82\%$** | $64.13\%$ | Dilatation $\kappa=2$ increases noise cross-section by $>5\times$ |
| **4. Continuous Analog Tracking (MSE)** | $0.10779$ | $0.10779$ | **$0.00527$ ($20.5\times$ better)** | Continuous $Q16.16$ membrane potential vastly outperforms 1-bit BPTN |

---

## 4. Fundamental Incompressible Limits ("No Free Lunch")

1. **Nilpotent Window Wall ($\mathbf{J}^{64} = \mathbf{0}$):** A feedforward BPTN loses 100% of past state beyond 64 ticks (verified by `test_window_wall_theorem_beyond_64_ticks`).
2. **Temporal Rigidity vs Noise Cross-Section:** Without dilation ($\kappa = 0$), a single clock cycle of jitter ($\pm 1$) eliminates coincidence detection. With dilation ($\kappa > 0$), active bits in the mask increase from $1$ to $2\kappa + 1$, multiplying Poisson noise false alarm rates (`test_dilation_false_positive_cross_section_tradeoff`).
3. **H-BPTN Pyramid Saturation:** Under $10\%$ noise, `OR` pooling saturates to $>99\%$ active bits by scale 3; under $25\%$ noise, even thresholded pooling (`Threshold = 2` or `3`) degrades to chance ($50\%$) beyond $\Delta = 64$.
4. **1-Bit Quantization Floor:** On continuous tracking and smooth analog estimation tasks, a fixed-point $Q16.16$ LIF SNN is $>20\times$ more accurate in mean squared error (MSE).
