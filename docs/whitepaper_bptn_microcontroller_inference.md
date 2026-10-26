# Bit-Parallel Temporal Neural Networks (BPTN): Multi-Scale Temporal Inference on Word-RAM Microcontrollers without FPU or MAC Operations

**Status:** Draft, not peer reviewed, not submitted. Claims limited to what `CLAIMS.md` lists as established.  
**Author:** Zuytan  
**Source Code & Associated Firmware:** [https://github.com/Zuytan/snn_core](https://github.com/Zuytan/snn_core)  
**Qualified Target Silicon:** Seeed Studio Wio-E5 (STMicroelectronics STM32WLE5JC, ARM Cortex-M4 @ 48 MHz)

---

## Abstract

Artificial intelligence inference on ultra-low-power edge microcontrollers (ARM Cortex-M0+, Cortex-M4 without FPU, RISC-V RV32I) is historically constrained by the heavy computational overhead of Multiply-Accumulate (*MAC*) operations and the accuracy degradation of 8-bit integer quantization. While bio-inspired Spiking Neural Networks (SNN) replace dense activation multiplications with event-driven additions, they impose sequential step-by-step updates of analog membrane state variables ($V(t)$), consuming precious clock cycles and requiring continuous SRAM retention for exponential decay modeling.

In this work, we formally introduce the **Bit-Parallel Temporal Neural Network (BPTN)** architecture. BPTN abandons the biomimetic emulation of analog membrane potentials in favor of a silicon-centric algebraic formalism operating directly on the Word-RAM architecture of 32/64-bit microprocessors. By leveraging intra-word vectorization (*SWAR*), discrete 1D temporal morphology ($\mathcal{O}_\kappa$), and 4-state saturating Smith automata (HS-BPTN), BPTN executes multi-scale spatio-temporal convolutions in **$\mathcal{O}(1)$ branchless logical instructions (`XOR`, `AND`, `POPCNT`)**.

Deployed under real physical conditions on an **STM32WLE5JC microcontroller (bare ARM Cortex-M4, `#![no_std]`, zero SRAM heap allocation)**, BPTN evaluates a layer of 256 temporal synapses (synthetic masks, not a trained keyword-spotting model) across a 64-step window in **11,972 CPU cycles (249.42 µs at 48 MHz, 2 Flash wait states)**, corresponding to **0.73 cycle per synapse per time-step** with **ZERO hardware multiplications (0 MACs)**. No comparison with LIF SNN, TFLite Micro or CMSIS-NN baselines has been measured on the same board yet, and no accuracy is reported on a public dataset; those are the open items of this draft.

---

## 1. Introduction and Hardware Motivation

The proliferation of the Internet of Things (IoT) and autonomous sentinel sensing nodes imposes stringent energy efficiency mandates:
1. **Incompatibility of Standard RNN / CNN Models:**  
   Recurrent architectures (GRU, LSTM) and 1D temporal convolutions (TFLite Micro, CMSIS-NN) require thousands of 8/16-bit MAC operations per inference. On processor cores lacking DSP hardware accelerators or FPUs, these computations monopolize the CPU for multiple milliseconds ($> 1\text{ ms}$ for Micro-GRU), rapidly draining battery reserves.
2. **The Microcontroller SNN Dilemma:**  
   Leaky Integrate-and-Fire (LIF) neurons replace MAC operations with sparse event-driven additions. However, at every discrete time-step $\Delta t$, the processor must apply membrane potential decay:
   $$V_i(t) = V_i(t-1) \cdot \lambda + I_i(t)$$
   Even when scaled in Q16.16 fixed-point (`Fix32`), this formulation requires integer multiplication or shifting per neuron at every step, and demands persistent storage of $V_i$ in dynamic SRAM ($5$ to $9$ bytes/neuron), preventing the microcontroller from entering deep sleep modes (*Deep Sleep*).
3. **The BPTN Paradigm:**  
   BPTN posits that digital microcontrollers are not continuous biophysical systems, but synchronous machines operating on discrete 32- or 64-bit register words. By mapping the temporal history of a signal directly across the 64 bits of a machine word (`u64`), complete spatio-temporal correlation collapses into a single `AND` instruction followed by a hardware `POPCNT` bit-count.

---

## 2. Algebraic Formalism of BPTN

### 2.1. Word-RAM Representation Space

Consider a discrete binary event stream $s_j(t) \in \{0, 1\}$. At each sampling step $t$, recent temporal history across a window of $W = 64$ steps is mapped into a single machine word:
$$\mathbf{h}_j(t) = \sum_{\tau=0}^{W-1} s_j(t - \tau) \cdot 2^\tau \in \mathbb{F}_2^{64}$$
Temporal updates are performed in a single 1-cycle logical shift instruction:
$$\mathbf{h}_j(t) = (\mathbf{h}_j(t-1) \ll 1) \mid s_j(t)$$

### 2.2. BPTN Synaptic Operator

A BPTN synapse $\Sigma_{i,j}$ connecting input channel $j$ to neuron $i$ is formalized by the 4-tuple:
$$\Sigma_{i,j} = (\mathbf{k}_{i,j}, \mathbf{p}_{i,j}, \kappa_{i,j}, d_{i,j})$$
* $\mathbf{k}_{i,j} \in \mathbb{F}_2^{64}$: Synaptic kernel support mask.
* $\mathbf{p}_{i,j} \in \mathbb{F}_2^{64}$: Logical polarity mask ($0$: excitatory, $1$: inhibitory).
* $\kappa_{i,j} \in \mathbb{N}$: Morphological radius for temporal jitter tolerance.
* $d_{i,j} \in [0, 63]$: Programmable axonal delay.

Temporal coincidence evaluation across the entire 64-step window is governed by:
$$\mathbf{u}_{i,j}(t) = \left( \left(\mathbf{h}_j(t) \gg d_{i,j}\right) \oplus \mathbf{p}_{i,j} \right) \ \& \ \mathbf{k}_{i,j}$$

### 2.3. Membrane Potential and Threshold Firing

Membrane potential $V_i(t)$ represents the aggregate sum of temporal coincidences across all $M$ input channels:
$$V_i(t) = \sum_{j=1}^{M} \operatorname{POPCNT}\left(\mathbf{u}_{i,j}(t)\right)$$
The neuron fires an output spike if and only if its potential reaches or exceeds integer threshold $\theta_i$:
$$y_i(t) = \mathbb{I}\left(V_i(t) \ge \theta_i\right)$$

---

## 3. Morphological Filtering and Noise-Resistant Automata

In physical transmission channels subjected to thermal noise or spurious spikes (shot noise), naive bitwise coincidence is susceptible to false alarms. BPTN introduces two micro-architectural hardware defenses:

### 3.1. 1D Temporal Morphological Opening ($\mathcal{O}_\kappa$)

The 1D morphological opening operator removes isolated glitches without suppressing coherent burst packets:
$$\mathcal{O}_\kappa(\mathbf{x}) = \mathcal{D}_\kappa(\mathcal{E}_\kappa(\mathbf{x}))$$
* **Erosion ($\mathcal{E}_\kappa$):** Eliminates any isolated spike burst of duration $< 2\kappa + 1$:
  $$\mathcal{E}_1(\mathbf{x}) = \mathbf{x} \ \& \ (\mathbf{x} \ll 1) \ \& \ (\mathbf{x} \gg 1)$$
* **Dilation ($\mathcal{D}_\kappa$):** Reconstructs the temporal envelope of signals surviving erosion:
  $$\mathcal{D}_1(\mathbf{x}) = \mathbf{x} \mid (\mathbf{x} \ll 1) \mid (\mathbf{x} \gg 1)$$

### 3.2. 4-State Saturating HS-BPTN Predictor (Bit-Parallel Smith Automaton)

For recurrent connections and ultra-long-horizon retention ($> 1,000$ steps), the stochastic 1-bit T-latch is replaced with a 2-bit saturating state machine encoded in two register words $C_1, C_0 \in \mathbb{F}_2^{64}$.

Updating **64 parallel automata simultaneously** executes in branchless bit-parallel logic:
$$\begin{cases}
\operatorname{eff\_inc} = \operatorname{inc} \ \& \ \neg\operatorname{dec} \\
\operatorname{eff\_dec} = \operatorname{dec} \ \& \ \neg\operatorname{inc} \\
\operatorname{none} = \neg(\operatorname{eff\_inc} \mid \operatorname{eff\_dec}) \\
C_1' = (\operatorname{none} \ \& \ C_1) \mid (\operatorname{eff\_inc} \ \& \ (C_1 \mid C_0)) \mid (\operatorname{eff\_dec} \ \& \ (C_1 \ \& \ C_0)) \\
C_0' = (\operatorname{none} \ \& \ C_0) \mid (\operatorname{eff\_inc} \ \& \ (\neg C_0 \mid C_1)) \mid (\operatorname{eff\_dec} \ \& \ (C_1 \ \& \ \neg C_0))
\end{cases}$$

---

## 4. Empirical Silicon Measurements (STM32WLE5JC)

Physical tests were conducted on a **Seeed Studio Wio-E5** clocked at 48 MHz (2 Flash wait states, set and read back by the firmware) without FPU, instrumented with the ARM DWT `CYCCNT` register and SEGGER RTT telemetry over an SWD probe.

### 4.1. Micro-Architectural Primitives

| Hardware Primitive | Cycles (DWT), including `black_box` harness | Time @ 48 MHz |
| :--- | :---: | :---: |
| **1D Dilation ($\mathcal{D}_1$)** | 20.0 | 417 ns |
| **1D Erosion ($\mathcal{E}_1$)** | 20.0 | 417 ns |
| **Temporal Opening ($\mathcal{O}_1$)** | 28.0 | 584 ns |
| **2-Bit Hysteresis Latch (HS-BPTN)** | 22.0 | 458 ns |

Earlier drafts gave 6 to 12 cycles and throughputs derived from them; those runs used constant inputs the
compiler could fold, at the 4 MHz reset clock. They are withdrawn.

### 4.2. BPTN layer cost (256 temporal synapses / 64 steps, synthetic masks)

| Metric | Silicon-Native 64-bit BPTN (STM32WLE5JC @ 48 MHz) |
| :--- | :---: |
| **CPU cycles / inference** (DWT CYCCNT) | 11,972 cycles |
| **Execution time @ 48 MHz, 2 wait states** | 249.42 µs |
| **Energy estimate, CPU active only** | ≈ 3.46 µJ (249.42 µs × 13.86 mW; assumed current, not measured with a current probe) |
| **Multiplications** | 0 in the BPTN kernel |

Earlier drafts compared these figures with a LIF SNN and a Micro-GRU. Those baseline numbers were not
measured and were removed. See `CLAIMS.md` and `comparative_benchmark_analysis.md` for what is established.

---

## 5. Conclusion

BPTN is a research direction: aligning temporal computation with the word-parallel operations of a
Cortex-M core (AND, shift, popcount). What is established so far is its cycle cost on one board for one
synthetic layer. Open questions: accuracy on real datasets, trainability beyond toy tasks, and a fair
comparison with TFLite Micro / CMSIS-NN and LIF SNNs on the same task, accuracy and measured energy.
In tick-by-tick streaming, an event-driven LIF is faster than BPTN on the author's own host benchmark.
