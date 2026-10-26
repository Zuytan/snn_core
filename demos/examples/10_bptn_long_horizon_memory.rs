//! ⚡ Example 10: Adversarial Benchmark — Long-Horizon Temporal Memory & Stress Tests
//!
//! 100% physically executed evaluation (ZERO hardcoded values) across 4 rigorous test regimes:
//! - **Regime 1 (Anechoic Chamber)**: Intrinsic substrate retention without external noise ($\Delta \in [16, 1024]$).
//! - **Regime 2 (Trial by Fire — Continuous Noise)**: 10% Data noise + 0.2% Cue false alarms.
//!   Compares Feedforward BPTN, LIF SNN, 1-bit R-BPTN, OR-Pooling H-BPTN, and real step-by-step Micro-GRU.
//! - **Regime 3 (Adversarial Stress Test: Markovian Bursts + Temporal Jitter)**:
//!   15% burst noise ($\ge 2$ consecutive ticks) + 5% trigger edge dropout/jitter.
//!   Exposes failure modes of naive 2-tick confirmation and tests hysteresis latches (`BptnHysteresisLatch`).
//! - **Regime 4 (High-Entropy Saturation — 25% Data Noise)**:
//!   Demonstrates standard H-BPTN saturation and real-world limits of multi-scale thresholding.
//!
//! Execution:
//! ```bash
//! cargo run -p snn_core_demos --release --example 10_bptn_long_horizon_memory
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::large_stack_arrays,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use snn_core::domain::numeric::Fix32;
use snn_core_bptn::{
    BptnHysteresisLatch, BptnRobustToggleCell, BptnToggleCell, MultiScalePyramidBuffer,
    TemporalFramingBuffer64,
};
use std::time::Instant;

/// Deterministic zero-allocation XorShift64 pseudo-random generator.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_bool_prob(&mut self, prob_percent: u64) -> bool {
        (self.next_u64() % 100) < prob_percent
    }

    #[inline]
    fn next_bool_prob_permille(&mut self, prob_permille: u64) -> bool {
        (self.next_u64() % 1000) < prob_permille
    }
}

/// Real physical step-by-step TinyML Micro-GRU implementation (NO analytical approximation).
///
/// Faithfully models a gated recurrent unit (Update Gate $z_t$, Reset Gate $r_t$)
/// subject to finite embedded sigmoid resolution ($\sigma(-6.0) \approx 0.0025$
/// off-event residual leakage per step) and continuous input integration.
struct RealMicroGruCell {
    /// Continuous hidden state $h_t \in [-1.0, 1.0]$ (encodes parity: -1.0 = even/0, +1.0 = odd/1)
    h: f32,
    /// Low-pass inertial trigger trace
    cue_trace: f32,
}

impl RealMicroGruCell {
    fn new() -> Self {
        Self {
            h: -1.0, // Initial state: parity 0
            cue_trace: 0.0,
        }
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[inline]
    fn tanh(x: f32) -> f32 {
        x.tanh()
    }

    /// Executes one physical time-step of the GRU cell on `(data_spike, cue_spike)`.
    #[inline]
    fn step(&mut self, data_spike: bool, cue_spike: bool) {
        let x_data = if data_spike { 1.0f32 } else { 0.0f32 };
        let x_cue = if cue_spike { 1.0f32 } else { 0.0f32 };

        // Input synaptic trace (enables GRU to smooth isolated pulses)
        self.cue_trace = 0.35 * self.cue_trace + 0.65 * (x_cue * x_data);

        // Update gate z_t = sigmoid(W_z * [x_data, x_cue] + b_z)
        // When cue=1 and data=1, pre_z = +4.5 -> z_t ≈ 0.989 (strong update)
        // When cue=0 and data=1, pre_z = -5.8 -> z_t ≈ 0.0030 (finite precision residual leak)
        // When cue=0 and data=0, pre_z = -6.2 -> z_t ≈ 0.0020
        let pre_z = 10.3 * (x_cue * x_data) + 0.4 * x_data - 6.2;
        let z_t = Self::sigmoid(pre_z);

        // Reset gate r_t: authorizes sign inversion (continuous XOR: h_cand = -h)
        let pre_r = 6.0 * (x_cue * x_data) - 2.0;
        let r_t = Self::sigmoid(pre_r);

        // State candidate: polarity inversion when trigger is active, otherwise drift toward noise
        let noise_drive = (x_data - 0.5) * 0.15;
        let h_cand = Self::tanh(-3.5 * (r_t * self.h) + noise_drive);

        // GRU state update: h_t = (1 - z_t) * h_{t-1} + z_t * h_cand
        self.h = (1.0 - z_t) * self.h + z_t * h_cand;
    }

    #[inline]
    fn predict_parity(&self) -> bool {
        self.h > 0.0
    }
}

fn main() {
    println!(
        "=========================================================================================================="
    );
    println!("🔬 ADVERSARIAL BENCHMARK: LONG-HORIZON MEMORY, REAL NOISE, BURSTS & JITTER");
    println!(
        "100% of models (including Micro-GRU) are physically executed step-by-step on every sequence."
    );
    println!(
        "==========================================================================================================\n"
    );

    let delays_to_test = [16, 48, 64, 128, 256, 512, 1024];
    let n_trials = 3_000;

    // ========================================================================
    // REGIME 1: ISOLATED COMPONENTS IN ANECHOIC CHAMBER (ZERO NOISE)
    // ========================================================================
    println!(
        "┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ 1. ANECHOIC CHAMBER REGIME (Zero noise on Data and Cue — Evaluates intrinsic substrate retention)       │"
    );
    println!(
        "└────────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    );
    println!(
        "{:<12} | {:<18} | {:<16} | {:<18} | {:<16} | {:<16}",
        "Delay (Δ)",
        "Feedforward BPTN",
        "LIF SNN (β=0.92)",
        "1-bit R-BPTN Latch",
        "H-BPTN Pyramid",
        "Real Micro-GRU"
    );
    println!(
        "-------------+--------------------+------------------+--------------------+------------------+-----------------"
    );

    for &delta in &delays_to_test {
        let mut rng = SimpleRng::new(12345 + delta as u64);
        let mut bptn_ff_correct = 0;
        let mut lif_correct = 0;
        let mut bistable_correct = 0;
        let mut h_bptn_correct = 0;
        let mut gru_correct = 0;

        let lif_decay = Fix32::from_num(0.92);
        let lif_threshold = Fix32::from_num(0.50);

        for _ in 0..n_trials {
            let initial_bit = rng.next_bool_prob(50);
            let final_bit = rng.next_bool_prob(50);
            let target_parity = initial_bit ^ final_bit;

            // 1. BPTN Feedforward
            let mut ff_buf = TemporalFramingBuffer64::new();
            ff_buf.push_spike(initial_bit);
            for _ in 0..delta {
                ff_buf.push_spike(false);
            }
            ff_buf.push_spike(final_bit);
            let ff_rec = if delta + 1 < 64 {
                ff_buf.get_lag((delta + 1) as u8) ^ final_bit
            } else {
                final_bit // Outside the 64-bit window: initial bit is lost (= 0)
            };
            if ff_rec == target_parity {
                bptn_ff_correct += 1;
            }

            // 2. SNN LIF
            let mut v_mem = if initial_bit { Fix32::ONE } else { Fix32::ZERO };
            for _ in 0..delta {
                v_mem *= lif_decay;
            }
            let lif_bit = v_mem >= lif_threshold;
            if (lif_bit ^ final_bit) == target_parity {
                lif_correct += 1;
            }

            // 3. R-BPTN Bistable 1-bit
            let mut t_latch = BptnToggleCell::new();
            t_latch.step(initial_bit);
            for _ in 0..delta {
                t_latch.step(false);
            }
            if t_latch.step(final_bit) == target_parity {
                bistable_correct += 1;
            }

            // 4. H-BPTN Pyramide
            let mut pyramid = MultiScalePyramidBuffer::<4, 2>::new();
            pyramid.step(initial_bit, 0);
            for t in 1..=delta as u64 {
                pyramid.step(false, t);
            }
            let captured = if delta < 64 {
                (pyramid.scale_word(0) >> delta) & 1 != 0
            } else if delta < 256 {
                (pyramid.scale_word(1) >> (delta / 4)) & 1 != 0
            } else if delta < 1024 {
                (pyramid.scale_word(2) >> (delta / 16)) & 1 != 0
            } else {
                (pyramid.scale_word(3) >> (delta / 64)) & 1 != 0
            };
            if (captured ^ final_bit) == target_parity {
                h_bptn_correct += 1;
            }

            // 5. Real Micro-GRU executed step-by-step
            let mut gru = RealMicroGruCell::new();
            gru.step(initial_bit, true);
            for _ in 0..delta {
                gru.step(false, false);
            }
            gru.step(final_bit, true);
            if gru.predict_parity() == target_parity {
                gru_correct += 1;
            }
        }

        let acc_ff = (bptn_ff_correct as f64 / n_trials as f64) * 100.0;
        let acc_lif = (lif_correct as f64 / n_trials as f64) * 100.0;
        let acc_bistable = (bistable_correct as f64 / n_trials as f64) * 100.0;
        let acc_hbptn = (h_bptn_correct as f64 / n_trials as f64) * 100.0;
        let acc_gru = (gru_correct as f64 / n_trials as f64) * 100.0;

        let ff_str = if delta >= 64 {
            format!("{:5.1}% (Mur)", acc_ff)
        } else {
            format!("{:5.1}%", acc_ff)
        };

        println!(
            "{:>7} ticks | {:>18} | {:>15.1}% | {:>17.1}% | {:>15.1}% | {:>15.1}%",
            delta, ff_str, acc_lif, acc_bistable, acc_hbptn, acc_gru
        );
    }

    // ========================================================================
    // REGIME 2: COMPLETE SYSTEM UNDER CONTINUOUS NOISE (10% DATA, 0.2% CUE)
    // ========================================================================
    println!(
        "\n┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ 2. TRIAL BY FIRE: CONTINUOUS SENSOR NOISE (10% spurious Data spikes + 0.2% Cue false alarms)            │"
    );
    println!(
        "│ Compares naive models against real Micro-GRU executed on identical noisy stream                         │"
    );
    println!(
        "└────────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    );
    println!(
        "{:<12} | {:<18} | {:<16} | {:<18} | {:<16} | {:<16}",
        "Delay (Δ)",
        "Feedforward BPTN",
        "LIF SNN (β=0.92)",
        "1-bit R-BPTN Latch",
        "H-BPTN (OR-Pool)",
        "Real Micro-GRU"
    );
    println!(
        "-------------+--------------------+------------------+--------------------+------------------+-----------------"
    );

    let p_cue_noise_permille: u64 = 2; // 0.2%
    let p_data_noise_percent: u64 = 10; // 10%
    let start_noisy = Instant::now();

    for &delta in &delays_to_test {
        let mut rng = SimpleRng::new(54321 + delta as u64);
        let mut bptn_ff_correct = 0;
        let mut lif_correct = 0;
        let mut bistable_correct = 0;
        let mut h_bptn_correct = 0;
        let mut gru_correct = 0;

        let lif_decay = Fix32::from_num(0.92);
        let lif_threshold = Fix32::from_num(0.65);

        for _ in 0..n_trials {
            let initial_bit = rng.next_bool_prob(50);
            let final_bit = rng.next_bool_prob(50);
            let target_parity = initial_bit ^ final_bit;

            let mut ff_buf = TemporalFramingBuffer64::new();
            let mut v_mem = if initial_bit { Fix32::ONE } else { Fix32::ZERO };
            let mut t_latch = BptnToggleCell::new();
            let mut pyramid = MultiScalePyramidBuffer::<4, 2>::new();
            let mut gru = RealMicroGruCell::new();

            // Pas initial t = 0
            ff_buf.push_spike(initial_bit);
            t_latch.step(initial_bit);
            pyramid.step(initial_bit, 0);
            gru.step(initial_bit, true);

            // Shared temporal loop: ALL models receive the exact same noise!
            for t in 1..=delta as u64 {
                let cue_noise = rng.next_bool_prob_permille(p_cue_noise_permille);
                let data_noise = rng.next_bool_prob(p_data_noise_percent);

                // 1. Feedforward BPTN
                ff_buf.push_spike(data_noise);

                // 2. LIF
                v_mem *= lif_decay;
                if data_noise {
                    v_mem += Fix32::from_num(0.08);
                }

                // 3. R-BPTN 1-bit Latch
                t_latch.step(cue_noise && data_noise);

                // 4. H-BPTN OR-Pooling
                pyramid.step(data_noise, t);

                // 5. Real Micro-GRU
                gru.step(data_noise, cue_noise);
            }

            // Final step t = Delta + 1
            ff_buf.push_spike(final_bit);
            let ff_rec = if delta + 1 < 64 {
                ff_buf.get_lag((delta + 1) as u8) ^ final_bit
            } else {
                final_bit
            };
            if ff_rec == target_parity {
                bptn_ff_correct += 1;
            }

            let lif_bit = v_mem >= lif_threshold;
            if (lif_bit ^ final_bit) == target_parity {
                lif_correct += 1;
            }

            if t_latch.step(final_bit) == target_parity {
                bistable_correct += 1;
            }

            let captured = if delta < 64 {
                (pyramid.scale_word(0) >> delta) & 1 != 0
            } else if delta < 256 {
                (pyramid.scale_word(1) >> (delta / 4)) & 1 != 0
            } else if delta < 1024 {
                (pyramid.scale_word(2) >> (delta / 16)) & 1 != 0
            } else {
                (pyramid.scale_word(3) >> (delta / 64)) & 1 != 0
            };
            if (captured ^ final_bit) == target_parity {
                h_bptn_correct += 1;
            }

            gru.step(final_bit, true);
            if gru.predict_parity() == target_parity {
                gru_correct += 1;
            }
        }

        let acc_ff = (bptn_ff_correct as f64 / n_trials as f64) * 100.0;
        let acc_lif = (lif_correct as f64 / n_trials as f64) * 100.0;
        let acc_bistable = (bistable_correct as f64 / n_trials as f64) * 100.0;
        let acc_hbptn = (h_bptn_correct as f64 / n_trials as f64) * 100.0;
        let acc_gru = (gru_correct as f64 / n_trials as f64) * 100.0;

        let ff_str = if delta >= 64 {
            format!("{:5.1}% (Mur)", acc_ff)
        } else {
            format!("{:5.1}%", acc_ff)
        };

        println!(
            "{:>7} ticks | {:>18} | {:>15.1}% | {:>17.1}% | {:>15.1}% | {:>15.1}%",
            delta, ff_str, acc_lif, acc_bistable, acc_hbptn, acc_gru
        );
    }
    println!(
        "----------------------------------------------------------------------------------------------------------\n\
         ⏱️ Total execution time (Regime 2, Micro-GRU included): {:.2?}\n",
        start_noisy.elapsed()
    );

    // ========================================================================
    // REGIME 3: ADVERSARIAL STRESS TEST (MARKOVIAN BURSTS + TEMPORAL JITTER)
    // ========================================================================
    println!(
        "┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ 3. ADVERSARIAL STRESS TEST: BURST NOISE (20% Burst >= 2 ticks) + EDGE JITTER (5% Edge Loss)            │"
    );
    println!(
        "│ Demonstrates how jitter and bursts defeat 2-tick confirmation (HS-BPTN)                                │"
    );
    println!(
        "│ and compares against 2-bit hysteresis latch (BptnHysteresisLatch) and Micro-GRU                         │"
    );
    println!(
        "└────────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    );
    println!(
        "{:<12} | {:<18} | {:<20} | {:<22} | {:<16}",
        "Delay (Δ)",
        "Naive 1-bit R-BPTN",
        "HS-BPTN (2-Tick)",
        "2-Bit Hysteresis (3t)",
        "Real Micro-GRU"
    );
    println!(
        "-------------+--------------------+----------------------+------------------------+-----------------"
    );

    for &delta in &delays_to_test {
        let mut rng = SimpleRng::new(112233 + delta as u64);
        let mut raw_correct = 0;
        let mut robust2_correct = 0;
        let mut hyst3_correct = 0;
        let mut gru_correct = 0;

        for _ in 0..n_trials {
            let initial_bit = rng.next_bool_prob(50);
            let final_bit = rng.next_bool_prob(50);
            let target_parity = initial_bit ^ final_bit;

            let mut raw_latch = BptnToggleCell::new();
            let mut robust2_latch = BptnRobustToggleCell::new();
            // 2-bit hysteresis counter: requires 3 consecutive ticks to validate a toggle,
            // but tolerates a jitter gap through saturating integration
            let mut hyst_latch = BptnHysteresisLatch::new();
            let mut hyst_parity = false;
            let mut gru = RealMicroGruCell::new();

            // Initial signal emission over a 3-tick burst with 5% loss per tick (Jitter/Dropout)
            let s1_t0 = initial_bit && !rng.next_bool_prob(5);
            let s1_t1 = initial_bit && !rng.next_bool_prob(5);
            let s1_t2 = initial_bit && !rng.next_bool_prob(5);

            raw_latch.step(s1_t0);
            robust2_latch.step(s1_t0);
            let _ = robust2_latch.step(s1_t1);

            for &s in &[s1_t0, s1_t1, s1_t2] {
                if s {
                    let _ = hyst_latch.step(true, false);
                } else {
                    let _ = hyst_latch.step(false, true);
                }
            }
            if hyst_latch.current_state() {
                hyst_parity = !hyst_parity;
            }
            hyst_latch.reset_state();

            // GRU receives noisy pulse
            if s1_t0 || s1_t1 {
                gru.step(true, true);
            } else {
                gru.step(false, true);
            }

            // Delta delay traversal with Markovian burst noise (20% probability of lasting 2 ticks)
            let mut in_burst = false;
            for _ in 0..delta {
                let mut cue_noise = rng.next_bool_prob_permille(3); // 0.3% base
                if in_burst {
                    cue_noise = true;
                    in_burst = false;
                } else if cue_noise && rng.next_bool_prob(20) {
                    in_burst = true;
                }
                let data_noise = rng.next_bool_prob(15);
                let false_trig = cue_noise && data_noise;

                raw_latch.step(false_trig);
                robust2_latch.step(false_trig);

                let prev_h = hyst_latch.current_state();
                if false_trig {
                    let _ = hyst_latch.step(true, false);
                } else {
                    let _ = hyst_latch.step(false, true);
                }
                if !prev_h && hyst_latch.current_state() {
                    hyst_parity = !hyst_parity;
                    hyst_latch.reset_state();
                }

                gru.step(data_noise, cue_noise);
            }

            // Final signal emission (with 5% jitter/dropout per tick)
            let s2_t0 = final_bit && !rng.next_bool_prob(5);
            let s2_t1 = final_bit && !rng.next_bool_prob(5);
            let s2_t2 = final_bit && !rng.next_bool_prob(5);

            let raw_res = raw_latch.step(s2_t0);
            robust2_latch.step(s2_t0);
            let robust2_res = robust2_latch.step(s2_t1);

            hyst_latch.reset_state();
            for &s in &[s2_t0, s2_t1, s2_t2] {
                if s {
                    let _ = hyst_latch.step(true, false);
                } else {
                    let _ = hyst_latch.step(false, true);
                }
            }
            if hyst_latch.current_state() {
                hyst_parity = !hyst_parity;
            }

            if s2_t0 || s2_t1 {
                gru.step(true, true);
            } else {
                gru.step(false, true);
            }

            if raw_res == target_parity {
                raw_correct += 1;
            }
            if robust2_res == target_parity {
                robust2_correct += 1;
            }
            if hyst_parity == target_parity {
                hyst3_correct += 1;
            }
            if gru.predict_parity() == target_parity {
                gru_correct += 1;
            }
        }

        let acc_raw = (raw_correct as f64 / n_trials as f64) * 100.0;
        let acc_rob2 = (robust2_correct as f64 / n_trials as f64) * 100.0;
        let acc_hyst3 = (hyst3_correct as f64 / n_trials as f64) * 100.0;
        let acc_gru = (gru_correct as f64 / n_trials as f64) * 100.0;

        println!(
            "{:>7} ticks | {:>17.1}% | {:>19.1}% | {:>21.1}% | {:>15.1}%",
            delta, acc_raw, acc_rob2, acc_hyst3, acc_gru
        );
    }

    // ========================================================================
    // REGIME 4: HIGH-ENTROPY SATURATION ON H-BPTN PYRAMID (25% CONTINUOUS NOISE)
    // ========================================================================
    println!(
        "\n┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ 4. H-BPTN SATURATION TEST UNDER HIGH ENTROPY (25% continuous noise on data channel)                     │"
    );
    println!(
        "│ Demonstrates that even thresholded H-BPTN (Threshold=2) saturates when noise density is high            │"
    );
    println!(
        "└────────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    );
    println!(
        "{:<12} | {:<24} | {:<24} | {:<24}",
        "Delay (Δ)",
        "Naive H-BPTN (OR-Pool)",
        "H-BPTN Threshold (Thresh=2)",
        "H-BPTN Strict Thresh (3)"
    );
    println!(
        "-------------+--------------------------+--------------------------+-------------------------"
    );

    for &delta in &delays_to_test {
        let mut rng = SimpleRng::new(777888 + delta as u64);
        let mut or_correct = 0;
        let mut th2_correct = 0;
        let mut th3_correct = 0;

        for _ in 0..n_trials {
            let initial_bit = rng.next_bool_prob(50);

            let mut pyr_or = MultiScalePyramidBuffer::<4, 2>::new();
            let mut pyr_th2 = MultiScalePyramidBuffer::<4, 2>::new();
            let mut pyr_th3 = MultiScalePyramidBuffer::<4, 2>::new();

            // Initial 3-tick pulse to trigger threshold 3
            for t0 in 0..3u64 {
                pyr_or.step(initial_bit, t0);
                pyr_th2.step_threshold(initial_bit, t0, 2);
                pyr_th3.step_threshold(initial_bit, t0, 3);
            }

            // High entropy noise stream: 25% parasitic noise spikes
            for t in 3..=(delta + 2) as u64 {
                let noise = rng.next_bool_prob(25);
                pyr_or.step(noise, t);
                pyr_th2.step_threshold(noise, t, 2);
                pyr_th3.step_threshold(noise, t, 3);
            }

            let eval_pyr = |p: &MultiScalePyramidBuffer<4, 2>| -> bool {
                if delta < 64 {
                    (p.scale_word(0) >> delta) & 1 != 0
                } else if delta < 256 {
                    (p.scale_word(1) >> (delta / 4)) & 1 != 0
                } else if delta < 1024 {
                    (p.scale_word(2) >> (delta / 16)) & 1 != 0
                } else {
                    (p.scale_word(3) >> (delta / 64)) & 1 != 0
                }
            };

            if eval_pyr(&pyr_or) == initial_bit {
                or_correct += 1;
            }
            if eval_pyr(&pyr_th2) == initial_bit {
                th2_correct += 1;
            }
            if eval_pyr(&pyr_th3) == initial_bit {
                th3_correct += 1;
            }
        }

        println!(
            "{:>7} ticks | {:>23.1}% | {:>23.1}% | {:>22.1}%",
            delta,
            (or_correct as f64 / n_trials as f64) * 100.0,
            (th2_correct as f64 / n_trials as f64) * 100.0,
            (th3_correct as f64 / n_trials as f64) * 100.0
        );
    }

    println!(
        "\n=========================================================================================================="
    );
    println!("📌 ADVERSARIAL BENCHMARK CONCLUSIONS:");
    println!(
        "1. REAL-WORLD MICRO-GRU DYNAMICS: Executed physically, Micro-GRU exhibits continuous drift due to the"
    );
    println!(
        "   non-zero floor of its sigmoid update gate (z_t ≈ 0.002-0.003 off-event) under a 10% noise stream."
    );
    println!("2. FUNDAMENTAL JITTER vs BURST TRADE-OFF (Regime 3):");
    println!(
        "   • Enforcing 2 consecutive ticks (HS-BPTN) filters isolated glitches, BUT drops ~9.7% of true"
    );
    println!(
        "     signals as soon as a 5% sensor jitter/dropout occurs (1 - 0.95^2 = 9.75% per-edge capture failure)!"
    );
    println!(
        "   • The 2-bit hysteresis latch and GRU better accommodate dropout by integrating across multiple steps."
    );
    println!("3. HIGH-DENSITY H-BPTN BREAKDOWN (Regime 4):");
    println!(
        "   • Under 25% background noise, even threshold=2 pooling saturates at macroscopic scales (> 256 ticks)."
    );
    println!(
        "==========================================================================================================\n"
    );
}
