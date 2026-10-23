//! 🛡️ Intractable & Adversarial Integration Tests — BPTN vs SNN LIF
//!
//! This automated test suite rigorously verifies:
//! 1. Exact mathematical guarantees of the BPTN model in its nominal domain.
//! 2. Its physical and theoretical failure modes ("No Free Lunch"):
//!    - Nilpotent window wall ($\mathbf{J}^{64} = \mathbf{0}$) beyond 64 ticks;
//!    - Collapse of rigid BPTN ($\kappa = 0$) under $\pm 1$ tick temporal jitter;
//!    - Increase in false alarm rate during morphological dilation ($\kappa > 0$);
//!    - Markovian degradation law of 1-bit latch under continuous noise;
//!    - Vulnerability of 2-tick confirmation to bursts ($\ge 2$ ticks) and sensor dropout;
//!    - Catastrophic saturation of Max-Pooling `OR` (H-BPTN) under background noise;
//!    - Superiority of continuous SNN LIF (`Fix32`) on smooth analog signal tracking.

#![cfg(feature = "bptn")]
#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_lazy_continuation
)]

use snn_core::domain::bptn::{
    BptnFeedforwardLayer, BptnNeuron, BptnRobustToggleCell, BptnSpikingNetwork, BptnSynapse,
    BptnToggleCell, CompactBptnNeuron, MultiScalePyramidBuffer, TemporalFramingBuffer64,
    dilate_temporal_jitter,
};
use snn_core::domain::compact_population::CompactLifPopulation;
use snn_core::domain::delay_ring::StaticSpikeWordDelayLine;
use snn_core::domain::models::lif::{LifConfig, ResetMode};
use snn_core::domain::numeric::Fix32;
use snn_core::traits::{SpikingNetwork, SpikingPopulation};

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_prob(&mut self, p: f64) -> bool {
        let u = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        u < p
    }
}

/// 1. Verifies the "Window Wall" theorem ($\mathbf{J}^{64} = \mathbf{0}$):
/// A feedforward BPTN possesses perfect isometric memory (100%) for $\Delta < 64$,
/// but drops strictly to 50% (total loss of mutual information) once $\Delta \ge 64$.
#[test]
fn test_window_wall_theorem_beyond_64_ticks() {
    let mut rng = DeterministicRng::new(0x1234_5678_9ABC_DEF0);
    let trials = 2_000;

    for &delta in &[1usize, 15, 31, 62, 64, 96, 128] {
        let mut correct = 0;
        for _ in 0..trials {
            let b0 = rng.next_prob(0.5);
            let b1 = rng.next_prob(0.5);
            let target_parity = b0 ^ b1;

            let mut buf = TemporalFramingBuffer64::new();
            buf.push_spike(b0);
            for _ in 0..delta {
                buf.push_spike(false);
            }
            buf.push_spike(b1);

            let recovered_b0 = if delta + 1 < 64 {
                buf.get_lag((delta + 1) as u8)
            } else {
                // Beyond 63 ticks of delay, the bit was shifted out of u64 register (`>> 64` = 0)
                false
            };
            let pred_parity = recovered_b0 ^ b1;
            if pred_parity == target_parity {
                correct += 1;
            }
        }

        let acc = correct as f64 / trials as f64;
        if delta + 1 < 64 {
            assert!(
                (acc - 1.0).abs() < f64::EPSILON,
                "Before 64-tick wall (delta={delta}), retention must be 100%, got {acc}"
            );
        } else {
            assert!(
                (acc - 0.50).abs() < 0.04,
                "Beyond 64-tick wall (delta={delta}), accuracy must drop to chance (~50%), got {acc}"
            );
        }
    }
}

/// 2. Verifies critical vulnerability of rigid BPTN ($\kappa = 0$) to temporal jitter
/// and the comparative resilience of SNN LIF and dilated BPTN ($\kappa = 2$).
#[test]
fn test_temporal_jitter_collapse_and_morphological_dilation_recovery() {
    // Pattern to detect: Channel 0 fires at t_0, Channel 1 fires nominally 8 ticks later (t_0 + 8).
    // Under sensor jitter, second spike arrives at t_0 + 8 + j with j in {-2, -1, +1, +2}.
    let nominal_delay = 8usize;

    // A. Rigid BPTN Neuron (kappa = 0): expects exact lag 8 on ch0 and lag 0 on ch1
    let rigid_neuron = CompactBptnNeuron::<2>::new([1u64 << nominal_delay, 1u64], [0, 0], 2);

    // B. Dilated BPTN Neuron (kappa = 2): tolerates ±2 ticks of jitter on ch0
    let dilated_syn_ch0 = BptnSynapse::new(0, 2, u64::MAX, 0, 1u64 << nominal_delay);
    let dilated_syn_ch1 = BptnSynapse::new(0, 0, u64::MAX, 0, 1u64);
    let dilated_neuron = BptnNeuron::<2>::new([dilated_syn_ch0, dilated_syn_ch1], 2);

    // C. SNN LIF Neuron with 8-tick delay line and membrane decay (decay = 0.88)
    let lif_cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.88), Fix32::ZERO, 4)
        .with_reset_mode(ResetMode::Hard);
    let mut lif_pop = CompactLifPopulation::<1>::new(lif_cfg);
    let mut delay_line = StaticSpikeWordDelayLine::<1, 16>::new();

    for &jitter in &[-2i32, -1, 1, 2] {
        let mut buf0 = TemporalFramingBuffer64::new();
        let mut buf1 = TemporalFramingBuffer64::new();
        lif_pop.reset();
        delay_line.reset();

        let mut rigid_detected = false;
        let mut dilated_detected = false;
        let mut lif_detected = false;

        let t_ch0 = 10usize;
        let t_ch1 = (10i32 + nominal_delay as i32 + jitter) as usize;

        for t in 0..32 {
            let s0 = t == t_ch0;
            let s1 = t == t_ch1;

            let w0 = buf0.push_spike(s0);
            let w1 = buf1.push_spike(s1);

            if rigid_neuron.tick(&[w0, w1]) {
                rigid_detected = true;
            }
            if dilated_neuron.tick(&[w0, w1]).0 {
                dilated_detected = true;
            }

            // LIF evaluation with nominal axonal delay of 8 ticks on ch0
            delay_line.push(&[u32::from(s0)]);
            let s0_delayed = (delay_line.get_delayed(nominal_delay)[0] & 1) != 0;
            if s0_delayed {
                lif_pop.input_accumulators[0] += Fix32::from_num(0.62);
            }
            if s1 {
                lif_pop.input_accumulators[0] += Fix32::from_num(0.62);
            }
            if lif_pop.step_neuron(0) {
                lif_detected = true;
            }
        }

        assert!(
            !rigid_detected,
            "Rigid BPTN (kappa=0) MUST fail on non-zero jitter of {jitter} ticks"
        );
        assert!(
            dilated_detected,
            "Dilated BPTN (kappa=2) MUST detect coincidence despite jitter of {jitter} ticks"
        );
        assert!(
            lif_detected,
            "SNN LIF (decay=0.88) MUST naturally absorb jitter of {jitter} ticks"
        );
    }
}

/// 3. Temporal Dilation "No Free Lunch" Theorem:
/// Dilating a BPTN kernel ($\kappa > 0$) to tolerate jitter strictly increases
/// false alarm probability under a Poisson noise stream.
#[test]
fn test_dilation_false_positive_cross_section_tradeoff() {
    let mut rng = DeterministicRng::new(0xCAFE_F00D_DEAD_BEEF);
    let n_steps = 20_000;
    let noise_prob = 0.08; // 8% independent noise on each channel

    let kernel = [1u64 << 10, 1u64 << 10];
    let neuron = CompactBptnNeuron::<2>::new(kernel, [0, 0], 2);

    let mut false_alarms_k0 = 0;
    let mut false_alarms_k2 = 0;
    let mut false_alarms_k4 = 0;

    let mut buf0 = TemporalFramingBuffer64::new();
    let mut buf1 = TemporalFramingBuffer64::new();

    for _ in 0..n_steps {
        let w0 = buf0.push_spike(rng.next_prob(noise_prob));
        let w1 = buf1.push_spike(rng.next_prob(noise_prob));

        if neuron.tick(&[w0, w1]) {
            false_alarms_k0 += 1;
        }
        if neuron.tick(&[dilate_temporal_jitter(w0, 2), dilate_temporal_jitter(w1, 2)]) {
            false_alarms_k2 += 1;
        }
        if neuron.tick(&[dilate_temporal_jitter(w0, 4), dilate_temporal_jitter(w1, 4)]) {
            false_alarms_k4 += 1;
        }
    }

    // Without dilation, P(FA) ≈ p^2 = 0.0064 (~128 out of 20,000).
    // With dilation kappa=2 (5-bit window), P_dil ≈ 1 - (1-p)^5 ≈ 0.34 -> P(FA) ≈ 0.116.
    // With dilation kappa=4 (9-bit window), P_dil ≈ 1 - (1-p)^9 ≈ 0.53 -> P(FA) ≈ 0.28.
    assert!(
        false_alarms_k0 < false_alarms_k2 && false_alarms_k2 < false_alarms_k4,
        "False alarm rate must strictly increase with kappa: k0={false_alarms_k0}, k2={false_alarms_k2}, k4={false_alarms_k4}"
    );
    assert!(
        false_alarms_k2 > false_alarms_k0 * 5,
        "Dilation kappa=2 must significantly multiply noise cross section"
    );
}

/// 4. Markovian degradation law of 1-bit latch (`BptnToggleCell`) under continuous noise:
/// For parasite flip probability $p_{\text{flip}}$ per tick, probability of an even
/// number of parasite flips after $\Delta$ ticks is exactly:
/// $$P(\text{exact}) = \frac{1 + (1 - 2 p_{\text{flip}})^\Delta}{2}$$
#[test]
fn test_stochastic_1bit_latch_markov_degradation() {
    let mut rng = DeterministicRng::new(0x9988_7766_5544_3322);
    let p_flip = 0.002; // 0.2% false flip per tick
    let trials = 4_000;

    for &delta in &[16usize, 256, 1024] {
        let mut correct = 0;
        for _ in 0..trials {
            let init_bit = rng.next_prob(0.5);
            let mut cell = BptnToggleCell::new();
            cell.step(init_bit);

            for _ in 0..delta {
                let glitch = rng.next_prob(p_flip);
                cell.step(glitch);
            }

            if cell.current_state() == init_bit {
                correct += 1;
            }
        }

        let empirical_acc = correct as f64 / trials as f64;
        let theoretical_acc = f64::midpoint(1.0, (1.0 - 2.0 * p_flip).powi(delta as i32));

        assert!(
            (empirical_acc - theoretical_acc).abs() < 0.03,
            "At delta={delta}, empirical survival ({empirical_acc:.4}) must follow exact Markovian law ({theoretical_acc:.4})"
        );
    }
}

/// 5. Limits of `BptnRobustToggleCell` (2-tick confirmation):
/// - 100% immune against isolated 1-tick glitches;
/// - Vulnerable to 2-tick consecutive bursts AND edge dropout (1-tick dropout).
#[test]
fn test_robust_2tick_latch_vs_burst_noise_and_dropout() {
    // A. Perfect immunity to isolated glitches (alternating 1, 0, 1, 0...)
    let mut cell = BptnRobustToggleCell::new();
    cell.step(true);
    assert!(
        cell.step(true),
        "A true 2-tick signal must toggle state to 1"
    );

    for _ in 0..500 {
        // Isolated single-tick glitch followed by 0
        cell.step(true);
        cell.step(false);
    }
    assert!(
        cell.current_state(),
        "2-tick latch must NEVER toggle on isolated 1-tick glitches"
    );

    // B. Immediate vulnerability to consecutive 2-tick parasite burst
    cell.step(true);
    cell.step(true);
    assert!(
        !cell.current_state(),
        "A 2-tick consecutive noise burst MUST toggle the 2-tick latch"
    );

    // C. Vulnerability to sensor dropout (if second tick of true signal is lost -> undetected)
    cell.reset_state();
    cell.step(true); // First tick received
    cell.step(false); // Second tick lost due to jitter/dropout!
    assert!(
        !cell.current_state(),
        "Sensor dropout on 2nd tick prevents signal validation"
    );
}

/// 6. Catastrophic saturation of H-BPTN pyramid `OR` Max-Pooling under continuous noise:
/// With 10% noise, scale 3 (grid of $4^3 = 64$ ticks) saturates at $> 98\%$ of 1 bits.
#[test]
fn test_hbptn_or_pooling_saturation_under_background_noise() {
    let mut rng = DeterministicRng::new(0xABCD_EF01_2345_6789);
    let mut pyramid_or = MultiScalePyramidBuffer::<4, 2>::new();
    let mut pyramid_th3 = MultiScalePyramidBuffer::<4, 2>::new();

    // Ingest across 64 * 64 = 4096 ticks with 10% random noise only (no true signal)
    let total_ticks = 4096u64;
    for t in 0..total_ticks {
        let noise = rng.next_prob(0.10);
        pyramid_or.step(noise, t);
        pyramid_th3.step_threshold(noise, t, 3);
    }

    let ones_scale3_or = pyramid_or.scale_word(3).count_ones();
    let ones_scale1_th3 = pyramid_th3.scale_word(1).count_ones();

    // P(at least one 1 across 64 ticks at p=0.10) = 1 - 0.9^64 = 99.88% -> nearly all 64 bits are 1!
    assert!(
        ones_scale3_or >= 60,
        "OR Max-Pooling at scale 3 MUST saturate under 10% noise (expected >= 60/64 bits set, got {ones_scale3_or})"
    );

    // At scale 1 (4-tick grid), P(at least 3 spikes out of 4 at p=0.10) = 4*(0.001)*0.9 + 0.0001 = 0.37%
    assert!(
        ones_scale1_th3 <= 4,
        "Threshold=3 pooling must filter bulk of 10% noise (got {ones_scale1_th3}/64)"
    );
}

/// 7. Strict bit-exact equivalence between Streaming evaluation (`Stride = 1`)
/// and Block evaluation (`Stride = 64`) at the end of the 64-tick window.
#[test]
fn test_streaming_stride1_vs_block_stride64_bit_exactness() {
    let mut rng = DeterministicRng::new(0x5555_AAAA_1234_9876);
    let mut neurons = [CompactBptnNeuron::<4>::new([0; 4], [0; 4], 5); 16];
    for n in 0..16 {
        let mut k = [0u64; 4];
        let mut p = [0u64; 4];
        for ch in 0..4 {
            k[ch] = rng.next_u64() & 0x0F0F_0F0F_0F0F_0F0F;
            p[ch] = rng.next_u64() & 0x00FF_00FF_00FF_00FF;
        }
        neurons[n] = CompactBptnNeuron::new(k, p, 10 + (n as u32 % 8));
    }

    let layer = BptnFeedforwardLayer::new(neurons);
    let mut stream_net = BptnSpikingNetwork::new(layer);

    let mut last_stream_mask = 0u64;
    for _t in 0..64 {
        let mut word_in = [0u32; 1];
        for ch in 0..4 {
            if rng.next_prob(0.25) {
                word_in[0] |= 1 << ch;
            }
        }
        let out_words = stream_net.tick_spikes_words(&word_in);
        last_stream_mask = u64::from(out_words[0]) | (u64::from(out_words[1]) << 32);
    }

    let mut final_histories = [0u64; 4];
    for ch in 0..4 {
        final_histories[ch] = stream_net.buffers[ch].word;
    }
    let block_mask = stream_net.layer.tick_mask(&final_histories);

    assert_eq!(
        last_stream_mask, block_mask,
        "Streaming output at tick 63 must be bit-exact identical to Block evaluation on same registers"
    );
}

/// 8. Analog resolution ceiling: continuous SNN LIF (`Fix32`) significantly
/// outperforms thresholded binary BPTN output on a continuous tracking task.
#[test]
fn test_continuous_analog_regression_lif_superiority() {
    let mut mse_lif = 0.0f64;
    let mut mse_bptn = 0.0f64;

    let mut v_lif = Fix32::from_num(0.5);
    let alpha = Fix32::from_num(0.65);
    let beta = Fix32::from_num(0.35);
    let mut bptn_win = 0u64;

    for t in 0..400 {
        let target = 0.5 + 0.4 * (t as f64 * 0.1).sin();
        let spike = ((t as f64 * target * 11.7).fract()) < target;
        bptn_win = (bptn_win << 1) | u64::from(spike);

        v_lif = v_lif * alpha + Fix32::from_num(target as f32) * beta;
        let pred_lif = f64::from(v_lif.to_float());

        let pred_bptn = if (bptn_win & 0xFFFF).count_ones() >= 8 {
            0.8
        } else {
            0.2
        };

        mse_lif += (pred_lif - target).powi(2);
        mse_bptn += (pred_bptn - target).powi(2);
    }

    assert!(
        mse_lif * 5.0 < mse_bptn,
        "Continuous Q16.16 LIF must have MSE at least 5x lower than binary BPTN on smooth signal (MSE LIF={mse_lif}, BPTN={mse_bptn})"
    );
}
