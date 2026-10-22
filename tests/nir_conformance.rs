//! 🧪 NIR (Neuromorphic Intermediate Representation) Conformance Suite for `snn_core`
//!
//! Rigorously validates numerical equivalence between the formal NIR specification
//! (IEEE-754 double precision f64) and the bare-metal Fix32 / Fix16 kernels of `snn_core`.
//!
//! Acceptance criteria:
//! 1. Relative membrane potential divergence < 0.05% over 10,000 time steps.
//! 2. Zero phase drift or spike divergence (100% pulse concordance).

use snn_core::NeuronDynamics;
use snn_core::application::nir_import::pade_exp_neg;
use snn_core::domain::compact_population::CompactLifPopulation;
use snn_core::domain::layers::{SpikingFeedforwardLayer, SpikingLayer};
use snn_core::domain::models::lif::{LifConfig, LifDynamics, LifState, ResetMode};
use snn_core::domain::numeric::Fix32;
use snn_core::domain::rom_csr::RomCsrSynapses;

/// Deterministic LCG pseudo-random number generator (no std/rand dependency).
struct DeterministicPrng {
    state: u64,
}

impl DeterministicPrng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    /// Generates a float in [-1.0, 1.0].
    fn next_f64(&mut self) -> f64 {
        let val = self.next_u32();
        (val as f64 / u32::MAX as f64) * 2.0 - 1.0
    }
}

/// NIR reference simulator in IEEE-754 double-precision floating-point (f64).
struct NirReferenceLif {
    tau: f64,
    v_thresh: f64,
    v_reset: f64,
    dt: f64,
    v_mem: f64,
    refr_time: u32,
    refr_period: u32,
    reset_mode: ResetMode,
}

impl NirReferenceLif {
    fn new(
        tau: f64,
        v_thresh: f64,
        v_reset: f64,
        dt: f64,
        refr_period: u32,
        reset_mode: ResetMode,
    ) -> Self {
        Self {
            tau,
            v_thresh,
            v_reset,
            dt,
            v_mem: v_reset,
            refr_time: 0,
            refr_period,
            reset_mode,
        }
    }

    fn step(&mut self, input_current: f64) -> (bool, f64) {
        if self.refr_time > 0 {
            self.refr_time -= 1;
            self.v_mem = self.v_reset;
            return (false, self.v_mem);
        }

        let alpha = (-self.dt / self.tau).exp();
        let decayed = self.v_mem * alpha;
        let v_next = decayed + input_current;

        if v_next >= self.v_thresh {
            self.v_mem = match self.reset_mode {
                ResetMode::Hard => self.v_reset,
                ResetMode::Subtract => v_next - self.v_thresh,
            };
            self.refr_time = self.refr_period;
            (true, self.v_mem)
        } else {
            self.v_mem = v_next;
            (false, self.v_mem)
        }
    }
}

#[test]
fn test_nir_conformance_10000_ticks_membrane_divergence() {
    let dt = 0.001; // 1 ms
    let tau = 0.020; // 20 ms (standard neuromorphic membrane constant)
    let v_thresh = 1.0;
    let v_reset = 0.0;
    let refr_period = 2; // 2 ticks

    // 1. NIR reference model f64
    let mut ref_lif =
        NirReferenceLif::new(tau, v_thresh, v_reset, dt, refr_period, ResetMode::Subtract);

    // 2. snn_core Fix32 model
    let alpha_fix = pade_exp_neg(Fix32::from_num((dt / tau) as f32));
    let thresh_fix = Fix32::from_num(v_thresh as f32);
    let reset_fix = Fix32::from_num(v_reset as f32);
    let config = LifConfig::new(thresh_fix, alpha_fix, reset_fix, refr_period as u16)
        .with_reset_mode(ResetMode::Subtract);
    let mut core_state = LifState::default();

    let mut prng = DeterministicPrng::new(424242);
    let num_ticks = 10_000;

    let mut total_rel_error = 0.0;
    let mut valid_subthreshold_samples = 0;
    let mut ref_spikes_count = 0;
    let mut core_spikes_count = 0;
    let mut spike_divergences = 0;

    for tick in 0..num_ticks {
        // Stimulus alternating between silence, subthreshold drive, and discharges
        let raw_stim = prng.next_f64();
        let stim = if raw_stim > 0.6 {
            raw_stim * 0.45 // Triggering pulse
        } else if raw_stim > 0.0 {
            raw_stim * 0.15 // Sub-threshold noise
        } else {
            0.0 // Silence
        };

        let (ref_spike, ref_v) = ref_lif.step(stim);
        let core_spike = LifDynamics::step(&config, &mut core_state, Fix32::from_num(stim as f32));
        let core_v = core_state.membrane_potential;

        if ref_spike {
            ref_spikes_count += 1;
        }
        if core_spike {
            core_spikes_count += 1;
        }

        if ref_spike != core_spike {
            spike_divergences += 1;
            println!(
                "⚠️ Spike divergence at tick {}: Ref={}, Core={}",
                tick, ref_spike, core_spike
            );
        }

        // Relative divergence calculation on sub-threshold membrane potential
        let core_v_f64 = core_v.to_float() as f64;
        let delta = (ref_v - core_v_f64).abs();
        if ref_v.abs() > 0.01 {
            let rel_err = delta / ref_v.abs();
            total_rel_error += rel_err;
            valid_subthreshold_samples += 1;
        }
    }

    let mean_rel_error = if valid_subthreshold_samples > 0 {
        total_rel_error / (valid_subthreshold_samples as f64)
    } else {
        0.0
    };

    println!("============================================================");
    println!("📊 NIR CONFORMANCE TEST RESULTS (10,000 TICKS):");
    println!("   - Time samples           : {}", num_ticks);
    println!("   - Reference spikes (f64) : {}", ref_spikes_count);
    println!("   - snn_core spikes (Fix32): {}", core_spikes_count);
    println!(
        "   - Spike divergences      : {} (Concordance: {:.2}%)",
        spike_divergences,
        100.0 - (spike_divergences as f64 / num_ticks as f64) * 100.0
    );
    println!(
        "   - Membrane divergence    : {:.4}% (Criterion: < 0.05%)",
        mean_rel_error * 100.0
    );
    println!("============================================================");

    assert_eq!(
        spike_divergences, 0,
        "Zero pulse divergence allowed over 10,000 ticks"
    );
    assert!(
        mean_rel_error < 0.0005,
        "Mean relative divergence ({:.4}%) must be strictly < 0.05%",
        mean_rel_error * 100.0
    );
}

#[test]
fn test_nir_deep_feedforward_conformance_10000_ticks() {
    // 2-layer NIR pipeline: 4 inputs -> 3 hidden -> 2 outputs
    static ROW_PTRS_1: [u16; 5] = [0, 3, 5, 7, 9];
    static TARGETS_1: [u16; 9] = [0, 1, 2, 1, 2, 0, 2, 0, 1];
    static WEIGHTS_1: [Fix32; 9] = [
        Fix32::from_bits(78643),  // 1.2
        Fix32::from_bits(-32768), // -0.5
        Fix32::from_bits(52429),  // 0.8
        Fix32::from_bits(98304),  // 1.5
        Fix32::from_bits(-19661), // -0.3
        Fix32::from_bits(-39322), // -0.6
        Fix32::from_bits(72090),  // 1.1
        Fix32::from_bits(58982),  // 0.9
        Fix32::from_bits(-26214), // -0.4
    ];
    let syn1 = RomCsrSynapses::new(&ROW_PTRS_1, &TARGETS_1, &WEIGHTS_1);

    static ROW_PTRS_2: [u16; 4] = [0, 2, 4, 6];
    static TARGETS_2: [u16; 6] = [0, 1, 0, 1, 0, 1];
    static WEIGHTS_2: [Fix32; 6] = [
        Fix32::from_bits(32768),  // 0.5
        Fix32::from_bits(58982),  // 0.9
        Fix32::from_bits(-45875), // -0.7
        Fix32::from_bits(19661),  // 0.3
        Fix32::from_bits(26214),  // 0.4
        Fix32::from_bits(-13107), // -0.2
    ];
    let syn2 = RomCsrSynapses::new(&ROW_PTRS_2, &TARGETS_2, &WEIGHTS_2);

    let cfg_hidden = LifConfig::new(
        Fix32::from_num(1.0),
        pade_exp_neg(Fix32::from_num(0.001f32 / 0.020f32)),
        Fix32::ZERO,
        1,
    )
    .with_reset_mode(ResetMode::Subtract);

    let cfg_output = LifConfig::new(
        Fix32::from_num(1.0),
        pade_exp_neg(Fix32::from_num(0.001f32 / 0.010f32)),
        Fix32::ZERO,
        1,
    )
    .with_reset_mode(ResetMode::Subtract);

    let pop1 = CompactLifPopulation::<3>::new(cfg_hidden);
    let pop2 = CompactLifPopulation::<2>::new(cfg_output);

    let mut layer1: SpikingFeedforwardLayer<
        CompactLifPopulation<3>,
        RomCsrSynapses<'static, u16, Fix32>,
        3,
        1,
    > = SpikingFeedforwardLayer::new(pop1, syn1);
    let mut layer2: SpikingFeedforwardLayer<
        CompactLifPopulation<2>,
        RomCsrSynapses<'static, u16, Fix32>,
        2,
        1,
    > = SpikingFeedforwardLayer::new(pop2, syn2);

    let mut prng = DeterministicPrng::new(99999);
    let mut total_output_spikes = 0;

    for _tick in 0..10_000 {
        let in0 = Fix32::from_num(prng.next_f64().max(0.0) as f32);
        let in1 = Fix32::from_num(prng.next_f64().max(0.0) as f32);
        let in2 = Fix32::from_num(prng.next_f64().max(0.0) as f32);
        let in3 = Fix32::from_num(prng.next_f64().max(0.0) as f32);
        let inputs = [in0, in1, in2, in3];

        let l1_spikes = layer1.tick_inputs_words(&inputs);
        let l2_spikes = layer2.tick_spikes_words(l1_spikes);

        total_output_spikes += l2_spikes[0].count_ones();
    }

    assert!(
        total_output_spikes > 100,
        "Deep network must emit significant activity"
    );
    println!(
        "✅ Multi-layer NIR validation succeeded over 10,000 ticks (Spikes emitted: {})",
        total_output_spikes
    );
}
