//! 🥊 Example 6: Quantitative Head-to-Head Comparison — TinyML (Micro-ANN) vs `snn_core` (SNN)
//!
//! Compares operation counts, memory footprint and host execution time on the SAME CPU
//! for a 24/7 sentinel monitoring task (95% quiescent baseline, 5% transient anomaly events).
//! Energy is not reported: this host demo measures neither cycles on a target nor current.
//!
//! Execution:
//! ```bash
//! cargo run -p snn_core_demos --example 06_tinyml_vs_snn_head_to_head
//! ```

use snn_core::metrics::{SparsityMeter, SynOpsTracker};
use snn_core::traits::Readout;
use snn_core::{
    CompactLifPopulation, Fix32, GenericStaticNetwork, LifConfig, ResetMode, StaticCsrSynapses,
    StaticReadoutLayer, Synapse,
};
use std::time::Instant;

const N_NEURONS: usize = 64;
const N_NEURONS_P1: usize = N_NEURONS + 1;
const MAX_INTERNAL: usize = 512;
const MAX_INPUT: usize = 64;
const N_INPUTS: usize = 4;
const N_INPUTS_P1: usize = N_INPUTS + 1;
const N_CLASSES: usize = 2;
const FEATURE_DIM: usize = N_NEURONS + 1;

static READOUT_WEIGHTS: [[Fix32; FEATURE_DIM]; N_CLASSES] = [
    [Fix32::from_bits(6554); FEATURE_DIM],
    [Fix32::from_bits(13107); FEATURE_DIM],
];

/// Conventional TinyML equivalent baseline (Micro-RNN / Dense INT8) at iso-synaptic capacity:
/// - Dense input layer: 4 inputs -> 64 neurons = 256 MACs
/// - Dense recurrent layer: 512 active connections = 512 MACs
/// - Dense classification layer: 64 -> 2 classes = 128 MACs
/// - Total equivalent capacity: 896 systematic MACs per frame (synchronous non-sparse model).
struct ConventionalTinyMlModel {
    weights_in: [i8; N_INPUTS * N_NEURONS],
    weights_rec: [i8; MAX_INTERNAL],
    weights_out: [i8; N_NEURONS * N_CLASSES],
    hidden_state: [i8; N_NEURONS],
}

impl ConventionalTinyMlModel {
    pub fn new() -> Self {
        Self {
            weights_in: [1; N_INPUTS * N_NEURONS],
            weights_rec: [1; MAX_INTERNAL],
            weights_out: [1; N_NEURONS * N_CLASSES],
            hidden_state: [0; N_NEURONS],
        }
    }

    /// Executes dense INT8 TinyML inference (896 constant MACs per frame).
    #[inline(never)]
    pub fn infer(&mut self, input: &[f32; 4]) -> (usize, u64) {
        let mut macs: u64 = 0;

        // 1. Dense input projection (4 -> 64)
        for i in 0..N_NEURONS {
            let mut acc: i32 = 0;
            for (ch, &val) in input.iter().enumerate() {
                let w = self.weights_in[ch * N_NEURONS + i] as i32;
                acc += ((val * 10.0) as i32) * w;
                macs += 1;
            }
            // 2. Dense recurrence (512 synapses)
            if i * 8 < MAX_INTERNAL {
                for k in 0..8 {
                    let w = self.weights_rec[i * 8 + k] as i32;
                    let prev_h = self.hidden_state[(i + k) % N_NEURONS] as i32;
                    acc += prev_h * w;
                    macs += 1;
                }
            }
            self.hidden_state[i] = (acc.clamp(-128, 127)) as i8;
        }

        // 3. Dense output classification (64 -> 2)
        let mut best_class = 0;
        let mut max_logit = i32::MIN;
        for c in 0..N_CLASSES {
            let mut logit: i32 = 0;
            for i in 0..N_NEURONS {
                let w = self.weights_out[c * N_NEURONS + i] as i32;
                logit += (self.hidden_state[i] as i32) * w;
                macs += 1;
            }
            if logit > max_logit {
                max_logit = logit;
                best_class = c;
            }
        }

        (best_class, macs)
    }
}

fn main() {
    println!("==========================================================================");
    println!("🥊 QUANTITATIVE HEAD-TO-HEAD BENCHMARK ON IDENTICAL CPU (x86_64 / ARM)");
    println!("Task: 24/7 Sentinel Monitoring (1,000 frames: 95% quiescent, 5% alert)");
    println!("==========================================================================\n");

    const TOTAL_FRAMES: usize = 1_000;
    const ANOMALY_FRAMES: usize = 50;

    // ------------------------------------------------------------------------
    // 1. Initialize snn_core (Static no_std Fix32 SNN with CompactLifPopulation)
    // ------------------------------------------------------------------------
    let config = LifConfig::new(
        Fix32::ONE,
        Fix32::from_bits(58982), // decay = 0.90
        Fix32::ZERO,
        1,
    )
    .with_reset_mode(ResetMode::Subtract);
    let neurons = CompactLifPopulation::<N_NEURONS>::new(config);

    let mut input_synapses = Vec::new();
    for ch in 0..N_INPUTS {
        for target in (ch * 8)..((ch + 1) * 8) {
            input_synapses.push(Synapse::new(ch, target, Fix32::from_num(1.5)));
        }
    }

    let mut internal_synapses = Vec::new();
    let mut rng = fastrand::Rng::with_seed(42);
    for i in 0..N_NEURONS {
        for j in 0..N_NEURONS {
            if i != j && rng.f32() < 0.15 && internal_synapses.len() < MAX_INTERNAL {
                let w = if rng.f32() < 0.2 {
                    Fix32::from_num(-0.6)
                } else {
                    Fix32::from_num(0.4)
                };
                internal_synapses.push(Synapse::new(i, j, w));
            }
        }
    }

    let internal_csr =
        StaticCsrSynapses::<N_NEURONS_P1, MAX_INTERNAL>::from_synapses(&internal_synapses);
    let input_csr = StaticCsrSynapses::<N_INPUTS_P1, MAX_INPUT>::from_synapses(&input_synapses);

    let mut snn_network =
        GenericStaticNetwork::<_, _, _, N_NEURONS, N_INPUTS>::new(neurons, internal_csr, input_csr);
    let snn_readout = StaticReadoutLayer::<N_CLASSES, FEATURE_DIM>::new(&READOUT_WEIGHTS);

    let mut snn_synops = SynOpsTracker::new();
    let mut snn_sparsity = SparsityMeter::new();

    // ------------------------------------------------------------------------
    // 2. Initialize Classical TinyML Baseline
    // ------------------------------------------------------------------------
    let mut tinyml_model = ConventionalTinyMlModel::new();

    // ------------------------------------------------------------------------
    // 3. Benchmark Execution under Identical Input Data
    // ------------------------------------------------------------------------
    let mut snn_state = [Fix32::ZERO; N_NEURONS];

    // Profile snn_core
    let start_snn = Instant::now();
    for frame in 0..TOTAL_FRAMES {
        let is_anomaly = frame >= (TOTAL_FRAMES - ANOMALY_FRAMES);
        let input = if is_anomaly {
            [
                Fix32::from_num(2.0),
                Fix32::from_num(1.8),
                Fix32::ZERO,
                Fix32::ZERO,
            ]
        } else {
            // Quiescent background fluctuation (low amplitude sub-threshold)
            [
                Fix32::from_num(0.05),
                Fix32::ZERO,
                Fix32::from_num(0.02),
                Fix32::ZERO,
            ]
        };

        let spikes = *snn_network.tick(&input);

        // Record afferent input SynOps (active channel propagation)
        for (ch, &val) in input.iter().enumerate() {
            if val > Fix32::ZERO && ch + 1 < snn_network.input_synapses.row_ptrs.len() {
                let start = snn_network.input_synapses.row_ptrs[ch] as usize;
                let end = snn_network.input_synapses.row_ptrs[ch + 1] as usize;
                snn_synops.record_synaptic_spikes(end.saturating_sub(start));
            }
        }

        // Record recurrent AC SynOps only when neurons spike
        for &spiker in &snn_network.active_spikes_buf[..snn_network.active_spikes_count] {
            let start = snn_network.internal_synapses.row_ptrs[spiker as usize] as usize;
            let end = snn_network.internal_synapses.row_ptrs[spiker as usize + 1] as usize;
            snn_synops.record_synaptic_spikes(end.saturating_sub(start));
        }

        // Membrane decay operations (1 MAC per neuron per time-step)
        snn_synops.record_dense_mac(N_NEURONS);

        for i in 0..N_NEURONS {
            snn_state[i] = (snn_state[i] * Fix32::from_num(0.85))
                + if spikes[i] { Fix32::ONE } else { Fix32::ZERO };
        }

        snn_sparsity.record_tick(N_NEURONS, snn_network.active_spikes_count);

        // Autonomous wake-up decision:
        // No ground-truth oracle! The network activates the Readout classifier
        // only when internal collective activity exceeds its event-driven threshold.
        let wakeup_event = snn_network.active_spikes_count >= 2;
        if wakeup_event {
            snn_synops.record_dense_mac(N_CLASSES * FEATURE_DIM);
            let _ = Readout::predict(&snn_readout, &snn_state);
        }
    }
    let duration_snn = start_snn.elapsed();

    // Profile TinyML baseline
    let start_tinyml = Instant::now();
    let mut tinyml_total_macs: u64 = 0;
    for frame in 0..TOTAL_FRAMES {
        let is_anomaly = frame >= (TOTAL_FRAMES - ANOMALY_FRAMES);
        let input_f32 = if is_anomaly {
            [2.0, 1.8, 0.0, 0.0]
        } else {
            [0.05, 0.0, 0.02, 0.0]
        };

        let (_, macs) = tinyml_model.infer(&input_f32);
        tinyml_total_macs += macs;
    }
    let duration_tinyml = start_tinyml.elapsed();

    // ------------------------------------------------------------------------
    // 4. Memory footprint
    // ------------------------------------------------------------------------
    // Strict memory separation: SRAM (dynamic state) vs Flash ROM (.rodata)
    let snn_sram_bytes = core::mem::size_of::<[Fix32; N_NEURONS]>() // Membrane potentials
        + core::mem::size_of::<[bool; N_NEURONS]>() // Spike buffer
        + core::mem::size_of_val(&snn_state); // Readout state traces
    let tinyml_sram_bytes = core::mem::size_of_val(&tinyml_model.hidden_state);

    let snn_flash_bytes = core::mem::size_of_val(&internal_synapses)
        + core::mem::size_of_val(&input_synapses)
        + core::mem::size_of_val(&READOUT_WEIGHTS);
    let tinyml_flash_bytes = core::mem::size_of_val(&tinyml_model.weights_in)
        + core::mem::size_of_val(&tinyml_model.weights_rec)
        + core::mem::size_of_val(&tinyml_model.weights_out);

    println!("📊 COMPARATIVE RESULTS (1,000 SENTINEL MONITORING FRAMES)");
    println!("----------------------------------------------------------------------------------");
    println!(
        "Metric                         | Conventional TinyML  | snn_core (SNN)       | Trade-offs & Analysis"
    );
    println!(
        "-------------------------------+----------------------+----------------------+--------------------"
    );
    println!(
        "Computed Operations            | {:12} MACs  | {:12} SynOps | x{:.1} fewer operations",
        tinyml_total_macs,
        snn_synops.total_synops(),
        (tinyml_total_macs as f64) / (snn_synops.total_synops().max(1) as f64)
    );
    println!(
        "Total CPU Time (1,000 frames)  | {:12.2?}      | {:12.2?}      | x{:.1} faster",
        duration_tinyml,
        duration_snn,
        duration_tinyml.as_secs_f64() / duration_snn.as_secs_f64()
    );
    println!(
        "SRAM Memory (Dynamic state)    | {:10} B        | {:10} B       | Micro-overhead SRAM",
        tinyml_sram_bytes, snn_sram_bytes,
    );
    println!(
        "Flash ROM Memory (Model weights)| {:10} B       | {:10} B       | Sparse CSR vs Dense INT8",
        tinyml_flash_bytes, snn_flash_bytes,
    );
    println!(
        "Sparsity Rate (Inactivity)     |          0.0%        |        {:5.1}%        | Event-driven sparse",
        snn_sparsity.sparsity_percent()
    );
    println!("----------------------------------------------------------------------------------");
    println!(
        "\nNote: fewer operations do not imply less energy on a CPU. Energy is not reported here:"
    );
    println!("it needs cycles on the target and a current measurement. See CLAIMS.md.");
    println!("==========================================================================");
}
