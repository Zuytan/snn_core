//! Example 9: LIF vs Lazy LIF vs BPTN on the same detection task
//!
//! Host (x86/ARM desktop) comparison on one shared, labelled task, so that the models are
//! judged on the same output and the same accuracy metric.
//!
//! **Task (synthetic, deterministic):** 16 input channels, windows of $W=64$ ticks.
//! Every channel fires with probability 1/16 per tick (background noise). In half of the windows,
//! a weak burst (1/8 per tick and per channel) is added on the target group (channels 0 to 3) during the last 16 ticks of the window.
//! Each model gives one decision per window, at its last tick: "burst present" or not.
//!
//! Models, each a single detector neuron reading the target group:
//! 1. **Dense LIF (`CompactLifPopulation` Q16.16)**: stepped at every tick; decision = the neuron
//!    fired during the last 16 ticks.
//! 2. **Lazy LIF (`LazyDecayLifPopulation`)**: same model, decay applied only when input arrives.
//! 3. **BPTN, block mode**: inputs are shifted into `TemporalFramingBuffer64` at every tick (cost
//!    counted), and the neuron is evaluated once, at the last tick (kernel = last 16 ticks).
//! 4. **BPTN, streaming mode**: same, but the neuron is evaluated at every tick, as when an output
//!    is required at each tick; the decision is the output at the last tick.
//!
//! The parameters are not trained: each model's best value was picked by a small grid search on
//! this data (LIF decay 0.85 to 0.96 and weight 0.10 to 0.25; BPTN threshold 7 to 10). The data is synthetic. Latencies are host
//! timings and vary between machines; they say nothing about a Cortex-M by themselves.
//!
//! Execution:
//! ```bash
//! cargo run -p snn_core_demos --release --example 09_bptn_vs_lif_head_to_head
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use snn_core::domain::compact_population::CompactLifPopulation;
use snn_core::domain::lazy_population::LazyDecayLifPopulation;
use snn_core::domain::models::lif::{LifConfig, ResetMode};
use snn_core::domain::numeric::Fix32;
use snn_core_bptn::{BptnFeedforwardLayer, CompactBptnNeuron, TemporalFramingBuffer64};
use std::hint::black_box;
use std::time::Instant;

const N_INPUT_CHANNELS: usize = 16;
const WINDOW_SIZE: usize = 64;
const N_WINDOWS: usize = 20_000;

/// Target group: channels 0 to 3.
const TARGET_CHANNELS: usize = 4;
/// The burst, and the decision horizon, cover the last 16 ticks of a window.
const DECISION_HORIZON: usize = 16;
/// Ticks `[0, DECISION_START)` of a window precede the decision horizon.
const DECISION_START: usize = WINDOW_SIZE - DECISION_HORIZON;

/// BPTN kernel: bits 0 to 15 of a framing buffer are the last 16 ticks (bit 0 = present).
const BPTN_KERNEL: u64 = (1u64 << DECISION_HORIZON) - 1;
/// A burst adds a spike with probability `1 / 2^BURST_LOG2_INV` per tick on each target channel.
const BURST_LOG2_INV: u32 = 3;
/// About 4 noise spikes are expected on the target group over 16 ticks, about 12 with a burst.
const BPTN_THRESHOLD: u32 = 8;

/// Deterministic xorshift64 generator, so that every run sees the same data.
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Returns true with probability `1 / 2^log2_inv`.
    fn chance(&mut self, log2_inv: u32) -> bool {
        self.next() & ((1u64 << log2_inv) - 1) == 0
    }
}

/// One labelled window: input spikes per tick (bit `ch` = channel `ch`) and the label.
struct Window {
    ticks: [u32; WINDOW_SIZE],
    burst: bool,
}

fn generate_windows() -> Vec<Window> {
    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
    (0..N_WINDOWS)
        .map(|_| {
            let burst = rng.chance(1);
            let mut ticks = [0u32; WINDOW_SIZE];
            for t in 0..WINDOW_SIZE {
                for ch in 0..N_INPUT_CHANNELS {
                    // Background noise: 1/16 per tick and per channel.
                    let noise = rng.chance(4);
                    // Burst: 1/8 per tick on the target group during the decision horizon.
                    let in_burst = burst && ch < TARGET_CHANNELS && t >= DECISION_START;
                    if noise || (in_burst && rng.chance(BURST_LOG2_INV)) {
                        ticks[t] |= 1 << ch;
                    }
                }
            }
            Window { ticks, burst }
        })
        .collect()
}

/// Confusion counts of one model over all windows.
#[derive(Default)]
struct Score {
    true_pos: usize,
    false_pos: usize,
    true_neg: usize,
    false_neg: usize,
}

impl Score {
    fn record(&mut self, predicted: bool, actual: bool) {
        match (predicted, actual) {
            (true, true) => self.true_pos += 1,
            (true, false) => self.false_pos += 1,
            (false, false) => self.true_neg += 1,
            (false, true) => self.false_neg += 1,
        }
    }

    fn accuracy_percent(&self) -> f64 {
        let correct = self.true_pos + self.true_neg;
        let total = correct + self.false_pos + self.false_neg;
        100.0 * correct as f64 / total as f64
    }
}

/// Result of one model: confusion counts and host time per decision (one 64-tick window).
struct Run {
    name: &'static str,
    score: Score,
    ns_per_decision: f64,
}

fn lif_config() -> LifConfig {
    let decay = Fix32::from_bits(58982); // about 0.90
    LifConfig::new(Fix32::ONE, decay, Fix32::ZERO, 1).with_reset_mode(ResetMode::Subtract)
}

/// Synaptic weight of a target channel; other channels are not connected.
fn lif_weight() -> Fix32 {
    Fix32::from_num(0.2)
}

fn run_dense_lif(windows: &[Window]) -> Run {
    let mut pop = CompactLifPopulation::<1>::new(lif_config());
    let weight = lif_weight();
    let target_mask = (1u32 << TARGET_CHANNELS) - 1;
    let mut score = Score::default();

    let start = Instant::now();
    for window in windows {
        let mut fired_in_horizon = false;
        for t in 0..WINDOW_SIZE {
            let mut active = black_box(window.ticks[t]) & target_mask;
            while active != 0 {
                active &= active - 1;
                pop.input_accumulators[0] += weight;
            }
            if pop.step_neuron(0) && t >= DECISION_START {
                fired_in_horizon = true;
            }
        }
        score.record(fired_in_horizon, window.burst);
    }
    let elapsed = start.elapsed();

    Run {
        name: "Dense LIF (step every tick)",
        score,
        ns_per_decision: elapsed.as_nanos() as f64 / windows.len() as f64,
    }
}

fn run_lazy_lif(windows: &[Window]) -> Run {
    let mut pop = LazyDecayLifPopulation::<1>::new(lif_config());
    let weight = lif_weight();
    let target_mask = (1u32 << TARGET_CHANNELS) - 1;
    let mut score = Score::default();

    let start = Instant::now();
    for window in windows {
        let mut fired_in_horizon = false;
        for t in 0..WINDOW_SIZE {
            let active = black_box(window.ticks[t]) & target_mask;
            if active != 0 {
                let input = weight * Fix32::from_int(active.count_ones() as i32);
                if pop.receive_input(0, input) && t >= DECISION_START {
                    fired_in_horizon = true;
                }
            }
            pop.advance_tick();
        }
        score.record(fired_in_horizon, window.burst);
    }
    let elapsed = start.elapsed();

    Run {
        name: "Lazy LIF (decay on input only)",
        score,
        ns_per_decision: elapsed.as_nanos() as f64 / windows.len() as f64,
    }
}

fn bptn_layer() -> BptnFeedforwardLayer<1, N_INPUT_CHANNELS> {
    let mut kernels = [0u64; N_INPUT_CHANNELS];
    for ch in 0..TARGET_CHANNELS {
        kernels[ch] = BPTN_KERNEL;
    }
    BptnFeedforwardLayer::new([CompactBptnNeuron::new(
        kernels,
        [0u64; N_INPUT_CHANNELS],
        BPTN_THRESHOLD,
    )])
}

/// Shifts the spikes of one tick into the framing buffers and returns their windows.
fn push_tick(
    buffers: &mut [TemporalFramingBuffer64; N_INPUT_CHANNELS],
    tick: u32,
) -> [u64; N_INPUT_CHANNELS] {
    let mut histories = [0u64; N_INPUT_CHANNELS];
    for ch in 0..N_INPUT_CHANNELS {
        histories[ch] = buffers[ch].push_spike((tick >> ch) & 1 != 0);
    }
    histories
}

fn run_bptn(windows: &[Window], evaluate_every_tick: bool) -> Run {
    let layer = bptn_layer();
    let mut buffers = [TemporalFramingBuffer64::new(); N_INPUT_CHANNELS];
    let mut score = Score::default();

    let start = Instant::now();
    for window in windows {
        let mut decision = false;
        for t in 0..WINDOW_SIZE {
            let histories = push_tick(&mut buffers, black_box(window.ticks[t]));
            if evaluate_every_tick || t == WINDOW_SIZE - 1 {
                // black_box keeps the per-tick evaluations: they are pure, and only the last
                // one is used for the decision.
                decision = black_box(layer.tick_mask(black_box(&histories))) & 1 != 0;
            }
        }
        score.record(decision, window.burst);
    }
    let elapsed = start.elapsed();

    Run {
        name: if evaluate_every_tick {
            "BPTN streaming (evaluate every tick)"
        } else {
            "BPTN block (evaluate once per window)"
        },
        score,
        ns_per_decision: elapsed.as_nanos() as f64 / windows.len() as f64,
    }
}

fn main() {
    println!("LIF vs Lazy LIF vs BPTN on one shared detection task (host timing)");
    println!(
        "Task: {} channels, burst on channels 0-{} during the last {} of {} ticks, {} windows (half with a burst)\n",
        N_INPUT_CHANNELS,
        TARGET_CHANNELS - 1,
        DECISION_HORIZON,
        WINDOW_SIZE,
        N_WINDOWS
    );

    let windows = generate_windows();
    let runs = [
        run_dense_lif(&windows),
        run_lazy_lif(&windows),
        run_bptn(&windows, false),
        run_bptn(&windows, true),
    ];
    let reference_ns = runs[0].ns_per_decision;

    println!("1. Accuracy and cost per decision");
    println!(
        "{:<40} | {:>9} | {:>6} | {:>6} | {:>13} | {:>12}",
        "Model", "Accuracy", "FP", "FN", "ns / decision", "vs dense LIF"
    );
    println!("{}", "-".repeat(100));
    for run in &runs {
        println!(
            "{:<40} | {:>8.2}% | {:>6} | {:>6} | {:>13.1} | {:>11.2}x",
            run.name,
            run.score.accuracy_percent(),
            run.score.false_pos,
            run.score.false_neg,
            run.ns_per_decision,
            reference_ns / run.ns_per_decision
        );
    }
    println!("(\"vs dense LIF\" > 1 means faster than dense LIF on this machine.)\n");

    // Memory of the models used above. BPTN stores a 64-bit kernel and a 64-bit polarity mask per
    // input channel and a 64-bit framing buffer per input channel; LIF stores one weight per
    // connected channel and 9 bytes of state per neuron (Fix32 layout).
    let flash_lif = TARGET_CHANNELS * 4;
    let sram_lif = 9;
    let flash_bptn = N_INPUT_CHANNELS * 16 + 4;
    let sram_bptn = N_INPUT_CHANNELS * 8;
    println!("2. Memory of these models (computed from their layout, not measured)");
    println!("{:<40} | {:>12} | {:>12}", "Model", "Flash", "SRAM");
    println!("{}", "-".repeat(70));
    println!(
        "{:<40} | {:>10} B | {:>10} B",
        "LIF, Fix32, 4 weights", flash_lif, sram_lif
    );
    println!(
        "{:<40} | {:>10} B | {:>10} B",
        "BPTN, 16 kernels + 16 polarity masks", flash_bptn, sram_bptn
    );
    println!(
        "\nLimits: synthetic data, parameters set by hand, a single detector neuron, host CPU.\n\
         This shows how to compare the models on equal terms; it is not evidence for a real task."
    );
}
