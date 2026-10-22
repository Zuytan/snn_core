//! ⚡ Example 9: Rigorous Micro-Architectural Comparison — LIF SNN vs Lazy LIF vs BPTN
//!
//! Unbiased and adversarial evaluation of a layer of $N=64$ neurons
//! processing $M=16$ input channels across a temporal receptive field of $W=64$ time-steps.
//!
//! This benchmark eliminates common evaluation flaws by measuring:
//! 1. **Dense LIF SNN (`CompactLifPopulation` Q16.16)**: Standard step-by-step evaluation.
//! 2. **Event-Driven Lazy LIF (`LazyDecayLifPopulation`)**: Optimized for high sparsity ($O(1)$ per spike).
//! 3. **Streaming BPTN (`Stride = 1`)**: Framing buffer `TemporalFramingBuffer64` + evaluation at EVERY tick $t \in [0, 63]$.
//! 4. **Block Windowed BPTN (`Stride = 64`)**: Single evaluation per complete 64-tick window.
//! 5. **Physical Memory Footprint (Flash ROM & SRAM)**: Without hiding the ROM cost of 64-bit masks.
//!
//! Execution:
//! ```bash
//! cargo run --release --features bptn --example 09_bptn_vs_lif_head_to_head
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::large_stack_arrays,
    clippy::no_effect_underscore_binding,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use snn_core::domain::bptn::{BptnFeedforwardLayer, BptnSpikingNetwork, CompactBptnNeuron};
use snn_core::domain::compact_population::CompactLifPopulation;
use snn_core::domain::lazy_population::LazyDecayLifPopulation;
use snn_core::domain::models::lif::{LifConfig, ResetMode};
use snn_core::domain::numeric::Fix32;
use snn_core::traits::SpikingNetwork;
use std::hint::black_box;
use std::time::Instant;

const N_NEURONS: usize = 64;
const N_INPUT_CHANNELS: usize = 16;
const WINDOW_SIZE: usize = 64;
const N_EVALUATIONS: usize = 25_000;

fn main() {
    println!(
        "=========================================================================================="
    );
    println!("⚖️  ADVERSARIAL MICRO-ARCHITECTURAL BENCHMARK (UNBIASED)");
    println!("Comparison: Dense LIF vs Lazy LIF vs Streaming BPTN (Stride=1) vs Block BPTN");
    println!(
        "Architecture: {} neurons, {} input channels, window T={} ticks ({} repetitions)",
        N_NEURONS, N_INPUT_CHANNELS, WINDOW_SIZE, N_EVALUATIONS
    );
    println!(
        "==========================================================================================\n"
    );

    // ------------------------------------------------------------------------
    // 1. Initialize LIF Models (Dense & Lazy) in Fixed-Point Q16.16
    // ------------------------------------------------------------------------
    let decay = Fix32::from_bits(58982); // ~0.90
    let config =
        LifConfig::new(Fix32::ONE, decay, Fix32::ZERO, 1).with_reset_mode(ResetMode::Subtract);
    let mut lif_pop = CompactLifPopulation::<N_NEURONS>::new(config);
    let mut lazy_lif_pop = LazyDecayLifPopulation::<N_NEURONS>::new(config);

    let mut weights_lif = [[Fix32::ZERO; N_NEURONS]; N_INPUT_CHANNELS];
    for ch in 0..N_INPUT_CHANNELS {
        for n in 0..N_NEURONS {
            let w = if (ch + n) % 3 == 0 { 0.35 } else { -0.15 };
            weights_lif[ch][n] = Fix32::from_num(w);
        }
    }

    // ------------------------------------------------------------------------
    // 2. Initialize BPTN Model (64 neurons × 16 channels)
    // ------------------------------------------------------------------------
    let mut bptn_neurons = [CompactBptnNeuron::<N_INPUT_CHANNELS>::new(
        [0u64; N_INPUT_CHANNELS],
        [0u64; N_INPUT_CHANNELS],
        12,
    ); N_NEURONS];

    for n in 0..N_NEURONS {
        let mut kernels = [0u64; N_INPUT_CHANNELS];
        let mut polarities = [0u64; N_INPUT_CHANNELS];
        for ch in 0..N_INPUT_CHANNELS {
            kernels[ch] = 0x0408_1020_4080_0102u64.rotate_left(((n + ch * 3) % 64) as u32);
            polarities[ch] = if (ch + n) % 4 == 0 {
                0x0100_0001_0001_0000u64
            } else {
                0
            };
        }
        bptn_neurons[n] = CompactBptnNeuron::new(kernels, polarities, 12);
    }

    let bptn_layer = BptnFeedforwardLayer::new(bptn_neurons);
    let mut bptn_streaming_net = BptnSpikingNetwork::new(bptn_layer);

    // ------------------------------------------------------------------------
    // 3. Generate Realistic Input Stream (Sparsity ~12.5% = 8 spikes / 64 ticks)
    // ------------------------------------------------------------------------
    let mut input_words = [0u64; N_INPUT_CHANNELS];
    for ch in 0..N_INPUT_CHANNELS {
        input_words[ch] = 0x0104_1040_0208_2080u64.rotate_left((ch * 5) as u32);
    }

    let mut input_bools = [[false; N_INPUT_CHANNELS]; WINDOW_SIZE];
    let mut input_words_u32 = [[0u32; 1]; WINDOW_SIZE];
    for t in 0..WINDOW_SIZE {
        let mut mask32 = 0u32;
        for ch in 0..N_INPUT_CHANNELS {
            let is_spike = ((input_words[ch] >> t) & 1) != 0;
            input_bools[t][ch] = is_spike;
            if is_spike {
                mask32 |= 1u32 << ch;
            }
        }
        input_words_u32[t][0] = mask32;
    }

    // ------------------------------------------------------------------------
    // BENCHMARK A: Standard LIF SNN (CompactLifPopulation, Stride = 1)
    // ------------------------------------------------------------------------
    let start_lif = Instant::now();
    let mut lif_spikes_total: u64 = 0;

    for _ in 0..N_EVALUATIONS {
        for t in 0..WINDOW_SIZE {
            let mut active_word = black_box(input_words_u32[t][0]);
            while active_word != 0 {
                let ch = active_word.trailing_zeros() as usize;
                active_word &= active_word - 1;
                let w_row = &weights_lif[ch];
                for n in 0..N_NEURONS {
                    lif_pop.input_accumulators[n] += w_row[n];
                }
            }
            for n in 0..N_NEURONS {
                if lif_pop.step_neuron(n) {
                    lif_spikes_total += 1;
                }
            }
        }
    }
    let duration_lif = start_lif.elapsed();
    let ns_win_lif = duration_lif.as_nanos() as f64 / N_EVALUATIONS as f64;
    let ns_tick_lif = ns_win_lif / WINDOW_SIZE as f64;

    // ------------------------------------------------------------------------
    // BENCHMARK B: Event-Driven Lazy LIF (LazyDecayLifPopulation)
    // ------------------------------------------------------------------------
    let start_lazy = Instant::now();
    let mut lazy_spikes_total: u64 = 0;

    for _ in 0..N_EVALUATIONS {
        for t in 0..WINDOW_SIZE {
            let active_word = black_box(input_words_u32[t][0]);
            if active_word != 0 {
                let mut acc = [Fix32::ZERO; N_NEURONS];
                let mut w_bits = active_word;
                while w_bits != 0 {
                    let ch = w_bits.trailing_zeros() as usize;
                    w_bits &= w_bits - 1;
                    let w_row = &weights_lif[ch];
                    for n in 0..N_NEURONS {
                        acc[n] += w_row[n];
                    }
                }
                for n in 0..N_NEURONS {
                    if acc[n] != Fix32::ZERO && lazy_lif_pop.receive_input(n, acc[n]) {
                        lazy_spikes_total += 1;
                    }
                }
            }
            lazy_lif_pop.advance_tick();
        }
    }
    let duration_lazy = start_lazy.elapsed();
    let ns_win_lazy = duration_lazy.as_nanos() as f64 / N_EVALUATIONS as f64;
    let ns_tick_lazy = ns_win_lazy / WINDOW_SIZE as f64;

    // ------------------------------------------------------------------------
    // BENCHMARK C: Streaming BPTN Mode (Stride = 1, Evaluated at EVERY tick)
    // ------------------------------------------------------------------------
    let start_bptn_stream = Instant::now();
    let mut bptn_stream_spikes: u64 = 0;

    for _ in 0..N_EVALUATIONS {
        for t in 0..WINDOW_SIZE {
            let out_words = bptn_streaming_net.tick_spikes_words(black_box(&input_words_u32[t]));
            bptn_stream_spikes += u64::from(out_words[0].count_ones() + out_words[1].count_ones());
        }
    }
    let duration_bptn_stream = start_bptn_stream.elapsed();
    let ns_win_bptn_stream = duration_bptn_stream.as_nanos() as f64 / N_EVALUATIONS as f64;
    let ns_tick_bptn_stream = ns_win_bptn_stream / WINDOW_SIZE as f64;

    // ------------------------------------------------------------------------
    // BENCHMARK D: Block Windowed BPTN Mode (Stride = 64, 1 evaluation per window)
    // ------------------------------------------------------------------------
    let start_bptn_block = Instant::now();
    let mut bptn_block_spikes: u64 = 0;

    for _ in 0..N_EVALUATIONS {
        let mask = bptn_streaming_net.layer.tick_mask(black_box(&input_words));
        bptn_block_spikes += u64::from(mask.count_ones());
    }
    let duration_bptn_block = start_bptn_block.elapsed();
    let ns_win_bptn_block = duration_bptn_block.as_nanos() as f64 / N_EVALUATIONS as f64;
    let ns_tick_bptn_block = ns_win_bptn_block / WINDOW_SIZE as f64;

    // ------------------------------------------------------------------------
    // 4. Physical Memory Footprint Audit
    // ------------------------------------------------------------------------
    let rom_lif_fix32 = N_INPUT_CHANNELS * N_NEURONS * 4; // 4,096 B
    let rom_lif_int8 = N_INPUT_CHANNELS * N_NEURONS; // 1,024 B
    let sram_lif = N_NEURONS * 9; // 576 B

    let rom_lazy_fix32 = rom_lif_fix32 + 64 * 4; // 4,352 B
    let sram_lazy = N_NEURONS * 9; // 576 B

    let rom_bptn_full = N_NEURONS * (N_INPUT_CHANNELS * 16 + 4);
    let rom_bptn_excitatory = N_NEURONS * (N_INPUT_CHANNELS * 8 + 4);
    let sram_bptn = N_INPUT_CHANNELS * 8; // 128 B

    println!(
        "📊 1. EXECUTION BENCHMARK (Averaged across {} windows of 64 ticks)",
        N_EVALUATIONS
    );
    println!(
        "------------------------------------------------------------------------------------------"
    );
    println!(
        "{:<34} | {:>15} | {:>15} | {:>18}",
        "Architecture & Evaluation Mode", "Latency / 64t", "Cost / Tick (t)", "Gain vs Dense LIF"
    );
    println!(
        "-----------------------------------+-----------------+-----------------+-------------------"
    );
    println!(
        "{:<34} | {:>12.1} ns | {:>12.2} ns | {:>18}",
        "1. Dense LIF SNN (CTZ, Stride=1)", ns_win_lif, ns_tick_lif, "1.00x (Reference)"
    );
    println!(
        "{:<34} | {:>12.1} ns | {:>12.2} ns | {:>17.2}x",
        "2. Lazy-LIF SNN (Event, Stride=1)",
        ns_win_lazy,
        ns_tick_lazy,
        ns_win_lif / ns_win_lazy
    );
    println!(
        "{:<34} | {:>12.1} ns | {:>12.2} ns | {:>17.2}x",
        "3. Streaming BPTN (Tick-by-Tick)",
        ns_win_bptn_stream,
        ns_tick_bptn_stream,
        ns_win_lif / ns_win_bptn_stream
    );
    println!(
        "{:<34} | {:>12.1} ns | {:>12.2} ns | {:>17.2}x",
        "4. Block Windowed BPTN (Stride=64)",
        ns_win_bptn_block,
        ns_tick_bptn_block,
        ns_win_lif / ns_win_bptn_block
    );
    println!(
        "------------------------------------------------------------------------------------------"
    );
    println!(
        "Emitted Spikes (Verification) -> LIF: {} | Lazy: {} | BPTN-Stream: {} | BPTN-Block: {}\n",
        lif_spikes_total, lazy_spikes_total, bptn_stream_spikes, bptn_block_spikes
    );

    println!("💾 2. PHYSICAL MEMORY AUDIT (Flash ROM vs SRAM)");
    println!(
        "------------------------------------------------------------------------------------------"
    );
    println!(
        "{:<34} | {:>15} | {:>15} | {:>18}",
        "Model", "Flash ROM Weights", "Dynamic SRAM", "SRAM Ratio vs LIF"
    );
    println!(
        "-----------------------------------+-----------------+-----------------+-------------------"
    );
    println!(
        "{:<34} | {:>13} B | {:>13} B | {:>18}",
        "LIF SNN (Fix32 Q16.16)", rom_lif_fix32, sram_lif, "1.00x (576 B)"
    );
    println!(
        "{:<34} | {:>13} B | {:>13} B | {:>18}",
        "Quantized LIF SNN (INT8)", rom_lif_int8, sram_lif, "1.00x (576 B)"
    );
    println!(
        "{:<34} | {:>13} B | {:>13} B | {:>18}",
        "Lazy-LIF SNN (Fix32 + LUT)", rom_lazy_fix32, sram_lazy, "1.00x (576 B)"
    );
    println!(
        "{:<34} | {:>13} B | {:>13} B | {:>13.2}x less",
        "Full BPTN (Kernel + Polarity)",
        rom_bptn_full,
        sram_bptn,
        sram_lif as f64 / sram_bptn as f64
    );
    println!(
        "{:<34} | {:>13} B | {:>13} B | {:>13.2}x less",
        "Excitatory BPTN (Kernel u64 only)",
        rom_bptn_excitatory,
        sram_bptn,
        sram_lif as f64 / sram_bptn as f64
    );
    println!(
        "------------------------------------------------------------------------------------------\n"
    );

    println!("🔍 HONEST ENGINEERING DIAGNOSTIC:");
    println!(
        "1. SPEED REALITY — STREAMING (Stride=1) vs BLOCK (Stride=64):\n\
         • In tick-by-tick streaming mode (Stride=1, output required at every tick t), LIF SNN (CTZ)\n\
           is {:.2}x FASTER than BPTN ({:.1} ns/tick vs {:.1} ns/tick)!\n\
           Why? Because at each tick, LIF only propagates channels that fired a spike (sparsity),\n\
           whereas BPTN re-evaluates all {} channels x {} neurons ({} u64 POPCNTs) on every step.\n\
         • Conversely, in Block Windowed mode (Stride=64, 1 decision every 64 ticks via DMA buffer),\n\
           BPTN is {:.1}x faster ({:.1} ns vs {:.1} ns) by compressing 64 steps into 1 evaluation call.",
        ns_tick_bptn_stream / ns_tick_lif,
        ns_tick_lif,
        ns_tick_bptn_stream,
        N_INPUT_CHANNELS,
        N_NEURONS,
        N_INPUT_CHANNELS * N_NEURONS,
        ns_win_lif / ns_win_bptn_block,
        ns_win_bptn_block,
        ns_win_lif
    );
    println!(
        "2. SRAM vs FLASH ROM TRADE-OFFS:\n\
         • BPTN slashes SRAM footprint by {:.1}x ({} B vs {} B) because neurons are stateless\n\
           (only the {} input channels maintain a u64 shift register).\n\
         • HOWEVER, in raw Flash ROM, a full dense BPTN occupies {} B vs {} B\n\
           for Fix32 LIF ({} B in INT8), because each channel stores a 64-bit temporal bitmask\n\
           instead of a single scalar weight.",
        sram_lif as f64 / sram_bptn as f64,
        sram_bptn,
        sram_lif,
        N_INPUT_CHANNELS,
        rom_bptn_full,
        rom_lif_fix32,
        rom_lif_int8
    );
    println!(
        "==========================================================================================\n"
    );
}
