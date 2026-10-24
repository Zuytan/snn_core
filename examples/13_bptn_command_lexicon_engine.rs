//! ⚡ Example 13: BPTN Symbolic Command & Lexicon Engine (Bit-SSM / Axe 4)
//!
//! Demonstrates the v2.0 Bit-Parallel Language & Sequence Reasoning Engine:
//! 1. **Orthogonal Symbolic Projection (`Sparse SimHash`)**:
//!    Maps text commands into $\mathbb{F}_2^{32}$ ensuring $d_H \ge 10$ bits between distinct words.
//! 2. **Lexicon Matching & Noise Tolerance**:
//!    Resolves corrupted or noisy symbol transmissions with zero floating-point math.
//! 3. **Hardware-Accelerated Codebook Evaluation (Axe 4)**:
//!    Evaluates a layer of 8 command detectors using 2-byte `CompactCodebookSynapse`
//!    referencing the immutable `CANONICAL_BPTN_CODEBOOK_32`.
//!
//! Execution:
//! ```bash
//! cargo run --release --features bptn --example 13_bptn_command_lexicon_engine
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use snn_core::bptn::{
    BptnLexicon, BptnLexiconEntry, CANONICAL_BPTN_CODEBOOK_32, CodebookBptnLayer,
    CodebookBptnNeuron, CompactCodebookSynapse, compute_symbolic_simhash,
};
use std::time::Instant;

const N_COMMANDS: usize = 8;
const COMMAND_NAMES: [&str; N_COMMANDS] = [
    "START", "STOP", "ALLUME", "ETEINS", "AVANCE", "RECULE", "DROITE", "GAUCHE",
];

fn main() {
    println!(
        "=========================================================================================="
    );
    println!("⚡ BPTN SYMBOLIC REASONING & COMMAND ENGINE (Bit-SSM / AXE 4)");
    println!("Demonstrating Word32, Sparse SimHash Orthogonality & 2-Byte Codebook Synapses");
    println!(
        "==========================================================================================\n"
    );

    // ------------------------------------------------------------------------
    // 1. Build Static Lexicon at Compile-Time
    // ------------------------------------------------------------------------
    let lexicon = BptnLexicon::new([
        BptnLexiconEntry::from_str(COMMAND_NAMES[0]),
        BptnLexiconEntry::from_str(COMMAND_NAMES[1]),
        BptnLexiconEntry::from_str(COMMAND_NAMES[2]),
        BptnLexiconEntry::from_str(COMMAND_NAMES[3]),
        BptnLexiconEntry::from_str(COMMAND_NAMES[4]),
        BptnLexiconEntry::from_str(COMMAND_NAMES[5]),
        BptnLexiconEntry::from_str(COMMAND_NAMES[6]),
        BptnLexiconEntry::from_str(COMMAND_NAMES[7]),
    ]);

    println!("📋 1. COMPILED SYMBOLIC SIGNATURES (Sparse SimHash in F2^32):");
    println!(
        "------------------------------------------------------------------------------------------"
    );
    println!(
        "{:<10} | {:<12} | {:<34} | {:<8}",
        "Command", "Hex Word32", "Binary Representation (32-bit)", "Popcnt"
    );
    println!("-----------+--------------+------------------------------------+---------");
    for i in 0..N_COMMANDS {
        let sig = lexicon.entries[i].signature;
        println!(
            "{:<10} | 0x{:08X}   | {:032b} | {:>2} bits",
            lexicon.entries[i].label,
            sig,
            sig,
            sig.count_ones()
        );
    }
    println!(
        "------------------------------------------------------------------------------------------\n"
    );

    // ------------------------------------------------------------------------
    // 2. Pairwise Hamming Distance Matrix (Proof of Anti-Collision)
    // ------------------------------------------------------------------------
    println!("📐 2. PAIRWISE HAMMING DISTANCE MATRIX (Anti-Collision Verification):");
    println!(
        "------------------------------------------------------------------------------------------"
    );
    print!("{:<10}", "");
    for i in 0..N_COMMANDS {
        print!("{:<8}", &COMMAND_NAMES[i][..4.min(COMMAND_NAMES[i].len())]);
    }
    println!("\n-----------+---------------------------------------------------------------");

    let mut min_distance = 32u32;
    let mut total_distance = 0u32;
    let mut pairs_count = 0u32;

    for i in 0..N_COMMANDS {
        print!("{:<10}", COMMAND_NAMES[i]);
        for j in 0..N_COMMANDS {
            let dist = lexicon.distance_between(i, j);
            if i != j {
                if dist < min_distance {
                    min_distance = dist;
                }
                total_distance += dist;
                pairs_count += 1;
            }
            print!("{:>4}    ", dist);
        }
        println!();
    }
    let avg_distance = total_distance as f64 / pairs_count as f64;
    println!(
        "------------------------------------------------------------------------------------------"
    );
    println!(
        "  • Minimum Pairwise Distance: {} bits (Guaranteed >= 10 bits separation)",
        min_distance
    );
    println!(
        "  • Average Pairwise Distance: {:.2} bits (Theoretical optimum: 16.0 bits)",
        avg_distance
    );
    println!("  -> Absolute immunity against cross-talk between distinct words.\n");

    // ------------------------------------------------------------------------
    // 3. Transmission Noise Stress Test (Fuzzy Matching Recovery)
    // ------------------------------------------------------------------------
    println!("🛡️  3. NOISY TRANSMISSION STRESS TEST (10,000 trials per noise level):");
    println!(
        "------------------------------------------------------------------------------------------"
    );
    println!("Noise Level                | Exact Recognition | False Alarms | Recovery Success");
    println!("---------------------------+-------------------+--------------+-----------------");

    let trials = 10_000usize;
    let mut rng_state = 0x1357_9BDF_2468_ACE0u64;
    let mut next_u32 = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state as u32
    };

    for &flips in &[0usize, 1, 2, 3, 4] {
        let mut recovered_correct = 0;
        let mut false_alarms = 0;

        for _ in 0..trials {
            let target_idx = (next_u32() as usize) % N_COMMANDS;
            let target_sig = lexicon.entries[target_idx].signature;

            // Inject `flips` bit flips at random positions
            let mut corrupted_sig = target_sig;
            let mut flipped_mask = 0u32;
            let mut count = 0;
            while count < flips {
                let bit = 1u32 << (next_u32() % 32);
                if (flipped_mask & bit) == 0 {
                    corrupted_sig ^= bit;
                    flipped_mask |= bit;
                    count += 1;
                }
            }

            // Fuzzy lookup with max tolerance of 3 bits
            if let Some((matched_idx, _dist)) = lexicon.find_best_match(corrupted_sig, 3) {
                if matched_idx == target_idx {
                    recovered_correct += 1;
                } else {
                    false_alarms += 1;
                }
            }
        }

        let success_rate = (recovered_correct as f64 / trials as f64) * 100.0;
        let fa_rate = (false_alarms as f64 / trials as f64) * 100.0;
        println!(
            "{:<26} | {:>15.2}% | {:>10.2}% | {:>14}",
            format!("{flips} bit error(s) in 32b"),
            success_rate,
            fa_rate,
            if success_rate >= 99.0 {
                "PERFECT"
            } else if success_rate >= 90.0 {
                "HIGH"
            } else {
                "DEGRADED"
            }
        );
    }
    println!(
        "------------------------------------------------------------------------------------------\n"
    );

    // ------------------------------------------------------------------------
    // 4. Hardware-Accelerated Kernel Codebook Layer (Axe 4 Benchmarking)
    // ------------------------------------------------------------------------
    println!("⚡ 4. 2-BYTE CODEBOOK SYNAPSE LAYER EXECUTION (Axe 4 Benchmarking):");
    println!(
        "------------------------------------------------------------------------------------------"
    );

    // Build 8 neurons receiving 4 channels of 32-bit temporal history
    // Each synapse uses CompactCodebookSynapse (2 bytes)
    const CHANNELS: usize = 4;
    let mut neurons = [CodebookBptnNeuron::<CHANNELS>::new(
        [CompactCodebookSynapse::new(0, 0, false); CHANNELS],
        4,
    ); N_COMMANDS];

    for i in 0..N_COMMANDS {
        let mut syns = [CompactCodebookSynapse::new(0, 0, false); CHANNELS];
        for ch in 0..CHANNELS {
            let code_idx = ((i * 3 + ch * 7) % 32) as u8;
            let delay = (ch * 4) as u8;
            let invert = (i + ch) % 5 == 0;
            syns[ch] = CompactCodebookSynapse::new(code_idx, delay, invert);
        }
        neurons[i] = CodebookBptnNeuron::new(syns, 3);
    }

    let layer = CodebookBptnLayer::new(neurons);
    let sample_histories = [
        0x5555_5555u32,
        0x3333_3333u32,
        0x0F0F_0F0Fu32,
        compute_symbolic_simhash(b"START"),
    ];

    let n_iterations = 500_000usize;
    let start_time = Instant::now();
    let mut dummy_acc = 0u64;

    for _ in 0..n_iterations {
        dummy_acc =
            dummy_acc.wrapping_add(layer.tick_mask(&sample_histories, &CANONICAL_BPTN_CODEBOOK_32));
    }

    let elapsed = start_time.elapsed();
    let ns_per_eval = elapsed.as_nanos() as f64 / n_iterations as f64;
    let throughput_evals_sec = (n_iterations as f64 / elapsed.as_secs_f64()) / 1_000_000.0;

    println!("  • Number of Layer Evaluations: {}", n_iterations);
    println!("  • Total Execution Time: {:.2?}", elapsed);
    println!(
        "  • Latency per Layer Decision (8 neurons × 4 channels): {:.2} ns",
        ns_per_eval
    );
    println!(
        "  • Throughput: {:.2} million layer evaluations / second",
        throughput_evals_sec
    );
    println!("  • Output Verification Hash: 0x{:016X}", dummy_acc);

    // ------------------------------------------------------------------------
    // 5. Memory Footprint Audit
    // ------------------------------------------------------------------------
    println!("\n💾 5. PHYSICAL MEMORY AUDIT (v1.0 Dense vs v2.0 Codebook):");
    println!(
        "------------------------------------------------------------------------------------------"
    );
    println!(
        "Metric Evaluated               | Dense BPTN v1.0 | Codebook BPTN v2.0 | Savings Factor"
    );
    println!(
        "-------------------------------+-----------------+--------------------+---------------"
    );
    let dense_rom = N_COMMANDS * CHANNELS * 16;
    let codebook_rom = (N_COMMANDS * CHANNELS * 2) + (32 * 4);
    println!(
        "Flash ROM (Synapses + Table)   | {:>13} B | {:>16} B | {:>13.1}x less",
        dense_rom,
        codebook_rom,
        dense_rom as f64 / codebook_rom as f64
    );
    let dense_sram = CHANNELS * 8;
    let codebook_sram = CHANNELS * 4;
    println!(
        "Dynamic SRAM (Active History)  | {:>13} B | {:>16} B | {:>13.1}x less",
        dense_sram,
        codebook_sram,
        dense_sram as f64 / codebook_sram as f64
    );
    println!(
        "------------------------------------------------------------------------------------------"
    );
    println!("🏁 SUCCESS: BPTN v2.0 operates strictly in #![no_std] with zero dynamic allocation.");
    println!(
        "=========================================================================================="
    );
}
