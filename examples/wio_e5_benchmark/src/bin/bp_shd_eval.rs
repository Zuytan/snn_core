//! Runs a trained forest of LUT decision trees (learned branching programs) on SHD test samples
//! on the Wio-E5.
//!
//! Each tree starts at its root; a node is a LUT neuron (6 input bits at learned positions of the
//! history, 64-entry table) whose output selects the child (heap order: 2i + 1 for 0, 2i + 2 for
//! 1). After `DEPTH` nodes the reached leaf adds its 20 int8 class scores. Only `TREES x DEPTH`
//! nodes and `TREES` leaves are read per inference. Model and test chunk come from
//! `BP_MODEL_DIR` (export of `train_shd_comparison.py --export-bp`); timing, output format and the
//! final breakpoint are the same as in `lut_shd_eval.rs`.

#![no_std]
#![no_main]

#[path = "../board.rs"]
mod board;

mod model {
    include!(concat!(env!("OUT_DIR"), "/bp_shd_model.rs"));
}

use core::hint::black_box;
use cortex_m_rt::entry;
use model::*;
use panic_halt as _;
use rtt_target::{rprint, rprintln, rtt_init_print, ChannelMode::BlockIfFull};

const CLASSES: usize = 20;
const NODES_PER_TREE: usize = (1 << DEPTH) - 1;
const LEAVES_PER_TREE: usize = 1 << DEPTH;

#[inline(always)]
fn get_cycles() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

#[inline(never)]
fn classify(w: &[u32; WORDS]) -> u8 {
    let mut scores = [0i32; CLASSES];
    for t in 0..TREES {
        let base = t * NODES_PER_TREE;
        let mut node = 0usize;
        for _ in 0..DEPTH {
            let n = base + node;
            let mut i = 0u32;
            for (bit, [word, pos]) in BP_ADDR[n].iter().enumerate() {
                i |= ((w[*word as usize] >> *pos) & 1) << bit;
            }
            node = 2 * node + 1 + ((BP_TABLE[n] >> i) & 1) as usize;
        }
        let leaf = &BP_LEAF[t * LEAVES_PER_TREE + node - NODES_PER_TREE];
        for (score, &v) in scores.iter_mut().zip(leaf.iter()) {
            *score += i32::from(v);
        }
    }
    let mut best = 0usize;
    for c in 1..CLASSES {
        if scores[c] > scores[best] {
            best = c;
        }
    }
    best as u8
}

#[entry]
fn main() -> ! {
    rtt_init_print!(BlockIfFull, 4096);
    let mut cp = cortex_m::Peripherals::take().unwrap();
    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    rprintln!("BP_SHD_EVAL trees={} depth={}", TREES, DEPTH);
    let clock_hz = board::init_and_report();
    if !MODEL_PRESENT {
        rprintln!("No model embedded: build with BP_MODEL_DIR set.");
        cortex_m::asm::bkpt();
    }

    let t0 = get_cycles();
    let t1 = get_cycles();
    let overhead = t1.wrapping_sub(t0);

    let mut buffer = [0u32; WORDS];
    let (mut correct, mut agree) = (0u32, 0u32);
    let (mut c_min, mut c_max, mut c_sum) = (u32::MAX, 0u32, 0u64);

    rprint!("PRED {} ", CHUNK_START);
    for (s, sample) in TEST_X.iter().enumerate() {
        buffer.copy_from_slice(sample);
        let input = black_box(&buffer);
        let start = get_cycles();
        let pred = black_box(classify(input));
        let cycles = get_cycles().wrapping_sub(start).saturating_sub(overhead);

        c_min = c_min.min(cycles);
        c_max = c_max.max(cycles);
        c_sum += u64::from(cycles);
        correct += u32::from(pred == TEST_Y[s]);
        agree += u32::from(pred == HOST_PRED[s]);
        rprint!("{:02x}", pred);
    }
    rprintln!();

    let n = TEST_X.len() as u32;
    let mean = c_sum as f32 / n as f32;
    rprintln!(
        "CHUNK start={} samples={} correct={} host_agree={} cycles_min={} cycles_mean={:.1} cycles_max={} us_mean={:.2} clock_hz={}",
        CHUNK_START,
        n,
        correct,
        agree,
        c_min,
        mean,
        c_max,
        board::cycles_to_us(mean, clock_hz),
        clock_hz
    );
    rprintln!("DONE");

    loop {
        cortex_m::asm::bkpt();
    }
}
