//! Runs a classic tree ensemble (random forest or gradient boosting) on SHD test samples on the
//! Wio-E5, as the baseline of the forest of LUT trees (`bp_shd_eval.rs`).
//!
//! Each internal node tests one input bit (`TREE_NODES[i] = [feature, left, right]`, left when the
//! bit is 0); a child or root with bit 15 set is a leaf id (low 15 bits). Random-forest leaves add 20 int8
//! class scores; boosting leaves add one int16 value to the class of their tree, on top of an
//! int32 baseline. Model and test chunk come from `TREE_MODEL_DIR` (export of
//! `tools/bptn_torch/tree_baselines.py --export`); timing, output format and the final breakpoint
//! are the same as in `lut_shd_eval.rs`.

#![no_std]
#![no_main]

#[path = "../board.rs"]
mod board;

mod model {
    include!(concat!(env!("OUT_DIR"), "/tree_shd_model.rs"));
}

use core::hint::black_box;
use cortex_m_rt::entry;
use model::*;
use panic_halt as _;
use rtt_target::{rprint, rprintln, rtt_init_print, ChannelMode::BlockIfFull};

const CLASSES: usize = 20;
const LEAF_FLAG: u16 = 0x8000;

#[inline(always)]
fn get_cycles() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

#[inline(never)]
fn classify(w: &[u32; WORDS]) -> u8 {
    let mut scores = TREE_BASE;
    for (t, &root) in TREE_ROOTS.iter().enumerate() {
        let mut node = root;
        while node & LEAF_FLAG == 0 {
            let [feature, left, right] = TREE_NODES[node as usize];
            let bit = (w[(feature >> 5) as usize] >> (feature & 31)) & 1;
            node = if bit == 0 { left } else { right };
        }
        let leaf = (node & !LEAF_FLAG) as usize;
        if LEAF_VECTOR {
            for (score, &v) in scores.iter_mut().zip(LEAF_VEC[leaf].iter()) {
                *score += i32::from(v);
            }
        } else {
            scores[TREE_CLASS[t] as usize] += i32::from(LEAF_VAL[leaf]);
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

    rprintln!("TREE_SHD_EVAL trees={} vector_leaves={}", TREE_ROOTS.len(), LEAF_VECTOR);
    let clock_hz = board::init_and_report();
    if !MODEL_PRESENT {
        rprintln!("No model embedded: build with TREE_MODEL_DIR set.");
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
