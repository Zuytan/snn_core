//! Runs a trained temporal LUT network on SHD test samples on the Wio-E5.
//!
//! The model (LUT input positions and tables) and one chunk of the packed test set are embedded
//! by `build.rs` from the export of `tools/bptn_torch/train_shd_comparison.py --export-lut`.
//! For each sample, the input history is copied into SRAM (not timed), then the LUT layer and the
//! group-sum argmax run under the DWT cycle counter. The binary prints the predictions, their
//! agreement with the host integer reference, the accuracy and the cycle statistics, then stops
//! on a breakpoint so `probe-rs run` exits. `tools/run_lut_shd_on_board.py` drives the chunks.

#![no_std]
#![no_main]

#[path = "../board.rs"]
mod board;

// Shared with `lut_shd_early_exit.rs`, which also uses `EE_DELTA`.
#[allow(dead_code)]
mod model {
    include!(concat!(env!("OUT_DIR"), "/lut_shd_model.rs"));
}

use core::hint::black_box;
use cortex_m_rt::entry;
use model::*;
use panic_halt as _;
use rtt_target::{rprint, rprintln, rtt_init_print, ChannelMode::BlockIfFull};

const GROUPS: usize = 20;
const PER_GROUP: usize = N_LUTS / GROUPS;

#[inline(always)]
fn get_cycles() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

/// LUT layer (address-table form) and group-sum argmax; lowest class wins ties.
#[inline(never)]
fn classify(w: &[u32; WORDS]) -> u8 {
    let mut best_class = 0u8;
    let mut best_count = 0u32;
    for g in 0..GROUPS {
        let mut count = 0u32;
        for n in g * PER_GROUP..(g + 1) * PER_GROUP {
            let mut i = 0u32;
            for (bit, [word, pos]) in LUT_ADDR[n].iter().enumerate() {
                i |= ((w[*word as usize] >> *pos) & 1) << bit;
            }
            count += ((LUT_TABLE[n] >> i) & 1) as u32;
        }
        if count > best_count || g == 0 {
            best_count = count;
            best_class = g as u8;
        }
    }
    best_class
}

#[entry]
fn main() -> ! {
    rtt_init_print!(BlockIfFull, 4096);
    let mut cp = cortex_m::Peripherals::take().unwrap();
    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    rprintln!("LUT_SHD_EVAL");
    let clock_hz = board::init_and_report();
    if !MODEL_PRESENT {
        rprintln!("No model embedded: build with LUT_MODEL_DIR set.");
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
