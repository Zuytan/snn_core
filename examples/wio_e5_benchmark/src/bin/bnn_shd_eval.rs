//! Runs a trained binarized network on SHD test samples on the Wio-E5.
//!
//! Counterpart of `lut_shd_eval.rs` for the binarized baseline exported by
//! `tools/bptn_torch/train_shd_comparison.py --export-bnn`. For each hidden neuron the firmware
//! counts the mismatches between the input history and the weight bits (XOR + popcount over the
//! u32 words), fires if the count is at most the neuron's integer limit, then adds the int8
//! readout weights of the firing neurons to the class scores and takes the argmax. Timing, output
//! format and the final breakpoint are the same as in `lut_shd_eval.rs`.

#![no_std]
#![no_main]

#[path = "../board.rs"]
mod board;

mod model {
    include!(concat!(env!("OUT_DIR"), "/bnn_shd_model.rs"));
}

use core::hint::black_box;
use cortex_m_rt::entry;
use model::*;
use panic_halt as _;
use rtt_target::{rprint, rprintln, rtt_init_print, ChannelMode::BlockIfFull};

const CLASSES: usize = 20;

#[inline(always)]
fn get_cycles() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

/// Binarized hidden layer and int8 readout; lowest class wins ties.
#[inline(never)]
fn classify(w: &[u32; WORDS]) -> u8 {
    let mut scores = READOUT_B;
    for n in 0..HIDDEN {
        let mut mismatches = 0u32;
        for (x, k) in w.iter().zip(BNN_WEIGHTS[n].iter()) {
            mismatches += (x ^ k).count_ones();
        }
        if mismatches as i32 <= BNN_MAX_MISMATCH[n] {
            for c in 0..CLASSES {
                scores[c] += i32::from(READOUT_W[c][n]);
            }
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

    rprintln!("BNN_SHD_EVAL");
    let clock_hz = board::init_and_report();
    if !MODEL_PRESENT {
        rprintln!("No model embedded: build with BNN_MODEL_DIR set.");
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
