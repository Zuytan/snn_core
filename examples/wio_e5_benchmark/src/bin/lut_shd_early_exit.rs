//! Temporal LUT network on SHD with early exit of the group-sum readout, on the Wio-E5.
//!
//! Same model and output format as `lut_shd_eval.rs`, but the LUTs are evaluated in rounds (one
//! LUT of every class group per round) and the evaluation stops early:
//! - `EE_DELTA = 0` (exact rule): as soon as no other class can overtake the leader with the LUTs
//!   it has left; the prediction is always the full-evaluation prediction;
//! - `EE_DELTA = d > 0` (margin rule): as soon as the leader is `d` votes ahead of the second.
//!
//! `EE_DELTA` comes from `LUT_EE_DELTA` at build time, and the expected predictions from
//! `LUT_HOST_PRED` (written by `tools/bptn_torch/lut_inference_analysis.py`). The CHUNK line also
//! reports `rounds_sum`, the total number of rounds over the chunk.

#![no_std]
#![no_main]

#[path = "../board.rs"]
mod board;

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

#[inline(always)]
fn lut_output(w: &[u32; WORDS], n: usize) -> u32 {
    let mut i = 0u32;
    for (bit, [word, pos]) in LUT_ADDR[n].iter().enumerate() {
        i |= ((w[*word as usize] >> *pos) & 1) << bit;
    }
    ((LUT_TABLE[n] >> i) & 1) as u32
}

/// Returns true when the evaluation can stop after a round, with `remaining` rounds left.
#[inline(always)]
fn can_stop(counts: &[u32; GROUPS], remaining: u32) -> bool {
    let mut leader = 0usize;
    for g in 1..GROUPS {
        if counts[g] > counts[leader] {
            leader = g;
        }
    }
    let lead = counts[leader];
    if EE_DELTA == 0 {
        // Exact: every other class, even winning all its remaining LUTs, stays behind the leader
        // (or ties with a higher class index, which the argmax resolves to the leader).
        (0..GROUPS).all(|g| {
            g == leader || lead > counts[g] + remaining || (lead == counts[g] + remaining && leader < g)
        })
    } else {
        let mut second = 0u32;
        for (g, &c) in counts.iter().enumerate() {
            if g != leader && c > second {
                second = c;
            }
        }
        lead >= second + EE_DELTA
    }
}

/// LUT layer in rounds with early exit; returns (prediction, rounds evaluated).
#[inline(never)]
fn classify(w: &[u32; WORDS]) -> (u8, u32) {
    let mut counts = [0u32; GROUPS];
    let mut rounds = PER_GROUP as u32;
    for j in 0..PER_GROUP {
        for (g, count) in counts.iter_mut().enumerate() {
            *count += lut_output(w, g * PER_GROUP + j);
        }
        let remaining = (PER_GROUP - 1 - j) as u32;
        if remaining > 0 && can_stop(&counts, remaining) {
            rounds = j as u32 + 1;
            break;
        }
    }
    let mut best = 0usize;
    for g in 1..GROUPS {
        if counts[g] > counts[best] {
            best = g;
        }
    }
    (best as u8, rounds)
}

#[entry]
fn main() -> ! {
    rtt_init_print!(BlockIfFull, 4096);
    let mut cp = cortex_m::Peripherals::take().unwrap();
    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    rprintln!("LUT_SHD_EARLY_EXIT delta={}", EE_DELTA);
    let clock_hz = board::init_and_report();
    if !MODEL_PRESENT {
        rprintln!("No model embedded: build with LUT_MODEL_DIR set.");
        cortex_m::asm::bkpt();
    }

    let t0 = get_cycles();
    let t1 = get_cycles();
    let overhead = t1.wrapping_sub(t0);

    let mut buffer = [0u32; WORDS];
    let (mut correct, mut agree, mut rounds_sum) = (0u32, 0u32, 0u32);
    let (mut c_min, mut c_max, mut c_sum) = (u32::MAX, 0u32, 0u64);

    rprint!("PRED {} ", CHUNK_START);
    for (s, sample) in TEST_X.iter().enumerate() {
        buffer.copy_from_slice(sample);
        let input = black_box(&buffer);
        let start = get_cycles();
        let (pred, rounds) = black_box(classify(input));
        let cycles = get_cycles().wrapping_sub(start).saturating_sub(overhead);

        c_min = c_min.min(cycles);
        c_max = c_max.max(cycles);
        c_sum += u64::from(cycles);
        rounds_sum += rounds;
        correct += u32::from(pred == TEST_Y[s]);
        agree += u32::from(pred == HOST_PRED[s]);
        rprint!("{:02x}", pred);
    }
    rprintln!();

    let n = TEST_X.len() as u32;
    let mean = c_sum as f32 / n as f32;
    rprintln!(
        "CHUNK start={} samples={} correct={} host_agree={} cycles_min={} cycles_mean={:.1} cycles_max={} us_mean={:.2} clock_hz={} rounds_sum={}",
        CHUNK_START,
        n,
        correct,
        agree,
        c_min,
        mean,
        c_max,
        board::cycles_to_us(mean, clock_hz),
        clock_hz,
        rounds_sum
    );
    rprintln!("DONE");

    loop {
        cortex_m::asm::bkpt();
    }
}
