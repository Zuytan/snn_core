//! Incremental streaming evaluation of a temporal LUT network on the Wio-E5.
//!
//! Each SHD test sample is streamed tick by tick into a 64-tick window that starts empty
//! (silence). At every tick the window slides by one bin, and only the LUTs whose input bits
//! changed are recomputed (dependency lists from `build.rs`); the class counts are updated by +1
//! or -1 and a prediction is produced at every tick. The timed region covers the window shift, the
//! change detection, the LUT updates and the per-tick argmax.
//!
//! After the last tick the window holds the whole sample, so the prediction must equal the block
//! evaluation (`HOST_PRED`), and the incremental counts are checked against a full recomputation
//! (not timed). The CHUNK line reports per-tick cycles and the LUTs updated per tick.

#![no_std]
#![no_main]

#[path = "../board.rs"]
mod board;

#[allow(dead_code)]
mod model {
    include!(concat!(env!("OUT_DIR"), "/lut_shd_model.rs"));
}

mod deps {
    include!(concat!(env!("OUT_DIR"), "/lut_deps.rs"));
}

use core::hint::black_box;
use cortex_m_rt::entry;
use deps::*;
use model::*;
use panic_halt as _;
use rtt_target::{rprint, rprintln, rtt_init_print, ChannelMode::BlockIfFull};

const GROUPS: usize = 20;
const PER_GROUP: usize = N_LUTS / GROUPS;
const TICKS: usize = 64;
const CHANNELS: usize = WORDS / 2; // one 64-bit window (2 words) per channel
const LUT_WORDS: usize = N_LUTS.div_ceil(32);

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

fn argmax(counts: &[i32; GROUPS]) -> u8 {
    let mut best = 0usize;
    for g in 1..GROUPS {
        if counts[g] > counts[best] {
            best = g;
        }
    }
    best as u8
}

/// Full evaluation of `w`: LUT output bits and class counts.
fn full_state(w: &[u32; WORDS], outs: &mut [u32; LUT_WORDS], counts: &mut [i32; GROUPS]) {
    *outs = [0; LUT_WORDS];
    *counts = [0; GROUPS];
    for n in 0..N_LUTS {
        let o = lut_output(w, n);
        outs[n / 32] |= o << (n % 32);
        counts[n / PER_GROUP] += o as i32;
    }
}

struct Stream {
    window: [u32; WORDS],
    outs: [u32; LUT_WORDS],
    counts: [i32; GROUPS],
    dirty: [u32; LUT_WORDS],
}

impl Stream {
    /// Pushes one tick (`column` bit c = channel c) and returns (prediction, LUTs recomputed).
    #[inline(never)]
    fn tick(&mut self, column: &[u32; 2]) -> (u8, u32) {
        // 1. Slide every channel window by one bin and mark the LUTs that read a changed bit.
        for c in 0..CHANNELS {
            let lo = self.window[2 * c];
            let hi = self.window[2 * c + 1];
            let new_bit = (column[c / 32] >> (c % 32)) & 1;
            let new_lo = (lo >> 1) | (hi << 31);
            let new_hi = (hi >> 1) | (new_bit << 31);
            self.window[2 * c] = new_lo;
            self.window[2 * c + 1] = new_hi;
            for (word, mut diff) in [(2 * c, lo ^ new_lo), (2 * c + 1, hi ^ new_hi)] {
                while diff != 0 {
                    let f = word * 32 + diff.trailing_zeros() as usize;
                    diff &= diff - 1;
                    for &n in &DEP_LUT[DEP_PTR[f] as usize..DEP_PTR[f + 1] as usize] {
                        self.dirty[n as usize / 32] |= 1 << (n % 32);
                    }
                }
            }
        }
        // 2. Recompute the marked LUTs and update the class counts by the change of their output.
        let mut updated = 0u32;
        for wi in 0..LUT_WORDS {
            let mut d = self.dirty[wi];
            self.dirty[wi] = 0;
            while d != 0 {
                let n = wi * 32 + d.trailing_zeros() as usize;
                d &= d - 1;
                updated += 1;
                let new = lut_output(&self.window, n);
                let old = (self.outs[wi] >> (n % 32)) & 1;
                if new != old {
                    self.outs[wi] ^= 1 << (n % 32);
                    self.counts[n / PER_GROUP] += if new == 1 { 1 } else { -1 };
                }
            }
        }
        (argmax(&self.counts), updated)
    }
}

/// Bits of tick `t` of a packed sample, one bit per channel.
fn column(sample: &[u32; WORDS], t: usize) -> [u32; 2] {
    let mut col = [0u32; 2];
    for c in 0..CHANNELS {
        let f = c * TICKS + t;
        let bit = (sample[f / 32] >> (f % 32)) & 1;
        col[c / 32] |= bit << (c % 32);
    }
    col
}

#[entry]
fn main() -> ! {
    rtt_init_print!(BlockIfFull, 4096);
    let mut cp = cortex_m::Peripherals::take().unwrap();
    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    rprintln!("LUT_STREAM_EVAL");
    let clock_hz = board::init_and_report();
    if !MODEL_PRESENT || DEP_LUT.len() != N_LUTS * 6 {
        rprintln!("No model embedded: build with LUT_MODEL_DIR set.");
        cortex_m::asm::bkpt();
    }

    let t0 = get_cycles();
    let t1 = get_cycles();
    let overhead = t1.wrapping_sub(t0);

    // State of the empty (silent) window, computed once.
    let zero = [0u32; WORDS];
    let mut zero_outs = [0u32; LUT_WORDS];
    let mut zero_counts = [0i32; GROUPS];
    full_state(&zero, &mut zero_outs, &mut zero_counts);

    let (mut correct, mut agree, mut counts_ok) = (0u32, 0u32, 0u32);
    let (mut c_min, mut c_max, mut c_sum, mut updated_sum) = (u32::MAX, 0u32, 0u64, 0u64);
    let mut check_outs = [0u32; LUT_WORDS];
    let mut check_counts = [0i32; GROUPS];

    rprint!("PRED {} ", CHUNK_START);
    for (s, sample) in TEST_X.iter().enumerate() {
        let mut stream = Stream { window: zero, outs: zero_outs, counts: zero_counts, dirty: [0; LUT_WORDS] };
        let mut pred = 0u8;
        for t in 0..TICKS {
            let col = black_box(column(sample, t));
            let start = get_cycles();
            let (p, updated) = black_box(stream.tick(&col));
            let cycles = get_cycles().wrapping_sub(start).saturating_sub(overhead);
            c_min = c_min.min(cycles);
            c_max = c_max.max(cycles);
            c_sum += u64::from(cycles);
            updated_sum += u64::from(updated);
            pred = p;
        }
        full_state(&stream.window, &mut check_outs, &mut check_counts);
        counts_ok += u32::from(check_counts == stream.counts && check_outs == stream.outs && stream.window == *sample);
        correct += u32::from(pred == TEST_Y[s]);
        agree += u32::from(pred == HOST_PRED[s]);
        rprint!("{:02x}", pred);
    }
    rprintln!();

    let n = TEST_X.len() as u32;
    let ticks = n * TICKS as u32;
    let mean = c_sum as f32 / ticks as f32;
    rprintln!(
        "CHUNK start={} samples={} correct={} host_agree={} cycles_min={} cycles_mean={:.1} cycles_max={} us_mean={:.2} clock_hz={} ticks={} updated_sum={} counts_ok={}",
        CHUNK_START,
        n,
        correct,
        agree,
        c_min,
        mean,
        c_max,
        board::cycles_to_us(mean, clock_hz),
        clock_hz,
        ticks,
        updated_sum,
        counts_ok
    );
    rprintln!("DONE");

    loop {
        cortex_m::asm::bkpt();
    }
}
