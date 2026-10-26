//! Runs an int8 CNN with CMSIS-NN on SHD test samples on the Wio-E5: the int8 baseline of the
//! forest of LUT trees.
//!
//! Network (NHWC, H = 1): conv(64 -> C1, K1, stride S1) + ReLU, conv(C1 -> C2, K2, stride S2) +
//! ReLU, global average pooling, fully connected (C2 -> 20). The layers are CMSIS-NN's
//! `arm_convolve_wrapper_s8`, `arm_avgpool_s8` and `arm_fully_connected_s8`, compiled from source by
//! `build.rs` (`CMSIS_NN_DIR`, `CMSIS_CORE_DIR`); the model and the test chunk come from
//! `CNN_MODEL_DIR` (export of `tools/bptn_torch/cnn_int8_baseline.py`).
//!
//! The packed input bits are expanded to int8 (0 or 127) before the timed region; that expansion is
//! timed separately (`unpack_mean`). Output format and the final breakpoint are the same as in
//! `lut_shd_eval.rs`.

#![no_std]
#![no_main]

#[path = "../board.rs"]
mod board;

#[allow(dead_code)]
mod model {
    include!(concat!(env!("OUT_DIR"), "/cnn_shd_model.rs"));
}

use core::hint::black_box;
use cortex_m_rt::entry;
use model::*;
use panic_halt as _;
use rtt_target::{rprint, rprintln, rtt_init_print, ChannelMode::BlockIfFull};

const CLASSES: usize = 20;
const CHANNELS: usize = 64;
const TICKS: usize = 64;

#[inline(always)]
fn get_cycles() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

/// CMSIS-NN types and functions (Include/arm_nn_types.h, Include/arm_nnfunctions.h, v8.0.0).
#[cfg(cmsis_nn)]
mod cmsis {
    #[repr(C)]
    pub struct Tile {
        pub w: i32,
        pub h: i32,
    }
    #[repr(C)]
    pub struct Context {
        pub buf: *mut core::ffi::c_void,
        pub size: i32,
    }
    #[repr(C)]
    pub struct Dims {
        pub n: i32,
        pub h: i32,
        pub w: i32,
        pub c: i32,
    }
    #[repr(C)]
    pub struct PerChannelQuant {
        pub multiplier: *mut i32,
        pub shift: *mut i32,
    }
    #[repr(C)]
    pub struct PerTensorQuant {
        pub multiplier: i32,
        pub shift: i32,
    }
    #[repr(C)]
    pub struct Activation {
        pub min: i32,
        pub max: i32,
    }
    #[repr(C)]
    pub struct ConvParams {
        pub input_offset: i32,
        pub output_offset: i32,
        pub stride: Tile,
        pub padding: Tile,
        pub dilation: Tile,
        pub activation: Activation,
    }
    #[repr(C)]
    pub struct PoolParams {
        pub stride: Tile,
        pub padding: Tile,
        pub activation: Activation,
    }
    #[repr(C)]
    pub struct FcParams {
        pub input_offset: i32,
        pub filter_offset: i32,
        pub output_offset: i32,
        pub activation: Activation,
    }

    extern "C" {
        pub fn arm_convolve_wrapper_s8_get_buffer_size(
            conv: *const ConvParams,
            input: *const Dims,
            filter: *const Dims,
            output: *const Dims,
        ) -> i32;
        pub fn arm_convolve_wrapper_s8(
            ctx: *const Context,
            conv: *const ConvParams,
            quant: *const PerChannelQuant,
            input_dims: *const Dims,
            input: *const i8,
            filter_dims: *const Dims,
            filter: *const i8,
            bias_dims: *const Dims,
            bias: *const i32,
            output_dims: *const Dims,
            output: *mut i8,
        ) -> i32;
        pub fn arm_avgpool_s8(
            ctx: *const Context,
            pool: *const PoolParams,
            input_dims: *const Dims,
            input: *const i8,
            filter_dims: *const Dims,
            output_dims: *const Dims,
            output: *mut i8,
        ) -> i32;
        pub fn arm_fully_connected_s8(
            ctx: *const Context,
            fc: *const FcParams,
            quant: *const PerTensorQuant,
            input_dims: *const Dims,
            input: *const i8,
            filter_dims: *const Dims,
            filter: *const i8,
            bias_dims: *const Dims,
            bias: *const i32,
            output_dims: *const Dims,
            output: *mut i8,
        ) -> i32;
    }

    pub const fn dims(n: usize, h: usize, w: usize, c: usize) -> Dims {
        Dims { n: n as i32, h: h as i32, w: w as i32, c: c as i32 }
    }
    pub const fn tile(w: usize, h: usize) -> Tile {
        Tile { w: w as i32, h: h as i32 }
    }
}

const SCRATCH_BYTES: usize = 16 * 1024;

// Only `input` is used when CMSIS-NN is not built in.
#[cfg_attr(not(cmsis_nn), allow(dead_code))]
struct Buffers {
    input: [i8; TICKS * CHANNELS],
    a1: [i8; W1 * C1],
    a2: [i8; W2 * C2],
    pooled: [i8; C2],
    logits: [i8; CLASSES],
    scratch: [u8; SCRATCH_BYTES],
}

/// Expands a packed sample (bit c * 64 + t) into the NHWC int8 input (index t * 64 + c).
fn unpack(sample: &[u32; WORDS], input: &mut [i8; TICKS * CHANNELS]) {
    for c in 0..CHANNELS {
        for t in 0..TICKS {
            let f = c * TICKS + t;
            input[t * CHANNELS + c] = if (sample[f / 32] >> (f % 32)) & 1 != 0 { 127 } else { 0 };
        }
    }
}

/// Runs the network on `b.input`; returns the prediction, or 0xFF if a CMSIS-NN call fails.
#[cfg(cmsis_nn)]
#[inline(never)]
fn classify(b: &mut Buffers) -> u8 {
    use cmsis::*;
    let relu = Activation { min: -128, max: 127 };
    let ctx = Context { buf: b.scratch.as_mut_ptr().cast(), size: SCRATCH_BYTES as i32 };
    let no_ctx = Context { buf: core::ptr::null_mut(), size: 0 };

    let conv1 = ConvParams {
        input_offset: 0,
        output_offset: -128,
        stride: tile(S1, 1),
        padding: tile(0, 0),
        dilation: tile(1, 1),
        activation: Activation { ..relu },
    };
    let conv2 = ConvParams {
        input_offset: 128,
        output_offset: -128,
        stride: tile(S2, 1),
        padding: tile(0, 0),
        dilation: tile(1, 1),
        activation: Activation { ..relu },
    };
    // SAFETY: CMSIS-NN only reads the multiplier and shift arrays; the casts from shared statics
    // to `*mut` match its C prototype (non-const pointers) without any write.
    let q1 = PerChannelQuant {
        multiplier: CONV1_MULT.as_ptr().cast_mut(),
        shift: CONV1_SHIFT.as_ptr().cast_mut(),
    };
    let q2 = PerChannelQuant {
        multiplier: CONV2_MULT.as_ptr().cast_mut(),
        shift: CONV2_SHIFT.as_ptr().cast_mut(),
    };
    let pool = PoolParams { stride: tile(1, 1), padding: tile(0, 0), activation: Activation { ..relu } };
    let fc = FcParams { input_offset: 128, filter_offset: 0, output_offset: 0, activation: relu };
    let fq = PerTensorQuant { multiplier: FC_MULT, shift: FC_SHIFT };

    // SAFETY: every pointer refers to a live buffer whose length matches the dimensions passed with
    // it (input TICKS x CHANNELS, a1 W1 x C1, a2 W2 x C2, pooled C2, logits 20); the scratch buffer
    // is checked against the size CMSIS-NN asks for in `main`.
    let status = unsafe {
        arm_convolve_wrapper_s8(
            &ctx,
            &conv1,
            &q1,
            &dims(1, 1, TICKS, CHANNELS),
            b.input.as_ptr(),
            &dims(C1, 1, K1, CHANNELS),
            CONV1_W.as_ptr(),
            &dims(1, 1, 1, C1),
            CONV1_B.as_ptr(),
            &dims(1, 1, W1, C1),
            b.a1.as_mut_ptr(),
        ) | arm_convolve_wrapper_s8(
            &ctx,
            &conv2,
            &q2,
            &dims(1, 1, W1, C1),
            b.a1.as_ptr(),
            &dims(C2, 1, K2, C1),
            CONV2_W.as_ptr(),
            &dims(1, 1, 1, C2),
            CONV2_B.as_ptr(),
            &dims(1, 1, W2, C2),
            b.a2.as_mut_ptr(),
        ) | arm_avgpool_s8(
            &ctx,
            &pool,
            &dims(1, 1, W2, C2),
            b.a2.as_ptr(),
            &dims(1, 1, W2, 1),
            &dims(1, 1, 1, C2),
            b.pooled.as_mut_ptr(),
        ) | arm_fully_connected_s8(
            &no_ctx,
            &fc,
            &fq,
            &dims(1, 1, 1, C2),
            b.pooled.as_ptr(),
            &dims(C2, 1, 1, CLASSES),
            FC_W.as_ptr(),
            &dims(1, 1, 1, CLASSES),
            FC_B.as_ptr(),
            &dims(1, 1, 1, CLASSES),
            b.logits.as_mut_ptr(),
        )
    };
    if status != 0 {
        return 0xFF;
    }
    let mut best = 0usize;
    for c in 1..CLASSES {
        if b.logits[c] > b.logits[best] {
            best = c;
        }
    }
    best as u8
}

#[cfg(not(cmsis_nn))]
fn classify(_: &mut Buffers) -> u8 {
    0xFF
}

#[entry]
fn main() -> ! {
    rtt_init_print!(BlockIfFull, 4096);
    let mut cp = cortex_m::Peripherals::take().unwrap();
    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    rprintln!("CNN_SHD_EVAL c1={} k1={} s1={} c2={} k2={} s2={}", C1, K1, S1, C2, K2, S2);
    let clock_hz = board::init_and_report();
    if !MODEL_PRESENT || !cfg!(cmsis_nn) {
        rprintln!("No model or no CMSIS-NN: build with CNN_MODEL_DIR, CMSIS_NN_DIR and CMSIS_CORE_DIR set.");
        cortex_m::asm::bkpt();
    }

    #[cfg(cmsis_nn)]
    {
        use cmsis::*;
        let p = |s: usize, o: i32, i: i32| ConvParams {
            input_offset: i,
            output_offset: o,
            stride: tile(s, 1),
            padding: tile(0, 0),
            dilation: tile(1, 1),
            activation: Activation { min: -128, max: 127 },
        };
        // SAFETY: pure size queries on stack values.
        let need = unsafe {
            arm_convolve_wrapper_s8_get_buffer_size(&p(S1, -128, 0), &dims(1, 1, TICKS, CHANNELS), &dims(C1, 1, K1, CHANNELS), &dims(1, 1, W1, C1))
                .max(arm_convolve_wrapper_s8_get_buffer_size(&p(S2, -128, 128), &dims(1, 1, W1, C1), &dims(C2, 1, K2, C1), &dims(1, 1, W2, C2)))
        };
        rprintln!("CMSIS-NN scratch needed: {} B (available {} B)", need, SCRATCH_BYTES);
        if need as usize > SCRATCH_BYTES {
            cortex_m::asm::bkpt();
        }
    }

    let t0 = get_cycles();
    let t1 = get_cycles();
    let overhead = t1.wrapping_sub(t0);

    let mut b = Buffers {
        input: [0; TICKS * CHANNELS],
        a1: [0; W1 * C1],
        a2: [0; W2 * C2],
        pooled: [0; C2],
        logits: [0; CLASSES],
        scratch: [0; SCRATCH_BYTES],
    };
    let (mut correct, mut agree) = (0u32, 0u32);
    let (mut c_min, mut c_max, mut c_sum, mut unpack_sum) = (u32::MAX, 0u32, 0u64, 0u64);

    rprint!("PRED {} ", CHUNK_START);
    for (s, sample) in TEST_X.iter().enumerate() {
        let start = get_cycles();
        unpack(black_box(sample), &mut b.input);
        unpack_sum += u64::from(get_cycles().wrapping_sub(start).saturating_sub(overhead));
        black_box(&b.input);

        let start = get_cycles();
        let pred = black_box(classify(&mut b));
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
        "CHUNK start={} samples={} correct={} host_agree={} cycles_min={} cycles_mean={:.1} cycles_max={} us_mean={:.2} clock_hz={} unpack_sum={}",
        CHUNK_START,
        n,
        correct,
        agree,
        c_min,
        mean,
        c_max,
        board::cycles_to_us(mean, clock_hz),
        clock_hz,
        unpack_sum
    );
    rprintln!("DONE");

    loop {
        cortex_m::asm::bkpt();
    }
}
