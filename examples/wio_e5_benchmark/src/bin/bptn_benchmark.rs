#![no_std]
#![no_main]

#[path = "../board.rs"]
mod board;

use core::hint::black_box;
use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print, ChannelMode::BlockIfFull};

use snn_core_bptn::{
    dilate_temporal_jitter, erode_temporal_noise, open_temporal_filter, BptnHysteresisLatch,
    BptnRobustToggleCell, CompactBptnNeuron, TemporalFramingBuffer64,
};

#[inline(always)]
fn get_cycles() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

// ============================================================================
// LAYER CONSTANTS (16 CHANNELS x 16 NEURONS, SYNTHETIC MASKS, NOT A TRAINED KWS MODEL)
// ============================================================================
const KWS_CHANNELS: usize = 16;
const KWS_NEURONS: usize = 16;
const ITER: u32 = 1000;

// Immutable Flash ROM synaptic masks (.rodata)
static KWS_KERNELS: [[u64; KWS_CHANNELS]; KWS_NEURONS] = [
    [0x0000_0000_0000_F000; KWS_CHANNELS],
    [0x0000_0000_000F_0000; KWS_CHANNELS],
    [0x0000_0000_00F0_0000; KWS_CHANNELS],
    [0x0000_0000_0F00_0000; KWS_CHANNELS],
    [0x0000_0000_F000_0000; KWS_CHANNELS],
    [0x0000_000F_0000_0000; KWS_CHANNELS],
    [0x0000_00F0_0000_0000; KWS_CHANNELS],
    [0x0000_0F00_0000_0000; KWS_CHANNELS],
    [0x0000_F000_0000_0000; KWS_CHANNELS],
    [0x000F_0000_0000_0000; KWS_CHANNELS],
    [0x00F0_0000_0000_0000; KWS_CHANNELS],
    [0x0F00_0000_0000_0000; KWS_CHANNELS],
    [0xF000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_FFFF_0000_FFFF; KWS_CHANNELS],
    [0xFFFF_0000_FFFF_0000; KWS_CHANNELS],
    [0xAAAA_AAAA_AAAA_AAAA; KWS_CHANNELS],
];

static KWS_POLARITIES: [[u64; KWS_CHANNELS]; KWS_NEURONS] = [
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
    [0x0000_0000_0000_0000; KWS_CHANNELS],
];

static KWS_THRESHOLDS: [u32; KWS_NEURONS] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16];

#[entry]
fn main() -> ! {
    rtt_init_print!(BlockIfFull, 4096);

    let mut cp = cortex_m::Peripherals::take().unwrap();
    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    rprintln!("\n================================================================================");
    rprintln!("🚀 BPTN SILICON BENCHMARK: WIO-E5 (STM32WLE5JC)");
    rprintln!("⚡ ARM Cortex-M4 without FPU (thumbv7em-none-eabi) - Cycle-Accurate DWT Profiling");
    rprintln!("================================================================================\n");
    let clock_hz = board::init_and_report();

    // DWT overhead measurement
    let t0 = get_cycles();
    let t1 = get_cycles();
    let overhead = t1.wrapping_sub(t0);
    rprintln!(
        "ℹ️  DWT counter access overhead: {} cycles\n",
        overhead
    );

    // ------------------------------------------------------------------------
    // SECTION 1: MORPHOLOGICAL OPERATORS & ROBUST CELLS (Unit Timing)
    // ------------------------------------------------------------------------
    rprintln!("┌──────────────────────────────────────────────────────────────────────────────┐");
    rprintln!("│ 1. MICRO-ARCHITECTURAL MEASUREMENT OF BPTN 1-CYCLE OPERATORS                  │");
    rprintln!("└──────────────────────────────────────────────────────────────────────────────┘");

    let sample_u64 = 0xAA55_AA55_AA55_AA55u64;

    // 1.1 Morphological Dilation (Anti-Jitter)
    let start = get_cycles();
    for _ in 0..ITER {
        black_box(dilate_temporal_jitter(black_box(sample_u64), 1));
    }
    let elapsed = get_cycles().wrapping_sub(start).saturating_sub(overhead);
    let cycles_dilate = (elapsed as f32) / (ITER as f32);
    rprintln!(
        "  • Morphological 1D Dilation (D_1)     : {:.2} cycles ({:.1} ns)",
        cycles_dilate,
        board::cycles_to_us(cycles_dilate, clock_hz) * 1000.0
    );

    // 1.2 Morphological Erosion (Noise Filter)
    let start = get_cycles();
    for _ in 0..ITER {
        black_box(erode_temporal_noise(black_box(sample_u64), 1));
    }
    let elapsed = get_cycles().wrapping_sub(start).saturating_sub(overhead);
    let cycles_erode = (elapsed as f32) / (ITER as f32);
    rprintln!(
        "  • Morphological 1D Erosion (E_1)      : {:.2} cycles ({:.1} ns)",
        cycles_erode,
        board::cycles_to_us(cycles_erode, clock_hz) * 1000.0
    );

    // 1.3 Full Morphological Opening
    let start = get_cycles();
    for _ in 0..ITER {
        black_box(open_temporal_filter(black_box(sample_u64), 1));
    }
    let elapsed = get_cycles().wrapping_sub(start).saturating_sub(overhead);
    let cycles_open = (elapsed as f32) / (ITER as f32);
    rprintln!(
        "  • Full Morphological Opening (O_1=DoE): {:.2} cycles ({:.1} ns)",
        cycles_open,
        board::cycles_to_us(cycles_open, clock_hz) * 1000.0
    );

    // 1.4 2-Bit Saturating Hysteresis Latch (HS-BPTN)
    let mut hs_latch = BptnHysteresisLatch::new();
    let start = get_cycles();
    for _ in 0..ITER {
        black_box(hs_latch.step(black_box(true), false));
    }
    let elapsed = get_cycles().wrapping_sub(start).saturating_sub(overhead);
    let cycles_hs = (elapsed as f32) / (ITER as f32);
    rprintln!(
        "  • 2-Bit Hysteresis Latch (HS-BPTN)    : {:.2} cycles / timestep",
        cycles_hs
    );

    // 1.5 Confirmed Anti-Glitch Parity Latch
    let mut robust_toggle = BptnRobustToggleCell::new();
    let start = get_cycles();
    for _ in 0..ITER {
        black_box(robust_toggle.step(black_box(true)));
    }
    let elapsed = get_cycles().wrapping_sub(start).saturating_sub(overhead);
    let cycles_toggle = (elapsed as f32) / (ITER as f32);
    rprintln!(
        "  • Anti-Glitch Parity Latch (2-tick)   : {:.2} cycles / timestep",
        cycles_toggle
    );

    // ------------------------------------------------------------------------
    // SECTION 2: FULL BPTN LAYER (16 Channels -> 16 Neurons = 256 Synapses)
    // ------------------------------------------------------------------------
    rprintln!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    rprintln!("│ 2. BPTN LAYER, 16 CHANNELS x 16 NEURONS (synthetic masks, not a KWS model)   │");
    rprintln!("└──────────────────────────────────────────────────────────────────────────────┘");

    let mut neurons: [CompactBptnNeuron<KWS_CHANNELS>; KWS_NEURONS] =
        [CompactBptnNeuron::new([0; KWS_CHANNELS], [0; KWS_CHANNELS], 0); KWS_NEURONS];

    for i in 0..KWS_NEURONS {
        neurons[i] = CompactBptnNeuron::new(KWS_KERNELS[i], KWS_POLARITIES[i], KWS_THRESHOLDS[i]);
    }

    let histories = [
        0x0000_0000_0000_F000u64,
        0x0000_0000_000F_0000u64,
        0x0000_0000_00F0_0000u64,
        0x0000_0000_0F00_0000u64,
        0x0000_0000_F000_0000u64,
        0x0000_000F_0000_0000u64,
        0x0000_00F0_0000_0000u64,
        0x0000_0F00_0000_0000u64,
        0x0000_F000_0000_0000u64,
        0x000F_0000_0000_0000u64,
        0x00F0_0000_0000_0000u64,
        0x0F00_0000_0000_0000u64,
        0xF000_0000_0000_0000u64,
        0x0000_FFFF_0000_FFFFu64,
        0xFFFF_0000_FFFF_0000u64,
        0xAAAA_AAAA_AAAA_AAAAu64,
    ];

    let mut out_mask: u32 = 0;

    let start = get_cycles();
    for _ in 0..ITER {
        let mut mask: u32 = 0;
        let histories = black_box(&histories);
        for (n_idx, n) in black_box(&neurons).iter().enumerate() {
            if n.tick(histories) {
                mask |= 1 << n_idx;
            }
        }
        out_mask = black_box(mask);
    }
    let elapsed = get_cycles().wrapping_sub(start).saturating_sub(overhead);
    let cycles_kws_layer = (elapsed as f32) / (ITER as f32);
    let us_inference = board::cycles_to_us(cycles_kws_layer, clock_hz);

    rprintln!(
        "  • One evaluation of a 64-tick window      : {:.1} cycles ({:.2} µs)",
        cycles_kws_layer,
        us_inference
    );
    rprintln!(
        "  • Output spike mask                       : 0x{:04X} (synthetic input, no accuracy)",
        out_mask
    );

    // ------------------------------------------------------------------------
    // SECTION 3: BPTN LAYER COST ON CORTEX-M4
    // ------------------------------------------------------------------------
    rprintln!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    rprintln!("│ 3. BPTN LAYER COST ON THIS BOARD (no baselines measured here)                │");
    rprintln!("└──────────────────────────────────────────────────────────────────────────────┘");

    // Energy estimate for the BPTN layer only (CPU active power, nothing else), from an assumed
    // run current, not a measurement (no current probe was used). See CLAIMS.md.
    let bptn_energy_uj = board::estimated_energy_uj(cycles_kws_layer, clock_hz);

    rprintln!("  Measured (DWT CYCCNT) on this board:");
    rprintln!("  {:<34} | {:>10.1}", "CPU cycles / inference", cycles_kws_layer);
    rprintln!("  {:<34} | {:>10.2}", "Duration (us)", us_inference);
    rprintln!(
        "  {:<34} | {:>10.2}",
        "Energy estimate, CPU active (uJ)",
        bptn_energy_uj
    );
    rprintln!(
        "  (energy assumes {} mA at {} mV; not probed)",
        board::ASSUMED_RUN_CURRENT_UA as f32 / 1000.0,
        board::ASSUMED_SUPPLY_MV
    );
    rprintln!("  Not measured here: LIF SNN and Micro-GRU baselines, SRAM, Flash.");
    rprintln!("  Run the same task with TFLite Micro / CMSIS-NN on this board to compare.");

    // ------------------------------------------------------------------------
    // SECTION 4: TEMPORAL SENTINEL STREAMING (Continuous Demonstrator)
    // ------------------------------------------------------------------------
    rprintln!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    rprintln!("│ 4. FRAMING BUFFER SANITY CHECK (one injected spike, not a keyword detector)  │");
    rprintln!("└──────────────────────────────────────────────────────────────────────────────┘");

    let mut framer = TemporalFramingBuffer64::new();
    let mut detections = 0;

    for tick in 0..100 {
        // Injection of a spike at t = 30
        let spike = tick == 30;
        framer.push_spike(spike);

        // Instantaneous detection evaluation
        let hist = framer.window();
        let detected = (hist & 0x0000_0000_0000_0001u64) != 0;
        if detected {
            detections += 1;
            rprintln!(
                "  [Tick {:2}] spike visible in the 64-bit window",
                tick
            );
        }
    }
    rprintln!("  Ticks where the injected spike was visible: {}", detections);

    rprintln!("\n================================================================================");
    rprintln!("WIO-E5 PROFILING SUMMARY:");
    rprintln!(
        "  BPTN layer: 256 temporal synapses in {:.1} CPU cycles ({:.2} us).",
        cycles_kws_layer,
        us_inference
    );
    rprintln!("  Energy estimate (CPU active only): {:.2} uJ per inference.", bptn_energy_uj);
    rprintln!("================================================================================\n");

    loop {
        cortex_m::asm::wfi();
    }
}
