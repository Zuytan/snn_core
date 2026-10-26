#![no_std]
#![no_main]

#[path = "../board.rs"]
mod board;

use core::hint::black_box as bb;
use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print, ChannelMode::BlockIfFull};

use snn_core::domain::bitset::SpikeBitset;
use snn_core_bptn::{
    bptn_parallel_smith_step_u64, dilate_temporal_jitter, erode_temporal_noise,
    open_temporal_filter, swar_majority_pool_8x8, BptnFeedforwardLayer, BptnHysteresisLatch,
    BptnRobustToggleCell, CompactBptnNeuron,
};
use snn_core::domain::compact_population::CompactLifPopulation;
use snn_core::domain::hardware::{qadd16, smlad, ssat_16};
use snn_core::domain::models::lif::LifConfig;
use snn_core::domain::numeric::{Fix16, Fix32};
use snn_core::domain::rom_csr::QuantizedRomCsrSynapses;

#[inline(always)]
fn get_cycles() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

// ============================================================================
// STRUCTURE DE GESTION DU RAPPORT DE TESTS ON-TARGET
// ============================================================================
struct TestRunner {
    passed: u32,
    failed: u32,
    overhead: u32,
}

impl TestRunner {
    fn new(overhead: u32) -> Self {
        Self {
            passed: 0,
            failed: 0,
            overhead,
        }
    }

    fn check(&mut self, name: &str, condition: bool, cycles: u32) {
        let net_cycles = cycles.saturating_sub(self.overhead);
        if condition {
            self.passed += 1;
            rprintln!("  ✅ [PASS] {:<50} ({} cycles)", name, net_cycles);
        } else {
            self.failed += 1;
            rprintln!("  ❌ [FAIL] {:<50} (ASSERTION FAILED)", name);
        }
    }
}

// ============================================================================
// HAND-WRITTEN TEST PATTERNS IN FLASH ROM (.rodata)
// (16 channels x 64 ticks per pattern). These bit patterns were written by hand to
// look like coarse spectro-temporal shapes of the words "yes", "no" and "stop". They are
// not recorded audio and not a keyword-spotting dataset: suite 4 is a functional test of
// the BPTN layer, not a measure of keyword-spotting accuracy.
// ============================================================================
const KWS_CHANNELS: usize = 16;

// "YES" pattern: low-mid frequency transition then high sibilant (Ch 12-15)
static SAMPLE_YES: [u64; KWS_CHANNELS] = [
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_000F_F000,
    0x0000_0000_00FF_F000,
    0x0000_0000_0FFF_0000,
    0x0000_0000_3FFC_0000,
    0x0000_0000_0FF0_0000,
    0x0000_0000_03C0_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_000F_0000_0000,
    0x0000_00FF_0000_0000,
    0x0000_3FF0_0000_0000,
    0x000F_FF00_0000_0000,
    0x003F_FC00_0000_0000,
    0x001F_F800_0000_0000,
];

// "NO" pattern: nasal resonance and vowel 'o' (Ch 1-6) without high-frequency activity
static SAMPLE_NO: [u64; KWS_CHANNELS] = [
    0x0000_0000_00FF_0000,
    0x0000_0000_0FFF_8000,
    0x0000_0000_1FFF_C000,
    0x0000_0000_3FFF_E000,
    0x0000_0000_1FFF_C000,
    0x0000_0000_0FFF_8000,
    0x0000_0000_00FF_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
];

// "STOP" pattern: sibilant attack 'S' (Ch 13-15), silence, explosion 'T', resonance 'O'
static SAMPLE_STOP: [u64; KWS_CHANNELS] = [
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_000F_FC00_0000,
    0x0000_001F_FE00_0000,
    0x0000_000F_FC00_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_00FF,
    0x0000_0000_0000_01FF,
    0x0000_0000_0000_03FF,
    0x0000_0000_0000_01FF,
];

// "NOISE" pattern: sporadic hand-written bits
static SAMPLE_NOISE: [u64; KWS_CHANNELS] = [
    0x0100_0020_0004_0001,
    0x0008_0001_0040_0002,
    0x0002_0080_0008_0004,
    0x0040_0004_0100_0008,
    0x0001_0200_0010_0000,
    0x0080_0010_0001_0020,
    0x0004_0000_0200_0040,
    0x0200_0008_0020_0001,
    0x0010_0040_0002_0080,
    0x0001_0100_0080_0004,
    0x0400_0002_0010_0002,
    0x0020_0800_0004_0010,
    0x0008_0004_0040_0100,
    0x0080_0020_0100_0001,
    0x0002_0100_0008_0020,
    0x0100_0001_0020_0008,
];

#[entry]
fn main() -> ! {
    rtt_init_print!(BlockIfFull, 8192);

    let mut cp = cortex_m::Peripherals::take().unwrap();
    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    rprintln!("\n================================================================================");
    rprintln!("🧪 REAL SILICON TEST SUITE: WIO-E5 (STM32WLE5JC)");
    rprintln!("⚡ ARM Cortex-M4 without FPU (thumbv7em-none-eabi) - Hardware Validation");
    rprintln!("================================================================================\n");
    board::init_and_report();
    rprintln!("Inputs and results go through black_box, so every check runs on the core.");

    let t0 = get_cycles();
    let t1 = get_cycles();
    let overhead = t1.wrapping_sub(t0);
    rprintln!("ℹ️  Measured DWT access overhead: {} cycles\n", overhead);

    let mut runner = TestRunner::new(overhead);

    // ========================================================================
    // SUITE 1: FIXED-POINT ARITHMETIC & ARM CORTEX-M4 DSP INTRINSICS
    // ========================================================================
    rprintln!("┌──────────────────────────────────────────────────────────────────────────────┐");
    rprintln!("│ SUITE 1: FIXED ARITHMETIC & ARM CORTEX-M4 DSP INTRINSICS                     │");
    rprintln!("└──────────────────────────────────────────────────────────────────────────────┘");

    // 1.1 Fix32 Positive and Negative Saturation
    let start = get_cycles();
    let a = Fix32::from_bits(bb(i32::MAX - 100));
    let b = Fix32::from_bits(bb(500));
    let c = bb(a.saturating_add(b));
    let cycles = get_cycles().wrapping_sub(start);
    runner.check(
        "Fix32::saturating_add to i32::MAX",
        c.to_bits() == i32::MAX,
        cycles,
    );

    // 1.2 Fix16 Saturation Q8.8
    let start = get_cycles();
    let fa = Fix16::from_bits(bb(i16::MIN + 50));
    let fb = Fix16::from_bits(bb(100));
    let fc = bb(fa.saturating_sub(fb));
    let cycles = get_cycles().wrapping_sub(start);
    runner.check(
        "Fix16::saturating_sub to i16::MIN",
        fc.to_bits() == i16::MIN,
        cycles,
    );

    // 1.3 Intrinsic ssat_16 (Branchless 16-bit hardware saturation)
    let start = get_cycles();
    let sat_val = bb(ssat_16(bb(70000)));
    let sat_neg = bb(ssat_16(bb(-70000)));
    let cycles = get_cycles().wrapping_sub(start);
    runner.check(
        "ARM ssat_16 clamp [+32767, -32768]",
        sat_val == 32767 && sat_neg == -32768,
        cycles,
    );

    // 1.4 Intrinsic smlad (Dual 16-bit Multiply-Accumulate in 1 cycle)
    let start = get_cycles();
    let p1 = (1000i16 as u16 as u32) | ((2000i16 as u16 as u32) << 16);
    let p2 = (3i16 as u16 as u32) | ((4i16 as u16 as u32) << 16);
    let res_smlad = bb(smlad(bb(p1), bb(p2), bb(10)));
    let cycles = get_cycles().wrapping_sub(start);
    // (1000 * 3) + (2000 * 4) + 10 = 3000 + 8000 + 10 = 11010
    runner.check(
        "ARM smlad dual-MAC (1000*3 + 2000*4 + 10)",
        res_smlad == 11010,
        cycles,
    );

    // 1.5 Intrinsic qadd16 (Dual 16-bit saturating addition)
    let start = get_cycles();
    let qa = (32000u16 as u32) | ((100u16 as u32) << 16);
    let qb = (2000u16 as u32) | ((200u16 as u32) << 16);
    let res_qadd = bb(qadd16(bb(qa), bb(qb)));
    let cycles = get_cycles().wrapping_sub(start);
    let low16 = (res_qadd & 0xFFFF) as i16;
    let high16 = ((res_qadd >> 16) & 0xFFFF) as i16;
    runner.check(
        "ARM qadd16 dual saturating addition",
        low16 == 32767 && high16 == 300,
        cycles,
    );

    // ========================================================================
    // SUITE 2: BIT OPERATIONS & HARDWARE CTZ INSTRUCTION
    // ========================================================================
    rprintln!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    rprintln!("│ SUITE 2: BIT VECTORIZATION & HARDWARE CTZ INSTRUCTION (CLZ/RBIT)             │");
    rprintln!("└──────────────────────────────────────────────────────────────────────────────┘");

    // 2.1 SpikeBitset Trailing Zeros Hardware (CTZ) via extract_active_indices
    let start = get_cycles();
    let mut bitset = SpikeBitset::<2>::new();
    bitset.set(bb(17));
    bitset.set(bb(42));
    let mut active = [0u32; 4];
    let n_active = bb(bb(&bitset).extract_active_indices(&mut active));
    bb(&active);
    let cycles = get_cycles().wrapping_sub(start);
    runner.check(
        "SpikeBitset CTZ extraction ([17, 42])",
        n_active == 2 && active[0] == 17 && active[1] == 42,
        cycles,
    );

    // 2.2 Hardware population count (count_ones)
    let start = get_cycles();
    let count = bb(bb(&bitset).count_ones());
    let cycles = get_cycles().wrapping_sub(start);
    runner.check("SpikeBitset count_ones() == 2", count == 2, cycles);

    // ========================================================================
    // SUITE 3: MORPHOLOGICAL OPERATORS & BPTN MEMORY CELLS
    // ========================================================================
    rprintln!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    rprintln!("│ SUITE 3: MORPHOLOGICAL OPERATORS & HIGH-RESILIENCE BPTN CELLS                │");
    rprintln!("└──────────────────────────────────────────────────────────────────────────────┘");

    // 3.1 1D Morphological Dilation (Anti-Jitter)
    let start = get_cycles();
    let d_res = bb(dilate_temporal_jitter(bb(0x0000_0000_0000_0002u64), 1));
    let cycles = get_cycles().wrapping_sub(start);
    // Bit 1 dilated by 1 -> Bits 0, 1, 2 = 0b111 = 7
    runner.check(
        "dilate_temporal_jitter(bit 1, 1) -> 0x07",
        d_res == 0x07,
        cycles,
    );

    // 3.2 1D Morphological Erosion (Noise Filter)
    let start = get_cycles();
    let e_res = bb(erode_temporal_noise(bb(0x0000_0000_0000_000Fu64), 1));
    let cycles = get_cycles().wrapping_sub(start);
    // 0x0F = bits 0,1,2,3. Only bits 1 and 2 have active left/right neighbors -> 0b0110 = 6
    runner.check(
        "erode_temporal_noise(0x0F, 1) -> 0x06",
        e_res == 0x06,
        cycles,
    );

    // 3.3 Full Opening on isolated impulse noise
    let start = get_cycles();
    let noisy_isolated_spike = 0x0000_0000_0000_0004u64; // isolated 1-tick impulse
    let filtered = bb(open_temporal_filter(bb(noisy_isolated_spike), 1));
    let cycles = get_cycles().wrapping_sub(start);
    runner.check(
        "open_temporal_filter suppresses isolated spike (0x04 -> 0x00)",
        filtered == 0,
        cycles,
    );

    // 3.4 2-Bit Hysteresis Latch HS-BPTN
    let start = get_cycles();
    let mut latch = BptnHysteresisLatch::new();
    // 1 excitation rises to state 1 (output still false as threshold >= 2)
    let s1 = bb(latch.step(bb(true), false));
    // 2nd excitation rises to state 2 (output toggles to true)
    let s2 = bb(latch.step(bb(true), false));
    // 1 inhibition lowers to state 1 (output falls back to false)
    let s3 = bb(latch.step(bb(false), true));
    let cycles = get_cycles().wrapping_sub(start);
    runner.check(
        "BptnHysteresisLatch toggles at 2 ticks and falls back",
        !s1 && s2 && !s3,
        cycles,
    );

    // 3.5 Anti-Glitch Parity Latch (2-tick confirmation and full cycle)
    let start = get_cycles();
    let mut toggle = BptnRobustToggleCell::new();
    let t1 = bb(toggle.step(bb(true))); // 1st candidate tick -> no toggle (remains 0)
    let t2 = bb(toggle.step(bb(true))); // 2nd consecutive confirmed tick -> toggles to 1
    let t3 = bb(toggle.step(bb(false))); // hold state (remains 1)
    let t4 = bb(toggle.step(bb(true))); // isolated 1-tick glitch (hold 1)
    let t5 = bb(toggle.step(bb(false))); // glitch dropped without confirmation (hold 1)
    let t6 = bb(toggle.step(bb(true))); // 1st candidate inversion tick -> hold 1
    let t7 = bb(toggle.step(bb(true))); // 2nd consecutive confirmed tick -> toggles 1 -> 0!
    let cycles = get_cycles().wrapping_sub(start);
    runner.check(
        "BptnRobustToggleCell full cycle (0 -> 1 -> hold -> 0)",
        !t1 && t2 && t3 && t4 && t5 && t6 && !t7,
        cycles,
    );

    // 3.6 SWAR Majority-Vote Pooling 8x8 Operator (Dense Noise Filter)
    let start = get_cycles();
    let swar_sample = 0x03u64 | (0x01u64 << 8) | (0x0Fu64 << 16);
    let pooled = bb(swar_majority_pool_8x8(bb(swar_sample), 2));
    let cycles = get_cycles().wrapping_sub(start);
    runner.check(
        "swar_majority_pool_8x8 threshold 2 (0b0000_0101)",
        pooled == 0b0000_0101,
        cycles,
    );

    // 3.7 Saturating Smith Automaton 64 Channels in Parallel (bptn_parallel_smith_step_u64)
    let start = get_cycles();
    let (c1_step1, c0_step1) = bb(bptn_parallel_smith_step_u64(bb(0), bb(0), bb(1), 0)); // 0 -> 1
    let (c1_step2, c0_step2) = bb(bptn_parallel_smith_step_u64(c1_step1, c0_step1, bb(1), 0)); // 1 -> 2
    let cycles = get_cycles().wrapping_sub(start);
    let smith_ok =
        (c1_step1 & 1 == 0) && (c0_step1 & 1 == 1) && (c1_step2 & 1 == 1) && (c0_step2 & 1 == 0);
    runner.check(
        "bptn_parallel_smith_step_u64 (64 bit-parallel channels)",
        smith_ok,
        cycles,
    );

    // ========================================================================
    // SUITE 4: BPTN LAYER ON HAND-WRITTEN PATTERNS (FUNCTIONAL TEST, NOT KWS ACCURACY)
    // ========================================================================
    rprintln!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    rprintln!("│ SUITE 4: BPTN LAYER ON 4 HAND-WRITTEN PATTERNS (not recorded audio)          │");
    rprintln!("└──────────────────────────────────────────────────────────────────────────────┘");

    // Definition of a 3-class BPTN classifier: 0: YES, 1: NO, 2: STOP
    // Neuron 0 ("YES" Detector): Threshold 30 (requires resonance + sibilant co-occurrence)
    let mut k_yes = [0u64; KWS_CHANNELS];
    k_yes[3] = 0x0000_0000_00FF_F000;
    k_yes[4] = 0x0000_0000_0FFF_0000;
    k_yes[13] = 0x000F_FF00_0000_0000;
    k_yes[14] = 0x003F_FC00_0000_0000;
    let n_yes = CompactBptnNeuron::new(k_yes, [0; KWS_CHANNELS], 30);

    // Neuron 1 ("NO" Detector): Threshold 30 (requires pure low nasal resonance)
    let mut k_no = [0u64; KWS_CHANNELS];
    k_no[1] = 0x0000_0000_0FFF_8000;
    k_no[2] = 0x0000_0000_1FFF_C000;
    k_no[3] = 0x0000_0000_3FFF_E000;
    let n_no = CompactBptnNeuron::new(k_no, [0; KWS_CHANNELS], 30);

    // Neuron 2 ("STOP" Detector): Threshold 22 (requires sibilant attack then resonance)
    let mut k_stop = [0u64; KWS_CHANNELS];
    k_stop[13] = 0x0000_0000_0000_01FF;
    k_stop[14] = 0x0000_0000_0000_03FF;
    k_stop[2] = 0x0000_000F_FC00_0000;
    let n_stop = CompactBptnNeuron::new(k_stop, [0; KWS_CHANNELS], 22);

    let kws_layer = BptnFeedforwardLayer::new([n_yes, n_no, n_stop]);

    // Test 4.1: Inference on "YES" sample
    let start = get_cycles();
    let mask_yes = bb(bb(&kws_layer).tick_mask(bb(&SAMPLE_YES)));
    let cycles = get_cycles().wrapping_sub(start);
    // Must activate neuron 0 (bit 0 = 1) and reject neurons 1 and 2
    let is_yes_correct = (mask_yes & 0b001) != 0 && (mask_yes & 0b110) == 0;
    runner.check(
        "Pattern 'YES' -> mask 0b001",
        is_yes_correct,
        cycles,
    );

    // Test 4.2: Inference on "NO" sample
    let start = get_cycles();
    let mask_no = bb(bb(&kws_layer).tick_mask(bb(&SAMPLE_NO)));
    let cycles = get_cycles().wrapping_sub(start);
    // Must activate neuron 1 (bit 1 = 1) and reject neurons 0 and 2
    let is_no_correct = (mask_no & 0b010) != 0 && (mask_no & 0b101) == 0;
    runner.check(
        "Pattern 'NO' -> mask 0b010",
        is_no_correct,
        cycles,
    );

    // Test 4.3: Inference on "STOP" sample
    let start = get_cycles();
    let mask_stop = bb(bb(&kws_layer).tick_mask(bb(&SAMPLE_STOP)));
    let cycles = get_cycles().wrapping_sub(start);
    // Must activate neuron 2 (bit 2 = 1) and reject neurons 0 and 1
    let is_stop_correct = (mask_stop & 0b100) != 0 && (mask_stop & 0b011) == 0;
    runner.check(
        "Pattern 'STOP' -> mask 0b100",
        is_stop_correct,
        cycles,
    );

    // Test 4.4: Thermal Noise Immunity / False Positives
    let start = get_cycles();
    let mask_noise = bb(bb(&kws_layer).tick_mask(bb(&SAMPLE_NOISE)));
    let cycles = get_cycles().wrapping_sub(start);
    // Total rejection: no neuron should fire (mask = 0)
    runner.check(
        "Pattern 'NOISE' -> mask 0x00",
        mask_noise == 0,
        cycles,
    );

    // ========================================================================
    // SUITE 5: COMPACT NEURON POPULATION & INT8 FLASH ROM QUANTIZED CSR
    // ========================================================================
    rprintln!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    rprintln!("│ SUITE 5: COMPACT SNN POPULATION (5B/9B) & ROM CSR INT8                       │");
    rprintln!("└──────────────────────────────────────────────────────────────────────────────┘");

    // 5.1 Compact LIF Population (5 bytes per neuron)
    let start = get_cycles();
    let config = LifConfig::new(Fix32::from_num(1.0), Fix32::from_num(0.95), Fix32::ZERO, 2);
    let mut pop: CompactLifPopulation<4> = CompactLifPopulation::new(config);
    // Injection of suprathreshold current into neuron 0
    pop.receive_input(0, bb(Fix32::from_num(1.5)));
    let fired_0 = bb(pop.step_neuron(0));
    let fired_1 = bb(pop.step_neuron(1));
    let cycles = get_cycles().wrapping_sub(start);
    runner.check(
        "CompactLifPopulation spike emission neuron 0",
        fired_0 && !fired_1,
        cycles,
    );

    // 5.2 Flash ROM INT8 CSR Synapse Matrix
    let start = get_cycles();
    static CSR_ROW_PTRS: [u16; 3] = [0, 2, 4];
    static CSR_TARGETS: [u16; 4] = [0, 1, 1, 2];
    static CSR_WEIGHTS: [i8; 4] = [120, -50, 80, 100];
    let csr = QuantizedRomCsrSynapses::new(&CSR_ROW_PTRS, &CSR_TARGETS, &CSR_WEIGHTS, Fix32::ONE);
    let mut acc_buf = [0i32; 3];
    // Active spikes on sources 0 and 1 (mask 0b0011 = 0x03)
    bb(&csr).propagate_spikes_words_i32(bb(&[0x03]), &mut acc_buf);
    bb(&acc_buf);
    let cycles = get_cycles().wrapping_sub(start);
    // Source 0 propagates +120 on target 0, -50 on target 1
    // Source 1 propagates +80 on target 1, +100 on target 2
    // Target 0: 120, Target 1: -50 + 80 = 30, Target 2: 100
    let csr_ok = acc_buf[0] == 120 && acc_buf[1] == 30 && acc_buf[2] == 100;
    runner.check(
        "QuantizedRomCsrSynapses INT8 Flash ROM propagation",
        csr_ok,
        cycles,
    );

    // ========================================================================
    // GLOBAL SILICON RUNNER SUMMARY
    // ========================================================================
    rprintln!("\n================================================================================");
    rprintln!("🏁 GLOBAL WIO-E5 SILICON RUNNER SUMMARY:");
    rprintln!(
        "   • Total tests executed on CPU         : {}",
        runner.passed + runner.failed
    );
    rprintln!(
        "   • Successfully passed tests           : {}",
        runner.passed
    );
    rprintln!(
        "   • Failed tests                        : {}",
        runner.failed
    );
    if runner.failed == 0 {
        rprintln!("🎉 TOTAL SUCCESS: ALL TESTS PASSED ON REAL HARDWARE WITHOUT ERROR!");
    } else {
        rprintln!("⚠️  FAILURE: SOME ASSERTIONS FAILED ON SILICON!");
    }
    rprintln!("================================================================================\n");

    loop {
        cortex_m::asm::wfi();
    }
}
