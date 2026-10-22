//! 🛡️ Critical Safety Tests, Deterministic Fuzzing, and Formal Properties (Phase 4 - v0.7.0)
//!
//! Verifies the absolute absence of undefined behavior, buffer overflows,
//! and panics on critical inference paths, even under malformed or corrupted inputs.

use snn_core::application::ffi::*;
use snn_core::domain::aer::{AerBitsetDecoder, AerSpike1d, AerSpike2d};
use snn_core::domain::bitset::SpikeBitset;
use snn_core::domain::compact_population::CompactLifPopulation;
use snn_core::domain::engine::{EngineError, LayerDescriptor, ModelDescriptor};
use snn_core::domain::hardware::FastDivider;
use snn_core::domain::layers::{Conv2dSynapses, ConvDataLayout};
use snn_core::domain::models::lif::LifConfig;
use snn_core::domain::numeric::{Fix16, Fix32};
use snn_core::domain::rom_csr::{QuantizedRomCsrSynapses, RomCsrSynapses};
use snn_core::traits::SynapseMatrix;

/// Lightweight, deterministic, reproducible pseudo-random LCG generator for fuzzing
struct FuzzRng {
    state: u64,
}

impl FuzzRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_u16(&mut self) -> u16 {
        self.next_u32() as u16
    }

    fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }

    fn next_usize(&mut self, max_exclusive: usize) -> usize {
        if max_exclusive == 0 {
            0
        } else {
            (self.next_u32() as usize) % max_exclusive
        }
    }
}

// ============================================================================
// 1. 1D AND 2D AER FUZZING: RESILIENCE TO CORRUPTED STREAMS
// ============================================================================

#[test]
fn test_fuzz_aer_1d_decoding_boundaries() {
    let mut rng = FuzzRng::new(0xDEAD_BEEF_CAFE_0001);
    let mut bitset = SpikeBitset::<4>::new(); // 128 neurons (0..127)

    // Test with 10,000 random events including out-of-bounds addresses (> 128)
    for _ in 0..10_000 {
        let count = rng.next_usize(50);
        let mut events = [AerSpike1d::default(); 50];
        for ev in &mut events[..count] {
            *ev = AerSpike1d::new(rng.next_u16(), rng.next_u16());
        }

        // Must never panic
        AerBitsetDecoder::decode_1d_batch(&events[..count], &mut bitset);

        // Verification of invariant properties
        for ev in &events[..count] {
            let addr = ev.address as usize;
            if addr < 128 {
                assert!(bitset.get(addr), "Bit {} should be active", addr);
            }
        }
        assert!(bitset.count_ones() <= 128);
    }
}

#[test]
fn test_fuzz_c_ffi_aer_robustness() {
    let mut rng = FuzzRng::new(0x1234_5678_9ABC_DEF0);
    let mut words = [0u32; 4];

    // 1. Null pointers and zero lengths
    unsafe {
        assert_eq!(
            snn_aer_decode_1d(core::ptr::null(), 10, words.as_mut_ptr(), 4),
            -1
        );
        assert_eq!(
            snn_aer_decode_1d(core::ptr::null(), 0, words.as_mut_ptr(), 4),
            0
        );
        assert_eq!(
            snn_aer_decode_1d(core::ptr::null(), 0, core::ptr::null_mut(), 4),
            -1
        );
        assert_eq!(
            snn_aer_decode_1d(core::ptr::null(), 0, words.as_mut_ptr(), 0),
            -1
        );
    }

    // 2. C FFI fuzzing with random event packets
    let mut events = [AerSpike1d::default(); 64];
    for _ in 0..2_000 {
        let n = rng.next_usize(64);
        for ev in &mut events[..n] {
            *ev = AerSpike1d::new(rng.next_u16(), rng.next_u16());
        }

        let ret = unsafe { snn_aer_decode_1d(events.as_ptr(), n, words.as_mut_ptr(), words.len()) };
        assert_eq!(ret, 0);

        for ev in &events[..n] {
            let addr = ev.address as usize;
            if addr < words.len() * 32 {
                assert!(unsafe { snn_bitset_get_spike(words.as_ptr(), words.len(), addr) });
            }
        }
    }
}

#[test]
fn test_fuzz_aer_2d_bipolar_extreme_coordinates() {
    let mut rng = FuzzRng::new(0xCAFE_BABE_0002);
    let mut bitset = SpikeBitset::<8>::new(); // 256 bits

    for _ in 0..5_000 {
        let width = rng.next_usize(64) + 1;
        let height = rng.next_usize(64) + 1;
        let count = rng.next_usize(20);
        let mut events = [AerSpike2d::default(); 20];

        for ev in &mut events[..count] {
            *ev = AerSpike2d::new(
                rng.next_u16(),
                rng.next_u16(),
                (rng.next_u32() & 1) == 1,
                rng.next_u32(),
            );
        }

        AerBitsetDecoder::decode_2d_bipolar_batch(&events[..count], width, height, &mut bitset);

        // Equivalent C FFI call
        let mut ffi_words = [0u32; 8];
        let ret = unsafe {
            snn_aer_decode_2d_bipolar(
                events.as_ptr(),
                count,
                width,
                height,
                ffi_words.as_mut_ptr(),
                ffi_words.len(),
            )
        };
        assert_eq!(ret, 0);
        assert_eq!(bitset.words(), &ffi_words);
    }
}

// ============================================================================
// 2. CORRUPTED CSR MATRICES AND INDEX OUT-OF-BOUNDS ATTACKS
// ============================================================================

#[test]
fn test_fuzz_rom_csr_corrupted_indices_and_monotonicity() {
    let mut rng = FuzzRng::new(0x9999_8888_7777_6666);
    let mut accumulators = [Fix32::ZERO; 16];

    for _ in 0..2_000 {
        // Deliberately construct corrupted CSR matrix
        let row_count = rng.next_usize(20) + 1;
        let mut row_ptrs = [0u16; 32];
        for r in &mut row_ptrs[..=row_count] {
            // Random offsets (can be non-monotonic or exceed targets.len())
            *r = rng.next_u16() % 100;
        }

        let total_synapses = rng.next_usize(30);
        let mut targets = [0u16; 30];
        let mut weights = [Fix32::ZERO; 30];
        for i in 0..total_synapses {
            targets[i] = rng.next_u16(); // Can easily exceed accumulators.len()
            weights[i] = Fix32::from_bits(rng.next_i32());
        }

        let csr = RomCsrSynapses::new(
            &row_ptrs[..=row_count],
            &targets[..total_synapses],
            &weights[..total_synapses],
        );

        // Generate random active spikes
        let mut active_spikes = [0u32; 10];
        let active_count = rng.next_usize(10);
        for s in &mut active_spikes[..active_count] {
            *s = rng.next_u32() % 50;
        }

        // MUST NEVER PANIC OR CRASH
        csr.propagate_active_spikes(&active_spikes[..active_count], &mut accumulators);

        // Test with bitset words
        let words = [rng.next_u32(), rng.next_u32()];
        csr.propagate_spikes_words(&words, &mut accumulators);
    }
}

#[test]
fn test_fuzz_quantized_csr_corrupted() {
    let mut rng = FuzzRng::new(0x5555_4444_3333_2222);
    let mut int_accumulators = [0i32; 16];
    let mut out_accumulators = [Fix32::ZERO; 16];

    for _ in 0..2_000 {
        let row_ptrs = [
            rng.next_u16() % 20,
            rng.next_u16() % 20,
            rng.next_u16() % 20,
        ];
        let targets = [rng.next_u16(), rng.next_u16(), rng.next_u16()];
        let weights = [
            rng.next_i32() as i8,
            rng.next_i32() as i8,
            rng.next_i32() as i8,
        ];
        let scale = Fix32::from_bits(rng.next_i32());

        let q_csr = QuantizedRomCsrSynapses::new(&row_ptrs, &targets, &weights, scale);

        let active = [0u32, 1, 5, 100];
        q_csr.propagate_active_spikes_i32(&active, &mut int_accumulators);
        q_csr.scale_accumulators(&int_accumulators, &mut out_accumulators);
    }
}

// ============================================================================
// 3. C FFI FUZZING: ARENA AND DEEP NETWORK SAFETY
// ============================================================================

#[test]
fn test_fuzz_c_ffi_deep_network_adversarial() {
    let mut rng = FuzzRng::new(0xABCD_1111_2222_3333);

    // 1. Corrupted descriptors
    let cfg = SnnLifConfig {
        threshold: 65536,
        decay_factor: 58982,
        reset_potential: 0,
        refractory_period: 2,
        bias: 0,
        reset_mode: 0,
    };

    let empty_syn = SnnCsrMatrix {
        row_ptrs: core::ptr::null(),
        targets: core::ptr::null(),
        weights: core::ptr::null(),
        in_features: 0,
        out_features: 0,
        total_synapses: 0,
    };

    let layer = SnnLayerDesc {
        layer_type: SnnLayerType::Feedforward,
        in_features: 1,
        num_neurons: 4,
        config: cfg,
        synapses: empty_syn,
        bias: core::ptr::null(),
        quant_scale: 0,
    };

    let desc = SnnDeepNetworkDesc {
        layers: &raw const layer,
        num_layers: 1,
        in_features: 1,
    };

    let req_size = unsafe { snn_deep_network_state_size(&raw const desc) };
    assert!(req_size > 0);

    // Allocate stack arena buffer with padding to test misalignment
    let mut raw_arena = [0u8; 1024];
    let mut state = SnnDeepNetworkState {
        membrane_potentials: core::ptr::null_mut(),
        refractory_times: core::ptr::null_mut(),
        arena_accumulators: core::ptr::null_mut(),
        ping_words: core::ptr::null_mut(),
        pong_words: core::ptr::null_mut(),
        total_neurons: 0,
        max_neurons: 0,
        max_words: 0,
    };

    // Test misaligned arena
    unsafe {
        let misaligned = raw_arena.as_mut_ptr().add(1);
        assert_eq!(
            snn_deep_network_init_state(&raw const desc, misaligned, 512, &raw mut state),
            -4
        );
    }

    // Test arena buffer too small
    unsafe {
        assert_eq!(
            snn_deep_network_init_state(
                &raw const desc,
                raw_arena.as_mut_ptr(),
                req_size - 1,
                &raw mut state
            ),
            -3
        );
    }

    // Valid initialization
    unsafe {
        let ret = snn_deep_network_init_state(
            &raw const desc,
            raw_arena.as_mut_ptr(),
            raw_arena.len(),
            &raw mut state,
        );
        assert_eq!(ret, 0);
    }

    // Fuzz inference ticks with extreme inputs
    let mut out_words = [0u32; 2];
    for _ in 0..1_000 {
        let in_val = rng.next_i32();
        let inputs = [in_val];
        let ret = unsafe {
            snn_deep_network_step(
                &raw const desc,
                &raw mut state,
                inputs.as_ptr(),
                inputs.len(),
                out_words.as_mut_ptr(),
                out_words.len(),
            )
        };
        assert_eq!(ret, 0);
    }
}

// ============================================================================
// 4. FIX32 AND FIX16 MATHEMATICAL INVARIANTS: NO OVERFLOW AND SATURATION
// ============================================================================

#[test]
fn test_property_fix32_addition_commutativity_and_saturation() {
    let mut rng = FuzzRng::new(0x7777_6666_5555_4444);

    for _ in 0..10_000 {
        let a = Fix32::from_bits(rng.next_i32());
        let b = Fix32::from_bits(rng.next_i32());

        // 1. Commutativity: a + b == b + a
        assert_eq!(
            a + b,
            b + a,
            "Non-commutative addition for a={:?}, b={:?}",
            a,
            b
        );

        // 2. Exact saturation invariant (compared to i64 clamp)
        let exact_sum = (a.to_bits() as i64) + (b.to_bits() as i64);
        let expected_sat = exact_sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        assert_eq!((a + b).to_bits(), expected_sat);
    }
}

#[test]
fn test_property_fix32_multiplication_extremes() {
    let extremes = [
        Fix32::from_bits(i32::MAX),
        Fix32::from_bits(i32::MIN),
        Fix32::from_bits(0),
        Fix32::ONE,
        -Fix32::ONE,
        Fix32::from_bits(1),
        Fix32::from_bits(-1),
    ];

    for &a in &extremes {
        for &b in &extremes {
            let prod = a * b;
            // Commutativity
            assert_eq!(prod, b * a);
            // Sign check
            if (a > Fix32::ZERO && b < Fix32::ZERO) || (a < Fix32::ZERO && b > Fix32::ZERO) {
                assert!(prod <= Fix32::ZERO);
            }
        }
    }
}

#[test]
fn test_property_fix16_addition_and_multiplication_saturation() {
    let mut rng = FuzzRng::new(0x3333_2222_1111_0000);

    for _ in 0..10_000 {
        let a = Fix16(rng.next_u16() as i16);
        let b = Fix16(rng.next_u16() as i16);

        // Commutativity
        assert_eq!(a + b, b + a);

        let exact_sum = (a.0 as i32) + (b.0 as i32);
        let expected_sat = exact_sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        assert_eq!((a + b).0, expected_sat);
    }
}

#[test]
fn test_property_fast_divider_exhaustive_fuzz() {
    let mut rng = FuzzRng::new(0xF00D_BEEF_0000_1234);

    for _ in 0..5_000 {
        let d = (rng.next_u16() as usize) % 1024;
        let n = (rng.next_u32() as usize) % 1_000_000;

        let fd = FastDivider::new(d);
        if d <= 1 {
            assert_eq!(fd.div(n), n);
        } else {
            assert_eq!(fd.div(n), n / d, "div mismatch for n={}, d={}", n, d);
            assert_eq!(fd.rem(n), n % d, "rem mismatch for n={}, d={}", n, d);
            assert_eq!(
                fd.div_ceil(n),
                n.div_ceil(d),
                "div_ceil mismatch for n={}, d={}",
                n,
                d
            );
        }
    }
}

// ============================================================================
// 5. PANIC-FREEDOM AUDIT: CRITICAL PATHS WITHOUT ANY POSSIBLE PANIC
// ============================================================================

#[test]
fn test_panic_freedom_compact_population_out_of_bounds() {
    let cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 2);
    let mut pop = CompactLifPopulation::<4>::new(cfg);

    // Deliberate calls with out-of-bounds indices (0..4 are valid)
    assert!(!pop.step_neuron(4));
    assert!(!pop.step_neuron(5));
    assert!(!pop.step_neuron(1000));
    assert!(!pop.step_neuron(usize::MAX));

    pop.receive_input(100, Fix32::ONE); // Must not corrupt memory nor panic

    // Verify valid neurons still function
    pop.receive_input(0, Fix32::from_num(1.5));
    assert!(pop.step_neuron(0)); // Triggers a spike
}

#[test]
fn test_panic_freedom_conv2d_zero_dimensions() {
    let kernel = [Fix32::ONE];
    let mut acc = [Fix32::ZERO; 10];
    let active = [0u32, 1, 2];

    // Conv2d with zero dimensions: must never divide by zero nor panic
    let conv = Conv2dSynapses::with_padding(&kernel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        .with_layout(ConvDataLayout::Hwc);
    conv.propagate_active_spikes(&active, &mut acc);

    // Equivalent C FFI
    let ret = unsafe {
        snn_conv2d_propagate_hwc(
            kernel.as_ptr().cast::<i32>(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            active.as_ptr(),
            active.len(),
            acc.as_mut_ptr().cast::<i32>(),
            acc.len(),
        )
    };
    assert_eq!(ret, 0);
}

#[test]
fn test_fuzz_model_descriptor_adversarial_inputs() {
    let mut rng = FuzzRng::new(0xCAFE_BABE_9999);

    static ROW_PTRS: [u16; 3] = [0, 2, 4];
    static TARGETS: [u16; 4] = [0, 1, 0, 1];
    static WEIGHTS: [Fix32; 4] = [
        Fix32::from_bits(i32::MAX),
        Fix32::from_bits(i32::MIN),
        Fix32::from_bits(1 << 16),
        Fix32::from_bits(-(1 << 16)),
    ];
    let syn = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS);
    let cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 2);
    let layer = LayerDescriptor::new(2, 2, syn, cfg, None);
    let layers = [layer];
    let model = ModelDescriptor::new(&layers);

    let mut sram = [0u8; 512];
    let mut ctx = model.init_context(&mut sram).expect("Valid SRAM context");

    let mut out_spikes = [0u32; 1];

    // Execution of 10,000 ticks with random extreme values (including MAX, MIN, 0)
    for _ in 0..10_000 {
        let in0 = Fix32::from_bits(rng.next_i32());
        let in1 = Fix32::from_bits(rng.next_i32());
        let inputs = [in0, in1];

        // Inference must NEVER panic
        let res = model.step(&mut ctx, &inputs, &mut out_spikes);
        assert!(res.is_ok());
    }
}

#[test]
fn test_fuzz_model_descriptor_buffer_safety() {
    static ROW_PTRS: [u16; 3] = [0, 1, 2];
    static TARGETS: [u16; 2] = [0, 1];
    static WEIGHTS: [Fix32; 2] = [Fix32::ONE, Fix32::ONE];
    let syn = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS);
    let cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
    let layer = LayerDescriptor::new(2, 2, syn, cfg, None);
    let layers = [layer];
    let model = ModelDescriptor::new(&layers);

    let required = model.memory_footprint.total_sram_bytes;

    // 1. Buffer too small
    let mut small_buf = [0u8; 8];
    assert!(matches!(
        model.init_context(&mut small_buf),
        Err(EngineError::BufferTooSmall { .. })
    ));

    // 2. Misaligned buffer
    let mut raw_buf = [0u8; 512];
    let unaligned_slice = &mut raw_buf[1..=required];
    assert!(matches!(
        model.init_context(unaligned_slice),
        Err(EngineError::MisalignedBuffer)
    ));

    // 3. Invalid input or output dimensions
    let mut valid_sram = [0u8; 512];
    let mut ctx = model
        .init_context(&mut valid_sram)
        .expect("Valid SRAM context");

    let short_inputs = [Fix32::ONE]; // Expected: 2
    let mut out_spikes = [0u32; 1];
    assert_eq!(
        model.step(&mut ctx, &short_inputs, &mut out_spikes),
        Err(EngineError::DimensionMismatch)
    );

    let valid_inputs = [Fix32::ONE, Fix32::ONE];
    let mut empty_out_spikes = [0u32; 0]; // Expected: >= 1
    assert_eq!(
        model.step(&mut ctx, &valid_inputs, &mut empty_out_spikes),
        Err(EngineError::DimensionMismatch)
    );
}

// ============================================================================
// BPTN PROPERTY TESTS AND FUZZING (PHASE 4 - BIT-PARALLEL ROBUSTNESS)
// ============================================================================

#[cfg(feature = "bptn")]
mod bptn_fuzz {
    use super::*;
    use snn_core::domain::bptn::{
        CompactBptnNeuron, TemporalFramingBuffer64, bptn_parallel_smith_step_u64,
        swar_majority_pool_8x8,
    };

    #[test]
    fn test_fuzz_bptn_temporal_translation_invariance() {
        let mut rng = FuzzRng::new(0xDEAD_BEEF_CAFE_BABE);
        let mut framer = TemporalFramingBuffer64::new();

        // Verification of strict temporal translation invariance property across 10,000 steps
        let mut prev_window = framer.window();
        for _ in 0..10_000 {
            let spike = (rng.next_u32() & 1) != 0;
            let new_window = framer.push_spike(spike);

            // Invariance theorem: new_window == (prev_window << 1) | spike
            let expected = (prev_window << 1) | (spike as u64);
            assert_eq!(
                new_window, expected,
                "Violation of linear temporal translation property!"
            );
            prev_window = new_window;
        }
    }

    #[test]
    fn test_fuzz_bptn_popcnt_no_overflow() {
        let mut rng = FuzzRng::new(0x1234_5678_9ABC_DEF0);

        // Intensive fuzzing of 50,000 random neurons and histories
        for _ in 0..50_000 {
            let k0 = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            let k1 = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            let p0 = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            let p1 = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            let th = rng.next_u32() % 150;

            let neuron = CompactBptnNeuron::new([k0, k1], [p0, p1], th);

            let h0 = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            let h1 = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);

            let potential = neuron.evaluate_potential(&[h0, h1]);
            // Property: potential <= 64 * channels (here 128)
            assert!(potential <= 128, "BPTN membrane potential overflow!");

            let fired = neuron.tick(&[h0, h1]);
            assert_eq!(fired, potential >= th, "Spike thresholding inconsistency!");
        }
    }

    #[test]
    fn test_fuzz_bptn_smith_counter_saturation_invariants() {
        let mut rng = FuzzRng::new(0x9988_7766_5544_3322);

        let mut c1 = 0u64;
        let mut c0 = 0u64;

        // Fuzzing of 100,000 simultaneous steps across 64 parallel channels
        for _ in 0..100_000 {
            let inc = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            let dec = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);

            let (next_c1, next_c0) = bptn_parallel_smith_step_u64(c1, c0, inc, dec);

            // Invariant for each of 64 channels:
            // - State (c1, c0) encodes an integer [0..3].
            // - next_c1 == 1 iff state >= 2.
            // - If state == 3 and inc (without dec) -> state remains 3 (upper saturation).
            // - If state == 0 and dec (without inc) -> state remains 0 (lower saturation).
            for bit in 0..64 {
                let prev_val = (((c1 >> bit) & 1) << 1) | ((c0 >> bit) & 1);
                let next_val = (((next_c1 >> bit) & 1) << 1) | ((next_c0 >> bit) & 1);
                let bit_inc = ((inc >> bit) & 1) != 0;
                let bit_dec = ((dec >> bit) & 1) != 0;

                if bit_inc && !bit_dec {
                    assert_eq!(next_val, (prev_val + 1).min(3));
                } else if bit_dec && !bit_inc {
                    assert_eq!(next_val, prev_val.saturating_sub(1));
                } else {
                    assert_eq!(next_val, prev_val);
                }
            }

            c1 = next_c1;
            c0 = next_c0;
        }
    }

    #[test]
    fn test_fuzz_swar_majority_pool_bounds() {
        let mut rng = FuzzRng::new(0xFEDC_BA98_7654_3210);

        for _ in 0..20_000 {
            let x = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            let th = (rng.next_u32() % 12) as u8;

            let pooled = swar_majority_pool_8x8(x, th);
            let effective_th = th.clamp(1, 8);

            // Byte-by-byte exact verification
            for i in 0..8 {
                let byte = ((x >> (i * 8)) & 0xFF) as u8;
                let ones = byte.count_ones() as u8;
                let expected_bit = (ones >= effective_th) as u8;
                let actual_bit = (pooled >> i) & 1;
                assert_eq!(
                    actual_bit, expected_bit,
                    "SWAR Majority Pool failure on byte {} (ones={}, th={})",
                    i, ones, effective_th
                );
            }
        }
    }
}
