//! 🌐 Industrial C FFI Interface for microcontrollers and RTOS (`extern "C"`, `no_std`)
//!
//! Enables direct integration of `snn_core` into industrial C / C++ projects
//! (FreeRTOS, Zephyr RTOS, STM32Cube, ESP-IDF) without rewriting firmware in Rust.

use crate::domain::aer::{AerSpike1d, AerSpike2d};
use crate::domain::numeric::Fix32;

/// Converts a C float to Fix32 Q16.16
#[unsafe(no_mangle)]
pub extern "C" fn snn_fix32_from_float(val: f32) -> i32 {
    Fix32::from_num(val).0
}

/// Converts a Fix32 Q16.16 to C float
#[unsafe(no_mangle)]
pub extern "C" fn snn_fix32_to_float(bits: i32) -> f32 {
    Fix32::from_bits(bits).to_float()
}

/// Decodes a 1D AER event stream into a 32-bit word buffer (SpikeBitset).
///
/// # Safety
/// `events` and `out_words` pointers must be valid and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_aer_decode_1d(
    events: *const AerSpike1d,
    event_count: usize,
    out_words: *mut u32,
    words_count: usize,
) -> i32 {
    if out_words.is_null() || words_count == 0 {
        return -1;
    }
    if event_count > 0 && events.is_null() {
        return -1;
    }

    let words_slice = unsafe { core::slice::from_raw_parts_mut(out_words, words_count) };

    words_slice.fill(0);

    if event_count == 0 {
        return 0;
    }

    let ev_slice = unsafe { core::slice::from_raw_parts(events, event_count) };

    for ev in ev_slice {
        let addr = ev.address as usize;
        let w_idx = addr / 32;
        let b_idx = addr % 32;
        if w_idx < words_count {
            words_slice[w_idx] |= 1 << b_idx;
        }
    }

    0
}

/// Decodes a 2D DVS AER event stream (neuromorphic vision) into a 32-bit word buffer.
///
/// # Safety
/// `events` and `out_words` pointers must be valid and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_aer_decode_2d_bipolar(
    events: *const AerSpike2d,
    event_count: usize,
    width: usize,
    height: usize,
    out_words: *mut u32,
    words_count: usize,
) -> i32 {
    if out_words.is_null() || words_count == 0 || width == 0 || height == 0 {
        return -1;
    }
    if event_count > 0 && events.is_null() {
        return -1;
    }

    let words_slice = unsafe { core::slice::from_raw_parts_mut(out_words, words_count) };

    words_slice.fill(0);

    if event_count == 0 {
        return 0;
    }

    let ev_slice = unsafe { core::slice::from_raw_parts(events, event_count) };

    for ev in ev_slice {
        let linear_idx = ev.to_linear_index_with_polarity(width, height);
        let w_idx = linear_idx / 32;
        let b_idx = linear_idx % 32;
        if w_idx < words_count {
            words_slice[w_idx] |= 1 << b_idx;
        }
    }

    0
}

/// Tests whether a bit is active in a C spike bitset mask.
///
/// # Safety
/// `words` pointer must be valid for reading `words_count` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_bitset_get_spike(
    words: *const u32,
    words_count: usize,
    neuron_index: usize,
) -> bool {
    if words.is_null() {
        return false;
    }
    let w_idx = neuron_index / 32;
    let b_idx = neuron_index % 32;
    if w_idx < words_count {
        let word = unsafe { *words.add(w_idx) };
        ((word >> b_idx) & 1) != 0
    } else {
        false
    }
}

/// Counts total number of active spikes via native hardware `POPCNT` instruction.
///
/// # Safety
/// `words` pointer must be valid for reading `words_count` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_bitset_count_spikes(words: *const u32, words_count: usize) -> u32 {
    if words.is_null() {
        return 0;
    }
    let slice = unsafe { core::slice::from_raw_parts(words, words_count) };
    slice
        .iter()
        .fold(0u32, |acc, &w| acc.saturating_add(w.count_ones()))
}

/// C configuration for a set of LIF neurons
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnnLifConfig {
    pub threshold: i32,
    pub decay_factor: i32,
    pub reset_potential: i32,
    pub refractory_period: u16,
    pub bias: i32,
    pub reset_mode: u8, // 0: Hard, 1: Subtract
}

/// CSR sparse synaptic matrix stored in immutable Flash ROM
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SnnCsrMatrix {
    pub row_ptrs: *const u16,
    pub targets: *const u16,
    pub weights: *const i32,
    pub in_features: usize,
    pub out_features: usize,
    pub total_synapses: usize,
}

/// Complete neural network descriptor in Flash ROM (.rodata)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SnnNetworkDesc {
    pub num_neurons: usize,
    pub in_features: usize,
    pub config: SnnLifConfig,
    pub input_synapses: SnnCsrMatrix,
    pub recurrent_synapses: SnnCsrMatrix,
    pub bias: *const i32,
}

/// Dynamic network state residing in SRAM (without heap allocation)
#[repr(C)]
#[derive(Debug)]
pub struct SnnNetworkState {
    pub membrane_potentials: *mut i32,
    pub input_accumulators: *mut i32,
    pub refractory_times: *mut u8,
    pub last_spikes: *mut u32,
    pub num_neurons: usize,
    pub words_count: usize,
}

/// Calculates exact size in bytes required to allocate dynamic state in SRAM
#[unsafe(no_mangle)]
pub extern "C" fn snn_network_state_size(num_neurons: usize) -> usize {
    if num_neurons == 0 {
        return 0;
    }
    let words = num_neurons.div_ceil(32);
    let mem_bytes = num_neurons * core::mem::size_of::<i32>();
    let acc_bytes = num_neurons * core::mem::size_of::<i32>();
    let refr_bytes = num_neurons * core::mem::size_of::<u8>();
    let refr_aligned = (refr_bytes + 3) & !3;
    let spikes_bytes = words * core::mem::size_of::<u32>();
    mem_bytes + acc_bytes + refr_aligned + spikes_bytes
}

/// Initializes dynamic network state in a user-provided SRAM buffer
///
/// # Safety
/// `desc`, `arena` and `out_state` pointers must be valid, and `arena` must be 4-byte aligned.
#[unsafe(no_mangle)]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe extern "C" fn snn_network_init_state(
    desc: *const SnnNetworkDesc,
    arena: *mut u8,
    arena_size: usize,
    out_state: *mut SnnNetworkState,
) -> i32 {
    if desc.is_null() || arena.is_null() || out_state.is_null() {
        return -1;
    }
    let d = unsafe { &*desc };
    let n = d.num_neurons;
    if n == 0 {
        return -2;
    }
    let required = snn_network_state_size(n);
    if arena_size < required {
        return -3;
    }
    if !(arena as usize).is_multiple_of(4) {
        return -4;
    }

    let words = n.div_ceil(32);
    let mem_ptr = arena.cast::<i32>();
    let acc_ptr = unsafe { mem_ptr.add(n) };
    let refr_ptr = unsafe { acc_ptr.add(n).cast::<u8>() };
    let refr_aligned = (n + 3) & !3;
    let spikes_ptr = (refr_ptr as usize + refr_aligned) as *mut u32;

    let reset_v = d.config.reset_potential;
    for i in 0..n {
        unsafe {
            *mem_ptr.add(i) = reset_v;
            *acc_ptr.add(i) = 0;
            *refr_ptr.add(i) = 0;
        }
    }
    for w in 0..words {
        unsafe {
            *spikes_ptr.add(w) = 0;
        }
    }

    unsafe {
        (*out_state).membrane_potentials = mem_ptr;
        (*out_state).input_accumulators = acc_ptr;
        (*out_state).refractory_times = refr_ptr;
        (*out_state).last_spikes = spikes_ptr;
        (*out_state).num_neurons = n;
        (*out_state).words_count = words;
    }

    0
}

/// Executes a complete network inference timestep (tick)
///
/// # Safety
/// Pointers must be valid for the declared sizes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_network_step(
    desc: *const SnnNetworkDesc,
    state: *mut SnnNetworkState,
    inputs: *const i32,
    inputs_len: usize,
    out_spikes_words: *mut u32,
    words_count: usize,
) -> i32 {
    if desc.is_null() || state.is_null() || out_spikes_words.is_null() {
        return -1;
    }
    let d = unsafe { &*desc };
    let s = unsafe { &mut *state };
    let n = d.num_neurons;
    let words = s.words_count;
    if words_count < words {
        return -2;
    }

    let mem_slice = unsafe { core::slice::from_raw_parts_mut(s.membrane_potentials, n) };
    let acc_slice = unsafe { core::slice::from_raw_parts_mut(s.input_accumulators, n) };
    let refr_slice = unsafe { core::slice::from_raw_parts_mut(s.refractory_times, n) };
    let last_spikes_slice = unsafe { core::slice::from_raw_parts_mut(s.last_spikes, words) };
    let out_words_slice = unsafe { core::slice::from_raw_parts_mut(out_spikes_words, words_count) };

    out_words_slice[..words.min(words_count)].fill(0);

    // 1. Input synapse propagation (weighted)
    if !inputs.is_null()
        && inputs_len > 0
        && !d.input_synapses.row_ptrs.is_null()
        && !d.input_synapses.targets.is_null()
        && !d.input_synapses.weights.is_null()
        && d.input_synapses.total_synapses > 0
    {
        let in_syn = &d.input_synapses;
        let in_slice =
            unsafe { core::slice::from_raw_parts(inputs, inputs_len.min(in_syn.in_features)) };
        let row_ptrs = unsafe { core::slice::from_raw_parts(in_syn.row_ptrs, in_slice.len() + 1) };
        let targets = unsafe { core::slice::from_raw_parts(in_syn.targets, in_syn.total_synapses) };
        let weights = unsafe { core::slice::from_raw_parts(in_syn.weights, in_syn.total_synapses) };

        for (in_idx, &in_val) in in_slice.iter().enumerate() {
            if in_val != 0 {
                let start = (row_ptrs[in_idx] as usize).min(in_syn.total_synapses);
                let end = (row_ptrs[in_idx + 1] as usize).min(in_syn.total_synapses);
                let in_fix = Fix32::from_bits(in_val);
                for syn_idx in start..end {
                    let target = targets[syn_idx] as usize;
                    let w_fix = Fix32::from_bits(weights[syn_idx]);
                    if target < n {
                        let weighted = (in_fix * w_fix).to_bits();
                        acc_slice[target] = acc_slice[target].saturating_add(weighted);
                    }
                }
            }
        }
    }

    // 2. Recurrent synapse propagation
    if !d.recurrent_synapses.row_ptrs.is_null()
        && !d.recurrent_synapses.targets.is_null()
        && !d.recurrent_synapses.weights.is_null()
        && d.recurrent_synapses.total_synapses > 0
    {
        let rec_syn = &d.recurrent_synapses;
        let rec_rows =
            unsafe { core::slice::from_raw_parts(rec_syn.row_ptrs, rec_syn.in_features + 1) };
        let rec_targets =
            unsafe { core::slice::from_raw_parts(rec_syn.targets, rec_syn.total_synapses) };
        let rec_weights =
            unsafe { core::slice::from_raw_parts(rec_syn.weights, rec_syn.total_synapses) };

        for (w_idx, &word) in last_spikes_slice.iter().enumerate() {
            let mut active = word;
            while active != 0 {
                let bit_idx = active.trailing_zeros() as usize;
                let pre_neuron = w_idx * 32 + bit_idx;
                if pre_neuron < rec_syn.in_features {
                    let start = (rec_rows[pre_neuron] as usize).min(rec_syn.total_synapses);
                    let end = (rec_rows[pre_neuron + 1] as usize).min(rec_syn.total_synapses);
                    for syn_idx in start..end {
                        let target = rec_targets[syn_idx] as usize;
                        let weight = rec_weights[syn_idx];
                        if target < n {
                            acc_slice[target] = acc_slice[target].saturating_add(weight);
                        }
                    }
                }
                active &= active - 1;
            }
        }
    }

    // 3. Dynamic neuron membrane evaluation
    let bias_slice = if d.bias.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(d.bias, n) })
    };

    #[cfg(not(feature = "cortex-m0"))]
    let cfg_decay = Fix32::from_bits(d.config.decay_factor);
    let cfg_thresh = Fix32::from_bits(d.config.threshold);
    let cfg_reset = Fix32::from_bits(d.config.reset_potential);
    let cfg_bias = Fix32::from_bits(d.config.bias);
    let is_soft_reset = d.config.reset_mode == 1;

    for i in 0..n {
        let refr = &mut refr_slice[i];
        let static_b = bias_slice.map_or(Fix32::ZERO, |b| Fix32::from_bits(b[i]));
        let total_bias = static_b + cfg_bias;
        let inp = Fix32::from_bits(acc_slice[i]) + total_bias;
        acc_slice[i] = 0;

        if *refr > 0 {
            *refr -= 1;
            mem_slice[i] = cfg_reset.to_bits();
            continue;
        }

        let current_v = Fix32::from_bits(mem_slice[i]);
        #[cfg(feature = "cortex-m0")]
        let decayed = crate::domain::hardware::fast_decay_m0(
            current_v,
            (d.config.decay_factor as u32 & 0xFFFF) as u16,
        );
        #[cfg(not(feature = "cortex-m0"))]
        let decayed = current_v * cfg_decay;

        let v = decayed + inp;

        if v >= cfg_thresh {
            let next_v = if is_soft_reset {
                v - cfg_thresh
            } else {
                cfg_reset
            };
            mem_slice[i] = next_v.to_bits();
            *refr = d.config.refractory_period as u8;

            let w_idx = i / 32;
            let b_idx = i % 32;
            if w_idx < words {
                out_words_slice[w_idx] |= 1 << b_idx;
                last_spikes_slice[w_idx] |= 1 << b_idx;
            }
        } else {
            mem_slice[i] = v.to_bits();
            let w_idx = i / 32;
            let b_idx = i % 32;
            if w_idx < words {
                last_spikes_slice[w_idx] &= !(1 << b_idx);
            }
        }
    }

    0
}

/// Resets all membrane potentials and refractory periods to rest
///
/// # Safety
/// `desc` and `state` pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_network_reset(
    desc: *const SnnNetworkDesc,
    state: *mut SnnNetworkState,
) -> i32 {
    if desc.is_null() || state.is_null() {
        return -1;
    }
    let d = unsafe { &*desc };
    let s = unsafe { &mut *state };
    let n = d.num_neurons;
    let words = s.words_count;
    let reset_v = d.config.reset_potential;

    let mem_slice = unsafe { core::slice::from_raw_parts_mut(s.membrane_potentials, n) };
    let acc_slice = unsafe { core::slice::from_raw_parts_mut(s.input_accumulators, n) };
    let refr_slice = unsafe { core::slice::from_raw_parts_mut(s.refractory_times, n) };
    let last_spikes_slice = unsafe { core::slice::from_raw_parts_mut(s.last_spikes, words) };

    mem_slice.fill(reset_v);
    acc_slice.fill(0);
    refr_slice.fill(0);
    last_spikes_slice.fill(0);
    0
}

/// Computes logits of a dense Readout layer from a spike mask
///
/// # Safety
/// Pointers must be valid for the required sizes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_readout_dense(
    weights_q16: *const i32,
    bias_q16: *const i32,
    spike_words: *const u32,
    num_neurons: usize,
    num_classes: usize,
    out_logits_q16: *mut i32,
) -> i32 {
    if weights_q16.is_null() || spike_words.is_null() || out_logits_q16.is_null() {
        return -1;
    }
    if num_neurons == 0 || num_classes == 0 {
        return -2;
    }
    let w_slice = unsafe { core::slice::from_raw_parts(weights_q16, num_classes * num_neurons) };
    let b_slice = if bias_q16.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(bias_q16, num_classes) })
    };
    let words_count = num_neurons.div_ceil(32);
    let spk_slice = unsafe { core::slice::from_raw_parts(spike_words, words_count) };
    let out_slice = unsafe { core::slice::from_raw_parts_mut(out_logits_q16, num_classes) };

    for c in 0..num_classes {
        let mut acc = b_slice.map_or(0i32, |b| b[c]);
        let class_offset = c * num_neurons;
        for (w_idx, &word) in spk_slice.iter().enumerate() {
            let mut active = word;
            while active != 0 {
                let bit_idx = active.trailing_zeros() as usize;
                let neuron_idx = w_idx * 32 + bit_idx;
                if neuron_idx < num_neurons {
                    let weight = w_slice[class_offset + neuron_idx];
                    acc = acc.saturating_add(weight);
                }
                active &= active - 1;
            }
        }
        out_slice[c] = acc;
    }
    0
}

/// Layer type in a deep multilayer C network
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnnLayerType {
    /// Standard feedforward linear layer with Fix32 synapses
    Feedforward = 0,
    /// INT8 quantized layer with shared Fix32 scale factor
    QuantizedInt8 = 1,
}

/// C descriptor for a layer in a deep network
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SnnLayerDesc {
    pub layer_type: SnnLayerType,
    pub in_features: usize,
    pub num_neurons: usize,
    pub config: SnnLifConfig,
    pub synapses: SnnCsrMatrix,
    pub bias: *const i32,
    pub quant_scale: i32,
}

/// Full C descriptor of a multilayer network operating on ActivationArena
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SnnDeepNetworkDesc {
    pub layers: *const SnnLayerDesc,
    pub num_layers: usize,
    pub in_features: usize,
}

/// Dynamic state of a deep network allocated in SRAM (zero heap)
#[repr(C)]
#[derive(Debug)]
pub struct SnnDeepNetworkState {
    pub membrane_potentials: *mut i32,
    pub refractory_times: *mut u8,
    pub arena_accumulators: *mut i32,
    pub ping_words: *mut u32,
    pub pong_words: *mut u32,
    pub total_neurons: usize,
    pub max_neurons: usize,
    pub max_words: usize,
}

/// Calculates exact size in bytes required for deep network state
///
/// # Safety
/// `desc` pointer must be valid for reading or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_deep_network_state_size(desc: *const SnnDeepNetworkDesc) -> usize {
    if desc.is_null() {
        return 0;
    }
    let d = unsafe { &*desc };
    if d.num_layers == 0 || d.layers.is_null() {
        return 0;
    }
    let layers = unsafe { core::slice::from_raw_parts(d.layers, d.num_layers) };
    let mut total_neurons = 0;
    let mut max_neurons = 0;
    let mut max_words = 0;

    for l in layers {
        total_neurons += l.num_neurons;
        max_neurons = max_neurons.max(l.num_neurons);
        let w = l.num_neurons.div_ceil(32);
        max_words = max_words.max(w);
    }

    if total_neurons == 0 {
        return 0;
    }

    let mem_bytes = total_neurons * core::mem::size_of::<i32>();
    let refr_bytes = (total_neurons * core::mem::size_of::<u8>() + 3) & !3;
    let acc_bytes = max_neurons * core::mem::size_of::<i32>();
    let ping_bytes = max_words * core::mem::size_of::<u32>();
    let pong_bytes = max_words * core::mem::size_of::<u32>();

    mem_bytes + refr_bytes + acc_bytes + ping_bytes + pong_bytes
}

/// Initializes deep network state in user-provided SRAM arena
///
/// # Safety
/// `desc`, `arena` and `out_state` pointers must be valid, and `arena` must be 4-byte aligned.
#[unsafe(no_mangle)]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe extern "C" fn snn_deep_network_init_state(
    desc: *const SnnDeepNetworkDesc,
    arena: *mut u8,
    arena_size: usize,
    out_state: *mut SnnDeepNetworkState,
) -> i32 {
    if desc.is_null() || arena.is_null() || out_state.is_null() {
        return -1;
    }
    if !(arena as usize).is_multiple_of(4) {
        return -4;
    }
    let d = unsafe { &*desc };
    if d.num_layers == 0 || d.layers.is_null() {
        return -2;
    }
    let required = unsafe { snn_deep_network_state_size(desc) };
    if arena_size < required || required == 0 {
        return -3;
    }

    let layers = unsafe { core::slice::from_raw_parts(d.layers, d.num_layers) };
    let mut total_neurons = 0;
    let mut max_neurons = 0;
    let mut max_words = 0;
    for l in layers {
        total_neurons += l.num_neurons;
        max_neurons = max_neurons.max(l.num_neurons);
        let w = l.num_neurons.div_ceil(32);
        max_words = max_words.max(w);
    }

    let mem_ptr = arena.cast::<i32>();
    let refr_ptr = unsafe { mem_ptr.add(total_neurons).cast::<u8>() };
    let refr_aligned = (total_neurons + 3) & !3;
    let acc_ptr = (refr_ptr as usize + refr_aligned) as *mut i32;
    let ping_ptr = unsafe { acc_ptr.add(max_neurons).cast::<u32>() };
    let pong_ptr = unsafe { ping_ptr.add(max_words) };

    // Initialize membrane potentials per layer
    let mut offset = 0;
    for l in layers {
        let reset_v = l.config.reset_potential;
        for i in 0..l.num_neurons {
            unsafe {
                *mem_ptr.add(offset + i) = reset_v;
                *refr_ptr.add(offset + i) = 0;
            }
        }
        offset += l.num_neurons;
    }

    for i in 0..max_neurons {
        unsafe {
            *acc_ptr.add(i) = 0;
        }
    }
    for w in 0..max_words {
        unsafe {
            *ping_ptr.add(w) = 0;
            *pong_ptr.add(w) = 0;
        }
    }

    unsafe {
        (*out_state).membrane_potentials = mem_ptr;
        (*out_state).refractory_times = refr_ptr;
        (*out_state).arena_accumulators = acc_ptr;
        (*out_state).ping_words = ping_ptr;
        (*out_state).pong_words = pong_ptr;
        (*out_state).total_neurons = total_neurons;
        (*out_state).max_neurons = max_neurons;
        (*out_state).max_words = max_words;
    }

    0
}

/// Executes a timestep (tick) on a full deep network with shared memory arena
///
/// # Safety
/// `desc`, `state`, `out_spikes_words` pointers must be valid, and `inputs` must point to at least `inputs_len` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_deep_network_step(
    desc: *const SnnDeepNetworkDesc,
    state: *mut SnnDeepNetworkState,
    inputs: *const i32,
    inputs_len: usize,
    out_spikes_words: *mut u32,
    words_count: usize,
) -> i32 {
    if desc.is_null() || state.is_null() || out_spikes_words.is_null() {
        return -1;
    }
    let d = unsafe { &*desc };
    let s = unsafe { &mut *state };
    if d.num_layers == 0 || d.layers.is_null() {
        return -2;
    }

    let layers = unsafe { core::slice::from_raw_parts(d.layers, d.num_layers) };
    let last_layer = &layers[d.num_layers - 1];
    let last_words = last_layer.num_neurons.div_ceil(32);
    if words_count < last_words {
        return -3;
    }

    let mem_slice =
        unsafe { core::slice::from_raw_parts_mut(s.membrane_potentials, s.total_neurons) };
    let refr_slice =
        unsafe { core::slice::from_raw_parts_mut(s.refractory_times, s.total_neurons) };
    let acc_slice = unsafe { core::slice::from_raw_parts_mut(s.arena_accumulators, s.max_neurons) };
    let ping_slice = unsafe { core::slice::from_raw_parts_mut(s.ping_words, s.max_words) };
    let pong_slice = unsafe { core::slice::from_raw_parts_mut(s.pong_words, s.max_words) };

    let mut neuron_offset = 0;

    for (layer_idx, layer) in layers.iter().enumerate() {
        let n = layer.num_neurons;
        let words = n.div_ceil(32);
        let is_even = (layer_idx % 2) == 0;

        // 1. Reset shared accumulators for this layer
        acc_slice[..n].fill(0);

        // Reset output buffer
        if is_even {
            pong_slice[..words].fill(0);
        } else {
            ping_slice[..words].fill(0);
        }

        // 2. Synaptic propagation
        let syn = &layer.synapses;
        if layer_idx == 0 {
            // Layer 0: receives continuous weighted inputs
            if !inputs.is_null()
                && inputs_len > 0
                && !syn.row_ptrs.is_null()
                && !syn.targets.is_null()
                && !syn.weights.is_null()
                && syn.total_synapses > 0
            {
                let in_slice =
                    unsafe { core::slice::from_raw_parts(inputs, inputs_len.min(syn.in_features)) };
                let row_ptrs =
                    unsafe { core::slice::from_raw_parts(syn.row_ptrs, in_slice.len() + 1) };
                let targets =
                    unsafe { core::slice::from_raw_parts(syn.targets, syn.total_synapses) };
                let weights =
                    unsafe { core::slice::from_raw_parts(syn.weights, syn.total_synapses) };

                for (in_idx, &in_val) in in_slice.iter().enumerate() {
                    if in_val != 0 {
                        let start = (row_ptrs[in_idx] as usize).min(syn.total_synapses);
                        let end = (row_ptrs[in_idx + 1] as usize).min(syn.total_synapses);
                        let in_fix = Fix32::from_bits(in_val);
                        for syn_idx in start..end {
                            let target = targets[syn_idx] as usize;
                            if target < n {
                                let weighted = if layer.layer_type == SnnLayerType::QuantizedInt8 {
                                    let raw_i8 = weights[syn_idx] as i8;
                                    let scale = Fix32::from_bits(layer.quant_scale);
                                    let syn_w = scale * Fix32::from_num(raw_i8 as f32);
                                    (in_fix * syn_w).to_bits()
                                } else {
                                    let w_fix = Fix32::from_bits(weights[syn_idx]);
                                    (in_fix * w_fix).to_bits()
                                };
                                acc_slice[target] = acc_slice[target].saturating_add(weighted);
                            }
                        }
                    }
                }
            }
        } else {
            // Subsequent layers: receive spikes from previous layer
            let prev_layer = &layers[layer_idx - 1];
            let prev_words = prev_layer.num_neurons.div_ceil(32);
            let in_spikes = if is_even {
                &ping_slice[..prev_words]
            } else {
                &pong_slice[..prev_words]
            };

            if !syn.row_ptrs.is_null()
                && !syn.targets.is_null()
                && !syn.weights.is_null()
                && syn.total_synapses > 0
            {
                let row_ptrs = unsafe {
                    core::slice::from_raw_parts(syn.row_ptrs, prev_layer.num_neurons + 1)
                };
                let targets =
                    unsafe { core::slice::from_raw_parts(syn.targets, syn.total_synapses) };
                let weights =
                    unsafe { core::slice::from_raw_parts(syn.weights, syn.total_synapses) };

                for (w_idx, &word) in in_spikes.iter().enumerate() {
                    let mut active = word;
                    while active != 0 {
                        let bit_idx = active.trailing_zeros() as usize;
                        let pre_neuron = w_idx * 32 + bit_idx;
                        if pre_neuron < prev_layer.num_neurons {
                            let start = (row_ptrs[pre_neuron] as usize).min(syn.total_synapses);
                            let end = (row_ptrs[pre_neuron + 1] as usize).min(syn.total_synapses);
                            for syn_idx in start..end {
                                let target = targets[syn_idx] as usize;
                                if target < n {
                                    let weight = if layer.layer_type == SnnLayerType::QuantizedInt8
                                    {
                                        let raw_i8 = weights[syn_idx] as i8;
                                        let scale = Fix32::from_bits(layer.quant_scale);
                                        (scale * Fix32::from_num(raw_i8 as f32)).to_bits()
                                    } else {
                                        weights[syn_idx]
                                    };
                                    acc_slice[target] = acc_slice[target].saturating_add(weight);
                                }
                            }
                        }
                        active &= active - 1;
                    }
                }
            }
        }

        // 3. Dynamic membrane evaluation of neurons in this layer
        let bias_slice = if layer.bias.is_null() {
            None
        } else {
            Some(unsafe { core::slice::from_raw_parts(layer.bias, n) })
        };

        let cfg_decay = Fix32::from_bits(layer.config.decay_factor);
        let cfg_thresh = Fix32::from_bits(layer.config.threshold);
        let cfg_reset = Fix32::from_bits(layer.config.reset_potential);
        let cfg_bias = Fix32::from_bits(layer.config.bias);
        let is_soft_reset = layer.config.reset_mode == 1;

        for i in 0..n {
            let global_idx = neuron_offset + i;
            let refr = &mut refr_slice[global_idx];
            let v_mem = &mut mem_slice[global_idx];
            let syn_input = Fix32::from_bits(acc_slice[i]);
            let b_val = bias_slice.map_or(cfg_bias, |b| Fix32::from_bits(b[i]) + cfg_bias);
            let total_in = syn_input + b_val;

            if *refr > 0 {
                *refr -= 1;
                *v_mem = cfg_reset.to_bits();
            } else {
                let v_cur = Fix32::from_bits(*v_mem);
                let decayed = v_cur * cfg_decay;
                let v_next = decayed + total_in;

                if v_next >= cfg_thresh {
                    let w_idx = i / 32;
                    let b_idx = i % 32;
                    if is_even {
                        pong_slice[w_idx] |= 1 << b_idx;
                    } else {
                        ping_slice[w_idx] |= 1 << b_idx;
                    }
                    *v_mem = if is_soft_reset {
                        (v_next - cfg_thresh).to_bits()
                    } else {
                        cfg_reset.to_bits()
                    };
                    *refr = layer.config.refractory_period as u8;
                } else {
                    *v_mem = v_next.to_bits();
                }
            }
        }

        neuron_offset += n;
    }

    // Copy final layer output into out_spikes_words
    let last_is_even = ((d.num_layers - 1) % 2) == 0;
    let final_words_slice = if last_is_even {
        unsafe { core::slice::from_raw_parts(s.pong_words, last_words) }
    } else {
        unsafe { core::slice::from_raw_parts(s.ping_words, last_words) }
    };

    let out_slice = unsafe { core::slice::from_raw_parts_mut(out_spikes_words, words_count) };
    out_slice[..last_words].copy_from_slice(&final_words_slice[..last_words]);
    out_slice[last_words..words_count].fill(0);

    0
}

/// Resets dynamic state of a deep network
///
/// # Safety
/// `desc` and `state` pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_deep_network_reset(
    desc: *const SnnDeepNetworkDesc,
    state: *mut SnnDeepNetworkState,
) -> i32 {
    if desc.is_null() || state.is_null() {
        return -1;
    }
    let d = unsafe { &*desc };
    let s = unsafe { &mut *state };
    if d.num_layers == 0 || d.layers.is_null() {
        return -2;
    }

    let layers = unsafe { core::slice::from_raw_parts(d.layers, d.num_layers) };
    let mem_slice =
        unsafe { core::slice::from_raw_parts_mut(s.membrane_potentials, s.total_neurons) };
    let refr_slice =
        unsafe { core::slice::from_raw_parts_mut(s.refractory_times, s.total_neurons) };
    let acc_slice = unsafe { core::slice::from_raw_parts_mut(s.arena_accumulators, s.max_neurons) };
    let ping_slice = unsafe { core::slice::from_raw_parts_mut(s.ping_words, s.max_words) };
    let pong_slice = unsafe { core::slice::from_raw_parts_mut(s.pong_words, s.max_words) };

    let mut offset = 0;
    for l in layers {
        let reset_v = l.config.reset_potential;
        mem_slice[offset..offset + l.num_neurons].fill(reset_v);
        refr_slice[offset..offset + l.num_neurons].fill(0);
        offset += l.num_neurons;
    }

    acc_slice.fill(0);
    ping_slice.fill(0);
    pong_slice.fill(0);

    0
}

/// Calculates required size (in bytes) for SRAM arena of an SnnModelDescriptor.
///
/// # Safety
/// `desc` pointer must be valid and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_model_memory_footprint(desc: *const SnnDeepNetworkDesc) -> usize {
    unsafe { snn_deep_network_state_size(desc) }
}

/// Initializes execution context of an SnnModelDescriptor in an SRAM buffer.
///
/// # Safety
/// `desc`, `arena` and `out_context` pointers must be valid and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_model_init_context(
    desc: *const SnnDeepNetworkDesc,
    arena: *mut u8,
    arena_size: usize,
    out_context: *mut SnnDeepNetworkState,
) -> i32 {
    unsafe { snn_deep_network_init_state(desc, arena, arena_size, out_context) }
}

/// Executes a complete inference timestep (tick) with an SnnModelDescriptor.
///
/// # Safety
/// `desc`, `context`, `inputs` and `out_spikes_words` pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_model_step(
    desc: *const SnnDeepNetworkDesc,
    context: *mut SnnDeepNetworkState,
    inputs: *const i32,
    inputs_len: usize,
    out_spikes_words: *mut u32,
    words_count: usize,
) -> i32 {
    unsafe {
        snn_deep_network_step(
            desc,
            context,
            inputs,
            inputs_len,
            out_spikes_words,
            words_count,
        )
    }
}

/// Resets dynamic state of an SnnModelDescriptor.
///
/// # Safety
/// `desc` and `context` pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_model_reset(
    desc: *const SnnDeepNetworkDesc,
    context: *mut SnnDeepNetworkState,
) -> i32 {
    unsafe { snn_deep_network_reset(desc, context) }
}

/// Propagates active spikes across a spatial 2D convolution with HWC layout directly in C
///
/// # Safety
/// `kernel_weights`, `active_spikes` and `accumulators` pointers must be valid and non-null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn snn_conv2d_propagate_hwc(
    kernel_weights: *const i32,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    in_h: usize,
    in_w: usize,
    active_spikes: *const u32,
    active_spikes_count: usize,
    accumulators: *mut i32,
    accumulators_len: usize,
) -> i32 {
    use crate::domain::layers::{Conv2dSynapses, ConvDataLayout};
    use crate::traits::SynapseMatrix;

    if accumulators.is_null() || accumulators_len == 0 || kernel_weights.is_null() {
        return -1;
    }
    if active_spikes_count > 0 && active_spikes.is_null() {
        return -1;
    }

    let k_size = in_channels * out_channels * kernel_h * kernel_w;
    if k_size == 0 || active_spikes_count == 0 {
        return 0;
    }

    let kernel_slice =
        unsafe { core::slice::from_raw_parts(kernel_weights.cast::<Fix32>(), k_size) };
    let active_slice = unsafe { core::slice::from_raw_parts(active_spikes, active_spikes_count) };
    let acc_slice =
        unsafe { core::slice::from_raw_parts_mut(accumulators.cast::<Fix32>(), accumulators_len) };

    let conv = Conv2dSynapses::with_padding(
        kernel_slice,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        in_h,
        in_w,
    )
    .with_layout(ConvDataLayout::Hwc);

    conv.propagate_active_spikes(active_slice, acc_slice);

    0
}

// ============================================================================
// BPTN (Bit-Parallel Temporal Neural Network) C FFI API
// ============================================================================

/// Pushes a binary spike into a 64-bit temporal history buffer and returns the new window.
///
/// # Safety
/// `buffer` pointer must point to a valid and aligned `u64` variable.
#[cfg(feature = "bptn")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_bptn_framing_push(buffer: *mut u64, spike: bool) -> u64 {
    if buffer.is_null() {
        return 0;
    }
    let current = unsafe { *buffer };
    let next = (current << 1) | (spike as u64);
    unsafe { *buffer = next };
    next
}

/// Evaluates a timestep of an R-BPTN synchronous bistable latch (0: ResetDominant, 1: SetDominant).
///
/// # Safety
/// `latch_state` pointer must point to a valid and aligned `u64` variable.
#[cfg(feature = "bptn")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_bptn_latch_step(
    latch_state: *mut u64,
    mode: i32,
    set: bool,
    reset: bool,
) -> bool {
    if latch_state.is_null() {
        return false;
    }
    let mut cell = crate::domain::bptn::BptnBistableCell {
        state_history: unsafe { *latch_state },
        mode: if mode == 1 {
            crate::domain::bptn::BptnLatchMode::SetDominant
        } else {
            crate::domain::bptn::BptnLatchMode::ResetDominant
        },
    };
    let res = cell.step(set, reset);
    unsafe { *latch_state = cell.state_history };
    res
}

/// Evaluates a timestep of a T-Latch synchronous parity toggle latch.
///
/// # Safety
/// `toggle_state` pointer must point to a valid and aligned `u64` variable.
#[cfg(feature = "bptn")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_bptn_toggle_step(toggle_state: *mut u64, toggle: bool) -> bool {
    if toggle_state.is_null() {
        return false;
    }
    let mut cell = crate::domain::bptn::BptnToggleCell {
        state_history: unsafe { *toggle_state },
    };
    let res = cell.step(toggle);
    unsafe { *toggle_state = cell.state_history };
    res
}

/// Evaluates a vectorized Word-RAM BPTN neuron across M input channels.
///
/// # Safety
/// `histories`, `kernels` and `polarities` pointers must point to slices of length `channels`.
#[cfg(feature = "bptn")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_bptn_neuron_eval(
    histories: *const u64,
    kernels: *const u64,
    polarities: *const u64,
    channels: usize,
    threshold: u32,
) -> bool {
    if histories.is_null() || kernels.is_null() || polarities.is_null() || channels == 0 {
        return false;
    }
    let hist_slice = unsafe { core::slice::from_raw_parts(histories, channels) };
    let kern_slice = unsafe { core::slice::from_raw_parts(kernels, channels) };
    let pol_slice = unsafe { core::slice::from_raw_parts(polarities, channels) };

    let mut potential: u32 = 0;
    for j in 0..channels {
        let active = (hist_slice[j] ^ pol_slice[j]) & kern_slice[j];
        potential += active.count_ones();
    }
    potential >= threshold
}

/// Evaluates a layer of N BPTN neurons receiving M channels of 64-bit history.
///
/// # Safety
/// `histories`, `kernels` (size N * channels), `polarities` (size N * channels)
/// and `thresholds` (size N) pointers must be valid.
#[cfg(feature = "bptn")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_bptn_layer_eval(
    histories: *const u64,
    channels: usize,
    kernels: *const u64,
    polarities: *const u64,
    thresholds: *const u32,
    num_neurons: usize,
    out_mask: *mut u64,
) -> i32 {
    if histories.is_null()
        || kernels.is_null()
        || polarities.is_null()
        || thresholds.is_null()
        || out_mask.is_null()
        || channels == 0
        || num_neurons == 0
        || num_neurons > 64
    {
        return -1;
    }
    let hist_slice = unsafe { core::slice::from_raw_parts(histories, channels) };
    let kern_slice = unsafe { core::slice::from_raw_parts(kernels, num_neurons * channels) };
    let pol_slice = unsafe { core::slice::from_raw_parts(polarities, num_neurons * channels) };
    let thresh_slice = unsafe { core::slice::from_raw_parts(thresholds, num_neurons) };

    let mut mask: u64 = 0;
    for (i, &thresh) in thresh_slice.iter().enumerate().take(num_neurons) {
        let offset = i * channels;
        let mut potential: u32 = 0;
        for j in 0..channels {
            let active = (hist_slice[j] ^ pol_slice[offset + j]) & kern_slice[offset + j];
            potential += active.count_ones();
        }
        if potential >= thresh {
            mask |= 1u64 << i;
        }
    }

    unsafe {
        *out_mask = mask;
    }
    0
}

/// Decodes predicted class from a `u64` spike mask (BPTN)
/// without multiplications, using pure `CTZ` (`trailing_zeros`) + saturating additions.
///
/// # Safety
/// `weights_q16` pointer must point to at least `num_classes * num_features` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn snn_readout_predict_spikes_mask(
    weights_q16: *const i32,
    bias_q16: *const i32,
    spikes_mask: u64,
    num_features: usize,
    num_classes: usize,
) -> i32 {
    if weights_q16.is_null() || num_classes == 0 || num_features == 0 {
        return -1;
    }
    let w_slice = unsafe { core::slice::from_raw_parts(weights_q16, num_classes * num_features) };
    let b_slice = if bias_q16.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(bias_q16, num_classes) })
    };

    let mut best_class: usize = 0;
    let mut max_score: i32 = i32::MIN;

    for c in 0..num_classes {
        let mut score = b_slice.map_or(0i32, |b| b[c]);
        let class_offset = c * num_features;
        let mut remaining = spikes_mask;

        while remaining != 0 {
            let feat_idx = remaining.trailing_zeros() as usize;
            if feat_idx < num_features {
                score = score.saturating_add(w_slice[class_offset + feat_idx]);
            }
            remaining &= remaining - 1;
        }

        if score > max_score {
            max_score = score;
            best_class = c;
        }
    }

    best_class as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_ffi_fix32() {
        let bits = snn_fix32_from_float(1.5);
        assert_eq!(bits, 98304);
        let flt = snn_fix32_to_float(bits);
        assert!((flt - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_c_ffi_aer_decoding_and_bitset() {
        let events = [AerSpike1d::new(5, 100), AerSpike1d::new(35, 102)];
        let mut words = [0u32; 2];

        unsafe {
            let res = snn_aer_decode_1d(
                events.as_ptr(),
                events.len(),
                words.as_mut_ptr(),
                words.len(),
            );
            assert_eq!(res, 0);

            assert!(snn_bitset_get_spike(words.as_ptr(), words.len(), 5));
            assert!(snn_bitset_get_spike(words.as_ptr(), words.len(), 35));
            assert!(!snn_bitset_get_spike(words.as_ptr(), words.len(), 0));
            assert_eq!(snn_bitset_count_spikes(words.as_ptr(), words.len()), 2);
        }
    }

    #[test]
    fn test_c_ffi_network_full_lifecycle() {
        // Definition of a 2-neuron C network with 1 input
        static IN_ROW: [u16; 2] = [0, 1];
        static IN_TGT: [u16; 1] = [0];
        static IN_W: [i32; 1] = [98304]; // 1.5 in Q16

        let in_syn = SnnCsrMatrix {
            row_ptrs: IN_ROW.as_ptr(),
            targets: IN_TGT.as_ptr(),
            weights: IN_W.as_ptr(),
            in_features: 1,
            out_features: 2,
            total_synapses: 1,
        };

        let null_syn = SnnCsrMatrix {
            row_ptrs: core::ptr::null(),
            targets: core::ptr::null(),
            weights: core::ptr::null(),
            in_features: 0,
            out_features: 0,
            total_synapses: 0,
        };

        let cfg = SnnLifConfig {
            threshold: 65536,    // 1.0
            decay_factor: 58982, // 0.9
            reset_potential: 0,
            refractory_period: 1,
            bias: 0,
            reset_mode: 0,
        };

        let desc = SnnNetworkDesc {
            num_neurons: 2,
            in_features: 1,
            config: cfg,
            input_synapses: in_syn,
            recurrent_synapses: null_syn,
            bias: core::ptr::null(),
        };

        let arena_size = snn_network_state_size(2);
        assert!(arena_size > 0);
        let mut arena = [0u8; 128];
        let mut state = core::mem::MaybeUninit::<SnnNetworkState>::uninit();

        unsafe {
            let init_res = snn_network_init_state(
                &raw const desc,
                arena.as_mut_ptr(),
                arena.len(),
                state.as_mut_ptr(),
            );
            assert_eq!(init_res, 0);
            let mut state = state.assume_init();

            let input_val = [65536i32]; // 1.0
            let mut out_spikes = [0u32; 1];

            // Tick 1 : input 1.0 * weight 1.5 = 1.5 > threshold 1.0 -> neuron 0 fires!
            let step_res = snn_network_step(
                &raw const desc,
                &raw mut state,
                input_val.as_ptr(),
                1,
                out_spikes.as_mut_ptr(),
                1,
            );
            assert_eq!(step_res, 0);
            assert_eq!(out_spikes[0] & 1, 1); // neuron 0 fired
            assert_eq!(out_spikes[0] & 2, 0); // neuron 1 quiet

            // Dense readout test: 2 classes, 2 neurons
            // Class 0: weights [100, 0] -> score 100
            // Class 1: weights [0, 500] -> score 0
            let readout_w = [100i32, 0, 0, 500];
            let readout_b = [10i32, 20];
            let mut logits = [0i32; 2];
            let r_res = snn_readout_dense(
                readout_w.as_ptr(),
                readout_b.as_ptr(),
                out_spikes.as_ptr(),
                2,
                2,
                logits.as_mut_ptr(),
            );
            assert_eq!(r_res, 0);
            assert_eq!(logits[0], 110);
            assert_eq!(logits[1], 20);

            // Reset test
            let rst_res = snn_network_reset(&raw const desc, &raw mut state);
            assert_eq!(rst_res, 0);
        }
    }

    #[test]
    fn test_c_ffi_deep_network_arena_lifecycle() {
        // Couche 0 : 2 neurones, 1 input
        static L0_ROW: [u16; 2] = [0, 1];
        static L0_TGT: [u16; 1] = [0];
        static L0_W: [i32; 1] = [98304]; // 1.5 en Q16

        let l0_syn = SnnCsrMatrix {
            row_ptrs: L0_ROW.as_ptr(),
            targets: L0_TGT.as_ptr(),
            weights: L0_W.as_ptr(),
            in_features: 1,
            out_features: 2,
            total_synapses: 1,
        };

        let cfg = SnnLifConfig {
            threshold: 65536,    // 1.0
            decay_factor: 58982, // 0.9
            reset_potential: 0,
            refractory_period: 1,
            bias: 0,
            reset_mode: 0,
        };

        let l0 = SnnLayerDesc {
            layer_type: SnnLayerType::Feedforward,
            in_features: 1,
            num_neurons: 2,
            config: cfg,
            synapses: l0_syn,
            bias: core::ptr::null(),
            quant_scale: 0,
        };

        // Layer 1: 2 neurons, connects neuron 0 of L0 to neuron 1 of L1 with weight 2.0
        static L1_ROW: [u16; 3] = [0, 1, 1];
        static L1_TGT: [u16; 1] = [1];
        static L1_W: [i32; 1] = [131072]; // 2.0 en Q16

        let l1_syn = SnnCsrMatrix {
            row_ptrs: L1_ROW.as_ptr(),
            targets: L1_TGT.as_ptr(),
            weights: L1_W.as_ptr(),
            in_features: 2,
            out_features: 2,
            total_synapses: 1,
        };

        let l1 = SnnLayerDesc {
            layer_type: SnnLayerType::Feedforward,
            in_features: 2,
            num_neurons: 2,
            config: cfg,
            synapses: l1_syn,
            bias: core::ptr::null(),
            quant_scale: 0,
        };

        let layers = [l0, l1];
        let deep_desc = SnnDeepNetworkDesc {
            layers: layers.as_ptr(),
            num_layers: 2,
            in_features: 1,
        };

        unsafe {
            let req_size = snn_deep_network_state_size(&raw const deep_desc);
            assert!(req_size > 0);

            let mut arena = [0u8; 256];
            let mut state = core::mem::MaybeUninit::<SnnDeepNetworkState>::uninit();
            let init_res = snn_deep_network_init_state(
                &raw const deep_desc,
                arena.as_mut_ptr(),
                arena.len(),
                state.as_mut_ptr(),
            );
            assert_eq!(init_res, 0);
            let mut state = state.assume_init();

            // Timestep with input = 1.0
            // L0 neuron 0 receives 1.5 -> spike!
            // L1 neuron 1 receives spike from L0(0) * 2.0 = 2.0 -> spike!
            let input_val = [65536i32];
            let mut out_spikes = [0u32; 1];
            let step_res = snn_deep_network_step(
                &raw const deep_desc,
                &raw mut state,
                input_val.as_ptr(),
                1,
                out_spikes.as_mut_ptr(),
                1,
            );
            assert_eq!(step_res, 0);
            // Verify that L1 neuron 1 fired a spike (bit 1 active)
            assert_eq!(out_spikes[0] & 2, 2);

            // Reset
            let rst_res = snn_deep_network_reset(&raw const deep_desc, &raw mut state);
            assert_eq!(rst_res, 0);

            // Test industrial snn_model_* aliases
            let model_size = snn_model_memory_footprint(&raw const deep_desc);
            assert_eq!(model_size, req_size);
            let mut model_state = core::mem::MaybeUninit::<SnnDeepNetworkState>::uninit();
            assert_eq!(
                snn_model_init_context(
                    &raw const deep_desc,
                    arena.as_mut_ptr(),
                    arena.len(),
                    model_state.as_mut_ptr()
                ),
                0
            );
            let mut model_state = model_state.assume_init();
            let mut model_out = [0u32; 1];
            assert_eq!(
                snn_model_step(
                    &raw const deep_desc,
                    &raw mut model_state,
                    input_val.as_ptr(),
                    1,
                    model_out.as_mut_ptr(),
                    1
                ),
                0
            );
            assert_eq!(model_out[0] & 2, 2);
            assert_eq!(
                snn_model_reset(&raw const deep_desc, &raw mut model_state),
                0
            );
        }
    }

    #[test]
    fn test_c_ffi_conv2d_hwc() {
        // Kernel 1x1, 1 in_channel -> 2 out_channels
        let weights = [65536i32, 131072i32]; // [1.0, 2.0]
        let active_spikes = [0u32]; // spike at (x=0, y=0, ch=0)
        let mut accumulators = [0i32; 2];

        unsafe {
            let res = snn_conv2d_propagate_hwc(
                weights.as_ptr(),
                1, // in_channels
                2, // out_channels
                1,
                1, // kernel_h, kernel_w
                1,
                1, // stride_h, stride_w
                0,
                0, // padding_h, padding_w
                1,
                1, // in_h, in_w
                active_spikes.as_ptr(),
                1,
                accumulators.as_mut_ptr(),
                2,
            );
            assert_eq!(res, 0);
            assert_eq!(accumulators[0], 65536);
            assert_eq!(accumulators[1], 131072);
        }
    }

    #[test]
    #[cfg(feature = "bptn")]
    fn test_c_ffi_bptn_api() {
        unsafe {
            // 1. Framing buffer push
            let mut buf: u64 = 0;
            let w1 = snn_bptn_framing_push(&raw mut buf, true);
            assert_eq!(w1, 1);
            let w2 = snn_bptn_framing_push(&raw mut buf, false);
            assert_eq!(w2, 2);

            // 2. Latch step (ResetDominant)
            let mut latch: u64 = 0;
            let s1 = snn_bptn_latch_step(&raw mut latch, 0, true, false);
            assert!(s1);
            assert_eq!(latch & 1, 1);
            let s2 = snn_bptn_latch_step(&raw mut latch, 0, false, false);
            assert!(s2);
            let s3 = snn_bptn_latch_step(&raw mut latch, 0, false, true);
            assert!(!s3);

            // 3. Toggle step
            let mut toggle: u64 = 0;
            let t1 = snn_bptn_toggle_step(&raw mut toggle, true);
            assert!(t1);
            let t2 = snn_bptn_toggle_step(&raw mut toggle, true);
            assert!(!t2);

            // 4. Neuron evaluation
            let histories = [0b101u64, 0b010u64];
            let kernels = [0b111u64, 0b111u64];
            let polarities = [0u64, 0u64];
            // potential = popcnt(101 & 111) + popcnt(010 & 111) = 2 + 1 = 3 >= 3 -> true
            assert!(snn_bptn_neuron_eval(
                histories.as_ptr(),
                kernels.as_ptr(),
                polarities.as_ptr(),
                2,
                3
            ));
            assert!(!snn_bptn_neuron_eval(
                histories.as_ptr(),
                kernels.as_ptr(),
                polarities.as_ptr(),
                2,
                4
            ));

            // 5. Layer evaluation: 2 neurons, 2 channels
            // Neuron 0 threshold 3 -> fires (bit 0 = 1)
            // Neuron 1 threshold 4 -> does not fire (bit 1 = 0)
            let layer_kernels = [0b111u64, 0b111u64, 0b111u64, 0b111u64];
            let layer_pols = [0u64; 4];
            let layer_thresh = [3u32, 4u32];
            let mut out_mask = 0u64;
            let l_res = snn_bptn_layer_eval(
                histories.as_ptr(),
                2,
                layer_kernels.as_ptr(),
                layer_pols.as_ptr(),
                layer_thresh.as_ptr(),
                2,
                &raw mut out_mask,
            );
            assert_eq!(l_res, 0);
            assert_eq!(out_mask, 1); // Only neuron 0 fired

            // 6. Multiplication-free CTZ Readout : 2 classes, 2 features
            // Class 0: w = [100, -50], bias = 0 -> score = 100
            // Class 1: w = [-50, 100], bias = 0 -> score = -50
            let r_w = [100i32, -50, -50, 100];
            let pred =
                snn_readout_predict_spikes_mask(r_w.as_ptr(), core::ptr::null(), out_mask, 2, 2);
            assert_eq!(pred, 0);
        }
    }
}
