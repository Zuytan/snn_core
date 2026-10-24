//! 🧠 STDP (Spike-Timing-Dependent Plasticity) Learning Rule in Fix32 Arithmetic
//!
//! Implements local event-driven synaptic plasticity:
//! - **LTP (Long-Term Potentiation)**: If the pre-synaptic neuron fires before the post-synaptic neuron (Δt > 0), the synapse is strengthened.
//! - **LTD (Long-Term Depression)**: If the post-synaptic neuron fires before the pre-synaptic neuron (Δt < 0), the synapse is weakened.
//!
//! Fully implemented in `Fix32` fixed-point arithmetic without FPU, compatible with `#![no_std]`.

use super::numeric::Fix32;
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Configuration of STDP hyperparameters
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StdpConfig {
    /// LTP strengthening amplitude (e.g., +0.015)
    pub a_plus: Fix32,
    /// LTD depression amplitude (e.g., -0.018, slightly asymmetric to prevent saturation)
    pub a_minus: Fix32,
    /// Pre-synaptic trace decay rate per tick (e.g., 0.90)
    pub trace_decay_pre: Fix32,
    /// Post-synaptic trace decay rate per tick (e.g., 0.90)
    pub trace_decay_post: Fix32,
    /// Cutoff threshold below which a residual trace is zeroed out (avoids O(E) on infinitesimal traces)
    pub trace_cutoff: Fix32,
    /// Minimum allowed synaptic weight (lower bound)
    pub w_min: Fix32,
    /// Maximum allowed synaptic weight (upper bound)
    pub w_max: Fix32,
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            a_plus: Fix32::from_num(0.02),
            a_minus: Fix32::from_num(-0.024),
            trace_decay_pre: Fix32::from_num(0.88),
            trace_decay_post: Fix32::from_num(0.88),
            trace_cutoff: Fix32::from_num(0.005),
            w_min: Fix32::from_num(-3.0),
            w_max: Fix32::from_num(3.0),
        }
    }
}

/// Static zero-allocation STDP synaptic plasticity learner (pure `no_std`)
#[derive(Clone, Copy, Debug)]
pub struct StaticStdpLearner<const N_INPUTS: usize, const N: usize> {
    pub config: StdpConfig,
    /// External input activation traces
    pub input_traces: [Fix32; N_INPUTS],
    /// Pre-synaptic activation traces of internal recurrent neurons
    pub neuron_pre_traces: [Fix32; N],
    /// Post-synaptic activation traces of internal neurons
    pub post_traces: [Fix32; N],
}

impl<const N_INPUTS: usize, const N: usize> StaticStdpLearner<N_INPUTS, N> {
    pub const fn new(config: StdpConfig) -> Self {
        Self {
            config,
            input_traces: [Fix32::ZERO; N_INPUTS],
            neuron_pre_traces: [Fix32::ZERO; N],
            post_traces: [Fix32::ZERO; N],
        }
    }

    #[inline]
    pub fn decay_traces(&mut self) {
        let cutoff = self.config.trace_cutoff;
        for trace in &mut self.input_traces {
            *trace *= self.config.trace_decay_pre;
            if *trace < cutoff {
                *trace = Fix32::ZERO;
            }
        }
        for trace in &mut self.neuron_pre_traces {
            *trace *= self.config.trace_decay_pre;
            if *trace < cutoff {
                *trace = Fix32::ZERO;
            }
        }
        for trace in &mut self.post_traces {
            *trace *= self.config.trace_decay_post;
            if *trace < cutoff {
                *trace = Fix32::ZERO;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_stdp(
        &mut self,
        active_inputs: &[u32],
        active_neurons: &[u32],
        input_row_ptrs: &[u32],
        input_targets: &[u32],
        input_weights: &mut [Fix32],
        internal_row_ptrs: &[u32],
        internal_targets: &[u32],
        internal_weights: &mut [Fix32],
    ) {
        // 1. Temporal trace decay with below-threshold cutoff
        self.decay_traces();

        // 2. LTD for input synapses (if target fired recently)
        for &in_idx in active_inputs {
            let idx = in_idx as usize;
            if idx < N_INPUTS && idx + 1 < input_row_ptrs.len() {
                let start = input_row_ptrs[idx] as usize;
                let end = (input_row_ptrs[idx + 1] as usize)
                    .min(input_targets.len())
                    .min(input_weights.len());
                for syn_idx in start..end {
                    let target = input_targets[syn_idx] as usize;
                    if target < N && self.post_traces[target] > Fix32::ZERO {
                        let dw = self.config.a_minus * self.post_traces[target];
                        input_weights[syn_idx] = (input_weights[syn_idx] + dw)
                            .max(self.config.w_min)
                            .min(self.config.w_max);
                    }
                }
            }
        }

        // 3. LTD for internal recurrent synapses (if target fired recently)
        for &neuron_idx in active_neurons {
            let n_idx = neuron_idx as usize;
            if n_idx < N && n_idx + 1 < internal_row_ptrs.len() {
                let start = internal_row_ptrs[n_idx] as usize;
                let end = (internal_row_ptrs[n_idx + 1] as usize)
                    .min(internal_targets.len())
                    .min(internal_weights.len());
                for syn_idx in start..end {
                    let target = internal_targets[syn_idx] as usize;
                    if target < N && self.post_traces[target] > Fix32::ZERO {
                        let dw = self.config.a_minus * self.post_traces[target];
                        internal_weights[syn_idx] = (internal_weights[syn_idx] + dw)
                            .max(self.config.w_min)
                            .min(self.config.w_max);
                    }
                }
            }
        }

        // 4. LTP for synapses (pre active in past -> post active now at tick t)
        // O(1) short-circuit: if no post-synaptic neuron fired, no LTP can occur
        if !active_neurons.is_empty() {
            let mut is_neuron_active = [false; N];
            for &neuron_idx in active_neurons {
                let n_idx = neuron_idx as usize;
                if n_idx < N {
                    is_neuron_active[n_idx] = true;
                }
            }

            // LTP for input synapses (pre active recently -> post active now)
            for (src_idx, window) in input_row_ptrs.windows(2).enumerate() {
                if src_idx < N_INPUTS {
                    let pre_trace = self.input_traces[src_idx];
                    if pre_trace > Fix32::ZERO {
                        let start = window[0] as usize;
                        let end = (window[1] as usize)
                            .min(input_targets.len())
                            .min(input_weights.len());
                        for syn_idx in start..end {
                            let target = input_targets[syn_idx] as usize;
                            if target < N && is_neuron_active[target] {
                                let dw = self.config.a_plus * pre_trace;
                                input_weights[syn_idx] = (input_weights[syn_idx] + dw)
                                    .max(self.config.w_min)
                                    .min(self.config.w_max);
                            }
                        }
                    }
                }
            }

            // LTP for internal recurrent synapses
            for (src_idx, window) in internal_row_ptrs.windows(2).enumerate() {
                if src_idx < N {
                    let pre_trace = self.neuron_pre_traces[src_idx];
                    if pre_trace > Fix32::ZERO {
                        let start = window[0] as usize;
                        let end = (window[1] as usize)
                            .min(internal_targets.len())
                            .min(internal_weights.len());
                        for syn_idx in start..end {
                            let target = internal_targets[syn_idx] as usize;
                            if target < N && is_neuron_active[target] {
                                let dw = self.config.a_plus * pre_trace;
                                internal_weights[syn_idx] = (internal_weights[syn_idx] + dw)
                                    .max(self.config.w_min)
                                    .min(self.config.w_max);
                            }
                        }
                    }
                }
            }
        }

        // 5. Increment pre and post traces for spikes emitted at current tick t
        for &in_idx in active_inputs {
            let idx = in_idx as usize;
            if idx < N_INPUTS {
                self.input_traces[idx] += Fix32::ONE;
            }
        }
        for &neuron_idx in active_neurons {
            let n_idx = neuron_idx as usize;
            if n_idx < N {
                self.post_traces[n_idx] += Fix32::ONE;
                self.neuron_pre_traces[n_idx] += Fix32::ONE;
            }
        }
    }

    /// Applies the STDP rule leveraging precalculated CSC indices for O(spikes * in_degree) complexity.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_stdp_csc<const MAX_IN: usize, const MAX_INT: usize, const N_PLUS_1: usize>(
        &mut self,
        active_inputs: &[u32],
        active_neurons: &[u32],
        input_row_ptrs: &[u32],
        input_targets: &[u32],
        input_weights: &mut [Fix32],
        input_csc: &StaticCscIndex<MAX_IN, N_PLUS_1>,
        internal_row_ptrs: &[u32],
        internal_targets: &[u32],
        internal_weights: &mut [Fix32],
        internal_csc: &StaticCscIndex<MAX_INT, N_PLUS_1>,
    ) {
        // 1. Temporal trace decay with below-threshold cutoff
        self.decay_traces();

        // 2. LTD for input synapses (post fired recently -> pre fires now)
        for &in_idx in active_inputs {
            let idx = in_idx as usize;
            if idx < N_INPUTS && idx + 1 < input_row_ptrs.len() {
                let start = input_row_ptrs[idx] as usize;
                let end = (input_row_ptrs[idx + 1] as usize)
                    .min(input_targets.len())
                    .min(input_weights.len());
                for syn_idx in start..end {
                    let target = input_targets[syn_idx] as usize;
                    if target < N && self.post_traces[target] > Fix32::ZERO {
                        let dw = self.config.a_minus * self.post_traces[target];
                        input_weights[syn_idx] = (input_weights[syn_idx] + dw)
                            .max(self.config.w_min)
                            .min(self.config.w_max);
                    }
                }
            }
        }

        // 3. LTD for internal recurrent synapses
        for &neuron_idx in active_neurons {
            let n_idx = neuron_idx as usize;
            if n_idx < N && n_idx + 1 < internal_row_ptrs.len() {
                let start = internal_row_ptrs[n_idx] as usize;
                let end = (internal_row_ptrs[n_idx + 1] as usize)
                    .min(internal_targets.len())
                    .min(internal_weights.len());
                for syn_idx in start..end {
                    let target = internal_targets[syn_idx] as usize;
                    if target < N && self.post_traces[target] > Fix32::ZERO {
                        let dw = self.config.a_minus * self.post_traces[target];
                        internal_weights[syn_idx] = (internal_weights[syn_idx] + dw)
                            .max(self.config.w_min)
                            .min(self.config.w_max);
                    }
                }
            }
        }

        // 4. LTP via CSC: traverse ONLY afferent synapses incoming to active neurons
        for &neuron_idx in active_neurons {
            let post_idx = neuron_idx as usize;
            if post_idx < N {
                // Input synapses afferent to post_idx
                let in_start = input_csc.col_ptrs[post_idx] as usize;
                let in_end = (input_csc.col_ptrs[post_idx + 1] as usize).min(input_csc.len);
                for i in in_start..in_end {
                    let src = input_csc.sources[i] as usize;
                    let syn_idx = input_csc.syn_indices[i] as usize;
                    if src < N_INPUTS && syn_idx < input_weights.len() {
                        let pre_trace = self.input_traces[src];
                        if pre_trace > Fix32::ZERO {
                            let dw = self.config.a_plus * pre_trace;
                            input_weights[syn_idx] = (input_weights[syn_idx] + dw)
                                .max(self.config.w_min)
                                .min(self.config.w_max);
                        }
                    }
                }

                // Internal recurrent synapses afferent to post_idx
                let int_start = internal_csc.col_ptrs[post_idx] as usize;
                let int_end = (internal_csc.col_ptrs[post_idx + 1] as usize).min(internal_csc.len);
                for i in int_start..int_end {
                    let src = internal_csc.sources[i] as usize;
                    let syn_idx = internal_csc.syn_indices[i] as usize;
                    if src < N && syn_idx < internal_weights.len() {
                        let pre_trace = self.neuron_pre_traces[src];
                        if pre_trace > Fix32::ZERO {
                            let dw = self.config.a_plus * pre_trace;
                            internal_weights[syn_idx] = (internal_weights[syn_idx] + dw)
                                .max(self.config.w_min)
                                .min(self.config.w_max);
                        }
                    }
                }
            }
        }

        // 5. Increment pre and post traces for spikes emitted at current tick t
        for &in_idx in active_inputs {
            let idx = in_idx as usize;
            if idx < N_INPUTS {
                self.input_traces[idx] += Fix32::ONE;
            }
        }
        for &neuron_idx in active_neurons {
            let n_idx = neuron_idx as usize;
            if n_idx < N {
                self.post_traces[n_idx] += Fix32::ONE;
                self.neuron_pre_traces[n_idx] += Fix32::ONE;
            }
        }
    }
}

/// Static precalculated CSC (Compressed Sparse Column) index for STDP acceleration.
///
/// Enables direct traversal of afferent (incoming) synapses for a post-synaptic neuron
/// in $\mathcal{O}(\text{spikes} \times \text{in-degree})$ without scanning all synapses in $\mathcal{O}(E)$.
#[derive(Clone, Copy, Debug)]
pub struct StaticCscIndex<const MAX_SYNAPSES: usize, const N_COLS_PLUS_1: usize> {
    /// Column start pointers (size = N_COLS_PLUS_1)
    pub col_ptrs: [u32; N_COLS_PLUS_1],
    /// Pre-synaptic neuron indices (sources)
    pub sources: [u32; MAX_SYNAPSES],
    /// Corresponding indices in the CSR weight array
    pub syn_indices: [u32; MAX_SYNAPSES],
    /// Actual number of registered synapses
    pub len: usize,
}

impl<const MAX_SYNAPSES: usize, const N_COLS_PLUS_1: usize>
    StaticCscIndex<MAX_SYNAPSES, N_COLS_PLUS_1>
{
    /// Builds the static CSC index from a CSR representation (`row_ptrs`, `targets`).
    #[must_use]
    #[allow(clippy::needless_range_loop)]
    pub fn from_csr(row_ptrs: &[u32], targets: &[u32]) -> Self {
        let n_cols = N_COLS_PLUS_1.saturating_sub(1);
        let mut col_ptrs = [0u32; N_COLS_PLUS_1];
        for &t in targets {
            let col = t as usize;
            if col < n_cols {
                col_ptrs[col + 1] += 1;
            }
        }

        for c in 0..n_cols {
            col_ptrs[c + 1] += col_ptrs[c];
        }

        let total_synapses = (col_ptrs[n_cols] as usize).min(MAX_SYNAPSES);
        let mut sources = [0u32; MAX_SYNAPSES];
        let mut syn_indices = [0u32; MAX_SYNAPSES];
        let mut next_pos = col_ptrs;

        for (src_idx, window) in row_ptrs.windows(2).enumerate() {
            let start = window[0] as usize;
            let end = (window[1] as usize).min(targets.len());
            for syn_idx in start..end {
                let col = targets[syn_idx] as usize;
                if col < n_cols {
                    let p = next_pos[col] as usize;
                    if p < MAX_SYNAPSES {
                        sources[p] = src_idx as u32;
                        syn_indices[p] = syn_idx as u32;
                        next_pos[col] += 1;
                    }
                }
            }
        }

        Self {
            col_ptrs,
            sources,
            syn_indices,
            len: total_synapses,
        }
    }
}

/// Dynamic STDP synaptic plasticity manager
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct StdpLearner {
    pub config: StdpConfig,
    /// Pre-synaptic activation traces (external and internal sources)
    pub pre_traces: Vec<Fix32>,
    /// Post-synaptic activation traces (network population)
    pub post_traces: Vec<Fix32>,
    /// Reusable mask for active neurons (eliminates per-tick dynamic allocations)
    pub active_neurons_mask: Vec<bool>,
}

#[cfg(feature = "alloc")]
impl StdpLearner {
    pub fn new(num_pre_inputs: usize, num_neurons: usize, config: StdpConfig) -> Self {
        Self {
            config,
            pre_traces: vec![Fix32::ZERO; num_pre_inputs + num_neurons],
            post_traces: vec![Fix32::ZERO; num_neurons],
            active_neurons_mask: vec![false; num_neurons],
        }
    }

    /// Decays synaptic traces at each time step
    #[inline]
    pub fn decay_traces(&mut self) {
        let cutoff = self.config.trace_cutoff;
        for trace in &mut self.pre_traces {
            *trace *= self.config.trace_decay_pre;
            if *trace < cutoff {
                *trace = Fix32::ZERO;
            }
        }
        for trace in &mut self.post_traces {
            *trace *= self.config.trace_decay_post;
            if *trace < cutoff {
                *trace = Fix32::ZERO;
            }
        }
    }

    /// Applies the STDP rule during a learning step
    #[allow(clippy::too_many_arguments)]
    pub fn apply_stdp(
        &mut self,
        active_inputs: &[u32],
        active_neurons: &[u32],
        input_row_ptrs: &[u32],
        input_targets: &[u32],
        input_weights: &mut [Fix32],
        internal_row_ptrs: &[u32],
        internal_targets: &[u32],
        internal_weights: &mut [Fix32],
    ) {
        let num_neurons = self.post_traces.len();
        let num_inputs = self.pre_traces.len().saturating_sub(num_neurons);

        // 1. Temporal trace decay
        self.decay_traces();

        // 2. LTD for input synapses (post fired recently, pre fires now)
        for &in_idx in active_inputs {
            let idx = in_idx as usize;
            if idx < num_inputs && idx + 1 < input_row_ptrs.len() {
                let start = input_row_ptrs[idx] as usize;
                let end = (input_row_ptrs[idx + 1] as usize)
                    .min(input_targets.len())
                    .min(input_weights.len());
                for syn_idx in start..end {
                    let target = input_targets[syn_idx] as usize;
                    if target < num_neurons && self.post_traces[target] > Fix32::ZERO {
                        let dw = self.config.a_minus * self.post_traces[target];
                        input_weights[syn_idx] = (input_weights[syn_idx] + dw)
                            .max(self.config.w_min)
                            .min(self.config.w_max);
                    }
                }
            }
        }

        // 3. LTD for internal recurrent synapses (post fired recently, pre fires now)
        for &neuron_idx in active_neurons {
            let n_idx = neuron_idx as usize;
            if n_idx < num_neurons && n_idx + 1 < internal_row_ptrs.len() {
                let start = internal_row_ptrs[n_idx] as usize;
                let end = (internal_row_ptrs[n_idx + 1] as usize)
                    .min(internal_targets.len())
                    .min(internal_weights.len());
                for syn_idx in start..end {
                    let target = internal_targets[syn_idx] as usize;
                    if target < num_neurons && self.post_traces[target] > Fix32::ZERO {
                        let dw = self.config.a_minus * self.post_traces[target];
                        internal_weights[syn_idx] = (internal_weights[syn_idx] + dw)
                            .max(self.config.w_min)
                            .min(self.config.w_max);
                    }
                }
            }
        }

        // 4. LTP for synapses (pre active in past -> post active now at tick t)
        // O(1) short-circuit: if no post-synaptic neuron fired, no LTP can occur
        if !active_neurons.is_empty() {
            if self.active_neurons_mask.len() < num_neurons {
                self.active_neurons_mask.resize(num_neurons, false);
            }
            self.active_neurons_mask.fill(false);
            for &n in active_neurons {
                let idx = n as usize;
                if idx < num_neurons {
                    self.active_neurons_mask[idx] = true;
                }
            }

            // LTP for input synapses (pre active recently -> post active now)
            for (src_idx, window) in input_row_ptrs.windows(2).enumerate() {
                if src_idx < num_inputs {
                    let pre_trace = self.pre_traces[src_idx];
                    if pre_trace > Fix32::ZERO {
                        let start = window[0] as usize;
                        let end = (window[1] as usize)
                            .min(input_targets.len())
                            .min(input_weights.len());
                        for syn_idx in start..end {
                            let target = input_targets[syn_idx] as usize;
                            if target < num_neurons && self.active_neurons_mask[target] {
                                let dw = self.config.a_plus * pre_trace;
                                input_weights[syn_idx] = (input_weights[syn_idx] + dw)
                                    .max(self.config.w_min)
                                    .min(self.config.w_max);
                            }
                        }
                    }
                }
            }

            // LTP for internal recurrent synapses
            for (src_idx, window) in internal_row_ptrs.windows(2).enumerate() {
                if src_idx < num_neurons {
                    let pre_trace = self.pre_traces[num_inputs + src_idx];
                    if pre_trace > Fix32::ZERO {
                        let start = window[0] as usize;
                        let end = (window[1] as usize)
                            .min(internal_targets.len())
                            .min(internal_weights.len());
                        for syn_idx in start..end {
                            let target = internal_targets[syn_idx] as usize;
                            if target < num_neurons && self.active_neurons_mask[target] {
                                let dw = self.config.a_plus * pre_trace;
                                internal_weights[syn_idx] = (internal_weights[syn_idx] + dw)
                                    .max(self.config.w_min)
                                    .min(self.config.w_max);
                            }
                        }
                    }
                }
            }
        }

        // 5. Increment pre and post traces for spikes emitted at current tick t
        for &in_idx in active_inputs {
            let idx = in_idx as usize;
            if idx < num_inputs {
                self.pre_traces[idx] += Fix32::ONE;
            }
        }
        for &neuron_idx in active_neurons {
            let n_idx = neuron_idx as usize;
            if n_idx < num_neurons {
                self.post_traces[n_idx] += Fix32::ONE;
                let pre_internal_idx = num_inputs + n_idx;
                if pre_internal_idx < self.pre_traces.len() {
                    self.pre_traces[pre_internal_idx] += Fix32::ONE;
                }
            }
        }
    }

    /// Applies the STDP rule leveraging precalculated CSC indices for O(spikes * in_degree) complexity.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_stdp_csc(
        &mut self,
        active_inputs: &[u32],
        active_neurons: &[u32],
        input_row_ptrs: &[u32],
        input_targets: &[u32],
        input_weights: &mut [Fix32],
        input_csc: &CscIndex,
        internal_row_ptrs: &[u32],
        internal_targets: &[u32],
        internal_weights: &mut [Fix32],
        internal_csc: &CscIndex,
    ) {
        let num_neurons = self.post_traces.len();
        let num_inputs = self.pre_traces.len().saturating_sub(num_neurons);

        // 1. Temporal trace decay
        self.decay_traces();

        // 2. LTD for input synapses
        for &in_idx in active_inputs {
            let idx = in_idx as usize;
            if idx < num_inputs && idx + 1 < input_row_ptrs.len() {
                let start = input_row_ptrs[idx] as usize;
                let end = (input_row_ptrs[idx + 1] as usize)
                    .min(input_targets.len())
                    .min(input_weights.len());
                for syn_idx in start..end {
                    let target = input_targets[syn_idx] as usize;
                    if target < num_neurons && self.post_traces[target] > Fix32::ZERO {
                        let dw = self.config.a_minus * self.post_traces[target];
                        input_weights[syn_idx] = (input_weights[syn_idx] + dw)
                            .max(self.config.w_min)
                            .min(self.config.w_max);
                    }
                }
            }
        }

        // 3. LTD for internal recurrent synapses
        for &neuron_idx in active_neurons {
            let n_idx = neuron_idx as usize;
            if n_idx < num_neurons && n_idx + 1 < internal_row_ptrs.len() {
                let start = internal_row_ptrs[n_idx] as usize;
                let end = (internal_row_ptrs[n_idx + 1] as usize)
                    .min(internal_targets.len())
                    .min(internal_weights.len());
                for syn_idx in start..end {
                    let target = internal_targets[syn_idx] as usize;
                    if target < num_neurons && self.post_traces[target] > Fix32::ZERO {
                        let dw = self.config.a_minus * self.post_traces[target];
                        internal_weights[syn_idx] = (internal_weights[syn_idx] + dw)
                            .max(self.config.w_min)
                            .min(self.config.w_max);
                    }
                }
            }
        }

        // 4. LTP via CSC in O(active_neurons * in_degree)
        for &neuron_idx in active_neurons {
            let post_idx = neuron_idx as usize;
            if post_idx < num_neurons {
                // Inputs
                if post_idx + 1 < input_csc.col_ptrs.len() {
                    let in_start = input_csc.col_ptrs[post_idx] as usize;
                    let in_end =
                        (input_csc.col_ptrs[post_idx + 1] as usize).min(input_csc.sources.len());
                    for i in in_start..in_end {
                        let src = input_csc.sources[i] as usize;
                        let syn_idx = input_csc.syn_indices[i] as usize;
                        if src < num_inputs && syn_idx < input_weights.len() {
                            let pre_trace = self.pre_traces[src];
                            if pre_trace > Fix32::ZERO {
                                let dw = self.config.a_plus * pre_trace;
                                input_weights[syn_idx] = (input_weights[syn_idx] + dw)
                                    .max(self.config.w_min)
                                    .min(self.config.w_max);
                            }
                        }
                    }
                }

                // Internal
                if post_idx + 1 < internal_csc.col_ptrs.len() {
                    let int_start = internal_csc.col_ptrs[post_idx] as usize;
                    let int_end = (internal_csc.col_ptrs[post_idx + 1] as usize)
                        .min(internal_csc.sources.len());
                    for i in int_start..int_end {
                        let src = internal_csc.sources[i] as usize;
                        let syn_idx = internal_csc.syn_indices[i] as usize;
                        if src < num_neurons && syn_idx < internal_weights.len() {
                            let pre_trace = self.pre_traces[num_inputs + src];
                            if pre_trace > Fix32::ZERO {
                                let dw = self.config.a_plus * pre_trace;
                                internal_weights[syn_idx] = (internal_weights[syn_idx] + dw)
                                    .max(self.config.w_min)
                                    .min(self.config.w_max);
                            }
                        }
                    }
                }
            }
        }

        // 5. Traces
        for &in_idx in active_inputs {
            let idx = in_idx as usize;
            if idx < num_inputs {
                self.pre_traces[idx] += Fix32::ONE;
            }
        }
        for &neuron_idx in active_neurons {
            let n_idx = neuron_idx as usize;
            if n_idx < num_neurons {
                self.post_traces[n_idx] += Fix32::ONE;
                let pre_internal_idx = num_inputs + n_idx;
                if pre_internal_idx < self.pre_traces.len() {
                    self.pre_traces[pre_internal_idx] += Fix32::ONE;
                }
            }
        }
    }
}

/// Dynamic CSC (Compressed Sparse Column) index for STDP acceleration.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CscIndex {
    pub col_ptrs: Vec<u32>,
    pub sources: Vec<u32>,
    pub syn_indices: Vec<u32>,
}

#[cfg(feature = "alloc")]
impl CscIndex {
    /// Builds the CSC index from a CSR representation (`row_ptrs`, `targets`).
    #[must_use]
    #[allow(clippy::needless_range_loop)]
    pub fn from_csr(num_cols: usize, row_ptrs: &[u32], targets: &[u32]) -> Self {
        let mut col_counts = vec![0usize; num_cols];
        for &t in targets {
            let col = t as usize;
            if col < num_cols {
                col_counts[col] += 1;
            }
        }

        let mut col_ptrs = vec![0u32; num_cols + 1];
        let mut sum = 0usize;
        for c in 0..num_cols {
            col_ptrs[c] = sum as u32;
            sum += col_counts[c];
        }
        col_ptrs[num_cols] = sum as u32;

        let mut sources = vec![0u32; sum];
        let mut syn_indices = vec![0u32; sum];
        let mut next_pos = col_ptrs.clone();

        for (src_idx, window) in row_ptrs.windows(2).enumerate() {
            let start = window[0] as usize;
            let end = (window[1] as usize).min(targets.len());
            for syn_idx in start..end {
                let col = targets[syn_idx] as usize;
                if col < num_cols {
                    let p = next_pos[col] as usize;
                    if p < sum {
                        sources[p] = src_idx as u32;
                        syn_indices[p] = syn_idx as u32;
                        next_pos[col] += 1;
                    }
                }
            }
        }

        Self {
            col_ptrs,
            sources,
            syn_indices,
        }
    }
}

/// Local reinforcement learning R-STDP (Reward-Modulated Spike-Timing-Dependent Plasticity).
///
/// Implements the Three-Factor Learning Rule:
/// 1. Pre-synaptic activity
/// 2. Post-synaptic activity generating a synaptic eligibility trace $e_{ij}$
/// 3. Global reinforcement/reward signal $R(t)$ (neuromodulator, performance score)
///
/// $$\dot{e}_{ij} = -\frac{e_{ij}}{\tau_e} + \Delta w_{\text{STDP}}(i, j)$$
/// $$\Delta w_{ij} = \eta \cdot R(t) \cdot e_{ij}$$
#[derive(Clone, Copy, Debug)]
pub struct RewardModulatedStdp<const MAX_SYNAPSES: usize> {
    /// Eligibility traces for each synapse
    pub eligibility_traces: [Fix32; MAX_SYNAPSES],
    /// Eligibility decay factor per tick (tau_e)
    pub eligibility_decay: Fix32,
    /// Modulated learning rate (eta)
    pub learning_rate: Fix32,
    /// Weight lower bound
    pub w_min: Fix32,
    /// Weight upper bound
    pub w_max: Fix32,
}

impl<const MAX_SYNAPSES: usize> RewardModulatedStdp<MAX_SYNAPSES> {
    pub const fn new(
        eligibility_decay: Fix32,
        learning_rate: Fix32,
        w_min: Fix32,
        w_max: Fix32,
    ) -> Self {
        Self {
            eligibility_traces: [Fix32::ZERO; MAX_SYNAPSES],
            eligibility_decay,
            learning_rate,
            w_min,
            w_max,
        }
    }

    /// Decays eligibility traces over time.
    #[inline]
    pub fn decay_traces(&mut self) {
        for e in &mut self.eligibility_traces {
            *e *= self.eligibility_decay;
        }
    }

    /// Accumulates an eligibility variation induced by local STDP on synapse `syn_idx`.
    #[inline]
    pub fn accumulate_eligibility(&mut self, syn_idx: usize, delta_stdp: Fix32) {
        if syn_idx < MAX_SYNAPSES {
            self.eligibility_traces[syn_idx] += delta_stdp;
        }
    }

    /// Applies the global `reward` signal to registered synapses.
    /// Updates `weights` proportionally to accumulated eligibility.
    #[allow(clippy::needless_range_loop)]
    pub fn apply_reward(&mut self, reward: Fix32, weights: &mut [Fix32]) {
        let n = weights.len().min(MAX_SYNAPSES);
        let factor = self.learning_rate * reward;
        for i in 0..n {
            let dw = factor * self.eligibility_traces[i];
            let new_w = (weights[i] + dw).clamp(self.w_min, self.w_max);
            weights[i] = new_w;
        }
    }
}

/// Membrane homeostasis (intrinsic excitability threshold regulation).
///
/// Dynamically adjusts each neuron's threshold to maintain a target firing rate $r_{\text{target}}$:
/// - If the neuron fires more frequently than the target rate, its threshold increases.
/// - If it is silent, its threshold decreases to restore excitability.
///
/// Prevents runaway activity or synaptic silencing during online learning.
#[derive(Clone, Copy, Debug)]
pub struct HomeostaticThreshold<const N: usize> {
    /// Current effective thresholds
    pub current_thresholds: [Fix32; N],
    /// Nominal baseline threshold
    pub base_threshold: Fix32,
    /// Target firing rate per tick (e.g., 0.05 = 5% firing)
    pub target_rate: Fix32,
    /// Homeostatic correction rate
    pub adaptation_rate: Fix32,
    /// Sliding decay factor
    pub decay: Fix32,
}

impl<const N: usize> HomeostaticThreshold<N> {
    pub const fn new(
        base_threshold: Fix32,
        target_rate: Fix32,
        adaptation_rate: Fix32,
        decay: Fix32,
    ) -> Self {
        Self {
            current_thresholds: [base_threshold; N],
            base_threshold,
            target_rate,
            adaptation_rate,
            decay,
        }
    }

    /// Updates thresholds after each tick based on firing neurons.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, spikes: &[bool]) {
        let n = N.min(spikes.len());
        for i in 0..n {
            let active = if spikes[i] { Fix32::ONE } else { Fix32::ZERO };
            let err = active - self.target_rate;
            let delta = err * self.adaptation_rate;
            let new_th = self.current_thresholds[i] * self.decay + delta;
            self.current_thresholds[i] = new_th.max(self.base_threshold / Fix32::from_num(2.0));
        }
    }

    /// Returns the effective threshold for neuron `index`.
    #[inline]
    pub fn get_threshold(&self, index: usize) -> Fix32 {
        if index < N {
            self.current_thresholds[index]
        } else {
            self.base_threshold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "alloc")]
    fn test_stdp_recurrent_ltd_and_ltp() {
        let config = StdpConfig {
            a_plus: Fix32::from_num(0.1),
            a_minus: Fix32::from_num(-0.1),
            trace_decay_pre: Fix32::from_num(0.5),
            trace_decay_post: Fix32::from_num(0.5),
            trace_cutoff: Fix32::from_num(0.005),
            w_min: Fix32::ZERO,
            w_max: Fix32::from_num(2.0),
        };

        // 2 neurons, internal synapse 0 -> 1 with weight 0.5
        let mut stdp = StdpLearner::new(0, 2, config);
        let internal_row_ptrs = [0u32, 1, 1]; // neuron 0 points to target 1
        let internal_targets = [1u32];
        let mut internal_weights = [Fix32::from_num(0.5)];

        // Tick 1: Neuron 1 fires first (post)
        stdp.apply_stdp(
            &[],
            &[1],
            &[0],
            &[],
            &mut [],
            &internal_row_ptrs,
            &internal_targets,
            &mut internal_weights,
        );
        // Weight unchanged at tick 1 since pre didn't fire
        assert_eq!(internal_weights[0], Fix32::from_num(0.5));
        assert!(stdp.post_traces[1] > Fix32::ZERO);

        // Tick 2: Neuron 0 fires next (pre after post -> LTD on synapse 0 -> 1!)
        stdp.apply_stdp(
            &[],
            &[0],
            &[0],
            &[],
            &mut [],
            &internal_row_ptrs,
            &internal_targets,
            &mut internal_weights,
        );
        // LTD must have decreased weight!
        assert!(
            internal_weights[0] < Fix32::from_num(0.5),
            "Internal LTD should decrease recurrent weight"
        );

        // Tick 3: Neuron 1 fires again (post after pre -> LTP on synapse 0 -> 1!)
        let weight_before_ltp = internal_weights[0];
        stdp.apply_stdp(
            &[],
            &[1],
            &[0],
            &[],
            &mut [],
            &internal_row_ptrs,
            &internal_targets,
            &mut internal_weights,
        );
        assert!(
            internal_weights[0] > weight_before_ltp,
            "Internal LTP should increase recurrent weight"
        );
    }

    #[test]
    fn test_static_stdp_learner() {
        let config = StdpConfig::default();
        let mut stdp = StaticStdpLearner::<1, 2>::new(config);
        let input_row_ptrs = [0u32, 1];
        let input_targets = [0u32];
        let mut input_weights = [Fix32::ONE];
        let internal_row_ptrs = [0u32, 1, 1];
        let internal_targets = [1u32];
        let mut internal_weights = [Fix32::ONE];

        stdp.apply_stdp(
            &[0],
            &[0],
            &input_row_ptrs,
            &input_targets,
            &mut input_weights,
            &internal_row_ptrs,
            &internal_targets,
            &mut internal_weights,
        );

        assert!(stdp.input_traces[0] > Fix32::ZERO);
        assert!(stdp.post_traces[0] > Fix32::ZERO);
        assert!(stdp.neuron_pre_traces[0] > Fix32::ZERO);
        // On first simultaneous tick (delta_t = 0), no false LTP should occur
        assert_eq!(
            input_weights[0],
            Fix32::ONE,
            "Simultaneous spikes at tick 0 should not trigger causal LTP"
        );
    }

    #[test]
    fn test_stdp_simultaneous_spikes_no_false_ltp() {
        let config = StdpConfig {
            a_plus: Fix32::from_num(0.2),
            a_minus: Fix32::from_num(-0.2),
            trace_decay_pre: Fix32::from_num(0.8),
            trace_decay_post: Fix32::from_num(0.8),
            trace_cutoff: Fix32::from_num(0.005),
            w_min: Fix32::ZERO,
            w_max: Fix32::from_num(2.0),
        };
        let mut stdp = StaticStdpLearner::<1, 1>::new(config);
        let input_row_ptrs = [0u32, 1];
        let input_targets = [0u32];
        let mut input_weights = [Fix32::ONE];
        let internal_row_ptrs = [0u32, 0];
        let internal_targets = [];
        let mut internal_weights = [];

        // Tick 0: Input 0 and Neuron 0 fire AT THE SAME TIME (delta_t = 0)
        stdp.apply_stdp(
            &[0],
            &[0],
            &input_row_ptrs,
            &input_targets,
            &mut input_weights,
            &internal_row_ptrs,
            &internal_targets,
            &mut internal_weights,
        );
        // Weight should NOT increase at tick 0
        assert_eq!(
            input_weights[0],
            Fix32::ONE,
            "Delta_t = 0 should not cause immediate LTP"
        );

        // Tick 1: Neuron 0 fires again (Tick 1 after input at Tick 0 -> causality preserved!)
        stdp.apply_stdp(
            &[],
            &[0],
            &input_row_ptrs,
            &input_targets,
            &mut input_weights,
            &internal_row_ptrs,
            &internal_targets,
            &mut internal_weights,
        );
        assert!(
            input_weights[0] > Fix32::ONE,
            "Firing at the next tick must benefit from causal trace (LTP)"
        );
    }

    #[test]
    fn test_reward_modulated_stdp() {
        let mut r_stdp = RewardModulatedStdp::<2>::new(
            Fix32::from_num(0.9), // decay
            Fix32::from_num(0.5), // learning rate
            Fix32::from_num(-2.0),
            Fix32::from_num(2.0),
        );

        let mut weights = [Fix32::from_num(0.5), Fix32::from_num(-0.5)];

        // STDP event on synapse 0 (eligible strengthening)
        r_stdp.accumulate_eligibility(0, Fix32::ONE);
        assert_eq!(r_stdp.eligibility_traces[0], Fix32::ONE);

        // Application of positive reward (+1.0)
        // dw = lr * R * e = 0.5 * 1.0 * 1.0 = +0.5
        // w[0] = 0.5 + 0.5 = 1.0
        r_stdp.apply_reward(Fix32::ONE, &mut weights);
        assert_eq!(weights[0], Fix32::ONE);

        // Synapse 1 had no eligibility -> weight unchanged
        assert_eq!(weights[1], Fix32::from_num(-0.5));
    }

    #[test]
    fn test_homeostatic_threshold_regulation() {
        let mut homeo = HomeostaticThreshold::<2>::new(
            Fix32::ONE,            // Nominal threshold 1.0
            Fix32::from_num(0.1),  // Target rate 10%
            Fix32::from_num(0.2),  // Adaptation rate
            Fix32::from_num(0.95), // Leak factor
        );

        // Neuron 0 fires continuously (hyperactivity: 100% instead of 10%)
        // Its threshold must rise to curb activity
        for _ in 0..5 {
            homeo.update(&[true, false]);
        }
        assert!(
            homeo.get_threshold(0) > Fix32::ONE,
            "Hyperactive neuron threshold must increase"
        );
        // Neuron 1 remains silent (0% instead of 10%) -> threshold must decrease
        assert!(
            homeo.get_threshold(1) < Fix32::ONE,
            "Silent neuron threshold must decrease"
        );
    }

    #[test]
    fn test_stdp_csc_equivalence() {
        let config = StdpConfig::default();
        let mut learner_csr = StaticStdpLearner::<2, 3>::new(config);
        let mut learner_csc = StaticStdpLearner::<2, 3>::new(config);

        // Topology: 2 inputs to 3 neurons, and 3 recurrent neurons
        let in_row_ptrs: [u32; 3] = [0, 2, 4];
        let in_targets: [u32; 4] = [0, 1, 1, 2];
        let mut in_weights_csr = [Fix32::from_num(0.5); 4];
        let mut in_weights_csc = [Fix32::from_num(0.5); 4];

        let int_row_ptrs: [u32; 4] = [0, 1, 2, 3];
        let int_targets: [u32; 3] = [1, 2, 0];
        let mut int_weights_csr = [Fix32::from_num(0.2); 3];
        let mut int_weights_csc = [Fix32::from_num(0.2); 3];

        let in_csc = StaticCscIndex::<4, 4>::from_csr(&in_row_ptrs, &in_targets);
        let int_csc = StaticCscIndex::<3, 4>::from_csr(&int_row_ptrs, &int_targets);

        // Tick 0: input 0 active
        learner_csr.apply_stdp(
            &[0],
            &[],
            &in_row_ptrs,
            &in_targets,
            &mut in_weights_csr,
            &int_row_ptrs,
            &int_targets,
            &mut int_weights_csr,
        );
        learner_csc.apply_stdp_csc(
            &[0],
            &[],
            &in_row_ptrs,
            &in_targets,
            &mut in_weights_csc,
            &in_csc,
            &int_row_ptrs,
            &int_targets,
            &mut int_weights_csc,
            &int_csc,
        );
        assert_eq!(in_weights_csr, in_weights_csc);
        assert_eq!(int_weights_csr, int_weights_csc);

        // Tick 1: neuron 0 active (LTP from input 0) + input 1 active
        learner_csr.apply_stdp(
            &[1],
            &[0],
            &in_row_ptrs,
            &in_targets,
            &mut in_weights_csr,
            &int_row_ptrs,
            &int_targets,
            &mut int_weights_csr,
        );
        learner_csc.apply_stdp_csc(
            &[1],
            &[0],
            &in_row_ptrs,
            &in_targets,
            &mut in_weights_csc,
            &in_csc,
            &int_row_ptrs,
            &int_targets,
            &mut int_weights_csc,
            &int_csc,
        );
        assert_eq!(in_weights_csr, in_weights_csc);
        assert_eq!(int_weights_csr, int_weights_csc);
        assert!(in_weights_csc[0] > Fix32::from_num(0.5)); // Validated LTP

        // Tick 2: neuron 1 active (LTP from neuron 0 and input 1)
        learner_csr.apply_stdp(
            &[],
            &[1],
            &in_row_ptrs,
            &in_targets,
            &mut in_weights_csr,
            &int_row_ptrs,
            &int_targets,
            &mut int_weights_csr,
        );
        learner_csc.apply_stdp_csc(
            &[],
            &[1],
            &in_row_ptrs,
            &in_targets,
            &mut in_weights_csc,
            &in_csc,
            &int_row_ptrs,
            &int_targets,
            &mut int_weights_csc,
            &int_csc,
        );
        assert_eq!(in_weights_csr, in_weights_csc);
        assert_eq!(int_weights_csr, int_weights_csc);
    }
}
