//! ⚡ Bit-Parallel Temporal Neuron (BPTN)
//!
//! ⚠️ **Positioning & Industrial Scope**:
//! BPTN is an **experimental ultra-lightweight** architecture specifically optimized for
//! ultra-short temporal triggers (*short-horizon triggers* $W \le 64$ ticks) on microcontrollers
//! lacking FPU (Wake-Word Detection, Keyword Spotting KWS, coincidence and edge detection).
//! For long temporal horizons, continuous neuromorphic vision, or online learning,
//! the biophysical neuron populations of `snn_core` ([`crate::domain::models::lif::LifDynamics`], [`crate::domain::models::alif::AlifDynamics`], [`crate::domain::models::cuba_lif::CubaLifDynamics`],
//! [`crate::domain::compact_population::CompactLif16Population`]) remain the standard industrial benchmark.
//!
//! Designed for conventional digital processor architectures (Word-RAM):
//! - Machine word width $W = 64$ bits (`u64`) representing local temporal history.
//! - Synaptic delays via hardware arithmetic shifts (`LSR` / `LSL` in 1 cycle).
//! - 1D morphological anti-jitter dilation ($\mathcal{D}_\kappa$).
//! - Polar spatio-temporal coincidences via boolean logic (`AND`, `XOR`).
//! - Membrane integration via hardware bit counting (`POPCNT` in 1 cycle).
//! - 100% branchless execution, zero dynamic allocation (`no_std`).

use crate::domain::numeric::Fix32;
use crate::traits::SpikingNetwork;

/// Standard machine temporal window width (64 bits).
pub const BPTN_WINDOW_SIZE: usize = 64;

/// Temporal delay shift operator towards the present ($\mathcal{S}_d^+(\mathbf{x}) = \mathbf{x} \gg d$).
///
/// Discards the $d$ most recent spikes and brings event $s(t-d)$ onto the LSB bit ($k=0$).
#[inline(always)]
pub const fn shift_temporal_delay(x: u64, delay: u8) -> u64 {
    if delay >= 64 { 0 } else { x >> delay }
}

/// Temporal aging operator ($\mathcal{S}_d^-(\mathbf{x}) = \mathbf{x} \ll d$).
///
/// Shifts history toward the past when advancing time.
#[inline(always)]
pub const fn shift_temporal_age(x: u64, delay: u8) -> u64 {
    if delay >= 64 { 0 } else { x << delay }
}

/// 1D temporal morphological dilation operator ($\mathcal{D}_\kappa$).
///
/// Widens each spike by $\pm \kappa$ time steps to provide tolerance
/// against temporal jitter without continuous analog integrators.
#[inline(always)]
pub const fn dilate_temporal_jitter(x: u64, kappa: u8) -> u64 {
    let mut res = x;
    let k_max = if kappa > 63 { 63 } else { kappa };
    let mut r = 1;
    while r <= k_max {
        res |= (x >> r) | (x << r);
        r += 1;
    }
    res
}

/// 1D temporal morphological erosion operator ($\mathcal{E}_\kappa$).
///
/// Eliminates isolated noise spikes whose temporal width is narrower
/// than support $2\kappa + 1$.
#[inline(always)]
pub const fn erode_temporal_noise(x: u64, kappa: u8) -> u64 {
    let mut res = x;
    let k_max = if kappa > 63 { 63 } else { kappa };
    let mut r = 1;
    while r <= k_max {
        res &= (x >> r) & (x << r);
        r += 1;
    }
    res
}

/// 1D temporal morphological opening operator ($\mathcal{O}_\kappa = \mathcal{D}_\kappa \circ \mathcal{E}_\kappa$).
///
/// Removes isolated thermal/shot noise spikes while preserving
/// the temporal envelope of structured robust signals.
#[inline(always)]
pub const fn open_temporal_filter(x: u64, kappa: u8) -> u64 {
    let eroded = erode_temporal_noise(x, kappa);
    dilate_temporal_jitter(eroded, kappa)
}

/// Complete silicon-centric synapse $\Sigma_j = (d_j, \kappa_j, \mathbf{m}_j, \mathbf{p}_j, \mathbf{k}_j)$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BptnSynapse {
    /// Discrete axonal delay $d_j \in [0, 63]$.
    pub delay: u8,
    /// Jitter tolerance radius $\kappa_j \in [0, 63]$.
    pub jitter_radius: u8,
    /// Pre-dilation selection mask $\mathbf{m}_j \in \mathbb{F}_2^{64}$.
    pub mask: u64,
    /// Logic polarity mask $\mathbf{p}_j \in \mathbb{F}_2^{64}$ (0: excitatory, 1: inhibitory / complementary).
    pub polarity: u64,
    /// Synaptic support kernel mask $\mathbf{k}_j \in \mathbb{F}_2^{64}$ (*kernel mask*).
    pub kernel: u64,
}

impl Default for BptnSynapse {
    #[inline(always)]
    fn default() -> Self {
        Self {
            delay: 0,
            jitter_radius: 0,
            mask: u64::MAX,
            polarity: 0,
            kernel: 0,
        }
    }
}

impl BptnSynapse {
    /// Creates a new parameterized BPTN synapse.
    #[inline(always)]
    pub const fn new(delay: u8, jitter_radius: u8, mask: u64, polarity: u64, kernel: u64) -> Self {
        Self {
            delay,
            jitter_radius,
            mask,
            polarity,
            kernel,
        }
    }

    /// Creates a direct synapse without delay or jitter ($\kappa_j = 0, d_j = 0$).
    #[inline(always)]
    pub const fn direct(kernel: u64, polarity: u64) -> Self {
        Self {
            delay: 0,
            jitter_radius: 0,
            mask: u64::MAX,
            polarity,
            kernel,
        }
    }

    /// Evaluates the polarized synaptic state $\mathbf{u}_j(t)$ from history register $\mathbf{x}_j(t)$.
    #[inline(always)]
    pub fn evaluate_state(&self, history: u64) -> u64 {
        let delayed = shift_temporal_delay(history, self.delay) & self.mask;
        let dilated = if self.jitter_radius > 0 {
            dilate_temporal_jitter(delayed, self.jitter_radius)
        } else {
            delayed
        };
        dilated ^ self.polarity
    }

    /// Computes the synaptic contribution to membrane potential $\operatorname{wt}(\mathbf{u}_j(t) \land \mathbf{k}_j)$.
    #[inline(always)]
    pub fn compute_contribution(&self, history: u64) -> u32 {
        let u = self.evaluate_state(history);
        (u & self.kernel).count_ones()
    }
}

/// Bit-Parallel Temporal Neuron integrating $M$ afferent synaptic channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BptnNeuron<const M: usize> {
    /// Afferent synapses connected to the neuron.
    pub synapses: [BptnSynapse; M],
    /// Integer firing threshold $\theta \in \mathbb{N}$.
    pub threshold: u32,
}

impl<const M: usize> BptnNeuron<M> {
    /// Initializes a BPTN neuron with its synapses and threshold.
    #[inline(always)]
    pub const fn new(synapses: [BptnSynapse; M], threshold: u32) -> Self {
        Self {
            synapses,
            threshold,
        }
    }

    /// Evaluates membrane potential $V(t) = \sum_{j=1}^M \operatorname{popcnt}(\mathbf{u}_j(t) \land \mathbf{k}_j)$.
    #[inline(always)]
    pub fn evaluate_potential(&self, histories: &[u64; M]) -> u32 {
        let mut potential: u32 = 0;
        let mut j = 0;
        while j < M {
            potential += self.synapses[j].compute_contribution(histories[j]);
            j += 1;
        }
        potential
    }

    /// Evaluates a tick: computes potential and emits a spike if $V(t) \ge \theta$.
    #[inline(always)]
    pub fn tick(&self, histories: &[u64; M]) -> (bool, u32) {
        let potential = self.evaluate_potential(histories);
        let spike = potential >= self.threshold;
        (spike, potential)
    }
}

/// Ultra-compact BPTN neuron in direct compiled regime ($\kappa_j = 0, d_j = 0, P = 1$).
///
/// Optimized for maximum throughput: evaluates each synaptic channel in only
/// 3 CPU clock instructions (`EOR`, `AND`, `POPCNT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactBptnNeuron<const M: usize> {
    /// Synaptic support masks $\mathbf{k}_j \in \mathbb{F}_2^{64}$.
    pub kernels: [u64; M],
    /// Polarity masks $\mathbf{p}_j \in \mathbb{F}_2^{64}$.
    pub polarities: [u64; M],
    /// Integer firing threshold $\theta$.
    pub threshold: u32,
}

impl<const M: usize> CompactBptnNeuron<M> {
    /// Creates a compact direct BPTN neuron.
    #[inline(always)]
    pub const fn new(kernels: [u64; M], polarities: [u64; M], threshold: u32) -> Self {
        Self {
            kernels,
            polarities,
            threshold,
        }
    }

    /// Evaluates membrane potential by accumulating `POPCNT` instructions.
    #[inline(always)]
    pub fn evaluate_potential(&self, histories: &[u64; M]) -> u32 {
        let mut potential: u32 = 0;
        let mut j = 0;
        while j < M {
            let active = (histories[j] ^ self.polarities[j]) & self.kernels[j];
            potential += active.count_ones();
            j += 1;
        }
        potential
    }

    /// Evaluates the neuron and returns output spike $y(t) \in \{0, 1\}$.
    #[inline(always)]
    pub fn tick(&self, histories: &[u64; M]) -> bool {
        self.evaluate_potential(histories) >= self.threshold
    }
}

/// Feedforward layer of $N$ BPTN neurons connected to $M$ inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BptnFeedforwardLayer<const N: usize, const M: usize> {
    pub neurons: [CompactBptnNeuron<M>; N],
}

impl<const N: usize, const M: usize> BptnFeedforwardLayer<N, M> {
    /// Creates a compact feedforward layer.
    #[inline(always)]
    pub const fn new(neurons: [CompactBptnNeuron<M>; N]) -> Self {
        Self { neurons }
    }

    /// Evaluates the complete layer on the input window vector.
    ///
    /// Returns a `u64` word where bit $i$ ($0 \le i < N$) encodes the spike of neuron $i$.
    #[inline(always)]
    pub fn tick_mask(&self, histories: &[u64; M]) -> u64 {
        let mut spike_word: u64 = 0;
        let mut i = 0;
        while i < N && i < 64 {
            if self.neurons[i].tick(histories) {
                spike_word |= 1 << i;
            }
            i += 1;
        }
        spike_word
    }
}

/// Recurrent R-BPTN layer of $N$ interconnected neurons (Definition 5.1).
///
/// Each neuron maintains its own history register $\mathbf{q}_i(t) \in \mathbb{F}_2^{64}$.
/// State transitions are performed via hardware shift `<< 1` without any heap allocation.
/// The `M_TOTAL` parameter corresponds to total inputs ($M_{\text{ext}} + N$).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RBptnLayer<const N: usize, const M_EXT: usize, const M_TOTAL: usize> {
    /// Sub-network of neurons receiving the $M_{\text{ext}}$ external inputs and $N$ recurrent feedback lines.
    pub neurons: [CompactBptnNeuron<M_TOTAL>; N],
    /// Internal history registers $\mathbf{Q}(t) = (\mathcal{F}_{64}[y_1](t), \dots, \mathcal{F}_{64}[y_N](t))$.
    pub recurrent_history: [u64; N],
}

impl<const N: usize, const M_EXT: usize, const M_TOTAL: usize> RBptnLayer<N, M_EXT, M_TOTAL> {
    /// Initializes the recurrent layer with all registers set to zero.
    #[inline(always)]
    pub const fn new(neurons: [CompactBptnNeuron<M_TOTAL>; N]) -> Self {
        Self {
            neurons,
            recurrent_history: [0; N],
        }
    }

    /// Resets internal state of recurrent registers.
    #[inline(always)]
    pub fn reset(&mut self) {
        let mut i = 0;
        while i < N {
            self.recurrent_history[i] = 0;
            i += 1;
        }
    }

    /// Executes transition step $t \to t+1$.
    ///
    /// 1. Concatenates external input windows and recurrent history $\mathbf{Q}(t)$.
    /// 2. Computes new spikes $\mathbf{y}(t+1)$.
    /// 3. Updates history registers: $\mathbf{q}_i(t+1) = (\mathbf{q}_i(t) \ll 1) \lor y_i(t+1)$.
    pub fn step(&mut self, external_inputs: &[u64; M_EXT]) -> u64 {
        // 1. Assemble combined input vector [M_TOTAL]
        let mut combined_inputs = [0u64; M_TOTAL];
        let ext_len = if M_EXT < M_TOTAL { M_EXT } else { M_TOTAL };
        let mut j = 0;
        while j < ext_len {
            combined_inputs[j] = external_inputs[j];
            j += 1;
        }
        let mut r = 0;
        while r < N && (M_EXT + r) < M_TOTAL {
            combined_inputs[M_EXT + r] = self.recurrent_history[r];
            r += 1;
        }

        // 2. Evaluate spikes for all neurons
        let mut new_spikes_mask: u64 = 0;
        let mut i = 0;
        while i < N && i < 64 {
            if self.neurons[i].tick(&combined_inputs) {
                new_spikes_mask |= 1 << i;
            }
            i += 1;
        }

        // 3. Isometric temporal update of De Bruijn register (<< 1 | spike)
        let mut k = 0;
        while k < N {
            let spike_bit = (new_spikes_mask >> k) & 1;
            self.recurrent_history[k] = (self.recurrent_history[k] << 1) | spike_bit;
            k += 1;
        }

        new_spikes_mask
    }
}

/// BPTN neural network implementing standard [`SpikingNetwork`] trait.
///
/// Enables integrating a BPTN layer into any `snn_core` pipeline or test bench.
/// Automatically manages sliding temporal framing buffers [`TemporalFramingBuffer64`]
/// across the $M$ input channels.
#[derive(Clone, Debug)]
pub struct BptnSpikingNetwork<const N: usize, const M: usize> {
    pub layer: BptnFeedforwardLayer<N, M>,
    pub buffers: [TemporalFramingBuffer64; M],
    pub threshold_input: Fix32,
    output_bools: [bool; N],
    output_words: [u32; 2],
}

impl<const N: usize, const M: usize> BptnSpikingNetwork<N, M> {
    /// Creates a BPTN network compatible with [`SpikingNetwork`].
    pub const fn new(layer: BptnFeedforwardLayer<N, M>) -> Self {
        Self {
            layer,
            buffers: [TemporalFramingBuffer64::new(); M],
            threshold_input: Fix32::ZERO,
            output_bools: [false; N],
            output_words: [0; 2],
        }
    }

    /// Specifies a threshold to convert scalar inputs into binary spikes.
    #[must_use]
    #[inline(always)]
    pub fn with_input_threshold(mut self, threshold: Fix32) -> Self {
        self.threshold_input = threshold;
        self
    }

    /// Resets temporal history buffers.
    #[inline(always)]
    pub fn reset(&mut self) {
        let mut i = 0;
        while i < M {
            self.buffers[i].reset();
            i += 1;
        }
        self.output_bools = [false; N];
        self.output_words = [0; 2];
    }

    /// Evaluates directly from a vector of `u64` temporal windows.
    #[inline(always)]
    pub fn tick_words(&mut self, words: &[u64; M]) -> u64 {
        let mask = self.layer.tick_mask(words);
        self.output_words[0] = mask as u32;
        self.output_words[1] = (mask >> 32) as u32;

        let mut i = 0;
        while i < N && i < 64 {
            self.output_bools[i] = ((mask >> i) & 1) != 0;
            i += 1;
        }
        mask
    }
}

impl<const N: usize, const M: usize> SpikingNetwork<Fix32> for BptnSpikingNetwork<N, M> {
    #[inline]
    fn tick(&mut self, inputs: &[Fix32]) -> &[bool] {
        let mut histories = [0u64; M];
        let mut j = 0;
        while j < M {
            let is_spike = if j < inputs.len() {
                inputs[j] > self.threshold_input
            } else {
                false
            };
            histories[j] = self.buffers[j].push_spike(is_spike);
            j += 1;
        }

        self.tick_words(&histories);
        &self.output_bools
    }

    #[inline]
    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        let mut histories = [0u64; M];
        let mut j = 0;
        while j < M {
            let word_idx = j / 32;
            let bit_idx = j % 32;
            let is_spike = if word_idx < prev_spikes_words.len() {
                ((prev_spikes_words[word_idx] >> bit_idx) & 1) != 0
            } else {
                false
            };
            histories[j] = self.buffers[j].push_spike(is_spike);
            j += 1;
        }

        self.tick_words(&histories);
        &self.output_words
    }

    #[inline]
    fn tick_inputs_words(&mut self, inputs: &[Fix32]) -> &[u32] {
        self.tick(inputs);
        &self.output_words
    }

    #[inline]
    fn reset(&mut self) {
        Self::reset(self);
    }
}

/// Recurrent R-BPTN network implementing standard [`SpikingNetwork`] trait.
#[derive(Clone, Debug)]
pub struct RBptnSpikingNetwork<const N: usize, const M_EXT: usize, const M_TOTAL: usize> {
    pub layer: RBptnLayer<N, M_EXT, M_TOTAL>,
    pub buffers: [TemporalFramingBuffer64; M_EXT],
    pub threshold_input: Fix32,
    output_bools: [bool; N],
    output_words: [u32; 2],
}

impl<const N: usize, const M_EXT: usize, const M_TOTAL: usize>
    RBptnSpikingNetwork<N, M_EXT, M_TOTAL>
{
    /// Initializes recurrent R-BPTN network.
    pub const fn new(layer: RBptnLayer<N, M_EXT, M_TOTAL>) -> Self {
        Self {
            layer,
            buffers: [TemporalFramingBuffer64::new(); M_EXT],
            threshold_input: Fix32::ZERO,
            output_bools: [false; N],
            output_words: [0; 2],
        }
    }

    /// Specifies scalar -> spike conversion threshold.
    #[must_use]
    #[inline(always)]
    pub fn with_input_threshold(mut self, threshold: Fix32) -> Self {
        self.threshold_input = threshold;
        self
    }

    /// Resets recurrent registers and input buffers.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.layer.reset();
        let mut i = 0;
        while i < M_EXT {
            self.buffers[i].reset();
            i += 1;
        }
        self.output_bools = [false; N];
        self.output_words = [0; 2];
    }
}

impl<const N: usize, const M_EXT: usize, const M_TOTAL: usize> SpikingNetwork<Fix32>
    for RBptnSpikingNetwork<N, M_EXT, M_TOTAL>
{
    #[inline]
    fn tick(&mut self, inputs: &[Fix32]) -> &[bool] {
        let mut histories = [0u64; M_EXT];
        let mut j = 0;
        while j < M_EXT {
            let is_spike = if j < inputs.len() {
                inputs[j] > self.threshold_input
            } else {
                false
            };
            histories[j] = self.buffers[j].push_spike(is_spike);
            j += 1;
        }

        let mask = self.layer.step(&histories);
        self.output_words[0] = mask as u32;
        self.output_words[1] = (mask >> 32) as u32;

        let mut i = 0;
        while i < N && i < 64 {
            self.output_bools[i] = ((mask >> i) & 1) != 0;
            i += 1;
        }
        &self.output_bools
    }

    #[inline]
    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        let mut histories = [0u64; M_EXT];
        let mut j = 0;
        while j < M_EXT {
            let word_idx = j / 32;
            let bit_idx = j % 32;
            let is_spike = if word_idx < prev_spikes_words.len() {
                ((prev_spikes_words[word_idx] >> bit_idx) & 1) != 0
            } else {
                false
            };
            histories[j] = self.buffers[j].push_spike(is_spike);
            j += 1;
        }

        let mask = self.layer.step(&histories);
        self.output_words[0] = mask as u32;
        self.output_words[1] = (mask >> 32) as u32;
        &self.output_words
    }

    #[inline]
    fn tick_inputs_words(&mut self, inputs: &[Fix32]) -> &[u32] {
        self.tick(inputs);
        &self.output_words
    }

    #[inline]
    fn reset(&mut self) {
        Self::reset(self);
    }
}

/// Scalar 64-bit temporal history buffer ($\mathcal{F}_{64}[s](t)$).
///
/// Ingests a temporal stream of scalar spikes $s(t) \in \{0, 1\}$
/// step by step and maintains the 64-bit sliding window in 1 clock cycle.
///
/// # Example
/// ```rust
/// use snn_core::bptn::TemporalFramingBuffer64;
///
/// let mut buf = TemporalFramingBuffer64::new();
/// buf.push_spike(true);
/// buf.push_spike(false);
/// buf.push_spike(true);
///
/// assert!(buf.get_lag(0)); // present: true
/// assert!(!buf.get_lag(1)); // t-1: false
/// assert!(buf.get_lag(2)); // t-2: true
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TemporalFramingBuffer64 {
    pub word: u64,
}

impl TemporalFramingBuffer64 {
    /// Creates an empty initialized buffer.
    #[inline(always)]
    pub const fn new() -> Self {
        Self { word: 0 }
    }

    /// Resets the buffer to zero.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.word = 0;
    }

    /// Ingests a new spike at present time $t$ and shifts history toward the past.
    #[inline(always)]
    pub fn push_spike(&mut self, spike: bool) -> u64 {
        self.word = (self.word << 1) | (spike as u64);
        self.word
    }

    /// Returns current window $\mathcal{F}_{64}[s](t)$.
    #[inline(always)]
    pub const fn window(&self) -> u64 {
        self.word
    }

    /// Indicates whether a spike occurred $k$ time steps ago ($0 \le k < 64$).
    #[inline(always)]
    pub const fn get_lag(&self, lag: u8) -> bool {
        if lag >= 64 {
            false
        } else {
            ((self.word >> lag) & 1) != 0
        }
    }
}

// ============================================================================
// SECTION 8: SYNCHRONOUS BISTABLE LATCHES (R-BPTN BISTABLE)
// ============================================================================

/// Priority mode for Set-Reset bistable latch (Lemma 8.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BptnLatchMode {
    /// Priority to Reset signal: q(t) = (S(t) | q(t-1)) & !R(t).
    ResetDominant,
    /// Priority to Set signal: q(t) = S(t) | (q(t-1) & !R(t)).
    SetDominant,
}

/// Synchronous bistable cell R-BPTN (Definition 8.1, Lemma 8.1, Theorem 8.1).
///
/// Implements 1-bit memory with infinite temporal persistence ($\Delta \to \infty$)
/// without membrane leak or analog drift.
///
/// Active state resides in LSB bit (k=0) of `state_history`, while bits
/// 1..63 retain recent firing history over 64 time steps.
///
/// # Example
/// ```rust
/// use snn_core::bptn::{BptnBistableCell, BptnLatchMode};
///
/// let mut cell = BptnBistableCell::new(BptnLatchMode::ResetDominant);
/// assert!(!cell.current_state());
///
/// // Set pulse: toggles to 1
/// cell.step(true, false);
/// assert!(cell.current_state());
///
/// // Maintain without inputs: retains state indefinitely
/// cell.step(false, false);
/// assert!(cell.current_state());
///
/// // Reset pulse: returns to 0
/// cell.step(false, true);
/// assert!(!cell.current_state());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BptnBistableCell {
    /// Temporal history of latch state (LSB = present t).
    pub state_history: u64,
    /// Priority mode of the latch.
    pub mode: BptnLatchMode,
}

impl BptnBistableCell {
    /// Initializes a bistable cell with a given mode and initial state 0.
    #[inline(always)]
    pub const fn new(mode: BptnLatchMode) -> Self {
        Self {
            state_history: 0,
            mode,
        }
    }

    /// Resets cell state.
    #[inline(always)]
    pub fn reset_state(&mut self) {
        self.state_history = 0;
    }

    /// Evaluates current latch state (LSB bit k=0).
    #[inline(always)]
    pub const fn current_state(&self) -> bool {
        (self.state_history & 1) != 0
    }

    /// Returns complete history of last 64 cell states.
    #[inline(always)]
    pub const fn history(&self) -> u64 {
        self.state_history
    }

    /// Executes synchronous time step of the latch (Lemma 8.1).
    ///
    /// Implemented according to exact single-neuron formulation:
    /// - Reset-Dominant: V = S + q_prev + 2*(1 - R) >= 3
    /// - Set-Dominant:   V = 2*S + q_prev + (1 - R) >= 2
    #[inline(always)]
    pub fn step(&mut self, set: bool, reset: bool) -> bool {
        let q_prev = (self.state_history & 1) as u32;
        let s = set as u32;
        let r_inv = (!reset) as u32; // Polarity inversion p_R = 1

        let new_q = match self.mode {
            BptnLatchMode::ResetDominant => {
                // V_RD = S + q + 2*(1 - R)
                let v = s + q_prev + (r_inv << 1);
                v >= 3
            }
            BptnLatchMode::SetDominant => {
                // V_SD = 2*S + q + (1 - R)
                let v = (s << 1) + q_prev + r_inv;
                v >= 2
            }
        };

        self.state_history = (self.state_history << 1) | (new_q as u64);
        new_q
    }
}

/// Synchronous parity inversion T-Latch (Definition 8.2, Theorem 8.1).
///
/// Resolves non-linear temporal parity over infinite horizon ($\Delta \to \infty$):
/// q(t) = q(t-1) ^ toggle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BptnToggleCell {
    pub state_history: u64,
}

impl BptnToggleCell {
    /// Initializes T-Latch to zero.
    #[inline(always)]
    pub const fn new() -> Self {
        Self { state_history: 0 }
    }

    /// Resets state to zero.
    #[inline(always)]
    pub fn reset_state(&mut self) {
        self.state_history = 0;
    }

    /// Current state at LSB bit.
    #[inline(always)]
    pub const fn current_state(&self) -> bool {
        (self.state_history & 1) != 0
    }

    /// History of last 64 states.
    #[inline(always)]
    pub const fn history(&self) -> u64 {
        self.state_history
    }

    /// Executes synchronous inversion step: q(t) = q(t-1) ^ toggle.
    #[inline(always)]
    pub fn step(&mut self, toggle: bool) -> bool {
        let q_prev = (self.state_history & 1) != 0;
        let new_q = q_prev ^ toggle;
        self.state_history = (self.state_history << 1) | (new_q as u64);
        new_q
    }
}

/// Conditional sampling D-Latch (Definition 8.2).
///
/// q(t) = (enable & data) | (!enable & q(t-1)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BptnGatedCell {
    pub state_history: u64,
}

impl BptnGatedCell {
    /// Initializes D-Latch cell to zero.
    #[inline(always)]
    pub const fn new() -> Self {
        Self { state_history: 0 }
    }

    /// Resets state to zero.
    #[inline(always)]
    pub fn reset_state(&mut self) {
        self.state_history = 0;
    }

    /// Current state at LSB bit.
    #[inline(always)]
    pub const fn current_state(&self) -> bool {
        (self.state_history & 1) != 0
    }

    /// History of last 64 states.
    #[inline(always)]
    pub const fn history(&self) -> u64 {
        self.state_history
    }

    /// Captures `data` if `enable` is active, otherwise holds `q(t-1)`.
    #[inline(always)]
    pub fn step(&mut self, data: bool, enable: bool) -> bool {
        let q_prev = (self.state_history & 1) != 0;
        let new_q = if enable { data } else { q_prev };
        self.state_history = (self.state_history << 1) | (new_q as u64);
        new_q
    }
}

/// 2-bit saturating hysteresis latch HS-BPTN (4-state Smith automaton).
///
/// Resolves stochastic vulnerability of 1-bit latch against false alarms:
/// Saturating state counter (0: confirmed OFF, 1: uncertain OFF, 2: uncertain ON, 3: confirmed ON).
/// Active output if state >= 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BptnHysteresisLatch {
    pub state: u8,
    pub state_history: u64,
}

impl BptnHysteresisLatch {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            state: 0,
            state_history: 0,
        }
    }

    #[inline(always)]
    pub fn reset_state(&mut self) {
        self.state = 0;
        self.state_history = 0;
    }

    #[inline(always)]
    pub const fn current_state(&self) -> bool {
        self.state >= 2
    }

    #[inline(always)]
    pub const fn history(&self) -> u64 {
        self.state_history
    }

    #[inline(always)]
    pub fn step(&mut self, set: bool, reset: bool) -> bool {
        if set && !reset && self.state < 3 {
            self.state += 1;
        } else if reset && !set && self.state > 0 {
            self.state -= 1;
        }
        let active = (self.state >= 2) as u64;
        self.state_history = (self.state_history << 1) | active;
        self.state >= 2
    }
}

/// Robust T-Latch with temporal confirmation filtering (Anti-Glitch / Anti-Noise).
///
/// Resolves R-BPTN vulnerability to stochastic false alarms:
/// Any isolated unit noise spike (1 tick duration) is immediately ignored.
/// Only a confirmed trigger pulse of duration >= 2 ticks toggles parity state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BptnRobustToggleCell {
    pub state_history: u64,
    pub prev_trigger: bool,
}

impl BptnRobustToggleCell {
    /// Initializes robust T-Latch.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            state_history: 0,
            prev_trigger: false,
        }
    }

    /// Resets state to zero.
    #[inline(always)]
    pub fn reset_state(&mut self) {
        self.state_history = 0;
        self.prev_trigger = false;
    }

    /// Current state at LSB bit.
    #[inline(always)]
    pub const fn current_state(&self) -> bool {
        (self.state_history & 1) != 0
    }

    /// History of last 64 states.
    #[inline(always)]
    pub const fn history(&self) -> u64 {
        self.state_history
    }

    /// Executes robust inversion step:
    /// Only a confirmed trigger edge across 2 successive steps toggles the latch.
    #[inline(always)]
    pub fn step(&mut self, trigger: bool) -> bool {
        let confirmed_trigger = trigger && self.prev_trigger;
        self.prev_trigger = if confirmed_trigger { false } else { trigger };

        let q_prev = (self.state_history & 1) != 0;
        let new_q = if confirmed_trigger {
            q_prev ^ true
        } else {
            q_prev
        };
        self.state_history = (self.state_history << 1) | (new_q as u64);
        new_q
    }

    /// Immediate direct toggle (mode without confirmation delay).
    #[inline(always)]
    pub fn step_direct(&mut self, toggle: bool) -> bool {
        let q_prev = (self.state_history & 1) != 0;
        let new_q = q_prev ^ toggle;
        self.state_history = (self.state_history << 1) | (new_q as u64);
        new_q
    }
}

// ============================================================================
// SECTION 9: MULTI-SCALE TEMPORAL PYRAMID (H-BPTN)
// ============================================================================

/// SWAR intra-word dyadic compression operator (Definition 9.2).
///
/// Merges each adjacent bit pair via boolean disjunction (x[2k] | x[2k+1])
/// and compacts the result into the lower 32 bits in O(log W) time without loops.
#[inline(always)]
pub const fn swar_compress_dyadic_u64(x: u64) -> u32 {
    let z0 = (x | (x >> 1)) & 0x5555_5555_5555_5555_u64;
    let z1 = (z0 | (z0 >> 1)) & 0x3333_3333_3333_3333_u64;
    let z2 = (z1 | (z1 >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_u64;
    let z3 = (z2 | (z2 >> 4)) & 0x00FF_00FF_00FF_00FF_u64;
    let z4 = (z3 | (z3 >> 8)) & 0x0000_FFFF_0000_FFFF_u64;
    let z5 = (z4 | (z4 >> 16)) & 0x0000_0000_FFFF_FFFF_u64;
    z5 as u32
}

/// SWAR majority-vote temporal decimation operator (*Majority-Vote Pooling* 8x8) (Section 1.2).
///
/// Divides the 64-bit word into 8 temporal slices of 8 bits each.
/// Each slice $k \in [0, 7]$ is summed in parallel without conditional branching.
/// Output bit $k$ is set to 1 if and only if the spike count in the slice
/// is greater than or equal to majority threshold $\theta_{\text{filter}} \in [1, 8]$.
///
/// Filters dense background noise without saturating higher scales as `OR` would.
#[inline(always)]
pub const fn swar_majority_pool_8x8(x: u64, threshold: u8) -> u8 {
    let th = if threshold == 0 {
        1
    } else if threshold > 8 {
        8
    } else {
        threshold
    };

    // 1. Bit-parallel intra-byte sum
    let x0 = x.wrapping_sub((x >> 1) & 0x5555_5555_5555_5555_u64);
    let x1 = (x0 & 0x3333_3333_3333_3333_u64) + ((x0 >> 2) & 0x3333_3333_3333_3333_u64);
    let byte_counts = (x1 + (x1 >> 4)) & 0x0F0F_0F0F_0F0F_0F0F_u64;

    // 2. Thresholding shift: add (128 - th) so bit 7 activates iff count >= th
    let offset = (128u64.wrapping_sub(th as u64)) & 0xFF;
    let offset_word = offset.wrapping_mul(0x0101_0101_0101_0101_u64);

    let res = byte_counts.wrapping_add(offset_word) & 0x8080_8080_8080_8080_u64;

    // 3. Branchless compaction of 8 decision bits into u8
    let b0 = ((res >> 7) & 1) as u8;
    let b1 = ((res >> 14) & 2) as u8;
    let b2 = ((res >> 21) & 4) as u8;
    let b3 = ((res >> 28) & 8) as u8;
    let b4 = ((res >> 35) & 16) as u8;
    let b5 = ((res >> 42) & 32) as u8;
    let b6 = ((res >> 49) & 64) as u8;
    let b7 = ((res >> 56) & 128) as u8;

    b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7
}

/// 2-bit Smith saturating automaton evaluated simultaneously on 64 parallel channels (HS-BPTN).
///
/// Models 64 saturating 4-state branch predictors without conditional jumps:
/// `00` (Strongly Inactive), `01` (Weakly Inactive), `10` (Weakly Active), `11` (Strongly Active).
///
/// Returns `(next_c1, next_c0)`. Active state corresponds to `next_c1` ($\text{state} \ge 2$).
#[inline(always)]
pub const fn bptn_parallel_smith_step_u64(c1: u64, c0: u64, inc: u64, dec: u64) -> (u64, u64) {
    // Symmetric cancellation in case of simultaneous excitation/inhibition co-activation
    let eff_inc = inc & !dec;
    let eff_dec = dec & !inc;
    let none = !(eff_inc | eff_dec);
    let next_c1 = (none & c1) | (eff_inc & (c1 | c0)) | (eff_dec & (c1 & c0));
    let next_c0 = (none & c0) | (eff_inc & (!c0 | c1)) | (eff_dec & (c1 & !c0));
    (next_c1, next_c0)
}

/// Multi-scale cascade history buffer in dyadic/b-adic layout (Definition 9.1, Lemma 9.1).
///
/// `SCALES`: number of temporal resolution levels ($L \ge 1$).
/// `STRIDE_POWER`: scaling factor $b = 2^m$ where $m = \text{STRIDE\_POWER}$ (e.g., 2 for $b=4$, 3 for $b=8$).
///
/// Maintains history via disjunctive aggregation (Max-Pooling) without spike loss (Theorem 9.1).
/// Covers an effective horizon of $W \cdot b^{L-1} = 64 \cdot 2^{(L-1) \cdot m}$ time steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiScalePyramidBuffer<const SCALES: usize, const STRIDE_POWER: usize = 2> {
    /// History registers for each scale (each of W=64 bits).
    pub registers: [u64; SCALES],
    /// Intra-window spike accumulators for robust threshold/majority temporal pooling.
    pub window_counts: [u8; SCALES],
}

impl<const SCALES: usize, const STRIDE_POWER: usize> Default
    for MultiScalePyramidBuffer<SCALES, STRIDE_POWER>
{
    #[inline(always)]
    fn default() -> Self {
        Self {
            registers: [0; SCALES],
            window_counts: [0; SCALES],
        }
    }
}

impl<const SCALES: usize, const STRIDE_POWER: usize> MultiScalePyramidBuffer<SCALES, STRIDE_POWER> {
    /// Initializes an empty multi-scale buffer.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            registers: [0; SCALES],
            window_counts: [0; SCALES],
        }
    }

    /// Resets all scales to zero.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.registers = [0; SCALES];
        self.window_counts = [0; SCALES];
    }

    /// Continuous maximal effective temporal horizon covered by the buffer (in time steps).
    #[inline(always)]
    pub const fn effective_horizon(&self) -> usize {
        if SCALES == 0 {
            0
        } else {
            let shift = (SCALES - 1) * STRIDE_POWER;
            if shift >= 58 { usize::MAX } else { 64 << shift }
        }
    }

    /// Updates the entire temporal pyramid in a 100% branchless fashion (Lemma 9.1).
    ///
    /// `spike`: current spike at time t ($s(t) \in \{0, 1\}$).
    /// `t`: global time step index ($t \in \mathbb{N}$).
    #[inline(always)]
    pub fn step(&mut self, spike: bool, t: u64) {
        let spike_u64 = spike as u64;

        // Scale 0 (finest step-by-step resolution, shifted at every step)
        if SCALES > 0 {
            self.registers[0] = (self.registers[0] << 1) | spike_u64;
        }

        // Scales 1 .. SCALES-1
        let mut l = 1;
        while l < SCALES {
            let shift_amount = l * STRIDE_POWER;
            let mask = if shift_amount >= 64 {
                u64::MAX
            } else {
                (1u64 << shift_amount) - 1
            };
            // sigma_l(t) = 1 if t is multiple of 2^(l * STRIDE_POWER), else 0
            let sigma = ((t & mask) == 0) as u8;

            // Lemma 9.1 equation: w_l(t) = (w_l(t-1) << sigma) | (spike & 1)
            self.registers[l] = (self.registers[l] << sigma) | spike_u64;
            l += 1;
        }
    }

    /// Updates the temporal pyramid with threshold-based temporal pooling (*Count-based / Threshold Pooling*).
    ///
    /// Resolves saturation of raw `OR` disjunctive pooling in noisy environments:
    /// rather than propagating a 1 bit on an isolated noise spike,
    /// a 1 bit is only emitted if spike count in the window reaches `min_spikes_threshold`.
    #[inline(always)]
    pub fn step_threshold(&mut self, spike: bool, t: u64, min_spikes_threshold: u8) {
        let spike_u64 = spike as u64;

        if SCALES > 0 {
            self.registers[0] = (self.registers[0] << 1) | spike_u64;
        }

        let mut l = 1;
        while l < SCALES {
            let shift_amount = l * STRIDE_POWER;
            let mask = if shift_amount >= 64 {
                u64::MAX
            } else {
                (1u64 << shift_amount) - 1
            };
            let is_window_end = (t & mask) == 0;

            if spike {
                self.window_counts[l] = self.window_counts[l].saturating_add(1);
            }

            if is_window_end {
                let emitted = (self.window_counts[l] >= min_spikes_threshold) as u64;
                self.registers[l] = (self.registers[l] << 1) | emitted;
                self.window_counts[l] = 0;
            }

            l += 1;
        }
    }

    /// Updates the temporal pyramid with temporal majority pooling (*Majority Pooling*).
    ///
    /// Sets detection threshold to half window size: $\text{threshold}_\ell = \max(1, 2^{(\ell \cdot m) - 1})$.
    #[inline(always)]
    pub fn step_majority(&mut self, spike: bool, t: u64) {
        let spike_u64 = spike as u64;

        if SCALES > 0 {
            self.registers[0] = (self.registers[0] << 1) | spike_u64;
        }

        let mut l = 1;
        while l < SCALES {
            let shift_amount = l * STRIDE_POWER;
            let mask = if shift_amount >= 64 {
                u64::MAX
            } else {
                (1u64 << shift_amount) - 1
            };
            let is_window_end = (t & mask) == 0;

            if spike {
                self.window_counts[l] = self.window_counts[l].saturating_add(1);
            }

            if is_window_end {
                let window_size = 1u64 << shift_amount;
                let majority_threshold = (window_size / 2).clamp(1, 255) as u8;
                let emitted = (self.window_counts[l] >= majority_threshold) as u64;
                self.registers[l] = (self.registers[l] << 1) | emitted;
                self.window_counts[l] = 0;
            }

            l += 1;
        }
    }

    /// Accesses history word $W=64$ of given scale $\ell \in [0, \text{SCALES}-1]$.
    #[inline(always)]
    pub const fn scale_word(&self, scale: usize) -> u64 {
        if scale < SCALES {
            self.registers[scale]
        } else {
            0
        }
    }
}

/// Multi-scale temporal neuron H-BPTN (Definition 9.3).
///
/// Integrates $M$ afferent synaptic channels across $L$ hierarchical temporal scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiScaleBptnNeuron<const CHANNELS: usize, const SCALES: usize> {
    /// Synaptic kernel masks for each scale and channel: `[SCALES][CHANNELS]`.
    pub kernels: [[u64; CHANNELS]; SCALES],
    /// Synaptic polarity masks for each scale and channel: `[SCALES][CHANNELS]`.
    pub polarities: [[u64; CHANNELS]; SCALES],
    /// Global integer firing threshold.
    pub threshold: u32,
}

impl<const CHANNELS: usize, const SCALES: usize> MultiScaleBptnNeuron<CHANNELS, SCALES> {
    /// Initializes multi-scale H-BPTN neuron.
    #[inline(always)]
    pub const fn new(
        kernels: [[u64; CHANNELS]; SCALES],
        polarities: [[u64; CHANNELS]; SCALES],
        threshold: u32,
    ) -> Self {
        Self {
            kernels,
            polarities,
            threshold,
        }
    }

    /// Evaluates multi-scale membrane potential $V_H(t)$ (Definition 9.3).
    #[inline(always)]
    pub fn evaluate_potential<const STRIDE_POWER: usize>(
        &self,
        buffers: &[MultiScalePyramidBuffer<SCALES, STRIDE_POWER>; CHANNELS],
    ) -> u32 {
        let mut total_potential: u32 = 0;
        let mut l = 0;
        while l < SCALES {
            let mut ch = 0;
            while ch < CHANNELS {
                let w = buffers[ch].scale_word(l);
                let k = self.kernels[l][ch];
                let p = self.polarities[l][ch];
                let active = (w ^ p) & k;
                total_potential += active.count_ones();
                ch += 1;
            }
            l += 1;
        }
        total_potential
    }

    /// Evaluates firing of multi-scale neuron at current time: y(t) = V_H(t) >= theta.
    #[inline(always)]
    pub fn tick<const STRIDE_POWER: usize>(
        &self,
        buffers: &[MultiScalePyramidBuffer<SCALES, STRIDE_POWER>; CHANNELS],
    ) -> bool {
        self.evaluate_potential(buffers) >= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_shift_operators() {
        let x: u64 = 0b0000_1000; // Spike at step k=3
        assert_eq!(shift_temporal_delay(x, 1), 0b0000_0100); // Becomes k=2
        assert_eq!(shift_temporal_delay(x, 3), 0b0000_0001); // Becomes k=0 (present)
        assert_eq!(shift_temporal_delay(x, 4), 0b0000_0000);
        assert_eq!(shift_temporal_delay(x, 64), 0);

        assert_eq!(shift_temporal_age(x, 1), 0b0001_0000); // Becomes k=4
    }

    #[test]
    fn test_morphological_dilation() {
        // Single spike at position k=4
        let x: u64 = 1 << 4;
        // With kappa = 1, dilation covers k=3, k=4, k=5
        let dilated = dilate_temporal_jitter(x, 1);
        let expected = (1 << 3) | (1 << 4) | (1 << 5);
        assert_eq!(dilated, expected);

        // With kappa = 0, identity
        assert_eq!(dilate_temporal_jitter(x, 0), x);
    }

    #[test]
    fn test_bptn_synapse_coincidence_and_polarity() {
        // Source with spike at t-2
        let history: u64 = 1 << 2;

        // Excitatory synapse with delay 2 (brings spike to k=0)
        let syn_exc = BptnSynapse::new(2, 0, u64::MAX, 0, 1 << 0);
        assert_eq!(syn_exc.compute_contribution(history), 1);

        // If input has no spike at t-2, zero contribution
        assert_eq!(syn_exc.compute_contribution(0), 0);

        // Synapse with inverted polarity (detects ABSENCE of spike at t-2)
        let syn_inh = BptnSynapse::new(2, 0, u64::MAX, 1 << 0, 1 << 0);
        assert_eq!(syn_inh.compute_contribution(0), 1); // 0 ^ 1 = 1 -> contribution = 1
        assert_eq!(syn_inh.compute_contribution(history), 0); // 1 ^ 1 = 0 -> contribution = 0
    }

    #[test]
    fn test_compact_bptn_neuron_xor_parity() {
        // Test BPTN neuron computing condition across 2 channels at t=0:
        // Detects (Channel 0 active AND Channel 1 inactive)
        // Channel 0: excitatory (polarity=0, kernel=1)
        // Channel 1: inhibitory (polarity=1, kernel=1)
        // Threshold theta = 2 (both conditions must be satisfied)
        let neuron = CompactBptnNeuron::<2>::new([1, 1], [0, 1], 2);

        // Case 1: ch0=1, ch1=0 -> match!
        assert!(neuron.tick(&[1, 0]));

        // Case 2: ch0=1, ch1=1 -> failure (ch1 active)
        assert!(!neuron.tick(&[1, 1]));

        // Case 3: ch0=0, ch1=0 -> failure (ch0 inactive)
        assert!(!neuron.tick(&[0, 0]));
    }

    #[test]
    fn test_recurrent_bptn_nlfsr_and_history_framing() {
        let mut buffer = TemporalFramingBuffer64::new();
        buffer.push_spike(true); // t=0: spike
        buffer.push_spike(false); // t=1: no spike
        buffer.push_spike(true); // t=2: spike

        // In window: t=2 is at bit 0 (true), t=1 is at bit 1 (false), t=0 is at bit 2 (true)
        assert!(buffer.get_lag(0));
        assert!(!buffer.get_lag(1));
        assert!(buffer.get_lag(2));
        assert_eq!(buffer.window() & 0b111, 0b101);

        // Recurrent layer of 2 neurons with 1 external input
        // Neuron 0: activates if external input active
        // Neuron 1: activates if Neuron 0 fired in previous step (lag 1 on recurrent history)
        let n0 = CompactBptnNeuron::<3>::new([1, 0, 0], [0, 0, 0], 1);
        let n1 = CompactBptnNeuron::<3>::new([0, 1 << 0, 0], [0, 0, 0], 1);

        let mut r_layer = RBptnLayer::<2, 1, 3>::new([n0, n1]);

        // Step 1: external pulse
        let spikes_t1 = r_layer.step(&[1]);
        assert_eq!(spikes_t1 & 1, 1); // Neuron 0 fired
        assert_eq!((spikes_t1 >> 1) & 1, 0); // Neuron 1 hasn't fired yet

        // Step 2: external silence
        let spikes_t2 = r_layer.step(&[0]);
        assert_eq!(spikes_t2 & 1, 0); // Neuron 0 silent
        assert_eq!((spikes_t2 >> 1) & 1, 1); // Neuron 1 fired due to past spike of Neuron 0!
    }

    #[test]
    fn test_bistable_latch_infinite_retention() {
        // Test Set-Reset synchronous latch with Reset priority (Lemma 8.1, Theorem 8.1)
        let mut latch = BptnBistableCell::new(BptnLatchMode::ResetDominant);
        assert!(!latch.current_state());

        // 1. Write (Set = true, Reset = false) -> state transitions to true
        assert!(latch.step(true, false));
        assert!(latch.current_state());

        // 2. Silent retention for 10,000 cycles (Set = false, Reset = false)
        // Verifies ZERO state leak or drift (Theorem 8.1)
        for _ in 0..10_000 {
            assert!(latch.step(false, false));
        }
        assert!(latch.current_state());

        // 3. Clear (Set = false, Reset = true) -> state transitions to false
        assert!(!latch.step(false, true));
        assert!(!latch.current_state());

        // 4. Retention of state 0 for 1,000 cycles
        for _ in 0..1_000 {
            assert!(!latch.step(false, false));
        }
        assert!(!latch.current_state());

        // 5. Conflict Set + Reset: Reset priority -> must remain false
        assert!(!latch.step(true, true));
        assert!(!latch.current_state());

        // 6. Test latch with Set priority
        let mut set_latch = BptnBistableCell::new(BptnLatchMode::SetDominant);
        // Conflict Set + Reset: Set priority -> must be true
        assert!(set_latch.step(true, true));
        assert!(set_latch.current_state());
    }

    #[test]
    fn test_toggle_latch_temporal_parity() {
        // Test T-Latch resolving temporal parity over infinite horizon
        let mut t_latch = BptnToggleCell::new();
        assert!(!t_latch.current_state());

        // Event at t=1: toggle -> 1
        assert!(t_latch.step(true));

        // Silence for 500 time steps (delay Delta = 500 >> 64)
        for _ in 0..500 {
            assert!(t_latch.step(false));
        }
        assert!(t_latch.current_state());

        // Event at t=502: toggle -> 0
        assert!(!t_latch.step(true));

        // Silence for 200 time steps
        for _ in 0..200 {
            assert!(!t_latch.step(false));
        }
        assert!(!t_latch.current_state());
    }

    #[test]
    fn test_gated_d_latch() {
        let mut d_latch = BptnGatedCell::new();

        // Enable = false -> state remains 0 even if data = true
        assert!(!d_latch.step(true, false));
        assert!(!d_latch.current_state());

        // Enable = true, Data = true -> capture 1
        assert!(d_latch.step(true, true));
        assert!(d_latch.current_state());

        // Enable = false, Data = false -> retain 1
        assert!(d_latch.step(false, false));
        assert!(d_latch.current_state());

        // Enable = true, Data = false -> capture 0
        assert!(!d_latch.step(false, true));
        assert!(!d_latch.current_state());
    }

    #[test]
    fn test_swar_dyadic_compression() {
        // Pair (1, 0) -> 1
        let x1: u64 = 0b01;
        assert_eq!(swar_compress_dyadic_u64(x1), 0b1);

        // Pair (0, 1) -> 1
        let x2: u64 = 0b10;
        assert_eq!(swar_compress_dyadic_u64(x2), 0b1);

        // Pair (0, 0) -> 0
        let x3: u64 = 0b00;
        assert_eq!(swar_compress_dyadic_u64(x3), 0b0);

        // Alternating pattern: 0b10_01_00_11 -> 4 pairs: (1,0)=1, (0,1)=1, (0,0)=0, (1,1)=1 -> 0b1101
        let x4: u64 = 0b10_01_00_11;
        assert_eq!(swar_compress_dyadic_u64(x4), 0b1101);
    }

    #[test]
    fn test_multi_scale_pyramid_buffer_anti_aliasing_and_horizon() {
        // 4 scales with stride factor b=4 (STRIDE_POWER = 2)
        // Scale 0: 64 steps (resolution 1 tick)
        // Scale 1: 256 steps (resolution 4 ticks)
        // Scale 2: 1024 steps (resolution 16 ticks)
        // Scale 3: 4096 steps (resolution 64 ticks)
        let mut pyramid = MultiScalePyramidBuffer::<4, 2>::new();
        assert_eq!(pyramid.effective_horizon(), 4096);

        // Inject isolated single spike at t = 10 (without any other spike)
        for t in 0..150 {
            let is_spike = t == 10;
            pyramid.step(is_spike, t);
        }

        // At t = 149:
        // Spike occurring Delta = 149 - 10 = 139 ticks ago:
        // - Has left scale 0 (since 139 >= 64, scale 0 bounded to 64 steps)
        assert_eq!(pyramid.scale_word(0) & (1 << 10), 0);

        // - But it IS STRICTLY PRESENT in scale 1!
        // At scale 1 (4-tick grain), 139 / 4 = 34.
        // Bit 34 of scale_word(1) MUST be 1 (Theorem 9.1: Anti-aliasing / zero false negatives)
        let w1 = pyramid.scale_word(1);
        assert_ne!(w1, 0, "Scale 1 must have captured the spike without loss");

        // - It is also present in scale 2 (16-tick grain, 139 / 16 = 8)
        let w2 = pyramid.scale_word(2);
        assert_ne!(w2, 0, "Scale 2 must have captured the spike without loss");
    }

    #[test]
    fn test_multi_scale_bptn_neuron_coincidence() {
        // H-BPTN neuron with 1 channel and 2 scales
        // Scale 0: expects recent spike at k=2 (2 ticks ago)
        // Scale 1: expects past activity at k=10 (~40 ticks ago)
        // Threshold theta = 2 (must observe coincidence across both scales)
        let mut kernels = [[0u64; 1]; 2];
        kernels[0][0] = 1 << 2;
        kernels[1][0] = 1 << 10;
        let polarities = [[0u64; 1]; 2];

        let neuron = MultiScaleBptnNeuron::<1, 2>::new(kernels, polarities, 2);

        let mut buffer = MultiScalePyramidBuffer::<2, 2>::new();

        // 1. Only scale 0 active -> potential = 1 < 2 -> false
        buffer.registers[0] = 1 << 2;
        buffer.registers[1] = 0;
        assert!(!neuron.tick(&[buffer]));

        // 2. Both scales coincide -> potential = 2 >= 2 -> true!
        buffer.registers[1] = 1 << 10;
        assert!(neuron.tick(&[buffer]));
    }

    #[test]
    fn test_bptn_spiking_network_trait_integration() {
        use crate::application::readout::StaticReadoutLayer;
        use crate::domain::numeric::Fix32;
        use crate::traits::SpikingNetwork;

        // Network of 2 BPTN neurons receiving 2 input channels
        // Neuron 0 detects channel 0 (kernel = 1)
        // Neuron 1 detects channel 1 (kernel = 1)
        let n0 = CompactBptnNeuron::<2>::new([1, 0], [0, 0], 1);
        let n1 = CompactBptnNeuron::<2>::new([0, 1], [0, 0], 1);
        let layer = BptnFeedforwardLayer::new([n0, n1]);

        let mut net = BptnSpikingNetwork::new(layer).with_input_threshold(Fix32::from_num(0.5));

        // Tick 1: continuous input [0.8, 0.2] -> only channel 0 exceeds 0.5
        let spikes = net.tick(&[Fix32::from_num(0.8), Fix32::from_num(0.2)]);
        assert!(spikes[0]);
        assert!(!spikes[1]);

        // Tick 2: continuous input [0.1, 0.9] -> only channel 1 exceeds 0.5
        let words = net.tick_inputs_words(&[Fix32::from_num(0.1), Fix32::from_num(0.9)]);
        assert_eq!(words[0], 0b10); // bit 1 is 1 (neuron 1)

        // Test with direct bitset words
        // prev_spikes_words: bit 0 and bit 1 active (0b11)
        let out_words = net.tick_spikes_words(&[0b11]);
        assert_eq!(out_words[0], 0b11); // both neurons fire

        // Reset
        net.reset();
        let words_after_reset = net.tick_inputs_words(&[Fix32::ZERO, Fix32::ZERO]);
        assert_eq!(words_after_reset[0], 0);

        // Chaining with StaticReadoutLayer
        static WEIGHTS: [[Fix32; 3]; 2] = [
            [
                Fix32::ZERO,
                Fix32::from_bits(2 << 16),
                Fix32::from_bits(-1 << 16),
            ], // Class 0
            [
                Fix32::ZERO,
                Fix32::from_bits(-1 << 16),
                Fix32::from_bits(2 << 16),
            ], // Class 1
        ];
        let readout = StaticReadoutLayer::<2, 3, Fix32>::new(&WEIGHTS);

        // If neuron 0 fires (mask = 1) -> class 0
        let pred0 = readout.predict_spikes_mask(1);
        assert_eq!(pred0, 0);

        // If neuron 1 fires (mask = 2) -> class 1
        let pred1 = readout.predict_spikes_mask(2);
        assert_eq!(pred1, 1);
    }

    #[test]
    fn test_rbptn_spiking_network_trait_integration() {
        use crate::domain::numeric::Fix32;
        use crate::traits::SpikingNetwork;

        // 1 recurrent neuron, 1 external channel, 2 total channels (1 ext + 1 rec)
        // Self-excitation / bistable retention:
        // Channel 0 = external input (k=0)
        // Channel 1 = recurrent return (k=0)
        // Threshold theta = 1: spike on input OR recurrence triggers firing
        let n0 = CompactBptnNeuron::<2>::new([1, 1], [0, 0], 1);
        let layer = RBptnLayer::<1, 1, 2>::new([n0]);

        let mut rnet = RBptnSpikingNetwork::new(layer).with_input_threshold(Fix32::from_num(0.5));

        // Tick 1: no external spike -> remains inactive (0)
        let spikes = rnet.tick(&[Fix32::ZERO]);
        assert!(!spikes[0]);

        // Tick 2: external spike (1.0 > 0.5) -> neuron activates (1)
        let spikes = rnet.tick(&[Fix32::from_num(1.0)]);
        assert!(spikes[0]);

        // Tick 3: no external input -> thanks to recurrence on Channel 1, neuron REMAINS at 1!
        let spikes = rnet.tick(&[Fix32::ZERO]);
        assert!(spikes[0]);

        // Tick 4: still no external input -> infinite retention
        let spikes = rnet.tick(&[Fix32::ZERO]);
        assert!(spikes[0]);

        // Test reset -> full shutdown
        rnet.reset();
        let words = rnet.tick_inputs_words(&[Fix32::ZERO]);
        assert_eq!(words[0], 0);
    }

    #[test]
    fn test_multiscale_threshold_noise_rejection() {
        // 2-scale buffer, stride power 2 (factor b = 4)
        let mut buffer_or = MultiScalePyramidBuffer::<2, 2>::new();
        let mut buffer_thresh = MultiScalePyramidBuffer::<2, 2>::new();

        // Inject 1 isolated noise spike during 4-tick window (t = 1)
        // t = 0 (start)
        buffer_or.step(false, 0);
        buffer_thresh.step_threshold(false, 0, 2);

        // t = 1 (noise spike)
        buffer_or.step(true, 1);
        buffer_thresh.step_threshold(true, 1, 2);

        // t = 2 (silence)
        buffer_or.step(false, 2);
        buffer_thresh.step_threshold(false, 2, 2);

        // t = 3 (silence)
        buffer_or.step(false, 3);
        buffer_thresh.step_threshold(false, 3, 2);

        // t = 4 (window end, multiple of 4)
        buffer_or.step(false, 4);
        buffer_thresh.step_threshold(false, 4, 2);

        // On raw OR operator: noise was shifted into scale 1 (bit 1 = 1)
        assert_eq!(
            (buffer_or.registers[1] >> 1) & 1,
            1,
            "Raw OR pooling saturates under noise"
        );

        // On threshold operator (threshold = 2): 1 spike < 2 -> noise rejected (bit 1 = 0)!
        assert_eq!(
            (buffer_thresh.registers[1] >> 1) & 1,
            0,
            "Threshold pooling rejects isolated noise"
        );

        // Now inject true burst of 2 spikes (t = 5, t = 6)
        buffer_thresh.step_threshold(true, 5, 2);
        buffer_thresh.step_threshold(true, 6, 2);
        buffer_thresh.step_threshold(false, 7, 2);
        buffer_thresh.step_threshold(false, 8, 2); // window 2 end

        // True burst reaches threshold of 2 -> bit = 1!
        assert_eq!(
            buffer_thresh.registers[1] & 1,
            1,
            "Valid burst is properly detected"
        );
    }

    #[test]
    fn test_morphological_erosion_and_opening() {
        // Binary pattern: isolated spike (noise) at pos 4, and packet of 3 spikes at pos 10, 11, 12
        let noisy = (1u64 << 4) | (1u64 << 10) | (1u64 << 11) | (1u64 << 12);

        // Order-1 erosion (3-bit support): only center of packet (pos 11) survives
        let eroded = erode_temporal_noise(noisy, 1);
        assert_eq!(
            eroded,
            1u64 << 11,
            "Erosion must remove isolated peak and borders"
        );

        // Order-1 opening: eliminates isolated peak (noise) and restores packet (pos 10, 11, 12)
        let opened = open_temporal_filter(noisy, 1);
        let expected_packet = (1u64 << 10) | (1u64 << 11) | (1u64 << 12);
        assert_eq!(
            opened, expected_packet,
            "Opening removes isolated noise and preserves packet"
        );
    }

    #[test]
    fn test_hysteresis_and_robust_toggle() {
        // Test BptnHysteresisLatch
        let mut latch = BptnHysteresisLatch::new();
        assert_eq!(latch.state, 0);
        assert!(!latch.current_state());

        // Single Set pulse: enters state 1 (Weakly Inactive), output remains false
        latch.step(true, false);
        assert_eq!(latch.state, 1);
        assert!(!latch.current_state());

        // 2nd Set pulse: enters state 2 (Weakly Active), output becomes true!
        latch.step(true, false);
        assert_eq!(latch.state, 2);
        assert!(latch.current_state());

        // 3rd Set pulse: saturates at state 3 (Strongly Active)
        latch.step(true, false);
        assert_eq!(latch.state, 3);
        assert!(latch.current_state());

        // 1 isolated Reset pulse: steps down to 2, output remains true!
        latch.step(false, true);
        assert_eq!(latch.state, 2);
        assert!(latch.current_state());

        // Test BptnRobustToggleCell (anti-glitch)
        let mut toggle = BptnRobustToggleCell::new();
        assert!(!toggle.current_state());

        // Isolated noise spike (1 tick): step(true) then step(false)
        toggle.step(true);
        assert!(
            !toggle.current_state(),
            "Isolated 1-tick peak should not toggle parity"
        );
        toggle.step(false);
        assert!(!toggle.current_state());

        // True confirmed pulse (2 consecutive trigger ticks)
        toggle.step(true);
        toggle.step(true);
        assert!(
            toggle.current_state(),
            "Confirmed 2-tick trigger toggles parity!"
        );
    }

    #[test]
    fn test_swar_majority_pool_8x8() {
        // Byte 0: 0b0000_0011 (2 ones) -> >= 2 ? Yes -> bit 0 = 1
        // Byte 1: 0b0000_0001 (1 one)  -> >= 2 ? No  -> bit 1 = 0
        // Byte 2: 0b0000_1111 (4 ones) -> >= 2 ? Yes -> bit 2 = 1
        // Byte 3: 0b0000_0000 (0 ones) -> >= 2 ? No  -> bit 3 = 0
        // Bytes 4..7: 0
        let x = 0x03u64 | (0x01u64 << 8) | (0x0Fu64 << 16);
        let pooled = swar_majority_pool_8x8(x, 2);
        assert_eq!(pooled, 0b0000_0101);

        // Test threshold 3 on byte 2 (4 ones -> active) and byte 0 (2 ones -> inactive)
        let pooled_th3 = swar_majority_pool_8x8(x, 3);
        assert_eq!(pooled_th3, 0b0000_0100);
    }

    #[test]
    fn test_parallel_smith_counter_u64() {
        // 64 automata in parallel
        let mut c1 = 0u64;
        let mut c0 = 0u64;

        // Increment on channel 0 (1 tick)
        let inc = 1u64;
        let dec = 0u64;
        let (n_c1, n_c0) = bptn_parallel_smith_step_u64(c1, c0, inc, dec);
        c1 = n_c1;
        c0 = n_c0;
        // State 1 (01) -> output c1 = 0
        assert_eq!(c1 & 1, 0);
        assert_eq!(c0 & 1, 1);

        // 2nd increment on channel 0 -> State 2 (10) -> output c1 = 1
        let (n_c1, n_c0) = bptn_parallel_smith_step_u64(c1, c0, inc, dec);
        c1 = n_c1;
        c0 = n_c0;
        assert_eq!(c1 & 1, 1);
        assert_eq!(c0 & 1, 0);

        // 3rd increment -> State 3 (11, saturated)
        let (n_c1, n_c0) = bptn_parallel_smith_step_u64(c1, c0, inc, dec);
        c1 = n_c1;
        c0 = n_c0;
        assert_eq!(c1 & 1, 1);
        assert_eq!(c0 & 1, 1);

        // 4th increment -> Saturation at 3 (11)
        let (n_c1, n_c0) = bptn_parallel_smith_step_u64(c1, c0, inc, dec);
        c1 = n_c1;
        c0 = n_c0;
        assert_eq!(c1 & 1, 1);
        assert_eq!(c0 & 1, 1);

        // Decrement on channel 0 -> State 2 (10)
        let (n_c1, n_c0) = bptn_parallel_smith_step_u64(c1, c0, 0, 1);
        c1 = n_c1;
        c0 = n_c0;
        assert_eq!(c1 & 1, 1);
        assert_eq!(c0 & 1, 0);
    }
}
