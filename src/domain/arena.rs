//! 🏟️ Multilayer Shared Memory Manager (`ActivationArena` & Memory Arena Planner).
//!
//! In deep spiking neural networks (Deep SNNs), allocating dedicated accumulation
//! buffers and spike bitmasks for every layer results in massive SRAM waste
//! proportional to the layer count: $\sum_l \text{SRAM}(L_l)$.
//!
//! This module implements the **Memory Arena Planner**:
//! 1. **`ActivationArena`**: Shared workspace sized to the largest layer:
//!    $$\text{SRAM}_{\text{accumulators}} = \max_l (N_l) \times \text{size}(T)$$
//! 2. **Double Ping-Pong Buffer**: Two bitmasks of size $\max_l (\text{words}_l)$
//!    alternate between consecutive layers, avoiding allocating a `SpikeBitset` per layer.
//! 3. **`ArenaLifPopulation` & `ArenaLif16Population`**: Lightweight populations reduced
//!    strictly to persistent states (5 bytes/neuron in Fix32, 3 bytes/neuron in Fix16).

use crate::domain::models::lif::{Lif16Config, LifConfig, ResetMode};
use crate::domain::numeric::{Fix16, Fix32, NumericScalar};
use crate::traits::{SpikingNetwork, SynapseMatrix};
use core::marker::PhantomData;

/// Shared memory arena for multilayer neuromorphic inference (Memory Arena Planner).
///
/// Enables SRAM sharing across all consecutive layers of a deep network:
/// - `accumulators`: Shared input accumulation buffer, sized to $\max_l(N_l)$ instead of $\sum_l N_l$.
/// - `ping_words` & `pong_words`: Ping-Pong spike double buffer avoiding per-layer `SpikeBitset` allocations.
///
/// # Example
/// ```rust
/// use snn_core::domain::arena::ActivationArena;
/// use snn_core::Fix32;
///
/// let mut arena = ActivationArena::<64, 2, Fix32>::new();
/// arena.accumulators[0] = Fix32::from_num(1.2);
/// arena.ping_words[0] = 0b101;
///
/// arena.clear_accumulators();
/// assert_eq!(arena.accumulators[0], Fix32::ZERO);
/// assert_eq!(arena.ping_words[0], 0b101);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ActivationArena<
    const MAX_NEURONS: usize,
    const MAX_WORDS: usize,
    T: NumericScalar = Fix32,
> {
    pub accumulators: [T; MAX_NEURONS],
    pub ping_words: [u32; MAX_WORDS],
    pub pong_words: [u32; MAX_WORDS],
}

impl<const MAX_NEURONS: usize, const MAX_WORDS: usize, T: NumericScalar> Default
    for ActivationArena<MAX_NEURONS, MAX_WORDS, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_NEURONS: usize, const MAX_WORDS: usize, T: NumericScalar>
    ActivationArena<MAX_NEURONS, MAX_WORDS, T>
{
    /// Creates a new static arena initialized to zero.
    pub const fn new() -> Self {
        Self {
            accumulators: [T::ZERO; MAX_NEURONS],
            ping_words: [0u32; MAX_WORDS],
            pong_words: [0u32; MAX_WORDS],
        }
    }

    /// Resets the entire arena to zero.
    pub fn reset(&mut self) {
        self.accumulators.fill(T::ZERO);
        self.ping_words.fill(0);
        self.pong_words.fill(0);
    }

    /// Clears only the shared accumulator buffer.
    #[inline(always)]
    pub fn clear_accumulators(&mut self) {
        self.accumulators.fill(T::ZERO);
    }
}

/// Ultra-dense LIF neuron population operating on the shared arena (5 bytes per neuron in Fix32).
///
/// Contains **no persistent input accumulator arrays**, as these are
/// provided temporarily by the [`ActivationArena`] during tick evaluation.
#[derive(Clone, Copy, Debug)]
pub struct ArenaLifPopulation<const N: usize> {
    pub config: LifConfig,
    pub membrane_potentials: [Fix32; N],
    pub refractory_times_left: [u8; N],
    pub bias: Option<&'static [Fix32; N]>,
}

impl<const N: usize> ArenaLifPopulation<N> {
    /// Creates a new lightweight population initialized at rest.
    pub const fn new(config: LifConfig) -> Self {
        Self {
            config,
            membrane_potentials: [config.reset_potential; N],
            refractory_times_left: [0; N],
            bias: None,
        }
    }

    /// Attaches a constant static bias vector stored in Flash ROM.
    #[must_use]
    pub const fn with_bias(mut self, bias: &'static [Fix32; N]) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Number of neurons in the population.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        N
    }

    /// Indicates whether the population is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Resets membrane potentials and refractory timers.
    pub fn reset(&mut self) {
        self.membrane_potentials.fill(self.config.reset_potential);
        self.refractory_times_left.fill(0);
    }

    /// Evaluates population using the shared arena scratchpad
    /// and directly writes generated spikes into `out_words`.
    #[inline]
    pub fn step_arena(&mut self, accumulators: &mut [Fix32], out_words: &mut [u32]) {
        let max_n = N.min(accumulators.len());
        let words_len = out_words.len();
        out_words.fill(0);

        for i in 0..max_n {
            let refractory = &mut self.refractory_times_left[i];
            let bias = self
                .bias
                .map_or(self.config.bias, |b| b[i] + self.config.bias);
            let input = accumulators[i] + bias;
            accumulators[i] = Fix32::ZERO;

            if *refractory > 0 {
                *refractory -= 1;
                self.membrane_potentials[i] = self.config.reset_potential;
                continue;
            }

            #[cfg(feature = "cortex-m0")]
            let decayed = crate::domain::hardware::fast_decay_m0(
                self.membrane_potentials[i],
                (self.config.decay_factor.0 as u32 & 0xFFFF) as u16,
            );
            #[cfg(not(feature = "cortex-m0"))]
            let decayed = self.membrane_potentials[i] * self.config.decay_factor;
            let v = decayed + input;

            if v >= self.config.threshold {
                self.membrane_potentials[i] = match self.config.reset_mode {
                    ResetMode::Hard => self.config.reset_potential,
                    ResetMode::Subtract => v - self.config.threshold,
                };
                *refractory = self.config.refractory_period as u8;

                let word_idx = i / 32;
                let bit_idx = i % 32;
                if word_idx < words_len {
                    out_words[word_idx] |= 1 << bit_idx;
                }
            } else {
                self.membrane_potentials[i] = v;
            }
        }
    }
}

/// Ultra-dense 16-bit LIF neuron population operating on the shared arena (3 bytes per neuron in Fix16).
#[derive(Clone, Copy, Debug)]
pub struct ArenaLif16Population<const N: usize> {
    pub config: Lif16Config,
    pub membrane_potentials: [Fix16; N],
    pub refractory_times_left: [u8; N],
    pub bias: Option<&'static [Fix16; N]>,
}

impl<const N: usize> ArenaLif16Population<N> {
    pub const fn new(config: Lif16Config) -> Self {
        Self {
            config,
            membrane_potentials: [config.reset_potential; N],
            refractory_times_left: [0; N],
            bias: None,
        }
    }

    #[must_use]
    pub const fn with_bias(mut self, bias: &'static [Fix16; N]) -> Self {
        self.bias = Some(bias);
        self
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        N
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    pub fn reset(&mut self) {
        self.membrane_potentials.fill(self.config.reset_potential);
        self.refractory_times_left.fill(0);
    }

    #[inline]
    pub fn step_arena(&mut self, accumulators: &mut [Fix16], out_words: &mut [u32]) {
        let max_n = N.min(accumulators.len());
        let words_len = out_words.len();
        out_words.fill(0);

        for i in 0..max_n {
            let refractory = &mut self.refractory_times_left[i];
            let bias = self
                .bias
                .map_or(self.config.bias, |b| b[i] + self.config.bias);
            let input = accumulators[i] + bias;
            accumulators[i] = Fix16::ZERO;

            if *refractory > 0 {
                *refractory -= 1;
                self.membrane_potentials[i] = self.config.reset_potential;
                continue;
            }

            let decayed = self.membrane_potentials[i] * self.config.decay_factor;
            let sum_raw = (decayed.0 as i32) + (input.0 as i32);
            let v = Fix16(crate::domain::hardware::ssat_16(sum_raw) as i16);

            if v >= self.config.threshold {
                self.membrane_potentials[i] = match self.config.reset_mode {
                    ResetMode::Hard => self.config.reset_potential,
                    ResetMode::Subtract => v - self.config.threshold,
                };
                *refractory = self.config.refractory_period as u8;

                let word_idx = i / 32;
                let bit_idx = i % 32;
                if word_idx < words_len {
                    out_words[word_idx] |= 1 << bit_idx;
                }
            } else {
                self.membrane_potentials[i] = v;
            }
        }
    }
}

/// Abstract trait defining a neuromorphic layer operating on a shared `ActivationArena`.
pub trait ArenaSpikingLayer<T: NumericScalar = Fix32> {
    /// Number of output neurons in the layer.
    fn len(&self) -> usize;

    /// Returns `true` if the layer is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of 32-bit words required to encode output spikes.
    #[inline]
    fn words_len(&self) -> usize {
        self.len().div_ceil(32)
    }

    /// Resets internal states (membrane potentials, refractory timers).
    fn reset(&mut self);

    /// Processes a time step from continuous weighted inputs.
    fn forward_inputs(&mut self, inputs: &[T], accumulators: &mut [T], out_words: &mut [u32]);

    /// Processes a time step from incident bitset spikes.
    fn forward_spikes(&mut self, in_words: &[u32], accumulators: &mut [T], out_words: &mut [u32]);

    /// Executes one step in the Ping-Pong chain.
    ///
    /// - `input_is_ping`: Indicates if input spikes reside in `ping_words` (`true`) or `pong_words` (`false`).
    /// - Returns `true` if output spikes are written to `ping_words`, `false` if written to `pong_words`.
    fn forward_step(
        &mut self,
        input_is_ping: bool,
        accumulators: &mut [T],
        ping_words: &mut [u32],
        pong_words: &mut [u32],
    ) -> bool {
        let words_count = self.words_len();
        if input_is_ping {
            self.forward_spikes(ping_words, accumulators, &mut pong_words[..words_count]);
            false
        } else {
            self.forward_spikes(pong_words, accumulators, &mut ping_words[..words_count]);
            true
        }
    }

    /// Executes initial input step of the chain from continuous inputs.
    fn forward_inputs_step(
        &mut self,
        inputs: &[T],
        accumulators: &mut [T],
        ping_words: &mut [u32],
        _pong_words: &mut [u32],
    ) -> bool {
        let words_count = self.words_len();
        self.forward_inputs(inputs, accumulators, &mut ping_words[..words_count]);
        true
    }
}

/// Feedforward layer connected to the shared arena (zero persistent RAM buffers).
#[derive(Clone, Copy, Debug)]
pub struct ArenaFeedforwardLayer<P, SIn, const N: usize, T: NumericScalar = Fix32> {
    pub neurons: P,
    pub synapses: SIn,
    _marker: PhantomData<T>,
}

impl<P, SIn, const N: usize, T: NumericScalar> ArenaFeedforwardLayer<P, SIn, N, T> {
    pub const fn new(neurons: P, synapses: SIn) -> Self {
        Self {
            neurons,
            synapses,
            _marker: PhantomData,
        }
    }
}

impl<SIn: SynapseMatrix<Fix32>, const N: usize> ArenaSpikingLayer<Fix32>
    for ArenaFeedforwardLayer<ArenaLifPopulation<N>, SIn, N, Fix32>
{
    #[inline]
    fn len(&self) -> usize {
        N
    }

    #[inline]
    fn reset(&mut self) {
        self.neurons.reset();
    }

    fn forward_inputs(
        &mut self,
        inputs: &[Fix32],
        accumulators: &mut [Fix32],
        out_words: &mut [u32],
    ) {
        let acc_slice = &mut accumulators[..N];
        self.synapses.propagate_weighted_inputs(inputs, acc_slice);
        self.neurons.step_arena(acc_slice, out_words);
    }

    fn forward_spikes(
        &mut self,
        in_words: &[u32],
        accumulators: &mut [Fix32],
        out_words: &mut [u32],
    ) {
        let acc_slice = &mut accumulators[..N];
        self.synapses.propagate_spikes_words(in_words, acc_slice);
        self.neurons.step_arena(acc_slice, out_words);
    }
}

impl<SIn: SynapseMatrix<Fix16>, const N: usize> ArenaSpikingLayer<Fix16>
    for ArenaFeedforwardLayer<ArenaLif16Population<N>, SIn, N, Fix16>
{
    #[inline]
    fn len(&self) -> usize {
        N
    }

    #[inline]
    fn reset(&mut self) {
        self.neurons.reset();
    }

    fn forward_inputs(
        &mut self,
        inputs: &[Fix16],
        accumulators: &mut [Fix16],
        out_words: &mut [u32],
    ) {
        let acc_slice = &mut accumulators[..N];
        self.synapses.propagate_weighted_inputs(inputs, acc_slice);
        self.neurons.step_arena(acc_slice, out_words);
    }

    fn forward_spikes(
        &mut self,
        in_words: &[u32],
        accumulators: &mut [Fix16],
        out_words: &mut [u32],
    ) {
        let acc_slice = &mut accumulators[..N];
        self.synapses.propagate_spikes_words(in_words, acc_slice);
        self.neurons.step_arena(acc_slice, out_words);
    }
}

/// Chain of interconnected layers sharing the same memory arena via Ping-Pong.
#[derive(Clone, Copy, Debug)]
pub struct ArenaLayerChain<
    Head: ArenaSpikingLayer<T>,
    Tail: ArenaSpikingLayer<T>,
    T: NumericScalar = Fix32,
> {
    pub head: Head,
    pub tail: Tail,
    _marker: PhantomData<T>,
}

impl<Head: ArenaSpikingLayer<T>, Tail: ArenaSpikingLayer<T>, T: NumericScalar>
    ArenaLayerChain<Head, Tail, T>
{
    pub const fn new(head: Head, tail: Tail) -> Self {
        Self {
            head,
            tail,
            _marker: PhantomData,
        }
    }
}

impl<Head: ArenaSpikingLayer<T>, Tail: ArenaSpikingLayer<T>, T: NumericScalar> ArenaSpikingLayer<T>
    for ArenaLayerChain<Head, Tail, T>
{
    #[inline]
    fn len(&self) -> usize {
        self.tail.len()
    }

    fn reset(&mut self) {
        self.head.reset();
        self.tail.reset();
    }

    fn forward_inputs(&mut self, inputs: &[T], accumulators: &mut [T], out_words: &mut [u32]) {
        let head_words_len = self.head.words_len();
        assert!(
            head_words_len <= 64,
            "ArenaLayerChain: intermediate layer exceeds 64 words (2048 neurons). Use ArenaSpikingNetwork with ActivationArena to support arbitrary layer sizes."
        );
        let mut temp_ping = [0u32; 64];
        self.head
            .forward_inputs(inputs, accumulators, &mut temp_ping[..head_words_len]);
        self.tail
            .forward_spikes(&temp_ping[..head_words_len], accumulators, out_words);
    }

    fn forward_spikes(&mut self, in_words: &[u32], accumulators: &mut [T], out_words: &mut [u32]) {
        let head_words_len = self.head.words_len();
        assert!(
            head_words_len <= 64,
            "ArenaLayerChain: intermediate layer exceeds 64 words (2048 neurons). Use ArenaSpikingNetwork with ActivationArena to support arbitrary layer sizes."
        );
        let mut temp_ping = [0u32; 64];
        self.head
            .forward_spikes(in_words, accumulators, &mut temp_ping[..head_words_len]);
        self.tail
            .forward_spikes(&temp_ping[..head_words_len], accumulators, out_words);
    }

    fn forward_step(
        &mut self,
        input_is_ping: bool,
        accumulators: &mut [T],
        ping_words: &mut [u32],
        pong_words: &mut [u32],
    ) -> bool {
        let head_out_is_ping =
            self.head
                .forward_step(input_is_ping, accumulators, ping_words, pong_words);
        self.tail
            .forward_step(head_out_is_ping, accumulators, ping_words, pong_words)
    }

    fn forward_inputs_step(
        &mut self,
        inputs: &[T],
        accumulators: &mut [T],
        ping_words: &mut [u32],
        pong_words: &mut [u32],
    ) -> bool {
        let head_out_is_ping =
            self.head
                .forward_inputs_step(inputs, accumulators, ping_words, pong_words);
        self.tail
            .forward_step(head_out_is_ping, accumulators, ping_words, pong_words)
    }
}

/// Deep SNN optimized by Memory Arena Planner.
///
/// Implements [`SpikingNetwork`]. Zero dynamic allocation during inference,
/// with strict SRAM arena reuse across all layers.
#[derive(Clone, Copy, Debug)]
pub struct ArenaSpikingNetwork<
    Chain: ArenaSpikingLayer<T>,
    const MAX_NEURONS: usize,
    const MAX_WORDS: usize,
    const OUT_N: usize,
    T: NumericScalar = Fix32,
> {
    pub chain: Chain,
    pub arena: ActivationArena<MAX_NEURONS, MAX_WORDS, T>,
    pub output_spikes: [bool; OUT_N],
    last_output_is_ping: bool,
    _marker: PhantomData<T>,
}

impl<
    Chain: ArenaSpikingLayer<T>,
    const MAX_NEURONS: usize,
    const MAX_WORDS: usize,
    const OUT_N: usize,
    T: NumericScalar,
> ArenaSpikingNetwork<Chain, MAX_NEURONS, MAX_WORDS, OUT_N, T>
{
    pub const fn new(chain: Chain, arena: ActivationArena<MAX_NEURONS, MAX_WORDS, T>) -> Self {
        Self {
            chain,
            arena,
            output_spikes: [false; OUT_N],
            last_output_is_ping: false,
            _marker: PhantomData,
        }
    }

    /// Processes a full tick crossing the entire layer chain within the arena.
    pub fn tick(&mut self, inputs: &[T]) -> &[bool; OUT_N] {
        let out_is_ping = self.chain.forward_inputs_step(
            inputs,
            &mut self.arena.accumulators,
            &mut self.arena.ping_words,
            &mut self.arena.pong_words,
        );
        self.last_output_is_ping = out_is_ping;

        let words = if out_is_ping {
            &self.arena.ping_words[..MAX_WORDS]
        } else {
            &self.arena.pong_words[..MAX_WORDS]
        };

        for (i, item) in self.output_spikes.iter_mut().enumerate().take(OUT_N) {
            let word_idx = i / 32;
            let bit_idx = i % 32;
            *item = word_idx < words.len() && (words[word_idx] & (1 << bit_idx)) != 0;
        }

        &self.output_spikes
    }

    /// Executes a time step traversing the chain via the alternating Ping-Pong mechanism.
    pub fn tick_spikes_words(&mut self, in_spikes_words: &[u32]) -> &[u32] {
        let in_len = in_spikes_words.len().min(MAX_WORDS);
        self.arena.ping_words[..in_len].copy_from_slice(&in_spikes_words[..in_len]);

        let out_is_ping = self.chain.forward_step(
            true,
            &mut self.arena.accumulators,
            &mut self.arena.ping_words,
            &mut self.arena.pong_words,
        );
        self.last_output_is_ping = out_is_ping;

        let out_words = self.chain.words_len().min(MAX_WORDS);
        if out_is_ping {
            &self.arena.ping_words[..out_words]
        } else {
            &self.arena.pong_words[..out_words]
        }
    }

    /// Resets the arena and layer states.
    pub fn reset(&mut self) {
        self.chain.reset();
        self.arena.reset();
        self.output_spikes.fill(false);
        self.last_output_is_ping = false;
    }
}

impl<
    Chain: ArenaSpikingLayer<T>,
    const MAX_NEURONS: usize,
    const MAX_WORDS: usize,
    const OUT_N: usize,
    T: NumericScalar,
> SpikingNetwork<T> for ArenaSpikingNetwork<Chain, MAX_NEURONS, MAX_WORDS, OUT_N, T>
{
    #[inline]
    fn tick(&mut self, inputs: &[T]) -> &[bool] {
        self.tick(inputs)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }

    #[inline]
    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        self.tick_spikes_words(prev_spikes_words)
    }
}

/// Macro chaining [`ArenaSpikingLayer`]-compatible layers.
#[macro_export]
macro_rules! arena_chain_layers {
    ($l1:expr, $l2:expr) => {
        $crate::domain::arena::ArenaLayerChain::new($l1, $l2)
    };
    ($l1:expr, $l2:expr, $($rest:expr),+) => {
        $crate::domain::arena::ArenaLayerChain::new($l1, $crate::arena_chain_layers!($l2, $($rest),+))
    };
}

/// Macro declaring a complete [`ArenaSpikingNetwork`] with integrated memory arena.
#[macro_export]
macro_rules! arena_spiking_network {
    (max_neurons: $max_n:expr, max_words: $max_w:expr, out_n: $out_n:expr; $($layer:expr),+ $(,)?) => {
        $crate::domain::arena::ArenaSpikingNetwork::<_, $max_n, $max_w, $out_n>::new(
            $crate::arena_chain_layers!($($layer),+),
            $crate::domain::arena::ActivationArena::new(),
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::rom_csr::RomCsrSynapses;

    #[test]
    fn test_arena_lif_population_footprint_and_savings() {
        // In Fix32: ArenaLifPopulation stores only V_mem (4B) and refractory (1B) = 5 bytes / neuron!
        // vs CompactLifPopulation which stores V_mem (4B), input_acc (4B), refractory (1B) = 9 bytes / neuron.
        const N: usize = 100;
        let arena_size = core::mem::size_of::<ArenaLifPopulation<N>>();
        let compact_size =
            core::mem::size_of::<crate::domain::compact_population::CompactLifPopulation<N>>();

        // Verify SRAM memory savings (>40% savings on the population)
        assert!(arena_size < compact_size);
        let saving_pct = ((compact_size - arena_size) as f64 / compact_size as f64) * 100.0;
        assert!(
            saving_pct >= 40.0,
            "Expected SRAM gain >= 40%, got: {:.1}%",
            saving_pct
        );

        // In Fix16: ArenaLif16Population = V_mem (2B) + refractory (1B) = 3 bytes / neuron!
        let arena16_size = core::mem::size_of::<ArenaLif16Population<N>>();
        let compact16_size =
            core::mem::size_of::<crate::domain::compact_population::CompactLif16Population<N>>();
        assert!(arena16_size < compact16_size);
        let saving16_pct = ((compact16_size - arena16_size) as f64 / compact16_size as f64) * 100.0;
        assert!(
            saving16_pct >= 35.0,
            "Expected Fix16 SRAM gain >= 35%, got: {:.1}%",
            saving16_pct
        );
    }

    #[test]
    fn test_arena_deep_snn_execution_and_reset() {
        // Layer 1: 4 inputs -> 3 neurons
        static L1_ROW_PTRS: [u16; 5] = [0, 1, 2, 3, 4];
        static L1_TARGETS: [u16; 4] = [0, 1, 2, 0];
        static L1_WEIGHTS: [Fix32; 4] = [
            Fix32::from_bits(98304),  // 1.5
            Fix32::from_bits(52429),  // 0.8
            Fix32::from_bits(131072), // 2.0
            Fix32::from_bits(32768),  // 0.5
        ];
        let l1_synapses = RomCsrSynapses::new(&L1_ROW_PTRS, &L1_TARGETS, &L1_WEIGHTS);
        let l1_cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
        let l1_pop = ArenaLifPopulation::<3>::new(l1_cfg);
        let l1 = ArenaFeedforwardLayer::new(l1_pop, l1_synapses);

        // Layer 2: 3 neurons -> 2 neurons
        static L2_ROW_PTRS: [u16; 4] = [0, 1, 2, 3];
        static L2_TARGETS: [u16; 3] = [0, 1, 0];
        static L2_WEIGHTS: [Fix32; 3] = [
            Fix32::from_bits(78643), // 1.2
            Fix32::from_bits(98304), // 1.5
            Fix32::from_bits(45875), // 0.7
        ];
        let l2_synapses = RomCsrSynapses::new(&L2_ROW_PTRS, &L2_TARGETS, &L2_WEIGHTS);
        let l2_cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.85), Fix32::ZERO, 1);
        let l2_pop = ArenaLifPopulation::<2>::new(l2_cfg);
        let l2 = ArenaFeedforwardLayer::new(l2_pop, l2_synapses);

        // 2-layer network in shared arena
        let mut net = arena_spiking_network!(
            max_neurons: 3,
            max_words: 1,
            out_n: 2;
            l1,
            l2,
        );

        // Strong input injection
        let inputs = [
            Fix32::from_num(2.0),
            Fix32::ZERO,
            Fix32::from_num(1.5),
            Fix32::ZERO,
        ];
        let spikes = net.tick(&inputs);
        assert_eq!(spikes.len(), 2);

        // Reset verification
        net.reset();
        assert_eq!(net.output_spikes, [false; 2]);
    }

    #[test]
    fn test_arena_ping_pong_step_alternation() {
        // Specifically tests Ping-Pong alternation on forward_step
        static ROW_PTRS: [u16; 3] = [0, 1, 2];
        static TARGETS: [u16; 2] = [0, 1];
        static WEIGHTS: [Fix32; 2] = [Fix32::from_bits(131072), Fix32::from_bits(131072)];
        let syn = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS);
        let cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
        let pop = ArenaLifPopulation::<2>::new(cfg);
        let mut layer = ArenaFeedforwardLayer::new(pop, syn);

        let mut acc = [Fix32::ZERO; 4];
        let mut ping = [0b11u32; 1]; // 2 incident spikes
        let mut pong = [0u32; 1];

        // Step 1: Input in ping -> Output in pong (must return false)
        let out_is_ping = layer.forward_step(true, &mut acc, &mut ping, &mut pong);
        assert!(!out_is_ping);
        assert_eq!(pong[0], 0b11); // Neurons 0 and 1 must fire

        // Step 2: Input in pong -> Output in ping (must return true)
        let out_is_ping_2 = layer.forward_step(false, &mut acc, &mut ping, &mut pong);
        assert!(out_is_ping_2);
    }
}
