//! 🧱 Multilayer Architectures (Deep SNN) and Stackable Layers (`no_std`)
//!
//! Allows chaining spiking layers (Feedforward, Recurrent, 1D/2D Convolutions)
//! and propagating spikes inter-layer **within the same time step $t$**
//! via the ultra-compact [`SpikeBitset`] format (zero layer parasite latency, zero allocation).

use crate::domain::bitset::SpikeBitset;
use crate::domain::delay_ring::StaticSpikeWordDelayLine;
use crate::domain::hardware::FastDivider;
use crate::domain::numeric::{Fix32, NumericScalar};
use crate::domain::rom_csr::{QuantizedRomCsrSynapses, RomIndex, SynapticWeight};
use crate::traits::{SpikingNetwork, SpikingPopulation, SynapseMatrix};
use core::marker::PhantomData;

/// Abstract trait defining neuromorphic layer operations in a Deep SNN.
pub trait SpikingLayer<T: NumericScalar = Fix32> {
    /// Resets internal states and accumulators of the layer.
    fn reset(&mut self);
    /// Processes a time step from continuous weighted inputs.
    fn tick_inputs_words(&mut self, inputs: &[T]) -> &[u32];
    /// Processes a time step from previous layer spikes.
    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32];
    /// Indicates whether the neuron at the specified index fired this tick.
    fn get_spike(&self, index: usize) -> bool;
}

/// Feedforward spiking layer (direct forward propagation).
/// Zero-allocation optimized: spikes are directly written into [`SpikeBitset`]
/// without any intermediate `[bool; N]` or `[u32; N]` array (massive SRAM and cycle savings).
#[derive(Clone, Copy, Debug)]
pub struct SpikingFeedforwardLayer<
    P: SpikingPopulation<T>,
    SIn: SynapseMatrix<T>,
    const N: usize,
    const WORDS: usize,
    T: NumericScalar = Fix32,
> {
    pub neurons: P,
    pub input_synapses: SIn,
    pub spikes: SpikeBitset<WORDS>,
    _marker: PhantomData<T>,
}

impl<
    P: SpikingPopulation<T>,
    SIn: SynapseMatrix<T>,
    const N: usize,
    const WORDS: usize,
    T: NumericScalar,
> SpikingFeedforwardLayer<P, SIn, N, WORDS, T>
{
    pub const fn new(neurons: P, input_synapses: SIn) -> Self {
        Self {
            neurons,
            input_synapses,
            spikes: SpikeBitset::new(),
            _marker: PhantomData,
        }
    }

    /// Resets membrane state and layer spikes.
    pub fn reset(&mut self) {
        self.neurons.reset();
        self.spikes.clear();
    }

    /// Processes a time step from continuous weighted inputs (e.g., first sensor layer).
    pub fn tick_inputs(&mut self, inputs: &[T]) -> &SpikeBitset<WORDS> {
        self.input_synapses
            .propagate_weighted_inputs(inputs, self.neurons.input_accumulators_mut());
        self.evaluate_neurons()
    }

    /// Processes a time step from previous layer spikes (via bitset words).
    pub fn tick_spikes(&mut self, prev_spikes_words: &[u32]) -> &SpikeBitset<WORDS> {
        self.input_synapses
            .propagate_spikes_words(prev_spikes_words, self.neurons.input_accumulators_mut());
        self.evaluate_neurons()
    }

    #[inline]
    fn evaluate_neurons(&mut self) -> &SpikeBitset<WORDS> {
        self.neurons.tick_bitset(&mut self.spikes.words);
        &self.spikes
    }
}

impl<
    P: SpikingPopulation<T>,
    SIn: SynapseMatrix<T>,
    const N: usize,
    const WORDS: usize,
    T: NumericScalar,
> SpikingLayer<T> for SpikingFeedforwardLayer<P, SIn, N, WORDS, T>
{
    #[inline]
    fn reset(&mut self) {
        self.reset();
    }

    #[inline]
    fn tick_inputs_words(&mut self, inputs: &[T]) -> &[u32] {
        self.tick_inputs(inputs).words()
    }

    #[inline]
    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        self.tick_spikes(prev_spikes_words).words()
    }

    #[inline]
    fn get_spike(&self, index: usize) -> bool {
        self.spikes.get(index)
    }
}

/// Ultra-fast INT8 quantized feedforward layer with [`QuantizedRomCsrSynapses`] synapses.
///
/// **Major silicon advantage over CMSIS-NN**:
/// Synapse accumulation is performed in raw `i32` integers (`ADD` / `LDRSB` in 1 CPU cycle),
/// **WITHOUT ANY MULTIPLICATION** while traversing active synapses.
/// The scaling factor is applied globally in a single pass at the end of the tick across the $N$ neurons.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedFeedforwardLayer<
    P: SpikingPopulation<T>,
    Idx: RomIndex + 'static,
    const N: usize,
    const WORDS: usize,
    T: NumericScalar = Fix32,
> {
    pub neurons: P,
    pub synapses: QuantizedRomCsrSynapses<'static, Idx>,
    pub spikes: SpikeBitset<WORDS>,
    int_accumulators: [i32; N],
    _marker: PhantomData<T>,
}

impl<
    P: SpikingPopulation<T>,
    Idx: RomIndex + 'static,
    const N: usize,
    const WORDS: usize,
    T: NumericScalar,
> QuantizedFeedforwardLayer<P, Idx, N, WORDS, T>
{
    pub const fn new(neurons: P, synapses: QuantizedRomCsrSynapses<'static, Idx>) -> Self {
        Self {
            neurons,
            synapses,
            spikes: SpikeBitset::new(),
            int_accumulators: [0; N],
            _marker: PhantomData,
        }
    }

    pub fn reset(&mut self) {
        self.neurons.reset();
        self.spikes.clear();
        self.int_accumulators.fill(0);
    }

    /// Processes a time step from previous layer spikes.
    /// Accumulates in 1 cycle/synapse integer arithmetic then applies scaling.
    pub fn tick_spikes(&mut self, prev_spikes_words: &[u32]) -> &SpikeBitset<WORDS> {
        // 1. Pure integer accumulation without multiplication
        self.synapses
            .propagate_spikes_words_i32(prev_spikes_words, &mut self.int_accumulators);

        // 2. Global scaling to neuron accumulators
        self.synapses.scale_accumulators(
            &self.int_accumulators,
            self.neurons.input_accumulators_mut(),
        );

        // 3. Fast reset of temporary buffer
        self.int_accumulators.fill(0);

        // 4. Membrane evaluation
        self.evaluate_neurons()
    }

    /// Processes a time step from continuous weighted inputs.
    pub fn tick_inputs(&mut self, inputs: &[T]) -> &SpikeBitset<WORDS> {
        self.synapses
            .propagate_weighted_inputs(inputs, self.neurons.input_accumulators_mut());
        self.evaluate_neurons()
    }

    #[inline]
    fn evaluate_neurons(&mut self) -> &SpikeBitset<WORDS> {
        self.neurons.tick_bitset(&mut self.spikes.words);
        &self.spikes
    }
}

impl<
    P: SpikingPopulation<T>,
    Idx: RomIndex + 'static,
    const N: usize,
    const WORDS: usize,
    T: NumericScalar,
> SpikingLayer<T> for QuantizedFeedforwardLayer<P, Idx, N, WORDS, T>
{
    #[inline]
    fn reset(&mut self) {
        self.reset();
    }

    #[inline]
    fn tick_inputs_words(&mut self, inputs: &[T]) -> &[u32] {
        self.tick_inputs(inputs).words()
    }

    #[inline]
    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        self.tick_spikes(prev_spikes_words).words()
    }

    #[inline]
    fn get_spike(&self, index: usize) -> bool {
        self.spikes.get(index)
    }
}

/// Recurrent spiking layer (with internal CTZ propagation on next tick).
/// Zero allocation and zero intermediate SRAM buffer.
#[derive(Clone, Copy, Debug)]
pub struct SpikingRecurrentLayer<
    P: SpikingPopulation<T>,
    SIn: SynapseMatrix<T>,
    SRec: SynapseMatrix<T>,
    const N: usize,
    const WORDS: usize,
    T: NumericScalar = Fix32,
> {
    pub neurons: P,
    pub input_synapses: SIn,
    pub recurrent_synapses: SRec,
    pub spikes: SpikeBitset<WORDS>,
    _marker: PhantomData<T>,
}

impl<
    P: SpikingPopulation<T>,
    SIn: SynapseMatrix<T>,
    SRec: SynapseMatrix<T>,
    const N: usize,
    const WORDS: usize,
    T: NumericScalar,
> SpikingRecurrentLayer<P, SIn, SRec, N, WORDS, T>
{
    pub const fn new(neurons: P, input_synapses: SIn, recurrent_synapses: SRec) -> Self {
        Self {
            neurons,
            input_synapses,
            recurrent_synapses,
            spikes: SpikeBitset::new(),
            _marker: PhantomData,
        }
    }

    pub fn reset(&mut self) {
        self.neurons.reset();
        self.spikes.clear();
    }

    /// Processes a time step from continuous weighted inputs (e.g., input reservoir).
    pub fn tick_inputs(&mut self, inputs: &[T]) -> &SpikeBitset<WORDS> {
        self.input_synapses
            .propagate_weighted_inputs(inputs, self.neurons.input_accumulators_mut());
        self.evaluate_recurrent()
    }

    /// Processes a time step from previous layer spikes.
    pub fn tick_spikes(&mut self, prev_spikes_words: &[u32]) -> &SpikeBitset<WORDS> {
        self.input_synapses
            .propagate_spikes_words(prev_spikes_words, self.neurons.input_accumulators_mut());
        self.evaluate_recurrent()
    }

    #[inline]
    fn evaluate_recurrent(&mut self) -> &SpikeBitset<WORDS> {
        self.neurons.tick_bitset(&mut self.spikes.words);

        // Recurrent propagation for next tick directly via hardware CTZ instructions
        self.recurrent_synapses
            .propagate_spikes_words(self.spikes.words(), self.neurons.input_accumulators_mut());

        &self.spikes
    }
}

impl<
    P: SpikingPopulation<T>,
    SIn: SynapseMatrix<T>,
    SRec: SynapseMatrix<T>,
    const N: usize,
    const WORDS: usize,
    T: NumericScalar,
> SpikingLayer<T> for SpikingRecurrentLayer<P, SIn, SRec, N, WORDS, T>
{
    #[inline]
    fn reset(&mut self) {
        self.reset();
    }

    #[inline]
    fn tick_inputs_words(&mut self, inputs: &[T]) -> &[u32] {
        self.tick_inputs(inputs).words()
    }

    #[inline]
    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        self.tick_spikes(prev_spikes_words).words()
    }

    #[inline]
    fn get_spike(&self, index: usize) -> bool {
        self.spikes.get(index)
    }
}

/// Spiking layer with zero-allocation synaptic delay line (`no_std`).
///
/// Unlike `RefCell`-based wrappers, delay line temporal state is encapsulated
/// within `&mut self`, ensuring zero dynamic overhead and absolute thread-safety.
#[derive(Clone, Copy, Debug)]
pub struct DelayedSpikingLayer<
    P: SpikingPopulation<T>,
    SIn: SynapseMatrix<T>,
    const N: usize,
    const WORDS: usize,
    const IN_WORDS: usize,
    const D: usize,
    T: NumericScalar = Fix32,
> {
    pub neurons: P,
    pub input_synapses: SIn,
    pub delay_line: StaticSpikeWordDelayLine<IN_WORDS, D>,
    pub delay_ticks: usize,
    pub spikes: SpikeBitset<WORDS>,
    _marker: PhantomData<T>,
}

impl<
    P: SpikingPopulation<T>,
    SIn: SynapseMatrix<T>,
    const N: usize,
    const WORDS: usize,
    const IN_WORDS: usize,
    const D: usize,
    T: NumericScalar,
> DelayedSpikingLayer<P, SIn, N, WORDS, IN_WORDS, D, T>
{
    pub const fn new(neurons: P, input_synapses: SIn, delay_ticks: usize) -> Self {
        Self {
            neurons,
            input_synapses,
            delay_line: StaticSpikeWordDelayLine::new(),
            delay_ticks,
            spikes: SpikeBitset::new(),
            _marker: PhantomData,
        }
    }

    pub fn reset(&mut self) {
        self.neurons.reset();
        self.delay_line.reset();
        self.spikes.clear();
    }

    /// Processes a time step from continuous weighted inputs (without delay).
    pub fn tick_inputs(&mut self, inputs: &[T]) -> &SpikeBitset<WORDS> {
        self.input_synapses
            .propagate_weighted_inputs(inputs, self.neurons.input_accumulators_mut());
        self.neurons.tick_bitset(&mut self.spikes.words);
        &self.spikes
    }

    /// Processes a time step from previous layer spikes applying temporal delay.
    pub fn tick_spikes(&mut self, prev_spikes_words: &[u32]) -> &SpikeBitset<WORDS> {
        let mut cur_in = [0u32; IN_WORDS];
        let copy_len = prev_spikes_words.len().min(IN_WORDS);
        cur_in[..copy_len].copy_from_slice(&prev_spikes_words[..copy_len]);

        self.delay_line.push(&cur_in);
        let delayed_words = self.delay_line.get_delayed(self.delay_ticks);

        self.input_synapses
            .propagate_spikes_words(delayed_words, self.neurons.input_accumulators_mut());
        self.neurons.tick_bitset(&mut self.spikes.words);
        &self.spikes
    }
}

impl<
    P: SpikingPopulation<T>,
    SIn: SynapseMatrix<T>,
    const N: usize,
    const WORDS: usize,
    const IN_WORDS: usize,
    const D: usize,
    T: NumericScalar,
> SpikingLayer<T> for DelayedSpikingLayer<P, SIn, N, WORDS, IN_WORDS, D, T>
{
    #[inline]
    fn reset(&mut self) {
        self.reset();
    }

    #[inline]
    fn tick_inputs_words(&mut self, inputs: &[T]) -> &[u32] {
        self.tick_inputs(inputs).words()
    }

    #[inline]
    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        self.tick_spikes(prev_spikes_words).words()
    }

    #[inline]
    fn get_spike(&self, index: usize) -> bool {
        self.spikes.get(index)
    }
}

/// Arbitrary stackable layer chain (allows chaining $N$ layers without depth limit).
#[derive(Clone, Copy, Debug)]
pub struct LayerChain<Head: SpikingLayer<T>, Tail: SpikingLayer<T>, T: NumericScalar = Fix32> {
    pub head: Head,
    pub tail: Tail,
    _marker: PhantomData<T>,
}

impl<Head: SpikingLayer<T>, Tail: SpikingLayer<T>, T: NumericScalar> LayerChain<Head, Tail, T> {
    pub const fn new(head: Head, tail: Tail) -> Self {
        Self {
            head,
            tail,
            _marker: PhantomData,
        }
    }
}

impl<Head: SpikingLayer<T>, Tail: SpikingLayer<T>, T: NumericScalar> SpikingLayer<T>
    for LayerChain<Head, Tail, T>
{
    fn reset(&mut self) {
        self.head.reset();
        self.tail.reset();
    }

    fn tick_inputs_words(&mut self, inputs: &[T]) -> &[u32] {
        let head_spikes = self.head.tick_inputs_words(inputs);
        self.tail.tick_spikes_words(head_spikes)
    }

    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        let head_spikes = self.head.tick_spikes_words(prev_spikes_words);
        self.tail.tick_spikes_words(head_spikes)
    }

    #[inline]
    fn get_spike(&self, index: usize) -> bool {
        self.tail.get_spike(index)
    }
}

/// Modular deep SNN composed of an arbitrary chain of [`LayerChain`] layers.
/// Implements the [`SpikingNetwork`] trait.
#[derive(Clone, Copy, Debug)]
pub struct ChainedSpikingNetwork<
    Chain: SpikingLayer<T>,
    const OUT_N: usize,
    T: NumericScalar = Fix32,
> {
    pub chain: Chain,
    pub output_spikes: [bool; OUT_N],
    _marker: PhantomData<T>,
}

impl<Chain: SpikingLayer<T>, const OUT_N: usize, T: NumericScalar>
    ChainedSpikingNetwork<Chain, OUT_N, T>
{
    pub const fn new(chain: Chain) -> Self {
        Self {
            chain,
            output_spikes: [false; OUT_N],
            _marker: PhantomData,
        }
    }

    /// Processes a full tick traversing the entire layer chain within the same time step.
    pub fn tick(&mut self, inputs: &[T]) -> &[bool; OUT_N] {
        let _ = self.chain.tick_inputs_words(inputs);
        for i in 0..OUT_N {
            self.output_spikes[i] = self.chain.get_spike(i);
        }
        &self.output_spikes
    }

    /// Processes a complete time step and directly returns bitset words (1 bit per neuron)
    /// without intermediate copies or boolean conversions.
    #[inline]
    pub fn tick_inputs_words(&mut self, inputs: &[T]) -> &[u32] {
        self.chain.tick_inputs_words(inputs)
    }

    /// Processes a complete time step from incoming bitset spikes (zero conversion latency).
    #[inline]
    pub fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        self.chain.tick_spikes_words(prev_spikes_words)
    }

    pub fn reset(&mut self) {
        self.chain.reset();
        self.output_spikes = [false; OUT_N];
    }
}

impl<Chain: SpikingLayer<T>, const OUT_N: usize, T: NumericScalar> SpikingNetwork<T>
    for ChainedSpikingNetwork<Chain, OUT_N, T>
{
    #[inline]
    fn tick(&mut self, inputs: &[T]) -> &[bool] {
        self.tick(inputs)
    }

    #[inline]
    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        self.tick_spikes_words(prev_spikes_words)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }
}

/// Macro recursively composing a chain of spiking layers [`LayerChain`].
#[macro_export]
macro_rules! chain_layers {
    ($l1:expr, $l2:expr) => {
        $crate::domain::layers::LayerChain::new($l1, $l2)
    };
    ($l1:expr, $l2:expr, $($rest:expr),+) => {
        $crate::domain::layers::LayerChain::new($l1, $crate::chain_layers!($l2, $($rest),+))
    };
}

/// Declarative macro instantiating a complete deep network [`ChainedSpikingNetwork`]
/// with $N$ arbitrary layers and output size `OUT_N`.
#[macro_export]
macro_rules! spiking_network {
    (out_n: $out_n:expr; $($layer:expr),+ $(,)?) => {
        $crate::domain::layers::ChainedSpikingNetwork::<_, $out_n>::new($crate::chain_layers!($($layer),+))
    };
}

/// 2-layer stacked deep SNN (Deep SNN).
/// Accepts any combination of feedforward and recurrent layers.
/// Spikes from layer 1 traverse layer 2 **within the same tick**.
#[deprecated(
    since = "0.2.0",
    note = "Use `ChainedSpikingNetwork` or the `spiking_network!` macro instead"
)]
#[derive(Clone, Copy, Debug)]
pub struct DeepSpikingNetwork2<
    L1: SpikingLayer<T>,
    L2: SpikingLayer<T>,
    const OUT_N: usize,
    T: NumericScalar = Fix32,
> {
    pub layer1: L1,
    pub layer2: L2,
    pub output_spikes: [bool; OUT_N],
    _marker: PhantomData<T>,
}

#[allow(deprecated)]
impl<L1: SpikingLayer<T>, L2: SpikingLayer<T>, const OUT_N: usize, T: NumericScalar>
    DeepSpikingNetwork2<L1, L2, OUT_N, T>
{
    pub const fn new(layer1: L1, layer2: L2) -> Self {
        Self {
            layer1,
            layer2,
            output_spikes: [false; OUT_N],
            _marker: PhantomData,
        }
    }

    /// Processes a complete time step traversing both layers sequentially in the same tick.
    pub fn tick(&mut self, inputs: &[T]) -> &[bool; OUT_N] {
        let l1_words = self.layer1.tick_inputs_words(inputs);
        let _ = self.layer2.tick_spikes_words(l1_words);

        for i in 0..OUT_N {
            self.output_spikes[i] = self.layer2.get_spike(i);
        }

        &self.output_spikes
    }

    pub fn reset(&mut self) {
        self.layer1.reset();
        self.layer2.reset();
        self.output_spikes = [false; OUT_N];
    }
}

#[allow(deprecated)]
impl<L1: SpikingLayer<T>, L2: SpikingLayer<T>, const OUT_N: usize, T: NumericScalar>
    SpikingNetwork<T> for DeepSpikingNetwork2<L1, L2, OUT_N, T>
{
    #[inline]
    fn tick(&mut self, inputs: &[T]) -> &[bool] {
        self.tick(inputs)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }
}

/// 3-layer stacked deep SNN (Deep SNN).
/// Chains 3 layers (e.g., Conv -> Recurrent -> Feedforward) intra-tick.
#[deprecated(
    since = "0.2.0",
    note = "Use `ChainedSpikingNetwork` or the `spiking_network!` macro instead"
)]
#[derive(Clone, Copy, Debug)]
pub struct DeepSpikingNetwork3<
    L1: SpikingLayer<T>,
    L2: SpikingLayer<T>,
    L3: SpikingLayer<T>,
    const OUT_N: usize,
    T: NumericScalar = Fix32,
> {
    pub layer1: L1,
    pub layer2: L2,
    pub layer3: L3,
    pub output_spikes: [bool; OUT_N],
    _marker: PhantomData<T>,
}

#[allow(deprecated)]
impl<
    L1: SpikingLayer<T>,
    L2: SpikingLayer<T>,
    L3: SpikingLayer<T>,
    const OUT_N: usize,
    T: NumericScalar,
> DeepSpikingNetwork3<L1, L2, L3, OUT_N, T>
{
    pub const fn new(layer1: L1, layer2: L2, layer3: L3) -> Self {
        Self {
            layer1,
            layer2,
            layer3,
            output_spikes: [false; OUT_N],
            _marker: PhantomData,
        }
    }

    pub fn tick(&mut self, inputs: &[T]) -> &[bool; OUT_N] {
        let l1_words = self.layer1.tick_inputs_words(inputs);
        let l2_words = self.layer2.tick_spikes_words(l1_words);
        let _ = self.layer3.tick_spikes_words(l2_words);

        for i in 0..OUT_N {
            self.output_spikes[i] = self.layer3.get_spike(i);
        }

        &self.output_spikes
    }

    pub fn reset(&mut self) {
        self.layer1.reset();
        self.layer2.reset();
        self.layer3.reset();
        self.output_spikes = [false; OUT_N];
    }
}

#[allow(deprecated)]
impl<
    L1: SpikingLayer<T>,
    L2: SpikingLayer<T>,
    L3: SpikingLayer<T>,
    const OUT_N: usize,
    T: NumericScalar,
> SpikingNetwork<T> for DeepSpikingNetwork3<L1, L2, L3, OUT_N, T>
{
    #[inline]
    fn tick(&mut self, inputs: &[T]) -> &[bool] {
        self.tick(inputs)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }
}

/// 4-layer stacked deep SNN (Deep SNN).
/// Chains 4 layers (e.g., Conv2D -> MaxPool2D -> Dense -> Readout) intra-tick.
#[deprecated(
    since = "0.2.0",
    note = "Use `ChainedSpikingNetwork` or the `spiking_network!` macro instead"
)]
#[derive(Clone, Copy, Debug)]
pub struct DeepSpikingNetwork4<
    L1: SpikingLayer<T>,
    L2: SpikingLayer<T>,
    L3: SpikingLayer<T>,
    L4: SpikingLayer<T>,
    const OUT_N: usize,
    T: NumericScalar = Fix32,
> {
    pub layer1: L1,
    pub layer2: L2,
    pub layer3: L3,
    pub layer4: L4,
    pub output_spikes: [bool; OUT_N],
    _marker: PhantomData<T>,
}

#[allow(deprecated)]
impl<
    L1: SpikingLayer<T>,
    L2: SpikingLayer<T>,
    L3: SpikingLayer<T>,
    L4: SpikingLayer<T>,
    const OUT_N: usize,
    T: NumericScalar,
> DeepSpikingNetwork4<L1, L2, L3, L4, OUT_N, T>
{
    pub const fn new(layer1: L1, layer2: L2, layer3: L3, layer4: L4) -> Self {
        Self {
            layer1,
            layer2,
            layer3,
            layer4,
            output_spikes: [false; OUT_N],
            _marker: PhantomData,
        }
    }

    pub fn tick(&mut self, inputs: &[T]) -> &[bool; OUT_N] {
        let l1_words = self.layer1.tick_inputs_words(inputs);
        let l2_words = self.layer2.tick_spikes_words(l1_words);
        let l3_words = self.layer3.tick_spikes_words(l2_words);
        let _ = self.layer4.tick_spikes_words(l3_words);

        for i in 0..OUT_N {
            self.output_spikes[i] = self.layer4.get_spike(i);
        }

        &self.output_spikes
    }

    pub fn reset(&mut self) {
        self.layer1.reset();
        self.layer2.reset();
        self.layer3.reset();
        self.layer4.reset();
        self.output_spikes = [false; OUT_N];
    }
}

#[allow(deprecated)]
impl<
    L1: SpikingLayer<T>,
    L2: SpikingLayer<T>,
    L3: SpikingLayer<T>,
    L4: SpikingLayer<T>,
    const OUT_N: usize,
    T: NumericScalar,
> SpikingNetwork<T> for DeepSpikingNetwork4<L1, L2, L3, L4, OUT_N, T>
{
    #[inline]
    fn tick(&mut self, inputs: &[T]) -> &[bool] {
        self.tick(inputs)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }
}

/// Residual spiking layer (Skip-connection / Spiking ResNet).
///
/// Merges via bitwise OR (`|`) spikes generated by the internal `inner` layer
/// with incoming spikes from the upstream layer (zero allocation, zero parasite delay).
#[derive(Clone, Copy, Debug)]
pub struct ResidualSpikingLayer<L: SpikingLayer<T>, const WORDS: usize, T: NumericScalar = Fix32> {
    pub inner: L,
    pub spikes: SpikeBitset<WORDS>,
    _marker: PhantomData<T>,
}

impl<L: SpikingLayer<T>, const WORDS: usize, T: NumericScalar> ResidualSpikingLayer<L, WORDS, T> {
    pub const fn new(inner: L) -> Self {
        Self {
            inner,
            spikes: SpikeBitset::new(),
            _marker: PhantomData,
        }
    }
}

impl<L: SpikingLayer<T>, const WORDS: usize, T: NumericScalar> SpikingLayer<T>
    for ResidualSpikingLayer<L, WORDS, T>
{
    fn reset(&mut self) {
        self.inner.reset();
        self.spikes.clear();
    }

    fn tick_inputs_words(&mut self, inputs: &[T]) -> &[u32] {
        let out_words = self.inner.tick_inputs_words(inputs);
        let len = WORDS.min(out_words.len());
        self.spikes.words_mut()[..len].copy_from_slice(&out_words[..len]);
        self.spikes.words()
    }

    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        let out_words = self.inner.tick_spikes_words(prev_spikes_words);
        let min_words = WORDS.min(out_words.len()).min(prev_spikes_words.len());
        let target = self.spikes.words_mut();
        for w in 0..min_words {
            target[w] = out_words[w] | prev_spikes_words[w];
        }
        self.spikes.words()
    }

    #[inline]
    fn get_spike(&self, index: usize) -> bool {
        self.spikes.get(index)
    }
}

/// 2D spike Max-Pooling layer (Spike Max Pooling).
///
/// Spatially reduces a 2D spike mask by applying logical OR over each $K_H \times K_W$ window.
/// Purely bitwise operation without heavy arithmetic computation or allocations.
#[derive(Clone, Copy, Debug)]
pub struct SpikeMaxPool2d<
    const CHANNELS: usize,
    const IN_H: usize,
    const IN_W: usize,
    const K_H: usize,
    const K_W: usize,
    const STRIDE_H: usize,
    const STRIDE_W: usize,
    const OUT_WORDS: usize,
    T: NumericScalar = Fix32,
> {
    pub out_h: usize,
    pub out_w: usize,
    pub spikes: SpikeBitset<OUT_WORDS>,
    _marker: PhantomData<T>,
}

impl<
    const CHANNELS: usize,
    const IN_H: usize,
    const IN_W: usize,
    const K_H: usize,
    const K_W: usize,
    const STRIDE_H: usize,
    const STRIDE_W: usize,
    const OUT_WORDS: usize,
    T: NumericScalar,
> Default for SpikeMaxPool2d<CHANNELS, IN_H, IN_W, K_H, K_W, STRIDE_H, STRIDE_W, OUT_WORDS, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    const CHANNELS: usize,
    const IN_H: usize,
    const IN_W: usize,
    const K_H: usize,
    const K_W: usize,
    const STRIDE_H: usize,
    const STRIDE_W: usize,
    const OUT_WORDS: usize,
    T: NumericScalar,
> SpikeMaxPool2d<CHANNELS, IN_H, IN_W, K_H, K_W, STRIDE_H, STRIDE_W, OUT_WORDS, T>
{
    pub const fn new() -> Self {
        let out_h = if IN_H >= K_H && STRIDE_H > 0 {
            (IN_H - K_H) / STRIDE_H + 1
        } else {
            0
        };
        let out_w = if IN_W >= K_W && STRIDE_W > 0 {
            (IN_W - K_W) / STRIDE_W + 1
        } else {
            0
        };
        Self {
            out_h,
            out_w,
            spikes: SpikeBitset::new(),
            _marker: PhantomData,
        }
    }

    /// Applies spatial pooling over the input spike stream.
    pub fn pool_spikes(&mut self, in_spikes_words: &[u32]) -> &SpikeBitset<OUT_WORDS> {
        self.spikes.clear();

        let mut out_idx = 0;
        for c in 0..CHANNELS {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let mut pooled = false;
                    for kh in 0..K_H {
                        for kw in 0..K_W {
                            let ih = oh * STRIDE_H + kh;
                            let iw = ow * STRIDE_W + kw;
                            let in_idx = (c * IN_H + ih) * IN_W + iw;
                            let w = in_idx / 32;
                            let b = in_idx % 32;
                            if w < in_spikes_words.len() && ((in_spikes_words[w] >> b) & 1) != 0 {
                                pooled = true;
                                break;
                            }
                        }
                        if pooled {
                            break;
                        }
                    }
                    if pooled {
                        self.spikes.set(out_idx);
                    }
                    out_idx += 1;
                }
            }
        }
        &self.spikes
    }
}

impl<
    const CHANNELS: usize,
    const IN_H: usize,
    const IN_W: usize,
    const K_H: usize,
    const K_W: usize,
    const STRIDE_H: usize,
    const STRIDE_W: usize,
    const OUT_WORDS: usize,
    T: NumericScalar,
> SpikingLayer<T>
    for SpikeMaxPool2d<CHANNELS, IN_H, IN_W, K_H, K_W, STRIDE_H, STRIDE_W, OUT_WORDS, T>
{
    fn reset(&mut self) {
        self.spikes.clear();
    }

    fn tick_inputs_words(&mut self, _inputs: &[T]) -> &[u32] {
        self.spikes.clear();
        self.spikes.words()
    }

    fn tick_spikes_words(&mut self, prev_spikes_words: &[u32]) -> &[u32] {
        self.pool_spikes(prev_spikes_words).words()
    }

    #[inline]
    fn get_spike(&self, index: usize) -> bool {
        self.spikes.get(index)
    }
}

/// Event-driven 1D convolutional synaptic matrix with shared kernel in Flash ROM (`.rodata`).
/// Enables applying temporal/spatial filters (audio, sensors) without weight duplication.
#[derive(Clone, Copy, Debug)]
pub struct Conv1dSynapses<'a, Weight: SynapticWeight<T> = Fix32, T: NumericScalar = Fix32> {
    pub kernel: &'a [Weight],
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub in_len: usize,
    pub out_len: usize,
    pub div_in_len: FastDivider,
    pub div_stride: FastDivider,
    _marker: PhantomData<T>,
}

impl<'a, Weight: SynapticWeight<T>, T: NumericScalar> Conv1dSynapses<'a, Weight, T> {
    pub const fn new(
        kernel: &'a [Weight],
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        in_len: usize,
    ) -> Self {
        Self::with_padding(
            kernel,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            in_len,
        )
    }

    pub const fn with_padding(
        kernel: &'a [Weight],
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        in_len: usize,
    ) -> Self {
        let padded_in_len = in_len + 2 * padding;
        let out_len = if padded_in_len >= kernel_size && stride > 0 {
            (padded_in_len - kernel_size) / stride + 1
        } else {
            0
        };
        let div_in_len = FastDivider::new(in_len);
        let div_stride = FastDivider::new(stride);
        Self {
            kernel,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            in_len,
            out_len,
            div_in_len,
            div_stride,
            _marker: PhantomData,
        }
    }
}

impl<Weight: SynapticWeight<T>, T: NumericScalar> SynapseMatrix<T>
    for Conv1dSynapses<'_, Weight, T>
{
    fn len(&self) -> usize {
        self.kernel.len()
    }

    fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [T]) {
        if self.stride == 0 || self.out_len == 0 {
            return;
        }
        let stride_weight_out_ch = self.in_channels * self.kernel_size;
        for &spiker in active_spikes {
            let spiker = spiker as usize;
            let (in_ch, in_pos) = self.div_in_len.div_rem(spiker);

            if in_ch < self.in_channels {
                let virt_in_pos = in_pos + self.padding;
                let (min_out, max_out) = if self.stride == 1 {
                    (
                        virt_in_pos.saturating_sub(self.kernel_size.saturating_sub(1)),
                        virt_in_pos.min(self.out_len.saturating_sub(1)),
                    )
                } else {
                    let min_o = if virt_in_pos + 1 >= self.kernel_size {
                        self.div_stride.div_ceil(virt_in_pos + 1 - self.kernel_size)
                    } else {
                        0
                    };
                    let max_o = self
                        .div_stride
                        .div(virt_in_pos)
                        .min(self.out_len.saturating_sub(1));
                    (min_o, max_o)
                };

                if min_out <= max_out {
                    for out_pos in min_out..=max_out {
                        let k_start = out_pos * self.stride;
                        let k_idx = virt_in_pos - k_start;
                        let mut weight_idx = in_ch * self.kernel_size + k_idx;
                        let mut target = out_pos;
                        for _out_ch in 0..self.out_channels {
                            if weight_idx < self.kernel.len()
                                && let Some(acc) = input_accumulators.get_mut(target)
                            {
                                *acc = acc.saturating_add(self.kernel[weight_idx].to_scalar());
                            }
                            weight_idx += stride_weight_out_ch;
                            target += self.out_len;
                        }
                    }
                }
            }
        }
    }

    fn propagate_weighted_inputs(&self, inputs: &[T], input_accumulators: &mut [T]) {
        if self.stride == 0 || self.out_len == 0 {
            return;
        }
        let stride_weight_out_ch = self.in_channels * self.kernel_size;
        for (src_idx, &val) in inputs.iter().enumerate() {
            if val != T::ZERO {
                let (in_ch, in_pos) = self.div_in_len.div_rem(src_idx);

                if in_ch < self.in_channels {
                    let virt_in_pos = in_pos + self.padding;
                    let (min_out, max_out) = if self.stride == 1 {
                        (
                            virt_in_pos.saturating_sub(self.kernel_size.saturating_sub(1)),
                            virt_in_pos.min(self.out_len.saturating_sub(1)),
                        )
                    } else {
                        let min_o = if virt_in_pos + 1 >= self.kernel_size {
                            self.div_stride.div_ceil(virt_in_pos + 1 - self.kernel_size)
                        } else {
                            0
                        };
                        let max_o = self
                            .div_stride
                            .div(virt_in_pos)
                            .min(self.out_len.saturating_sub(1));
                        (min_o, max_o)
                    };

                    if min_out <= max_out {
                        for out_pos in min_out..=max_out {
                            let k_start = out_pos * self.stride;
                            let k_idx = virt_in_pos - k_start;
                            let mut weight_idx = in_ch * self.kernel_size + k_idx;
                            let mut target = out_pos;
                            for _out_ch in 0..self.out_channels {
                                if weight_idx < self.kernel.len()
                                    && let Some(acc) = input_accumulators.get_mut(target)
                                {
                                    *acc = acc
                                        .saturating_add(val * self.kernel[weight_idx].to_scalar());
                                }
                                weight_idx += stride_weight_out_ch;
                                target += self.out_len;
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn propagate_spikes_words(&self, words: &[u32], input_accumulators: &mut [T]) {
        if self.stride == 0 || self.out_len == 0 {
            return;
        }
        let stride_weight_out_ch = self.in_channels * self.kernel_size;
        for (w_idx, &word) in words.iter().enumerate() {
            if word == 0 {
                continue;
            }
            let mut remaining = word;
            let base_idx = w_idx * 32;
            while remaining != 0 {
                let tz = remaining.trailing_zeros() as usize;
                let spiker = base_idx + tz;
                remaining &= remaining - 1;

                let (in_ch, in_pos) = self.div_in_len.div_rem(spiker);
                if in_ch < self.in_channels {
                    let virt_in_pos = in_pos + self.padding;
                    let (min_out, max_out) = if self.stride == 1 {
                        (
                            virt_in_pos.saturating_sub(self.kernel_size.saturating_sub(1)),
                            virt_in_pos.min(self.out_len.saturating_sub(1)),
                        )
                    } else {
                        let min_o = if virt_in_pos + 1 >= self.kernel_size {
                            self.div_stride.div_ceil(virt_in_pos + 1 - self.kernel_size)
                        } else {
                            0
                        };
                        let max_o = self
                            .div_stride
                            .div(virt_in_pos)
                            .min(self.out_len.saturating_sub(1));
                        (min_o, max_o)
                    };

                    if min_out <= max_out {
                        for out_pos in min_out..=max_out {
                            let k_start = out_pos * self.stride;
                            let k_idx = virt_in_pos - k_start;
                            let mut weight_idx = in_ch * self.kernel_size + k_idx;
                            let mut target = out_pos;
                            for _out_ch in 0..self.out_channels {
                                if weight_idx < self.kernel.len()
                                    && let Some(acc) = input_accumulators.get_mut(target)
                                {
                                    *acc = acc.saturating_add(self.kernel[weight_idx].to_scalar());
                                }
                                weight_idx += stride_weight_out_ch;
                                target += self.out_len;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Memory layout of convolutional tensors on microcontroller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ConvDataLayout {
    /// Channels-first (CHW or NCHW): standard PyTorch/NIR format.
    #[default]
    Chw,
    /// Channels-last (HWC or NHWC): standard CMSIS-NN and microcontroller format,
    /// ensuring absolute memory contiguity of channels and SIMD auto-vectorization.
    Hwc,
}

/// Event-driven 2D convolutional synaptic matrix with shared kernel in Flash ROM (`.rodata`).
/// Optimized for neuromorphic vision sensors (DVS cameras, N-MNIST, Gesture).
#[derive(Clone, Copy, Debug)]
pub struct Conv2dSynapses<'a, Weight: SynapticWeight<T> = Fix32, T: NumericScalar = Fix32> {
    pub kernel: &'a [Weight],
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub layout: ConvDataLayout,
    pub div_in_w: FastDivider,
    pub div_in_h: FastDivider,
    pub div_in_channels: FastDivider,
    pub div_stride_h: FastDivider,
    pub div_stride_w: FastDivider,
    _marker: PhantomData<T>,
}

impl<'a, Weight: SynapticWeight<T>, T: NumericScalar> Conv2dSynapses<'a, Weight, T> {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        kernel: &'a [Weight],
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        in_h: usize,
        in_w: usize,
    ) -> Self {
        Self::with_padding(
            kernel,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            0,
            0,
            in_h,
            in_w,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn with_padding(
        kernel: &'a [Weight],
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
    ) -> Self {
        let padded_h = in_h + 2 * padding_h;
        let padded_w = in_w + 2 * padding_w;
        let out_h = if padded_h >= kernel_h && stride_h > 0 {
            (padded_h - kernel_h) / stride_h + 1
        } else {
            0
        };
        let out_w = if padded_w >= kernel_w && stride_w > 0 {
            (padded_w - kernel_w) / stride_w + 1
        } else {
            0
        };
        let div_in_w = FastDivider::new(in_w);
        let div_in_h = FastDivider::new(in_h);
        let div_in_channels = FastDivider::new(in_channels);
        let div_stride_h = FastDivider::new(stride_h);
        let div_stride_w = FastDivider::new(stride_w);
        Self {
            kernel,
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
            out_h,
            out_w,
            layout: ConvDataLayout::Chw,
            div_in_w,
            div_in_h,
            div_in_channels,
            div_stride_h,
            div_stride_w,
            _marker: PhantomData,
        }
    }

    /// Specifies tensor memory layout (`Chw` or `Hwc`).
    #[must_use]
    pub const fn with_layout(mut self, layout: ConvDataLayout) -> Self {
        self.layout = layout;
        self
    }
}

impl<Weight: SynapticWeight<T>, T: NumericScalar> SynapseMatrix<T>
    for Conv2dSynapses<'_, Weight, T>
{
    fn len(&self) -> usize {
        self.kernel.len()
    }

    fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [T]) {
        if self.stride_h == 0
            || self.stride_w == 0
            || self.out_h == 0
            || self.out_w == 0
            || self.in_w == 0
            || self.in_h == 0
        {
            return;
        }
        let kernel_spatial_size = self.kernel_h * self.kernel_w;
        let stride_weight_out_ch = self.in_channels * kernel_spatial_size;
        let out_spatial_size = self.out_h * self.out_w;

        for &spiker in active_spikes {
            let spiker = spiker as usize;
            let (in_x, in_y, in_ch) = match self.layout {
                ConvDataLayout::Chw => {
                    let (rem_w, in_x) = self.div_in_w.div_rem(spiker);
                    let (in_ch, in_y) = self.div_in_h.div_rem(rem_w);
                    (in_x, in_y, in_ch)
                }
                ConvDataLayout::Hwc => {
                    let (rem_c, in_ch) = self.div_in_channels.div_rem(spiker);
                    let (in_y, in_x) = self.div_in_w.div_rem(rem_c);
                    (in_x, in_y, in_ch)
                }
            };

            if in_ch < self.in_channels {
                let virt_in_y = in_y + self.padding_h;
                let virt_in_x = in_x + self.padding_w;

                let (min_out_y, max_out_y) = if self.stride_h == 1 {
                    (
                        virt_in_y.saturating_sub(self.kernel_h.saturating_sub(1)),
                        virt_in_y.min(self.out_h.saturating_sub(1)),
                    )
                } else {
                    let min_y = if virt_in_y + 1 >= self.kernel_h {
                        self.div_stride_h.div_ceil(virt_in_y + 1 - self.kernel_h)
                    } else {
                        0
                    };
                    let max_y = self
                        .div_stride_h
                        .div(virt_in_y)
                        .min(self.out_h.saturating_sub(1));
                    (min_y, max_y)
                };

                let (min_out_x, max_out_x) = if self.stride_w == 1 {
                    (
                        virt_in_x.saturating_sub(self.kernel_w.saturating_sub(1)),
                        virt_in_x.min(self.out_w.saturating_sub(1)),
                    )
                } else {
                    let min_x = if virt_in_x + 1 >= self.kernel_w {
                        self.div_stride_w.div_ceil(virt_in_x + 1 - self.kernel_w)
                    } else {
                        0
                    };
                    let max_x = self
                        .div_stride_w
                        .div(virt_in_x)
                        .min(self.out_w.saturating_sub(1));
                    (min_x, max_x)
                };

                if min_out_y <= max_out_y && min_out_x <= max_out_x {
                    for out_y in min_out_y..=max_out_y {
                        let ky = virt_in_y - (out_y * self.stride_h);
                        for out_x in min_out_x..=max_out_x {
                            let kx = virt_in_x - (out_x * self.stride_w);

                            match self.layout {
                                ConvDataLayout::Chw => {
                                    let k_offset = ky * self.kernel_w + kx;
                                    let mut weight_idx = in_ch * kernel_spatial_size + k_offset;
                                    let mut target = out_y * self.out_w + out_x;
                                    for _out_ch in 0..self.out_channels {
                                        if weight_idx < self.kernel.len()
                                            && let Some(acc) = input_accumulators.get_mut(target)
                                        {
                                            *acc = acc.saturating_add(
                                                self.kernel[weight_idx].to_scalar(),
                                            );
                                        }
                                        weight_idx += stride_weight_out_ch;
                                        target += out_spatial_size;
                                    }
                                }
                                ConvDataLayout::Hwc => {
                                    let target_base =
                                        (out_y * self.out_w + out_x) * self.out_channels;
                                    let weight_base =
                                        ((ky * self.kernel_w + kx) * self.in_channels + in_ch)
                                            * self.out_channels;
                                    if target_base + self.out_channels <= input_accumulators.len()
                                        && weight_base + self.out_channels <= self.kernel.len()
                                    {
                                        let acc_slice = &mut input_accumulators
                                            [target_base..target_base + self.out_channels];
                                        let w_slice = &self.kernel
                                            [weight_base..weight_base + self.out_channels];
                                        let (chunks_acc, rem_acc) = acc_slice.as_chunks_mut::<4>();
                                        let (chunks_w, rem_w) = w_slice.as_chunks::<4>();
                                        for (ca, cw) in chunks_acc.iter_mut().zip(chunks_w.iter()) {
                                            ca[0] = ca[0].saturating_add(cw[0].to_scalar());
                                            ca[1] = ca[1].saturating_add(cw[1].to_scalar());
                                            ca[2] = ca[2].saturating_add(cw[2].to_scalar());
                                            ca[3] = ca[3].saturating_add(cw[3].to_scalar());
                                        }
                                        for (acc, w) in rem_acc.iter_mut().zip(rem_w.iter()) {
                                            *acc = acc.saturating_add(w.to_scalar());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn propagate_weighted_inputs(&self, inputs: &[T], input_accumulators: &mut [T]) {
        if self.stride_h == 0
            || self.stride_w == 0
            || self.out_h == 0
            || self.out_w == 0
            || self.in_w == 0
            || self.in_h == 0
        {
            return;
        }
        let kernel_spatial_size = self.kernel_h * self.kernel_w;
        let stride_weight_out_ch = self.in_channels * kernel_spatial_size;
        let out_spatial_size = self.out_h * self.out_w;

        for (src_idx, &val) in inputs.iter().enumerate() {
            if val != T::ZERO {
                let (in_x, in_y, in_ch) = match self.layout {
                    ConvDataLayout::Chw => {
                        let (rem_w, in_x) = self.div_in_w.div_rem(src_idx);
                        let (in_ch, in_y) = self.div_in_h.div_rem(rem_w);
                        (in_x, in_y, in_ch)
                    }
                    ConvDataLayout::Hwc => {
                        let (rem_c, in_ch) = self.div_in_channels.div_rem(src_idx);
                        let (in_y, in_x) = self.div_in_w.div_rem(rem_c);
                        (in_x, in_y, in_ch)
                    }
                };

                if in_ch < self.in_channels {
                    let virt_in_y = in_y + self.padding_h;
                    let virt_in_x = in_x + self.padding_w;

                    let (min_out_y, max_out_y) = if self.stride_h == 1 {
                        (
                            virt_in_y.saturating_sub(self.kernel_h.saturating_sub(1)),
                            virt_in_y.min(self.out_h.saturating_sub(1)),
                        )
                    } else {
                        let min_y = if virt_in_y + 1 >= self.kernel_h {
                            self.div_stride_h.div_ceil(virt_in_y + 1 - self.kernel_h)
                        } else {
                            0
                        };
                        let max_y = self
                            .div_stride_h
                            .div(virt_in_y)
                            .min(self.out_h.saturating_sub(1));
                        (min_y, max_y)
                    };

                    let (min_out_x, max_out_x) = if self.stride_w == 1 {
                        (
                            virt_in_x.saturating_sub(self.kernel_w.saturating_sub(1)),
                            virt_in_x.min(self.out_w.saturating_sub(1)),
                        )
                    } else {
                        let min_x = if virt_in_x + 1 >= self.kernel_w {
                            self.div_stride_w.div_ceil(virt_in_x + 1 - self.kernel_w)
                        } else {
                            0
                        };
                        let max_x = self
                            .div_stride_w
                            .div(virt_in_x)
                            .min(self.out_w.saturating_sub(1));
                        (min_x, max_x)
                    };

                    if min_out_y <= max_out_y && min_out_x <= max_out_x {
                        for out_y in min_out_y..=max_out_y {
                            let ky = virt_in_y - (out_y * self.stride_h);
                            for out_x in min_out_x..=max_out_x {
                                let kx = virt_in_x - (out_x * self.stride_w);

                                match self.layout {
                                    ConvDataLayout::Chw => {
                                        let k_offset = ky * self.kernel_w + kx;
                                        let mut weight_idx = in_ch * kernel_spatial_size + k_offset;
                                        let mut target = out_y * self.out_w + out_x;
                                        for _out_ch in 0..self.out_channels {
                                            if weight_idx < self.kernel.len()
                                                && let Some(acc) =
                                                    input_accumulators.get_mut(target)
                                            {
                                                *acc = acc.saturating_add(
                                                    val * self.kernel[weight_idx].to_scalar(),
                                                );
                                            }
                                            weight_idx += stride_weight_out_ch;
                                            target += out_spatial_size;
                                        }
                                    }
                                    ConvDataLayout::Hwc => {
                                        let target_base =
                                            (out_y * self.out_w + out_x) * self.out_channels;
                                        let weight_base =
                                            ((ky * self.kernel_w + kx) * self.in_channels + in_ch)
                                                * self.out_channels;
                                        if target_base + self.out_channels
                                            <= input_accumulators.len()
                                            && weight_base + self.out_channels <= self.kernel.len()
                                        {
                                            let acc_slice = &mut input_accumulators
                                                [target_base..target_base + self.out_channels];
                                            let w_slice = &self.kernel
                                                [weight_base..weight_base + self.out_channels];
                                            for (acc, w) in acc_slice.iter_mut().zip(w_slice.iter())
                                            {
                                                *acc = acc.saturating_add(val * w.to_scalar());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn propagate_spikes_words(&self, words: &[u32], input_accumulators: &mut [T]) {
        if self.stride_h == 0
            || self.stride_w == 0
            || self.out_h == 0
            || self.out_w == 0
            || self.in_w == 0
            || self.in_h == 0
        {
            return;
        }
        let kernel_spatial_size = self.kernel_h * self.kernel_w;
        let stride_weight_out_ch = self.in_channels * kernel_spatial_size;
        let out_spatial_size = self.out_h * self.out_w;

        for (w_idx, &word) in words.iter().enumerate() {
            if word == 0 {
                continue;
            }
            let mut remaining = word;
            let base_idx = w_idx * 32;

            while remaining != 0 {
                let tz = remaining.trailing_zeros() as usize;
                let spiker = base_idx + tz;
                remaining &= remaining - 1;

                let (in_x, in_y, in_ch) = match self.layout {
                    ConvDataLayout::Chw => {
                        let (rem_w, in_x) = self.div_in_w.div_rem(spiker);
                        let (in_ch, in_y) = self.div_in_h.div_rem(rem_w);
                        (in_x, in_y, in_ch)
                    }
                    ConvDataLayout::Hwc => {
                        let (rem_c, in_ch) = self.div_in_channels.div_rem(spiker);
                        let (in_y, in_x) = self.div_in_w.div_rem(rem_c);
                        (in_x, in_y, in_ch)
                    }
                };

                if in_ch < self.in_channels {
                    let virt_in_y = in_y + self.padding_h;
                    let virt_in_x = in_x + self.padding_w;

                    let (min_out_y, max_out_y) = if self.stride_h == 1 {
                        (
                            virt_in_y.saturating_sub(self.kernel_h.saturating_sub(1)),
                            virt_in_y.min(self.out_h.saturating_sub(1)),
                        )
                    } else {
                        let min_y = if virt_in_y + 1 >= self.kernel_h {
                            self.div_stride_h.div_ceil(virt_in_y + 1 - self.kernel_h)
                        } else {
                            0
                        };
                        let max_y = self
                            .div_stride_h
                            .div(virt_in_y)
                            .min(self.out_h.saturating_sub(1));
                        (min_y, max_y)
                    };

                    let (min_out_x, max_out_x) = if self.stride_w == 1 {
                        (
                            virt_in_x.saturating_sub(self.kernel_w.saturating_sub(1)),
                            virt_in_x.min(self.out_w.saturating_sub(1)),
                        )
                    } else {
                        let min_x = if virt_in_x + 1 >= self.kernel_w {
                            self.div_stride_w.div_ceil(virt_in_x + 1 - self.kernel_w)
                        } else {
                            0
                        };
                        let max_x = self
                            .div_stride_w
                            .div(virt_in_x)
                            .min(self.out_w.saturating_sub(1));
                        (min_x, max_x)
                    };

                    if min_out_y <= max_out_y && min_out_x <= max_out_x {
                        for out_y in min_out_y..=max_out_y {
                            let ky = virt_in_y - (out_y * self.stride_h);
                            for out_x in min_out_x..=max_out_x {
                                let kx = virt_in_x - (out_x * self.stride_w);

                                match self.layout {
                                    ConvDataLayout::Chw => {
                                        let k_offset = ky * self.kernel_w + kx;
                                        let mut weight_idx = in_ch * kernel_spatial_size + k_offset;
                                        let mut target = out_y * self.out_w + out_x;
                                        for _out_ch in 0..self.out_channels {
                                            if weight_idx < self.kernel.len()
                                                && let Some(acc) =
                                                    input_accumulators.get_mut(target)
                                            {
                                                *acc = acc.saturating_add(
                                                    self.kernel[weight_idx].to_scalar(),
                                                );
                                            }
                                            weight_idx += stride_weight_out_ch;
                                            target += out_spatial_size;
                                        }
                                    }
                                    ConvDataLayout::Hwc => {
                                        let target_base =
                                            (out_y * self.out_w + out_x) * self.out_channels;
                                        let weight_base =
                                            ((ky * self.kernel_w + kx) * self.in_channels + in_ch)
                                                * self.out_channels;
                                        if target_base + self.out_channels
                                            <= input_accumulators.len()
                                            && weight_base + self.out_channels <= self.kernel.len()
                                        {
                                            let acc_slice = &mut input_accumulators
                                                [target_base..target_base + self.out_channels];
                                            let w_slice = &self.kernel
                                                [weight_base..weight_base + self.out_channels];
                                            for (acc, w) in acc_slice.iter_mut().zip(w_slice.iter())
                                            {
                                                *acc = acc.saturating_add(w.to_scalar());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::generic_static_neuron::GenericStaticNeuronPopulation;
    use crate::domain::models::lif::{LifConfig, LifDynamics, LifState};
    use crate::domain::rom_csr::RomCsrSynapses;

    #[test]
    #[allow(deprecated)]
    fn test_deep_snn_2_layers_intra_tick() {
        // Layer 1: 2 LIF neurons
        static L1_ROW_PTRS: [u16; 2] = [0, 1];
        static L1_TARGETS: [u16; 1] = [0];
        static L1_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(98304)]; // 1.5
        let l1_synapses = RomCsrSynapses::new(&L1_ROW_PTRS, &L1_TARGETS, &L1_WEIGHTS);
        let l1_neurons = GenericStaticNeuronPopulation::<LifDynamics, 2>::new(
            LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1),
            LifState::default(),
        );
        let layer1 = SpikingFeedforwardLayer::<_, _, 2, 1>::new(l1_neurons, l1_synapses);

        // Layer 2: 2 LIF neurons connected to Layer 1
        static L2_ROW_PTRS: [u16; 3] = [0, 1, 1];
        static L2_TARGETS: [u16; 1] = [1];
        static L2_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(98304)]; // 1.5
        let l2_synapses = RomCsrSynapses::new(&L2_ROW_PTRS, &L2_TARGETS, &L2_WEIGHTS);
        let l2_neurons = GenericStaticNeuronPopulation::<LifDynamics, 2>::new(
            LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1),
            LifState::default(),
        );
        let layer2 = SpikingFeedforwardLayer::<_, _, 2, 1>::new(l2_neurons, l2_synapses);

        let mut deep_net = DeepSpikingNetwork2::<_, _, 2>::new(layer1, layer2);

        // At tick 0, input activates Layer 1 (neuron 0), which immediately activates Layer 2 (neuron 1) WITHIN THE SAME TICK!
        let out_spikes = deep_net.tick(&[Fix32::ONE]);
        assert!(!out_spikes[0]);
        assert!(
            out_spikes[1],
            "Layer 2 must fire at tick 0 thanks to intra-tick propagation"
        );
    }

    #[test]
    fn test_conv1d_synapses_fast_propagation() {
        // Kernel of size 3, 1 input channel, 1 output channel, stride 1
        let kernel = [
            Fix32::from_num(1.0),
            Fix32::from_num(2.0),
            Fix32::from_num(0.5),
        ];
        let conv = Conv1dSynapses::new(&kernel, 1, 1, 3, 1, 5); // in_len = 5 -> out_len = (5 - 3)/1 + 1 = 3
        assert_eq!(conv.out_len, 3);

        let mut acc = [Fix32::ZERO; 3];
        // Spike at input neuron 2 (middle)
        conv.propagate_active_spikes(&[2], &mut acc);
        // in_pos = 2 touches:
        // out_pos 0 (k_idx = 2 - 0 = 2 -> weight 0.5)
        // out_pos 1 (k_idx = 2 - 1 = 1 -> weight 2.0)
        // out_pos 2 (k_idx = 2 - 2 = 0 -> weight 1.0)
        assert_eq!(acc[0], Fix32::from_num(0.5));
        assert_eq!(acc[1], Fix32::from_num(2.0));
        assert_eq!(acc[2], Fix32::from_num(1.0));
    }

    #[test]
    fn test_conv2d_synapses_propagation() {
        // Kernel 2x2, in 3x3, stride 1, 1 channel
        let kernel = [
            Fix32::ONE,
            Fix32::from_num(2.0),
            Fix32::from_num(0.5),
            Fix32::from_num(1.5),
        ];
        let conv = Conv2dSynapses::new(&kernel, 1, 1, 2, 2, 1, 1, 3, 3);
        assert_eq!(conv.out_h, 2);
        assert_eq!(conv.out_w, 2);

        let mut acc = [Fix32::ZERO; 4];
        // Spike at point (y=1, x=1) -> spiker index = 1*3 + 1 = 4
        conv.propagate_active_spikes(&[4], &mut acc);
        // Touch out (0,0) at kx=1, ky=1 -> weight 1.5
        // Touch out (0,1) at kx=0, ky=1 -> weight 0.5
        // Touch out (1,0) at kx=1, ky=0 -> weight 2.0
        // Touch out (1,1) at kx=0, ky=0 -> weight 1.0
        assert_eq!(acc[0], Fix32::from_num(1.5)); // target 0*2+0 = 0
        assert_eq!(acc[1], Fix32::from_num(0.5)); // target 0*2+1 = 1
        assert_eq!(acc[2], Fix32::from_num(2.0)); // target 1*2+0 = 2
        assert_eq!(acc[3], Fix32::ONE); // target 1*2+1 = 3
    }

    #[test]
    fn test_conv1d_synapses_with_padding() {
        // Kernel of size 3, 1 channel, stride 1, padding 1, in_len = 3
        // Padded in_len = 3 + 2 = 5 -> out_len = (5 - 3) / 1 + 1 = 3 ("same" convolution)
        let kernel = [
            Fix32::from_num(1.0),
            Fix32::from_num(2.0),
            Fix32::from_num(0.5),
        ];
        let conv = Conv1dSynapses::with_padding(&kernel, 1, 1, 3, 1, 1, 3);
        assert_eq!(conv.out_len, 3);

        let mut acc = [Fix32::ZERO; 3];
        // Spike at input position 0 (virt_in_pos = 0 + 1 = 1)
        conv.propagate_active_spikes(&[0], &mut acc);
        // virt_in_pos = 1 touches:
        // out_pos 0: k_idx = 1 - 0 = 1 -> weight 2.0
        // out_pos 1: k_idx = 1 - 1 = 0 -> weight 1.0
        assert_eq!(acc[0], Fix32::from_num(2.0));
        assert_eq!(acc[1], Fix32::from_num(1.0));
        assert_eq!(acc[2], Fix32::ZERO);
    }

    #[test]
    fn test_conv2d_synapses_with_padding() {
        // Kernel 3x3, in 2x2, stride 1, padding 1 ("same" convolution)
        // Padded 4x4 -> out 2x2
        let kernel = [Fix32::ONE; 9];
        let conv = Conv2dSynapses::with_padding(&kernel, 1, 1, 3, 3, 1, 1, 1, 1, 2, 2);
        assert_eq!(conv.out_h, 2);
        assert_eq!(conv.out_w, 2);

        let mut acc = [Fix32::ZERO; 4];
        // Spike at in (0, 0) -> virt (1, 1)
        conv.propagate_active_spikes(&[0], &mut acc);
        // With padding 1, corner (0,0) touches all 4 outputs in a 2x2 output map
        assert_eq!(acc[0], Fix32::ONE);
        assert_eq!(acc[1], Fix32::ONE);
        assert_eq!(acc[2], Fix32::ONE);
        assert_eq!(acc[3], Fix32::ONE);
    }

    #[test]
    fn test_residual_spiking_layer() {
        use crate::domain::compact_population::CompactLifPopulation;
        use crate::domain::models::lif::LifConfig;
        use crate::domain::rom_csr::RomCsrSynapses;

        let cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.5), Fix32::ZERO, 0);
        let neurons = CompactLifPopulation::<2>::new(cfg);

        // Identity synapses of size 2: neuron 0 -> neuron 0 (weight 2.0), neuron 1 -> neuron 1 (weight 0.1)
        static ROW_PTR: [usize; 3] = [0, 1, 2];
        static COL_IDX: [usize; 2] = [0, 1];
        // 2.0 in Q16.16 = 131072, 0.1 in Q16.16 = 6554
        static WEIGHTS: [Fix32; 2] = [Fix32::from_bits(131072), Fix32::from_bits(6554)];
        let syn = RomCsrSynapses::new(&ROW_PTR, &COL_IDX, &WEIGHTS);

        let feedforward = SpikingFeedforwardLayer::<_, _, 2, 1>::new(neurons, syn);
        let mut residual = ResidualSpikingLayer::<_, 1>::new(feedforward);

        // Input: neuron 0 = 0 (no spike), neuron 1 = 1 (input spike = 0b10 = 2)
        let in_words = [2u32];
        let out_words = residual.tick_spikes_words(&in_words);

        // Input had spike on neuron 1 (bit 1)
        // Inner layer receives this spike on neuron 1 -> weight 0.1 < threshold 1.0 -> no spike generated by inner
        // Thanks to residual skip-connection, bit 1 is preserved (0b10)!
        assert_eq!(out_words[0], 2);
    }

    #[test]
    fn test_spike_max_pool_2d() {
        // Input: 1 channel, 4x4 image -> MaxPool 2x2 with stride 2 -> Output 2x2
        let mut pool = SpikeMaxPool2d::<1, 4, 4, 2, 2, 2, 2, 1>::new();
        assert_eq!(pool.out_h, 2);
        assert_eq!(pool.out_w, 2);

        // Set spike at (ih=1, iw=1), so in_idx = 1 * 4 + 1 = 5
        let in_words = [1u32 << 5];
        let out_words = pool.tick_spikes_words(&in_words);

        // Window (oh=0, ow=0) covers ih in [0..2], iw in [0..2], including (1,1).
        // Therefore output (0,0) (bit 0) must have fired!
        assert_eq!(out_words[0] & 1, 1);
        // The other 3 quadrants (0,1), (1,0), (1,1) must not have fired
        assert_eq!(out_words[0] & !1, 0);
    }

    #[test]
    fn test_delayed_spiking_layer() {
        // Synaptic matrix 1 -> 1 with weight 1.5
        static ROW_PTRS: [u16; 2] = [0, 1];
        static TARGETS: [u16; 1] = [0];
        static WEIGHTS: [Fix32; 1] = [Fix32::from_bits(98304)]; // 1.5
        let synapses = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS);

        let neurons = GenericStaticNeuronPopulation::<LifDynamics, 1>::new(
            LifConfig::new(Fix32::ONE, Fix32::ZERO, Fix32::ZERO, 0),
            LifState::default(),
        );

        // Delay of 2 ticks (delay_ticks = 2)
        let mut delayed_layer = DelayedSpikingLayer::<_, _, 1, 1, 1, 4>::new(neurons, synapses, 2);

        // Tick 0: input with spike (bit 0 = 1)
        let spikes_t0 = delayed_layer.tick_spikes(&[1]);
        assert!(!spikes_t0.get(0), "At tick 0, spike is delayed");

        // Tick 1: no input
        let spikes_t1 = delayed_layer.tick_spikes(&[0]);
        assert!(!spikes_t1.get(0), "At tick 1, spike has not arrived yet");

        // Tick 2: no input -> spike injected at tick 0 arrives (delay of 2)
        let spikes_t2 = delayed_layer.tick_spikes(&[0]);
        assert!(
            spikes_t2.get(0),
            "At tick 2, spike delayed by 2 ticks excites neuron"
        );
    }

    #[test]
    fn test_spiking_network_macro_chain() {
        // Creates 3 chained layers with spiking_network! macro
        static W: [Fix32; 1] = [Fix32::from_bits(98304)]; // 1.5
        static ROW: [u16; 2] = [0, 1];
        static TGT: [u16; 1] = [0];

        let s1 = RomCsrSynapses::new(&ROW, &TGT, &W);
        let n1 = GenericStaticNeuronPopulation::<LifDynamics, 1>::new(
            LifConfig::new(Fix32::ONE, Fix32::ZERO, Fix32::ZERO, 0),
            LifState::default(),
        );
        let l1 = SpikingFeedforwardLayer::<_, _, 1, 1>::new(n1, s1);

        let s2 = RomCsrSynapses::new(&ROW, &TGT, &W);
        let n2 = GenericStaticNeuronPopulation::<LifDynamics, 1>::new(
            LifConfig::new(Fix32::ONE, Fix32::ZERO, Fix32::ZERO, 0),
            LifState::default(),
        );
        let l2 = SpikingFeedforwardLayer::<_, _, 1, 1>::new(n2, s2);

        let s3 = RomCsrSynapses::new(&ROW, &TGT, &W);
        let n3 = GenericStaticNeuronPopulation::<LifDynamics, 1>::new(
            LifConfig::new(Fix32::ONE, Fix32::ZERO, Fix32::ZERO, 0),
            LifState::default(),
        );
        let l3 = SpikingFeedforwardLayer::<_, _, 1, 1>::new(n3, s3);

        let mut net = spiking_network!(out_n: 1; l1, l2, l3);

        let out = net.tick(&[Fix32::ONE]);
        assert!(
            out[0],
            "Intra-tick propagation across 3 chained layers via macro"
        );
    }

    #[test]
    fn test_conv2d_synapses_hwc_layout() {
        // Kernel 2x2, in 3x3, 1 channel in, 2 channels out, stride 1
        // Weights for out_ch 0 and out_ch 1
        // Shape of HWC kernel: [K_H * K_W * C_IN * C_OUT] = [2 * 2 * 1 * 2] = 8 weights
        let kernel = [
            Fix32::ONE,
            Fix32::from_num(2.0), // (ky=0, kx=0) -> out_ch 0: 1.0, out_ch 1: 2.0
            Fix32::from_num(0.5),
            Fix32::from_num(1.5), // (ky=0, kx=1) -> out_ch 0: 0.5, out_ch 1: 1.5
            Fix32::from_num(3.0),
            Fix32::from_num(0.0), // (ky=1, kx=0) -> out_ch 0: 3.0, out_ch 1: 0.0
            Fix32::from_num(0.2),
            Fix32::from_num(0.8), // (ky=1, kx=1) -> out_ch 0: 0.2, out_ch 1: 0.8
        ];
        let conv =
            Conv2dSynapses::new(&kernel, 1, 2, 2, 2, 1, 1, 3, 3).with_layout(ConvDataLayout::Hwc);
        assert_eq!(conv.out_h, 2);
        assert_eq!(conv.out_w, 2);

        // Accumulators: 2x2 outputs * 2 channels = 8 accumulators
        let mut acc = [Fix32::ZERO; 8];
        // Spike at point (y=1, x=1, c=0) -> spiker index = (1*3 + 1)*1 + 0 = 4
        conv.propagate_active_spikes(&[4], &mut acc);

        // At output point (0,0): ky=1, kx=1 -> w0=0.2, w1=0.8
        let target_base_00 = 0;
        assert_eq!(acc[target_base_00], Fix32::from_num(0.2));
        assert_eq!(acc[target_base_00 + 1], Fix32::from_num(0.8));

        // At output point (1,1): ky=0, kx=0 -> w0=1.0, w1=2.0
        let target_base_11 = 6;
        assert_eq!(acc[target_base_11], Fix32::ONE);
        assert_eq!(acc[target_base_11 + 1], Fix32::from_num(2.0));
    }

    #[test]
    fn test_conv2d_synapses_hwc_weighted_inputs() {
        let kernel = [
            Fix32::ONE,
            Fix32::from_num(2.0),
            Fix32::from_num(0.5),
            Fix32::from_num(1.5),
            Fix32::from_num(3.0),
            Fix32::from_num(0.0),
            Fix32::from_num(0.2),
            Fix32::from_num(0.8),
        ];
        let conv =
            Conv2dSynapses::new(&kernel, 1, 2, 2, 2, 1, 1, 3, 3).with_layout(ConvDataLayout::Hwc);

        let mut inputs = [Fix32::ZERO; 9];
        // Inject 1.5 at point (y=1, x=1, c=0) -> index = 4
        inputs[4] = Fix32::from_num(1.5);

        let mut acc = [Fix32::ZERO; 8];
        conv.propagate_weighted_inputs(&inputs, &mut acc);

        // At output point (0,0): ky=1, kx=1 -> w0=0.2 * 1.5 = 0.3, w1=0.8 * 1.5 = 1.2
        assert!((acc[0].to_float() - 0.3).abs() < 0.001);
        assert!((acc[1].to_float() - 1.2).abs() < 0.001);

        // At output point (1,1): ky=0, kx=0 -> w0=1.0 * 1.5 = 1.5, w1=2.0 * 1.5 = 3.0
        assert!((acc[6].to_float() - 1.5).abs() < 0.001);
        assert!((acc[7].to_float() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_quantized_feedforward_layer() {
        use crate::domain::compact_population::CompactLifPopulation;
        use crate::domain::models::lif::LifConfig;

        static ROW_PTRS: [u16; 3] = [0, 2, 2];
        static TARGETS: [u16; 2] = [0, 1];
        static WEIGHTS_I8: [i8; 2] = [100, 50];
        let scale = Fix32::from_num(0.02); // 100 * 0.02 = 2.0, 50 * 0.02 = 1.0

        let q_synapses = QuantizedRomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS_I8, scale);
        let config = LifConfig::new(
            Fix32::from_num(1.5), // threshold
            Fix32::from_num(0.8), // decay
            Fix32::ZERO,
            1,
        );
        let neurons = CompactLifPopulation::<2>::new(config);
        let mut layer = QuantizedFeedforwardLayer::<_, u16, 2, 1>::new(neurons, q_synapses);

        // Tick with spike from neuron 0 (word = [1])
        let spikes = layer.tick_spikes(&[1]);
        // Neuron 0 receives 2.0 >= 1.5 -> spike!
        // Neuron 1 receives 1.0 < 1.5 -> no spike
        assert!(spikes.get(0));
        assert!(!spikes.get(1));

        layer.reset();
        // Test continuous weighted inputs via SpikingLayer trait
        let in_vals = [Fix32::ONE];
        let out_words = layer.tick_inputs_words(&in_vals);
        assert_eq!(out_words[0] & 1, 1);
        assert_eq!(out_words[0] & 2, 0);
    }

    #[test]
    fn test_conv2d_spikes_words_equivalence() {
        let kernel = [
            Fix32::ONE,
            Fix32::from_num(2.0),
            Fix32::from_num(0.5),
            Fix32::from_num(1.5),
            Fix32::from_num(3.0),
            Fix32::from_num(0.0),
            Fix32::from_num(0.2),
            Fix32::from_num(0.8),
        ];
        let conv =
            Conv2dSynapses::new(&kernel, 1, 2, 2, 2, 1, 1, 3, 3).with_layout(ConvDataLayout::Hwc);

        let mut acc_active = [Fix32::ZERO; 8];
        let mut acc_words = [Fix32::ZERO; 8];

        // Spikes at indices 0 and 4:
        conv.propagate_active_spikes(&[0, 4], &mut acc_active);

        let words = [(1 << 0) | (1 << 4)];
        conv.propagate_spikes_words(&words, &mut acc_words);

        assert_eq!(acc_active, acc_words);
    }

    #[test]
    fn test_conv1d_spikes_words_equivalence() {
        let kernel = [Fix32::ONE, Fix32::from_num(0.5), Fix32::from_num(2.0)];
        let conv = Conv1dSynapses::new(&kernel, 1, 1, 3, 1, 5);

        let mut acc_active = [Fix32::ZERO; 3];
        let mut acc_words = [Fix32::ZERO; 3];

        conv.propagate_active_spikes(&[1, 3], &mut acc_active);

        let words = [(1 << 1) | (1 << 3)];
        conv.propagate_spikes_words(&words, &mut acc_words);

        assert_eq!(acc_active, acc_words);
    }
}
