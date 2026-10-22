use super::numeric::Fix32;
use super::static_csr::StaticCsrSynapses;
#[allow(deprecated)]
use super::static_neuron::StaticNeuronPopulation;
use super::stdp::{StaticStdpLearner, StdpConfig};
use super::synapse::Synapse;

/// 100% static Spiking Neural Network (SNN) without heap allocation during inference (`no_std`).
/// - `N`: Number of neurons in the reservoir
/// - `N_PLUS_1`: N + 1 (for internal row_ptrs)
/// - `MAX_INTERNAL`: Max capacity of internal recurrent synapses
/// - `MAX_INPUT`: Max capacity of input synapses
/// - `N_INPUTS`: Number of input channels
/// - `N_INPUTS_PLUS_1`: N_INPUTS + 1 (for input row_ptrs)
#[allow(deprecated)]
#[derive(Clone, Debug)]
pub struct StaticNetwork<
    const N: usize,
    const N_PLUS_1: usize,
    const MAX_INTERNAL: usize,
    const MAX_INPUT: usize,
    const N_INPUTS: usize,
    const N_INPUTS_PLUS_1: usize,
> {
    pub neurons: StaticNeuronPopulation<N>,
    pub internal_synapses: StaticCsrSynapses<N_PLUS_1, MAX_INTERNAL>,
    pub input_synapses: StaticCsrSynapses<N_INPUTS_PLUS_1, MAX_INPUT>,
    pub current_spikes: [bool; N],
    pub active_spikes_buf: [u32; N],
    pub active_spikes_count: usize,
    pub stdp: Option<StaticStdpLearner<N_INPUTS, N>>,
}

#[allow(deprecated)]
impl<
    const N: usize,
    const N_PLUS_1: usize,
    const MAX_INTERNAL: usize,
    const MAX_INPUT: usize,
    const N_INPUTS: usize,
    const N_INPUTS_PLUS_1: usize,
> StaticNetwork<N, N_PLUS_1, MAX_INTERNAL, MAX_INPUT, N_INPUTS, N_INPUTS_PLUS_1>
{
    /// Creates a static network from a neuron population and synapse slices.
    pub fn new(
        neurons: StaticNeuronPopulation<N>,
        internal_synapses: &[Synapse],
        input_synapses: &[Synapse],
    ) -> Self {
        let internal_csr = StaticCsrSynapses::from_synapses(internal_synapses);
        let input_csr = StaticCsrSynapses::from_synapses(input_synapses);

        Self::new_csr(neurons, internal_csr, input_csr)
    }

    /// Creates a static network directly from static CSR matrices.
    pub const fn new_csr(
        neurons: StaticNeuronPopulation<N>,
        internal_synapses: StaticCsrSynapses<N_PLUS_1, MAX_INTERNAL>,
        input_synapses: StaticCsrSynapses<N_INPUTS_PLUS_1, MAX_INPUT>,
    ) -> Self {
        assert!(N_PLUS_1 == N + 1, "N_PLUS_1 must equal N + 1");
        assert!(
            N_INPUTS_PLUS_1 == N_INPUTS + 1,
            "N_INPUTS_PLUS_1 must equal N_INPUTS + 1"
        );

        StaticNetwork {
            neurons,
            internal_synapses,
            input_synapses,
            current_spikes: [false; N],
            active_spikes_buf: [0u32; N],
            active_spikes_count: 0,
            stdp: None,
        }
    }

    /// Enables STDP learning in pure no_std mode.
    pub fn enable_stdp(&mut self, config: StdpConfig) {
        self.stdp = Some(StaticStdpLearner::new(config));
    }

    /// Processes a time step (tick) with `N_INPUTS` input spikes/values.
    /// Guaranteed zero dynamic allocation.
    #[inline]
    pub fn tick(&mut self, inputs: &[Fix32; N_INPUTS]) -> &[bool; N] {
        self.tick_slice(inputs)
    }

    /// Processes a time step with an arbitrary input slice without copying on the stack.
    #[inline]
    pub fn tick_slice(&mut self, inputs: &[Fix32]) -> &[bool; N] {
        // 1. Accumulate external inputs via static CSR
        self.input_synapses
            .propagate_weighted_inputs(inputs, &mut self.neurons.input_accumulators);

        // 2. Evaluate neurons and extract active spikes without allocation
        self.neurons.tick_with_active_spikes(
            &mut self.current_spikes,
            &mut self.active_spikes_buf,
            &mut self.active_spikes_count,
        );

        // 3. Ultra-fast event-driven propagation of internal recurrent spikes
        let active_slice = &self.active_spikes_buf[..self.active_spikes_count];
        self.internal_synapses
            .propagate_active_spikes(active_slice, &mut self.neurons.input_accumulators);

        &self.current_spikes
    }

    /// Processes a time step with online STDP weight updates.
    pub fn tick_with_learning(&mut self, inputs: &[Fix32; N_INPUTS]) -> &[bool; N] {
        let mut active_inputs = [0u32; N_INPUTS];
        let mut active_count = 0;
        for (idx, &v) in inputs.iter().enumerate() {
            if v >= Fix32::ONE && active_count < N_INPUTS {
                active_inputs[active_count] = idx as u32;
                active_count += 1;
            }
        }

        let _ = self.tick(inputs);

        if let Some(ref mut stdp) = self.stdp {
            let active_spikes = &self.active_spikes_buf[..self.active_spikes_count];
            stdp.apply_stdp(
                &active_inputs[..active_count],
                active_spikes,
                &self.input_synapses.row_ptrs,
                &self.input_synapses.targets,
                &mut self.input_synapses.weights,
                &self.internal_synapses.row_ptrs,
                &self.internal_synapses.targets,
                &mut self.internal_synapses.weights,
            );
        }

        &self.current_spikes
    }

    /// Resets internal state of neuron population and spike buffers.
    pub fn reset(&mut self) {
        self.neurons.reset();
        self.current_spikes = [false; N];
        self.active_spikes_buf = [0u32; N];
        self.active_spikes_count = 0;
    }
}

impl<
    const N: usize,
    const N_PLUS_1: usize,
    const MAX_INTERNAL: usize,
    const MAX_INPUT: usize,
    const N_INPUTS: usize,
    const N_INPUTS_PLUS_1: usize,
> crate::traits::SpikingNetwork
    for StaticNetwork<N, N_PLUS_1, MAX_INTERNAL, MAX_INPUT, N_INPUTS, N_INPUTS_PLUS_1>
{
    #[inline]
    fn tick(&mut self, inputs: &[Fix32]) -> &[bool] {
        self.tick_slice(inputs)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }
}

/// Generic static spiking neural network over neuron population and synapse matrices (`no_std`).
///
/// Orchestrates any population conforming to the [`crate::traits::SpikingPopulation`] trait
/// (LIF, ALIF, Izhikevich, heterogeneous population) and any topology conforming to the [`crate::traits::SynapseMatrix`] trait
/// (static CSR matrices or immutable Flash ROM matrices [`crate::domain::rom_csr::RomCsrSynapses`]).
#[derive(Clone, Debug)]
pub struct GenericStaticNetwork<
    P: crate::traits::SpikingPopulation<T>,
    SIn: crate::traits::SynapseMatrix<T>,
    SRec: crate::traits::SynapseMatrix<T>,
    const N: usize,
    const N_INPUTS: usize,
    T: crate::domain::numeric::NumericScalar + 'static = Fix32,
> {
    pub neurons: P,
    pub internal_synapses: SRec,
    pub input_synapses: SIn,
    pub current_spikes: [bool; N],
    pub active_spikes_buf: [u32; N],
    pub active_spikes_count: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<
    P: crate::traits::SpikingPopulation<T>,
    SIn: crate::traits::SynapseMatrix<T>,
    SRec: crate::traits::SynapseMatrix<T>,
    const N: usize,
    const N_INPUTS: usize,
    T: crate::domain::numeric::NumericScalar + 'static,
> GenericStaticNetwork<P, SIn, SRec, N, N_INPUTS, T>
{
    /// Constructs a generic static network.
    pub const fn new(neurons: P, internal_synapses: SRec, input_synapses: SIn) -> Self {
        Self {
            neurons,
            internal_synapses,
            input_synapses,
            current_spikes: [false; N],
            active_spikes_buf: [0u32; N],
            active_spikes_count: 0,
            _marker: core::marker::PhantomData,
        }
    }

    /// Processes a simulation time step (Tick) without dynamic allocation.
    #[inline]
    pub fn tick(&mut self, inputs: &[T; N_INPUTS]) -> &[bool; N] {
        self.tick_slice(inputs)
    }

    /// Processes a time step with an arbitrary input slice without copying on the stack.
    #[inline]
    pub fn tick_slice(&mut self, inputs: &[T]) -> &[bool; N] {
        // 1. External accumulation
        self.input_synapses
            .propagate_weighted_inputs(inputs, self.neurons.input_accumulators_mut());

        // 2. Neuron evaluation
        self.neurons.tick_active(
            &mut self.current_spikes,
            &mut self.active_spikes_buf,
            &mut self.active_spikes_count,
        );

        // 3. Internal recurrent propagation
        let active_slice = &self.active_spikes_buf[..self.active_spikes_count];
        self.internal_synapses
            .propagate_active_spikes(active_slice, self.neurons.input_accumulators_mut());

        &self.current_spikes
    }

    /// Resets population state and spike buffers.
    pub fn reset(&mut self) {
        self.neurons.reset();
        self.current_spikes = [false; N];
        self.active_spikes_buf = [0u32; N];
        self.active_spikes_count = 0;
    }
}

impl<
    P: crate::traits::SpikingPopulation<T>,
    SIn: crate::traits::SynapseMatrix<T>,
    SRec: crate::traits::SynapseMatrix<T>,
    const N: usize,
    const N_INPUTS: usize,
    T: crate::domain::numeric::NumericScalar + 'static,
> crate::traits::SpikingNetwork<T> for GenericStaticNetwork<P, SIn, SRec, N, N_INPUTS, T>
{
    #[inline]
    fn tick(&mut self, inputs: &[T]) -> &[bool] {
        self.tick_slice(inputs)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }
}

/// Ultra-lightweight neuromorphic recurrent reservoir with bitset vector spike bus (`no_std`).
///
/// Unlike legacy structures using boolean buffers `[bool; N]`,
/// `StaticReservoir` directly utilizes [`crate::domain::bitset::SpikeBitset<WORDS>`].
/// Internal spikes are propagated via native CPU instruction CTZ (`trailing_zeros`).
#[derive(Clone, Copy, Debug)]
pub struct StaticReservoir<
    P: crate::traits::SpikingPopulation<T>,
    SRec: crate::traits::SynapseMatrix<T>,
    SIn: crate::traits::SynapseMatrix<T>,
    const WORDS: usize,
    T: crate::domain::numeric::NumericScalar + 'static = Fix32,
> {
    pub neurons: P,
    pub internal_synapses: SRec,
    pub input_synapses: SIn,
    pub spikes: crate::domain::bitset::SpikeBitset<WORDS>,
    _marker: core::marker::PhantomData<T>,
}

impl<
    P: crate::traits::SpikingPopulation<T>,
    SRec: crate::traits::SynapseMatrix<T>,
    SIn: crate::traits::SynapseMatrix<T>,
    const WORDS: usize,
    T: crate::domain::numeric::NumericScalar + 'static,
> StaticReservoir<P, SRec, SIn, WORDS, T>
{
    pub const fn new(neurons: P, internal_synapses: SRec, input_synapses: SIn) -> Self {
        Self {
            neurons,
            internal_synapses,
            input_synapses,
            spikes: crate::domain::bitset::SpikeBitset::new(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Executes a full simulation time step (tick):
    /// 1. Propagation of continuous synaptic inputs
    /// 2. Neuron evaluation and direct write to bitset
    /// 3. Internal recurrent propagation via CTZ on bitset words
    #[inline]
    pub fn tick(&mut self, inputs: &[T]) -> &crate::domain::bitset::SpikeBitset<WORDS> {
        self.input_synapses
            .propagate_weighted_inputs(inputs, self.neurons.input_accumulators_mut());

        self.neurons.tick_bitset(self.spikes.words_mut());

        self.internal_synapses
            .propagate_spikes_words(self.spikes.words(), self.neurons.input_accumulators_mut());

        &self.spikes
    }

    /// Resets membrane state and clears spikes.
    pub fn reset(&mut self) {
        self.neurons.reset();
        self.spikes.clear();
    }
}
