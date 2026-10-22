//! 📦 Generic static neuron population over a `NeuronDynamics` model (`no_std` pure stack/static).

use crate::domain::numeric::{Fix32, NumericScalar};
use crate::traits::{NeuronDynamics, SpikingPopulation};
use core::marker::PhantomData;

/// Homogeneous population of $N$ neurons, entirely static without dynamic allocation,
/// sharing a single physical configuration (maximum SRAM optimization).
#[derive(Clone, Copy, Debug)]
pub struct GenericStaticNeuronPopulation<
    M: NeuronDynamics<T>,
    const N: usize,
    T: NumericScalar + 'static = Fix32,
> {
    pub config: M::Config,
    pub states: [M::State; N],
    pub input_accumulators: [T; N],
    pub bias: Option<&'static [T; N]>,
    _marker: PhantomData<(M, T)>,
}

impl<M: NeuronDynamics<T>, const N: usize, T: NumericScalar + 'static>
    GenericStaticNeuronPopulation<M, N, T>
{
    /// Creates a homogeneous population where all neurons share the same configuration (single copy in memory).
    pub const fn new(config: M::Config, initial_state: M::State) -> Self {
        Self {
            config,
            states: [initial_state; N],
            input_accumulators: [T::ZERO; N],
            bias: None,
            _marker: PhantomData,
        }
    }

    /// Attaches a static bias vector stored in ROM (e.g. from `nir.Affine`).
    #[must_use]
    pub const fn with_bias(mut self, bias: &'static [T; N]) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Number of neurons in the population.
    pub const fn len(&self) -> usize {
        N
    }

    /// Indicates whether the population is empty.
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Injects input current into neuron `index`.
    #[inline(always)]
    pub fn receive_input(&mut self, index: usize, input: T) {
        if index < N {
            self.input_accumulators[index] += input;
        }
    }
}

impl<M: NeuronDynamics<T>, const N: usize, T: NumericScalar> SpikingPopulation<T>
    for GenericStaticNeuronPopulation<M, N, T>
{
    #[inline]
    fn len(&self) -> usize {
        N
    }

    #[inline]
    fn receive_input(&mut self, index: usize, input: T) {
        self.receive_input(index, input);
    }

    fn reset(&mut self) {
        self.states.fill(M::State::default());
        self.input_accumulators.fill(T::ZERO);
    }

    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn tick_active(
        &mut self,
        current_spikes: &mut [bool],
        active_spikes_buf: &mut [u32],
        active_spikes_count: &mut usize,
    ) {
        let max_n = N.min(current_spikes.len()).min(active_spikes_buf.len());
        *active_spikes_count = 0;

        for i in 0..max_n {
            let bias = self.bias.map_or(T::ZERO, |b| b[i]);
            let input = self.input_accumulators[i] + bias;
            self.input_accumulators[i] = T::ZERO;

            let spike = M::step(&self.config, &mut self.states[i], input);
            current_spikes[i] = spike;

            if spike {
                active_spikes_buf[*active_spikes_count] = i as u32;
                *active_spikes_count += 1;
            }
        }
    }

    #[inline]
    fn tick_bitset(&mut self, bitset_words: &mut [u32]) {
        bitset_words.fill(0);
        for i in 0..N {
            let bias = self.bias.map_or(T::ZERO, |b| b[i]);
            let input = self.input_accumulators[i] + bias;
            self.input_accumulators[i] = T::ZERO;

            let spike = M::step(&self.config, &mut self.states[i], input);
            if spike {
                let w_idx = i / 32;
                let b_idx = i % 32;
                if w_idx < bitset_words.len() {
                    bitset_words[w_idx] |= 1 << b_idx;
                }
            }
        }
    }

    #[inline]
    fn input_accumulators_mut(&mut self) -> &mut [T] {
        &mut self.input_accumulators
    }

    #[inline]
    fn input_accumulators(&self) -> &[T] {
        &self.input_accumulators
    }
}

/// Heterogeneous population of $N$ neurons with individual configurations stored in Flash ROM (`.rodata`).
/// SRAM footprint is strictly zero for configuration data.
#[derive(Clone, Copy, Debug)]
pub struct HeterogeneousStaticNeuronPopulation<
    'a,
    M: NeuronDynamics<T>,
    const N: usize,
    T: NumericScalar + 'static = Fix32,
> {
    pub configs: &'a [M::Config; N],
    pub states: [M::State; N],
    pub input_accumulators: [T; N],
    pub bias: Option<&'static [T; N]>,
    _marker: PhantomData<(M, T)>,
}

impl<'a, M: NeuronDynamics<T>, const N: usize, T: NumericScalar + 'static>
    HeterogeneousStaticNeuronPopulation<'a, M, N, T>
{
    /// Creates a heterogeneous population pointing to immutable configurations in Flash ROM.
    pub const fn new(configs: &'a [M::Config; N], initial_states: [M::State; N]) -> Self {
        Self {
            configs,
            states: initial_states,
            input_accumulators: [T::ZERO; N],
            bias: None,
            _marker: PhantomData,
        }
    }

    /// Attaches a static bias vector stored in ROM.
    #[must_use]
    pub const fn with_bias(mut self, bias: &'static [T; N]) -> Self {
        self.bias = Some(bias);
        self
    }

    pub const fn len(&self) -> usize {
        N
    }

    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    #[inline(always)]
    pub fn receive_input(&mut self, index: usize, input: T) {
        if index < N {
            self.input_accumulators[index] += input;
        }
    }
}

impl<M: NeuronDynamics<T>, const N: usize, T: NumericScalar> SpikingPopulation<T>
    for HeterogeneousStaticNeuronPopulation<'_, M, N, T>
{
    #[inline]
    fn len(&self) -> usize {
        N
    }

    #[inline]
    fn receive_input(&mut self, index: usize, input: T) {
        self.receive_input(index, input);
    }

    fn reset(&mut self) {
        self.states.fill(M::State::default());
        self.input_accumulators.fill(T::ZERO);
    }

    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn tick_active(
        &mut self,
        current_spikes: &mut [bool],
        active_spikes_buf: &mut [u32],
        active_spikes_count: &mut usize,
    ) {
        let max_n = N.min(current_spikes.len()).min(active_spikes_buf.len());
        *active_spikes_count = 0;

        for i in 0..max_n {
            let bias = self.bias.map_or(T::ZERO, |b| b[i]);
            let input = self.input_accumulators[i] + bias;
            self.input_accumulators[i] = T::ZERO;

            let spike = M::step(&self.configs[i], &mut self.states[i], input);
            current_spikes[i] = spike;

            if spike {
                active_spikes_buf[*active_spikes_count] = i as u32;
                *active_spikes_count += 1;
            }
        }
    }

    #[inline]
    fn tick_bitset(&mut self, bitset_words: &mut [u32]) {
        bitset_words.fill(0);
        for i in 0..N {
            let bias = self.bias.map_or(T::ZERO, |b| b[i]);
            let input = self.input_accumulators[i] + bias;
            self.input_accumulators[i] = T::ZERO;

            let spike = M::step(&self.configs[i], &mut self.states[i], input);
            if spike {
                let w_idx = i / 32;
                let b_idx = i % 32;
                if w_idx < bitset_words.len() {
                    bitset_words[w_idx] |= 1 << b_idx;
                }
            }
        }
    }

    #[inline]
    fn input_accumulators_mut(&mut self) -> &mut [T] {
        &mut self.input_accumulators
    }

    #[inline]
    fn input_accumulators(&self) -> &[T] {
        &self.input_accumulators
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::lif::{LifConfig, LifDynamics, LifState};

    #[test]
    fn test_generic_static_population_lif() {
        let config = LifConfig::new(Fix32::from_num(1.0), Fix32::from_num(0.9), Fix32::ZERO, 2);
        let mut pop =
            GenericStaticNeuronPopulation::<LifDynamics, 3>::new(config, LifState::default());

        let mut current_spikes = [false; 3];
        let mut active_spikes = [0u32; 3];
        let mut active_count = 0;

        // Inject current only into neuron 1
        pop.receive_input(1, Fix32::from_num(1.5));
        pop.tick_active(&mut current_spikes, &mut active_spikes, &mut active_count);

        assert_eq!(active_count, 1);
        assert_eq!(active_spikes[0], 1);
        assert!(!current_spikes[0]);
        assert!(current_spikes[1]);
        assert!(!current_spikes[2]);
    }
}
