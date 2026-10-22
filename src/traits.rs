//! 🧠 Fundamental traits for embedded neuromorphic computing (`no_std` pure Fix32)
//!
//! This module defines abstract interfaces for:
//! - [`NeuronDynamics`]: State equations of an individual neuron model (LIF, ALIF, Izhikevich...)
//! - [`SpikingPopulation`]: A population of neurons (static or dynamic)
//! - [`SynapseMatrix`]: Synaptic propagation topology (static CSR, dynamic CSR, etc.)
//! - [`Readout`]: Readout and classification layer at network output
//! - [`SpikingNetwork`]: Full orchestration of a spiking neural network

use crate::domain::numeric::{Fix32, NumericScalar};

/// Trait defining membrane dynamics and state equations of a spiking neuron model.
pub trait NeuronDynamics<T: NumericScalar = Fix32> {
    /// Configurable physical constants and parameters of the model.
    type Config: Copy;

    /// Dynamic neuron state variables (membrane potential, adaptation, refractory state...).
    type State: Copy + Default;

    /// Evaluates a single timestep (tick) for an individual neuron.
    ///
    /// Updates `state` based on `config` and accumulated synaptic `input`.
    /// Returns `true` if the neuron fires an action potential (spike), `false` otherwise.
    fn step(config: &Self::Config, state: &mut Self::State, input: T) -> bool;
}

/// Abstract trait for a spiking neuron population.
pub trait SpikingPopulation<T: NumericScalar = Fix32> {
    /// Returns total number of neurons in the population.
    fn len(&self) -> usize;

    /// Indicates whether the population is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Accumulates synaptic input into neuron `index` accumulator.
    fn receive_input(&mut self, index: usize, input: T);

    /// Resets membrane states and accumulators of the population.
    fn reset(&mut self);

    /// Evaluates all neurons for this tick.
    ///
    /// - `current_spikes`: boolean buffer receiving firing state of each neuron.
    /// - `active_spikes_buf`: buffer receiving compact list of firing neuron indices.
    /// - `active_spikes_count`: counter updated with number of firing neurons.
    fn tick_active(
        &mut self,
        current_spikes: &mut [bool],
        active_spikes_buf: &mut [u32],
        active_spikes_count: &mut usize,
    );

    /// Evaluates all neurons and writes spikes directly into a compact bitset mask (1 bit per neuron).
    /// Eliminates intermediate `[bool; N]` and `[u32; N]` buffers (massive SRAM and CPU cycle saving).
    fn tick_bitset(&mut self, bitset_words: &mut [u32]);

    /// Returns a mutable reference to the neuron input accumulators buffer.
    fn input_accumulators_mut(&mut self) -> &mut [T];

    /// Returns a reference to the neuron input accumulators buffer.
    fn input_accumulators(&self) -> &[T];
}

/// Trait defining operations of a synaptic matrix (spike propagation).
pub trait SynapseMatrix<T: NumericScalar = Fix32> {
    /// Effective number of synapses stored in the structure.
    fn len(&self) -> usize;

    /// Indicates whether the synaptic matrix is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Propagates spikes from active source neurons to target accumulators.
    fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [T]);

    /// Propagates weighted continuous inputs to target accumulators.
    fn propagate_weighted_inputs(&self, inputs: &[T], input_accumulators: &mut [T]);

    /// Propagates spikes encoded as bitset words (1 bit per neuron)
    /// directly to target accumulators (using CTZ instruction).
    #[inline]
    fn propagate_spikes_words(&self, words: &[u32], input_accumulators: &mut [T]) {
        for (w_idx, &word) in words.iter().enumerate() {
            let mut remaining = word;
            let base_idx = (w_idx * 32) as u32;
            while remaining != 0 {
                let tz = remaining.trailing_zeros();
                let src_idx = base_idx + tz;
                self.propagate_active_spikes(&[src_idx], input_accumulators);
                remaining &= remaining - 1;
            }
        }
    }
}

/// Trait for a classification or decoding layer (Readout) of an SNN.
pub trait Readout<T: NumericScalar = Fix32> {
    /// Predicts the class corresponding to the reservoir state vector.
    fn predict(&self, state: &[T]) -> usize;
}

/// Abstract trait for a recurrent or feedforward neuromorphic network.
pub trait SpikingNetwork<T: NumericScalar = Fix32> {
    /// Processes a timestep (tick) with provided inputs.
    /// Returns slice of booleans indicating neurons firing at this tick.
    fn tick(&mut self, inputs: &[T]) -> &[bool];

    /// Processes a timestep (tick) from spikes encoded as bitset words (1 bit per neuron).
    /// Enables 100% event-driven execution without converting to booleans.
    #[inline]
    fn tick_spikes_words(&mut self, _prev_spikes_words: &[u32]) -> &[u32] {
        &[]
    }

    /// Processes a timestep (tick) from continuous inputs and directly returns bitset words (1 bit per neuron).
    #[inline]
    fn tick_inputs_words(&mut self, _inputs: &[T]) -> &[u32] {
        &[]
    }

    /// Resets internal network states.
    fn reset(&mut self);
}
