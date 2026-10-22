use super::csr::CsrSynapses;
use super::neuron::NeuronPopulation;
use super::numeric::Fix32;
use super::stdp::{StdpConfig, StdpLearner};
use super::synapse::Synapse;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone)]
pub struct Network {
    pub neurons: NeuronPopulation,
    pub internal_synapses: CsrSynapses,
    pub input_synapses: CsrSynapses,
    pub current_spikes: Vec<bool>,
    pub active_spikes_buf: Vec<u32>,
    pub stdp: Option<StdpLearner>,
}

impl Network {
    /// Creates an SNN network from AoS synapse lists (automatically converted to CSR).
    pub fn new(
        neurons: NeuronPopulation,
        internal_synapses: &[Synapse],
        input_synapses: &[Synapse],
    ) -> Self {
        let n_len = neurons.len();
        let num_inputs = input_synapses
            .iter()
            .map(|s| s.source_idx)
            .max()
            .map_or(0, |m| m + 1);

        let internal_csr = CsrSynapses::from_synapses(n_len, internal_synapses);
        let input_csr = CsrSynapses::from_synapses(num_inputs, input_synapses);

        Self::new_csr(neurons, internal_csr, input_csr)
    }

    /// Creates an SNN network directly from CSR synaptic structures.
    pub fn new_csr(
        neurons: NeuronPopulation,
        internal_synapses: CsrSynapses,
        input_synapses: CsrSynapses,
    ) -> Self {
        let n_len = neurons.len();
        Network {
            neurons,
            internal_synapses,
            input_synapses,
            current_spikes: vec![false; n_len],
            active_spikes_buf: Vec::with_capacity(n_len),
            stdp: None,
        }
    }

    /// Enables STDP synaptic plasticity learning.
    pub fn enable_stdp(&mut self, num_inputs: usize, config: StdpConfig) {
        self.stdp = Some(StdpLearner::new(num_inputs, self.neurons.len(), config));
    }

    /// Processes a simulation time step (Tick) driven by events (Spike-Driven CSR).
    /// Zero dynamic allocation during execution.
    pub fn tick(&mut self, inputs: &[Fix32]) -> &[bool] {
        // 1. Accumulate external inputs via secure CSR
        self.input_synapses
            .propagate_weighted_inputs(inputs, &mut self.neurons.input_accumulators);

        // 2. Evaluate neurons & extract active firing neurons without allocation
        self.neurons
            .tick_with_active_spikes(&mut self.current_spikes, &mut self.active_spikes_buf);

        // 3. Ultra-fast EVENT-DRIVEN propagation of internal spikes
        self.internal_synapses.propagate_active_spikes(
            &self.active_spikes_buf,
            &mut self.neurons.input_accumulators,
        );

        &self.current_spikes
    }

    /// Processes a time step with online STDP weight updates.
    pub fn tick_with_learning(&mut self, inputs: &[Fix32]) -> &[bool] {
        let mut active_inputs = Vec::new();
        for (idx, &v) in inputs.iter().enumerate() {
            if v >= Fix32::ONE {
                active_inputs.push(idx as u32);
            }
        }

        let _ = self.tick(inputs);

        if let Some(ref mut stdp) = self.stdp {
            stdp.apply_stdp(
                &active_inputs,
                &self.active_spikes_buf,
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
        self.current_spikes.fill(false);
        self.active_spikes_buf.clear();
    }
}

impl crate::traits::SpikingNetwork for Network {
    #[inline]
    fn tick(&mut self, inputs: &[Fix32]) -> &[bool] {
        self.tick(inputs)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }
}
