use super::csr::CsrSynapses;
use super::neuron::NeuronPopulation;
use super::numeric::Fix32;
use super::stdp::{CscIndex, StdpConfig, StdpLearner};
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
    pub active_inputs_buf: Vec<u32>,
    pub stdp: Option<StdpLearner>,
    pub input_csc: Option<CscIndex>,
    pub internal_csc: Option<CscIndex>,
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
            active_inputs_buf: Vec::new(),
            stdp: None,
            input_csc: None,
            internal_csc: None,
        }
    }

    /// Enables STDP synaptic plasticity learning.
    pub fn enable_stdp(&mut self, num_inputs: usize, config: StdpConfig) {
        self.stdp = Some(StdpLearner::new(num_inputs, self.neurons.len(), config));
        self.input_csc = Some(CscIndex::from_csr(
            self.neurons.len(),
            &self.input_synapses.row_ptrs,
            &self.input_synapses.targets,
        ));
        self.internal_csc = Some(CscIndex::from_csr(
            self.neurons.len(),
            &self.internal_synapses.row_ptrs,
            &self.internal_synapses.targets,
        ));
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
    /// Executes in O(spikes * in-degree) without per-tick dynamic allocations.
    pub fn tick_with_learning(&mut self, inputs: &[Fix32]) -> &[bool] {
        self.active_inputs_buf.clear();
        for (idx, &v) in inputs.iter().enumerate() {
            if v >= Fix32::ONE {
                self.active_inputs_buf.push(idx as u32);
            }
        }

        let _ = self.tick(inputs);

        if let (Some(stdp), Some(in_csc), Some(int_csc)) =
            (&mut self.stdp, &self.input_csc, &self.internal_csc)
        {
            stdp.apply_stdp_csc(
                &self.active_inputs_buf,
                &self.active_spikes_buf,
                &self.input_synapses.row_ptrs,
                &self.input_synapses.targets,
                &mut self.input_synapses.weights,
                in_csc,
                &self.internal_synapses.row_ptrs,
                &self.internal_synapses.targets,
                &mut self.internal_synapses.weights,
                int_csc,
            );
        }

        &self.current_spikes
    }

    /// Resets internal state of neuron population and spike buffers.
    pub fn reset(&mut self) {
        self.neurons.reset();
        self.current_spikes.fill(false);
        self.active_spikes_buf.clear();
        self.active_inputs_buf.clear();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_tick_with_learning_csc() {
        let neurons = NeuronPopulation::new(
            2,
            Fix32::from_num(1.0),
            Fix32::from_num(0.5),
            Fix32::ZERO,
            1,
        );
        let in_syn = [
            Synapse::new(0, 0, Fix32::from_num(1.5)),
            Synapse::new(0, 1, Fix32::from_num(0.5)),
        ];
        let int_syn = [Synapse::new(0, 1, Fix32::from_num(0.2))];
        let mut net = Network::new(neurons, &int_syn, &in_syn);
        let cfg = StdpConfig::default();
        net.enable_stdp(1, cfg);

        // Tick 1: input 1.0 -> neuron 0 fires
        let spikes = net.tick_with_learning(&[Fix32::ONE]);
        assert!(spikes[0]);
        assert!(!spikes[1]);

        // Tick 2: neuron 1 receives input and spikes
        let spikes2 = net.tick_with_learning(&[Fix32::from_num(2.0)]);
        assert!(spikes2[0] || spikes2[1]);
        assert!(net.input_csc.is_some());
        assert!(net.internal_csc.is_some());
    }
}
