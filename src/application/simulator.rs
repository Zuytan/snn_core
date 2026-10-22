use super::time::TimeProvider;
use crate::domain::network::Network;
use crate::domain::numeric::Fix32;
use alloc::vec;
use alloc::vec::Vec;

pub struct SimulationRunner<T: TimeProvider> {
    network: Network,
    time_provider: T,
    total_steps: usize,
    tick_inputs_buf: Vec<Fix32>,
    spike_history: Vec<bool>,
}

impl<T: TimeProvider> SimulationRunner<T> {
    pub fn new(network: Network, time_provider: T, total_steps: usize, num_inputs: usize) -> Self {
        let num_neurons = network.neurons.len();
        SimulationRunner {
            network,
            time_provider,
            total_steps,
            tick_inputs_buf: vec![Fix32::ZERO; num_inputs],
            spike_history: vec![false; num_neurons * total_steps],
        }
    }

    /// Runs the simulation until `TimeProvider::should_continue` returns false.
    /// `input_scenario` provides the inputs for each time step.
    /// Zero dynamic allocation in the simulation loop.
    pub fn run<S: AsRef<[Fix32]>>(&mut self, inputs: &[Fix32], input_scenario: &[S]) -> &Vec<bool> {
        while self.time_provider.should_continue(self.total_steps) {
            let t = self.time_provider.current_step();

            // Build the specific input vector for this tick using pre-allocated buffer
            let scenario = input_scenario[t % input_scenario.len()].as_ref();

            for (j, input) in inputs.iter().enumerate() {
                if j < self.tick_inputs_buf.len() && j < scenario.len() {
                    self.tick_inputs_buf[j] = *input * scenario[j];
                }
            }

            // Tick the network
            let current_spikes = self.network.tick(&self.tick_inputs_buf);

            // Record the spikes
            for (i, &spike) in current_spikes.iter().enumerate() {
                self.spike_history[i * self.total_steps + t] = spike;
            }

            // Advance time
            self.time_provider.advance();
        }

        &self.spike_history
    }
}
