use crate::domain::numeric::Fix32;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Debug)]
pub struct NeuronPopulation {
    pub membrane_potentials: Vec<Fix32>,
    pub thresholds: Vec<Fix32>,
    pub decay_factors: Vec<Fix32>,
    pub reset_potentials: Vec<Fix32>,
    pub refractory_periods: Vec<u16>,
    pub refractory_times_left: Vec<u16>,
    pub input_accumulators: Vec<Fix32>,
}

impl NeuronPopulation {
    pub fn new(
        size: usize,
        threshold: Fix32,
        decay_factor: Fix32,
        reset_potential: Fix32,
        refractory_period: u16,
    ) -> Self {
        NeuronPopulation {
            membrane_potentials: vec![reset_potential; size],
            thresholds: vec![threshold; size],
            decay_factors: vec![decay_factor; size],
            reset_potentials: vec![reset_potential; size],
            refractory_periods: vec![refractory_period; size],
            refractory_times_left: vec![0; size],
            input_accumulators: vec![Fix32::ZERO; size],
        }
    }

    pub fn len(&self) -> usize {
        self.membrane_potentials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.membrane_potentials.is_empty()
    }

    pub fn receive_input(&mut self, index: usize, input: Fix32) {
        if index < self.input_accumulators.len() {
            self.input_accumulators[index] += input;
        }
    }

    /// Evaluates spikes for all neurons.
    /// `current_spikes` must be correctly sized to the population.
    pub fn tick(&mut self, current_spikes: &mut [bool]) {
        let mut dummy = Vec::new();
        self.tick_with_active_spikes(current_spikes, &mut dummy);
    }

    /// Evaluates spikes for all neurons and populates `active_spikes_buf` with indices of firing neurons.
    /// Zero allocations when `active_spikes_buf` has pre-allocated capacity.
    #[allow(clippy::needless_range_loop)]
    pub fn tick_with_active_spikes(
        &mut self,
        current_spikes: &mut [bool],
        active_spikes_buf: &mut Vec<u32>,
    ) {
        let len = self
            .membrane_potentials
            .len()
            .min(self.thresholds.len())
            .min(self.decay_factors.len())
            .min(self.reset_potentials.len())
            .min(self.refractory_periods.len())
            .min(self.refractory_times_left.len())
            .min(self.input_accumulators.len())
            .min(current_spikes.len());
        active_spikes_buf.clear();

        for i in 0..len {
            let refractory_time_left = &mut self.refractory_times_left[i];
            let input_accumulator = &mut self.input_accumulators[i];
            let membrane_potential = &mut self.membrane_potentials[i];
            let decay_factor = self.decay_factors[i];
            let threshold = self.thresholds[i];
            let reset_potential = self.reset_potentials[i];
            let refractory_period = self.refractory_periods[i];

            if *refractory_time_left > 0 {
                *refractory_time_left -= 1;
                *input_accumulator = Fix32::ZERO;
                *membrane_potential *= decay_factor;
                current_spikes[i] = false;
                continue;
            }

            *membrane_potential = *membrane_potential * decay_factor + *input_accumulator;
            *input_accumulator = Fix32::ZERO;

            let is_spike = *membrane_potential >= threshold;
            if is_spike {
                *membrane_potential = reset_potential;
                *refractory_time_left = refractory_period;
                active_spikes_buf.push(i as u32);
            }
            current_spikes[i] = is_spike;
        }
    }

    /// Resets membrane potentials and clears accumulators.
    pub fn reset(&mut self) {
        let len = self.len();
        for i in 0..len {
            self.membrane_potentials[i] = self.reset_potentials[i];
            self.refractory_times_left[i] = 0;
            self.input_accumulators[i] = Fix32::ZERO;
        }
    }
}

impl crate::traits::SpikingPopulation for NeuronPopulation {
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn receive_input(&mut self, index: usize, input: Fix32) {
        self.receive_input(index, input);
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }

    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn tick_active(
        &mut self,
        current_spikes: &mut [bool],
        active_spikes_buf: &mut [u32],
        active_spikes_count: &mut usize,
    ) {
        let len = self
            .membrane_potentials
            .len()
            .min(self.thresholds.len())
            .min(self.decay_factors.len())
            .min(self.reset_potentials.len())
            .min(self.refractory_periods.len())
            .min(self.refractory_times_left.len())
            .min(self.input_accumulators.len())
            .min(current_spikes.len())
            .min(active_spikes_buf.len());
        *active_spikes_count = 0;

        for i in 0..len {
            let refractory_time_left = &mut self.refractory_times_left[i];
            let input_accumulator = &mut self.input_accumulators[i];
            let membrane_potential = &mut self.membrane_potentials[i];
            let decay_factor = self.decay_factors[i];
            let threshold = self.thresholds[i];
            let reset_potential = self.reset_potentials[i];
            let refractory_period = self.refractory_periods[i];

            if *refractory_time_left > 0 {
                *refractory_time_left -= 1;
                *input_accumulator = Fix32::ZERO;
                *membrane_potential *= decay_factor;
                current_spikes[i] = false;
                continue;
            }

            *membrane_potential = *membrane_potential * decay_factor + *input_accumulator;
            *input_accumulator = Fix32::ZERO;

            let is_spike = *membrane_potential >= threshold;
            if is_spike {
                *membrane_potential = reset_potential;
                *refractory_time_left = refractory_period;
                active_spikes_buf[*active_spikes_count] = i as u32;
                *active_spikes_count += 1;
            }
            current_spikes[i] = is_spike;
        }
    }

    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn tick_bitset(&mut self, bitset_words: &mut [u32]) {
        for w in bitset_words.iter_mut() {
            *w = 0;
        }
        let len = self
            .membrane_potentials
            .len()
            .min(self.thresholds.len())
            .min(self.decay_factors.len())
            .min(self.reset_potentials.len())
            .min(self.refractory_periods.len())
            .min(self.refractory_times_left.len())
            .min(self.input_accumulators.len())
            .min(bitset_words.len() * 32);

        for i in 0..len {
            let refractory_time_left = &mut self.refractory_times_left[i];
            let input_accumulator = &mut self.input_accumulators[i];
            let membrane_potential = &mut self.membrane_potentials[i];
            let decay_factor = self.decay_factors[i];
            let threshold = self.thresholds[i];
            let reset_potential = self.reset_potentials[i];
            let refractory_period = self.refractory_periods[i];

            if *refractory_time_left > 0 {
                *refractory_time_left -= 1;
                *input_accumulator = Fix32::ZERO;
                *membrane_potential *= decay_factor;
                continue;
            }

            *membrane_potential = *membrane_potential * decay_factor + *input_accumulator;
            *input_accumulator = Fix32::ZERO;

            if *membrane_potential >= threshold {
                *membrane_potential = reset_potential;
                *refractory_time_left = refractory_period;
                let w_idx = i / 32;
                let b_idx = i % 32;
                if w_idx < bitset_words.len() {
                    bitset_words[w_idx] |= 1 << b_idx;
                }
            }
        }
    }

    #[inline]
    fn input_accumulators_mut(&mut self) -> &mut [Fix32] {
        &mut self.input_accumulators
    }

    #[inline]
    fn input_accumulators(&self) -> &[Fix32] {
        &self.input_accumulators
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_spikes_and_refractory() {
        let mut pop = NeuronPopulation::new(
            1,
            Fix32::from_num(1.0),
            Fix32::from_num(1.0),
            Fix32::ZERO,
            2,
        );
        let mut spikes = vec![false; 1];

        pop.receive_input(0, Fix32::from_num(0.5));
        pop.tick(&mut spikes);
        assert!(!spikes[0]); // 0.5 < 1.0

        pop.receive_input(0, Fix32::from_num(0.6));
        pop.tick(&mut spikes);
        assert!(spikes[0]); // 0.5 + 0.6 = 1.1 >= 1.0 (spike!)

        // Refractory tick 1
        pop.receive_input(0, Fix32::from_num(2.0)); // massive input
        pop.tick(&mut spikes);
        assert!(!spikes[0]); // Ignored due to refractory period
        assert_eq!(pop.membrane_potentials[0], Fix32::ZERO);

        // Refractory tick 2
        pop.receive_input(0, Fix32::from_num(2.0));
        pop.tick(&mut spikes);
        assert!(!spikes[0]); // Ignored

        // Out of refractory period
        pop.receive_input(0, Fix32::from_num(1.2));
        pop.tick(&mut spikes);
        assert!(spikes[0]); // Spike!
    }

    #[test]
    fn test_neuron_decay_order() {
        let mut pop = NeuronPopulation::new(
            1,
            Fix32::from_num(1.0),
            Fix32::from_num(0.5),
            Fix32::ZERO,
            0,
        );
        let mut spikes = vec![false; 1];

        pop.receive_input(0, Fix32::from_num(0.4));
        pop.tick(&mut spikes);
        assert!(!spikes[0]); // V = 0 * 0.5 + 0.4 = 0.4
        assert_eq!(pop.membrane_potentials[0], Fix32::from_num(0.4));

        pop.receive_input(0, Fix32::from_num(0.2));
        pop.tick(&mut spikes);
        assert!(!spikes[0]); // V = 0.4 * 0.5 + 0.2 = 0.4
        assert_eq!(pop.membrane_potentials[0], Fix32::from_num(0.4));
    }
}
