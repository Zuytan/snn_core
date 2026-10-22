use crate::domain::numeric::Fix32;

/// Fully static LIF neuron population without dynamic allocation (`no_std` pure stack/static).
///
/// # Deprecation Note
/// This structure duplicates immutable configuration for each neuron in SRAM (~28 bytes/neuron).
/// Prefer [`CompactLifPopulation`](crate::domain::compact_population::CompactLifPopulation) or
/// [`GenericStaticNeuronPopulation`](crate::domain::generic_static_neuron::GenericStaticNeuronPopulation)
/// which store configuration in Flash ROM (60% to 75% SRAM reduction).
#[deprecated(
    since = "0.4.0",
    note = "Use CompactLifPopulation or GenericStaticNeuronPopulation to reduce SRAM footprint by over 60%"
)]
#[derive(Clone, Copy, Debug)]
pub struct StaticNeuronPopulation<const N: usize> {
    pub membrane_potentials: [Fix32; N],
    pub thresholds: [Fix32; N],
    pub decay_factors: [Fix32; N],
    pub reset_potentials: [Fix32; N],
    pub refractory_periods: [u16; N],
    pub refractory_times_left: [u16; N],
    pub input_accumulators: [Fix32; N],
}

#[allow(deprecated)]
impl<const N: usize> StaticNeuronPopulation<N> {
    pub const fn new(
        threshold: Fix32,
        decay_factor: Fix32,
        reset_potential: Fix32,
        refractory_period: u16,
    ) -> Self {
        StaticNeuronPopulation {
            membrane_potentials: [reset_potential; N],
            thresholds: [threshold; N],
            decay_factors: [decay_factor; N],
            reset_potentials: [reset_potential; N],
            refractory_periods: [refractory_period; N],
            refractory_times_left: [0; N],
            input_accumulators: [Fix32::ZERO; N],
        }
    }

    pub const fn len(&self) -> usize {
        N
    }

    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    #[inline(always)]
    pub fn receive_input(&mut self, index: usize, input: Fix32) {
        if index < N {
            self.input_accumulators[index] += input;
        }
    }

    /// Evaluates spikes for all neurons and populates active spikes buffer.
    /// Zero dynamic allocation.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub fn tick_with_active_spikes(
        &mut self,
        current_spikes: &mut [bool; N],
        active_spikes_buf: &mut [u32; N],
        active_spikes_count: &mut usize,
    ) {
        *active_spikes_count = 0;

        for i in 0..N {
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
                if *active_spikes_count < N {
                    active_spikes_buf[*active_spikes_count] = i as u32;
                    *active_spikes_count += 1;
                }
            }
            current_spikes[i] = is_spike;
        }
    }

    /// Resets membrane potentials and clears accumulators.
    pub fn reset(&mut self) {
        for i in 0..N {
            self.membrane_potentials[i] = self.reset_potentials[i];
            self.refractory_times_left[i] = 0;
            self.input_accumulators[i] = Fix32::ZERO;
        }
    }
}

#[allow(deprecated)]
impl<const N: usize> crate::traits::SpikingPopulation for StaticNeuronPopulation<N> {
    #[inline]
    fn len(&self) -> usize {
        N
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
        let max_n = N.min(current_spikes.len()).min(active_spikes_buf.len());
        *active_spikes_count = 0;

        for i in 0..max_n {
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
        bitset_words.fill(0);
        for i in 0..N {
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
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_static_neuron_spikes_and_refractory() {
        let mut pop = StaticNeuronPopulation::<1>::new(
            Fix32::from_num(1.0),
            Fix32::from_num(1.0),
            Fix32::ZERO,
            2,
        );
        let mut spikes = [false; 1];
        let mut active_buf = [0u32; 1];
        let mut active_count = 0;

        pop.receive_input(0, Fix32::from_num(0.5));
        pop.tick_with_active_spikes(&mut spikes, &mut active_buf, &mut active_count);
        assert!(!spikes[0]);
        assert_eq!(active_count, 0);

        pop.receive_input(0, Fix32::from_num(0.6));
        pop.tick_with_active_spikes(&mut spikes, &mut active_buf, &mut active_count);
        assert!(spikes[0]);
        assert_eq!(active_count, 1);
        assert_eq!(active_buf[0], 0);

        // Refractory tick 1
        pop.receive_input(0, Fix32::from_num(2.0));
        pop.tick_with_active_spikes(&mut spikes, &mut active_buf, &mut active_count);
        assert!(!spikes[0]);
        assert_eq!(pop.membrane_potentials[0], Fix32::ZERO);

        // Refractory tick 2
        pop.receive_input(0, Fix32::from_num(2.0));
        pop.tick_with_active_spikes(&mut spikes, &mut active_buf, &mut active_count);
        assert!(!spikes[0]);

        // Exiting refractory period
        pop.receive_input(0, Fix32::from_num(1.2));
        pop.tick_with_active_spikes(&mut spikes, &mut active_buf, &mut active_count);
        assert!(spikes[0]);
    }
}
