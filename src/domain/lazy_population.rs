//! ⚡ Pure event-driven neuromorphic inference engine with lazy evaluation (`LazyDecayLifPopulation`).
//!
//! On von Neumann microcontrollers (STM32, RP2040, ESP32, nRF52), ticking and decaying
//! all silent neurons every time step causes massive wasted CPU cycles and energy.
//!
//! This structure implements lazy evaluation:
//! - Each neuron stores its potential $V$ and the tick of its last update $t_{\text{last}}$.
//! - When a synaptic spike arrives at current tick $t$, the potential is updated in a single operation:
//!   $$V(t) = V(t_{\text{last}}) \cdot \text{LUT}[t - t_{\text{last}}] + W$$
//! - Unstimulated neurons consume zero clock cycles.

use crate::domain::bitset::SpikeBitset;
use crate::domain::models::lif::{LifConfig, ResetMode};
use crate::domain::numeric::Fix32;
use crate::traits::SynapseMatrix;

/// LIF neuron population with event-driven lazy evaluation (Lazy Evaluation).
///
/// ONLY updates a neuron when an incoming action potential is delivered to it.
/// Uses a static lookup table (LUT) for membrane decay $\alpha^{\Delta t}$.
#[derive(Clone, Copy, Debug)]
pub struct LazyDecayLifPopulation<const N: usize, const MAX_LUT: usize = 64> {
    pub config: LifConfig,
    pub membrane_potentials: [Fix32; N],
    pub last_update_tick: [u32; N],
    pub current_tick: u32,
    pub decay_lut: [Fix32; MAX_LUT],
    pub bias_lut: [Fix32; MAX_LUT],
    /// Remaining refractory period per neuron (in number of ticks).
    pub refractory_times_left: [u8; N],
    /// Synaptic input accumulators for integration with SpikingPopulation.
    pub input_accumulators: [Fix32; N],
    /// Optional static bias in Flash ROM.
    pub bias: Option<&'static [Fix32; N]>,
}

impl<const N: usize, const MAX_LUT: usize> LazyDecayLifPopulation<N, MAX_LUT> {
    /// Constructs a lazy population and precomputes exponential LUTs $\alpha^{\Delta t}$ and $\sum \alpha^k$.
    pub fn new(config: LifConfig) -> Self {
        let mut decay_lut = [Fix32::ZERO; MAX_LUT];
        let mut bias_lut = [Fix32::ZERO; MAX_LUT];
        if MAX_LUT > 0 {
            decay_lut[0] = Fix32::ONE;
            bias_lut[0] = Fix32::ZERO; // No continuous bias accumulated if dt == 0
            for dt in 1..MAX_LUT {
                decay_lut[dt] = decay_lut[dt - 1] * config.decay_factor;
                bias_lut[dt] = bias_lut[dt - 1] + decay_lut[dt - 1];
            }
        }

        Self {
            config,
            membrane_potentials: [config.reset_potential; N],
            last_update_tick: [0; N],
            current_tick: 0,
            decay_lut,
            bias_lut,
            refractory_times_left: [0; N],
            input_accumulators: [Fix32::ZERO; N],
            bias: None,
        }
    }

    /// Attaches a constant static bias vector stored in Flash ROM.
    #[must_use]
    pub const fn with_bias(mut self, bias: &'static [Fix32; N]) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Advances the global network time by one discretization time step (tick).
    #[inline(always)]
    pub fn advance_tick(&mut self) {
        self.current_tick = self.current_tick.wrapping_add(1);
    }

    /// Resets dynamic state of the population.
    pub fn reset(&mut self) {
        self.membrane_potentials = [self.config.reset_potential; N];
        self.last_update_tick = [0; N];
        self.refractory_times_left = [0; N];
        self.input_accumulators = [Fix32::ZERO; N];
        self.current_tick = 0;
    }

    /// Computes decay factor for a time difference of `dt` ticks.
    #[inline(always)]
    pub fn decay_for_dt(&self, dt: u32) -> Fix32 {
        let dt_idx = dt as usize;
        if dt_idx < MAX_LUT {
            self.decay_lut[dt_idx]
        } else {
            Fix32::ZERO
        }
    }

    /// Computes geometric accumulation factor of bias for a time difference of `dt` ticks:
    /// $$\sum_{k=0}^{\Delta t - 1} \alpha^k = \frac{1 - \alpha^{\Delta t}}{1 - \alpha}$$
    #[inline(always)]
    pub fn bias_decay_for_dt(&self, dt: u32) -> Fix32 {
        let dt_idx = dt as usize;
        if dt_idx < MAX_LUT {
            self.bias_lut[dt_idx]
        } else if let Some(&last_bias) = self.bias_lut.last() {
            last_bias
        } else {
            Fix32::ZERO
        }
    }

    /// Retrieves potential of a neuron updated to the current tick (for readout or inspection).
    #[inline]
    pub fn get_potential_at_current_tick(&self, neuron_idx: usize) -> Fix32 {
        if neuron_idx >= N {
            return Fix32::ZERO;
        }
        let dt = self
            .current_tick
            .saturating_sub(self.last_update_tick[neuron_idx]);
        if dt == 0 {
            self.membrane_potentials[neuron_idx]
        } else {
            let bias = self
                .bias
                .map_or(self.config.bias, |b| b[neuron_idx] + self.config.bias);
            (self.membrane_potentials[neuron_idx] * self.decay_for_dt(dt))
                + (bias * self.bias_decay_for_dt(dt))
        }
    }

    /// Receives a synaptic spike on neuron `neuron_idx` at the current tick.
    /// Only updates membrane potential for this specific neuron without touching others.
    /// Returns `true` if neuron fires a spike, `false` otherwise.
    #[inline]
    pub fn receive_input(&mut self, neuron_idx: usize, input: Fix32) -> bool {
        if neuron_idx >= N {
            return false;
        }

        let dt = self
            .current_tick
            .saturating_sub(self.last_update_tick[neuron_idx]);

        // Update refractory state
        if dt > 0 && self.refractory_times_left[neuron_idx] > 0 {
            let elapsed_refr = dt.min(self.refractory_times_left[neuron_idx] as u32) as u8;
            self.refractory_times_left[neuron_idx] -= elapsed_refr;
        }

        if self.refractory_times_left[neuron_idx] > 0 {
            self.membrane_potentials[neuron_idx] = self.config.reset_potential;
            self.last_update_tick[neuron_idx] = self.current_tick;
            return false;
        }

        let decayed_v = if dt == 0 {
            self.membrane_potentials[neuron_idx]
        } else {
            self.membrane_potentials[neuron_idx] * self.decay_for_dt(dt)
        };

        let bias = self
            .bias
            .map_or(self.config.bias, |b| b[neuron_idx] + self.config.bias);
        let bias_factor = self.bias_decay_for_dt(dt);
        let accumulated_bias = bias * bias_factor;

        let new_v = decayed_v + input + accumulated_bias;
        self.last_update_tick[neuron_idx] = self.current_tick;

        if new_v >= self.config.threshold {
            self.membrane_potentials[neuron_idx] = match self.config.reset_mode {
                ResetMode::Hard => self.config.reset_potential,
                ResetMode::Subtract => new_v - self.config.threshold,
            };
            self.refractory_times_left[neuron_idx] = self.config.refractory_period as u8;
            true
        } else {
            self.membrane_potentials[neuron_idx] = new_v;
            false
        }
    }

    /// Processes active spikes via a synapse matrix (CSR / ROM CSR).
    /// Purely event-driven: single allocation in O(spikes + targets) without re-allocation per spike.
    pub fn propagate_active_spikes<S: SynapseMatrix<Fix32>, const WORDS: usize>(
        &mut self,
        active_spikes: &[u32],
        synapses: &S,
        spikes: &mut SpikeBitset<WORDS>,
    ) {
        spikes.clear();
        let mut acc = [Fix32::ZERO; N];
        synapses.propagate_active_spikes(active_spikes, &mut acc);
        for (tgt, &w) in acc.iter().enumerate() {
            if w != Fix32::ZERO && self.receive_input(tgt, w) {
                spikes.set(tgt);
            }
        }
    }

    /// Directly processes a ROM CSR matrix without stack intermediate buffers (absolute zero allocation).
    pub fn propagate_active_spikes_rom<
        Idx: crate::domain::rom_csr::RomIndex,
        Weight: crate::domain::rom_csr::SynapticWeight<Fix32>,
        const WORDS: usize,
    >(
        &mut self,
        active_spikes: &[u32],
        synapses: &crate::domain::rom_csr::RomCsrSynapses<'_, Idx, Weight>,
        spikes: &mut SpikeBitset<WORDS>,
    ) {
        spikes.clear();
        for &src_idx in active_spikes {
            let src = src_idx as usize;
            if src + 1 < synapses.row_ptrs.len() {
                let start = synapses.row_ptrs[src].to_usize();
                let end = synapses.row_ptrs[src + 1]
                    .to_usize()
                    .min(synapses.targets.len())
                    .min(synapses.weights.len());

                for i in start..end {
                    let target = synapses.targets[i].to_usize();
                    let weight = synapses.weights[i].to_scalar();
                    if self.receive_input(target, weight) {
                        spikes.set(target);
                    }
                }
            }
        }
    }
}

impl<const N: usize, const MAX_LUT: usize> crate::traits::SpikingPopulation<Fix32>
    for LazyDecayLifPopulation<N, MAX_LUT>
{
    #[inline(always)]
    fn len(&self) -> usize {
        N
    }

    #[inline(always)]
    fn receive_input(&mut self, index: usize, input: Fix32) {
        if index < N {
            self.input_accumulators[index] += input;
        }
    }

    fn reset(&mut self) {
        self.reset();
    }

    #[allow(clippy::needless_range_loop)]
    fn tick_active(
        &mut self,
        current_spikes: &mut [bool],
        active_spikes_buf: &mut [u32],
        active_spikes_count: &mut usize,
    ) {
        self.advance_tick();
        *active_spikes_count = 0;
        let limit = N.min(current_spikes.len()).min(active_spikes_buf.len());

        for i in 0..limit {
            let inp = self.input_accumulators[i];
            self.input_accumulators[i] = Fix32::ZERO;

            let spiked = self.receive_input(i, inp);
            current_spikes[i] = spiked;
            if spiked {
                active_spikes_buf[*active_spikes_count] = i as u32;
                *active_spikes_count += 1;
            }
        }
    }

    fn tick_bitset(&mut self, bitset_words: &mut [u32]) {
        self.advance_tick();
        bitset_words.fill(0);

        for i in 0..N {
            let inp = self.input_accumulators[i];
            self.input_accumulators[i] = Fix32::ZERO;

            let spiked = self.receive_input(i, inp);
            if spiked {
                let w_idx = i / 32;
                let b_idx = i % 32;
                if w_idx < bitset_words.len() {
                    bitset_words[w_idx] |= 1 << b_idx;
                }
            }
        }
    }

    #[inline(always)]
    fn input_accumulators_mut(&mut self) -> &mut [Fix32] {
        &mut self.input_accumulators
    }

    #[inline(always)]
    fn input_accumulators(&self) -> &[Fix32] {
        &self.input_accumulators
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::lif::LifDynamics;
    use crate::traits::NeuronDynamics;

    #[test]
    fn test_lazy_decay_bias_accumulation() {
        let decay = Fix32::from_num(0.5);
        let bias = Fix32::from_num(0.2);
        let mut config = LifConfig::new(Fix32::from_num(10.0), decay, Fix32::ZERO, 0);
        config.bias = bias;

        let mut lazy_pop = LazyDecayLifPopulation::<1, 64>::new(config);
        let mut eager_state = crate::domain::models::lif::LifState::default();

        // 3 steps without external spike, but with continuous bias
        for _ in 0..3 {
            lazy_pop.advance_tick();
            LifDynamics::step(&config, &mut eager_state, Fix32::ZERO);
        }

        // Lazy potential at tick 3 must match step-by-step eager evaluation exactly
        let lazy_v = lazy_pop.get_potential_at_current_tick(0);
        let diff = (lazy_v - eager_state.membrane_potential).abs();
        assert!(
            diff <= Fix32::from_bits(2),
            "Bias accumulation error in lazy pop: got lazy={:?}, eager={:?}",
            lazy_v,
            eager_state.membrane_potential
        );
    }

    #[test]
    fn test_lazy_decay_vs_eager_lif_equivalence() {
        let decay = Fix32::from_num(0.8);
        let config = LifConfig::new(Fix32::from_num(1.0), decay, Fix32::ZERO, 0);

        let mut lazy_pop = LazyDecayLifPopulation::<1, 64>::new(config);
        let mut eager_state = crate::domain::models::lif::LifState::default();

        // Tick 0: inject 0.5
        assert!(!lazy_pop.receive_input(0, Fix32::from_num(0.5)));
        assert!(!LifDynamics::step(
            &config,
            &mut eager_state,
            Fix32::from_num(0.5)
        ));
        assert_eq!(
            lazy_pop.get_potential_at_current_tick(0),
            eager_state.membrane_potential
        );

        // Ticks 1..5: silence (lazy pop is untouched, just advance_tick)
        for _ in 1..=4 {
            lazy_pop.advance_tick();
            LifDynamics::step(&config, &mut eager_state, Fix32::ZERO);
        }

        // Verify lazy potential evaluated on-demand matches step-by-step eager computation
        let lazy_v = lazy_pop.get_potential_at_current_tick(0);
        let diff = (lazy_v - eager_state.membrane_potential).abs();
        assert!(
            diff <= Fix32::from_bits(2),
            "Difference between lazy and eager: {:?}",
            diff
        );

        // Tick 5: inject 0.9 -> Triggers spike
        lazy_pop.advance_tick();
        let eager_spiked = LifDynamics::step(&config, &mut eager_state, Fix32::from_num(0.9));
        let lazy_spiked = lazy_pop.receive_input(0, Fix32::from_num(0.9));
        assert_eq!(lazy_spiked, eager_spiked);
    }
}
