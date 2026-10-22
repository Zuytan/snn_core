//! ⚡ Ultra-compact LIF neuron populations in Structure-of-Arrays (SoA) layout.
//!
//! Designed to minimize SRAM footprint on bare-metal microcontrollers (`no_std`):
//! - Total elimination of inter-neuron padding (unlike AoS struct arrays)
//! - Direct support for Flash ROM static bias vectors (absorbing `nir.Affine` nodes)
//! - Available in 32-bit (`CompactLifPopulation`) and 16-bit (`CompactLif16Population`) precision.

use crate::domain::models::lif::{Lif16Config, LifConfig, ResetMode};
use crate::domain::numeric::{Fix16, Fix32};
use crate::traits::SpikingPopulation;

/// Homogeneous population of $N$ Leaky Integrate-and-Fire neurons (Fix32)
/// arranged in Structure-of-Arrays (SoA) for minimal and vectorizable SRAM footprint.
/// SRAM footprint: 9 bytes per neuron (4B membrane + 4B acc + 1B refractory).
///
/// # Example
/// ```rust
/// use snn_core::domain::compact_population::CompactLifPopulation;
/// use snn_core::models::lif::LifConfig;
/// use snn_core::Fix32;
///
/// let config = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
/// let mut pop = CompactLifPopulation::<4>::new(config);
///
/// pop.receive_input(0, Fix32::from_num(1.5));
/// let fired = pop.step_neuron(0);
/// assert!(fired); // 1.5 >= 1.0 -> Spikes
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CompactLifPopulation<const N: usize> {
    /// Shared physical configuration (threshold, decay, reset potential, refractory period).
    pub config: LifConfig,
    /// Membrane potentials (4 bytes per neuron).
    pub membrane_potentials: [Fix32; N],
    /// Synaptic input accumulators (4 bytes per neuron).
    pub input_accumulators: [Fix32; N],
    /// Remaining refractory time (1 byte per neuron).
    pub refractory_times_left: [u8; N],
    /// Optional constant static bias in Flash ROM.
    pub bias: Option<&'static [Fix32; N]>,
}

impl<const N: usize> CompactLifPopulation<N> {
    /// Creates a new compact population initialized at rest.
    pub const fn new(config: LifConfig) -> Self {
        Self {
            config,
            membrane_potentials: [config.reset_potential; N],
            input_accumulators: [Fix32::ZERO; N],
            refractory_times_left: [0; N],
            bias: None,
        }
    }

    /// Attaches a constant static bias vector stored in Flash ROM.
    #[must_use]
    pub const fn with_bias(mut self, bias: &'static [Fix32; N]) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Number of neurons in the population.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        N
    }

    /// Indicates whether the population is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Injects synaptic current into neuron `index`.
    #[inline(always)]
    pub fn receive_input(&mut self, index: usize, input: Fix32) {
        if index < N {
            self.input_accumulators[index] += input;
        }
    }

    /// Evaluates one tick for an individual neuron `i`.
    #[inline(always)]
    pub fn step_neuron(&mut self, i: usize) -> bool {
        if i >= N {
            return false;
        }
        let refractory = &mut self.refractory_times_left[i];
        let bias = self
            .bias
            .map_or(self.config.bias, |b| b[i] + self.config.bias);
        let input = self.input_accumulators[i] + bias;
        self.input_accumulators[i] = Fix32::ZERO;

        if *refractory > 0 {
            *refractory -= 1;
            self.membrane_potentials[i] = self.config.reset_potential;
            return false;
        }

        #[cfg(feature = "cortex-m0")]
        let decayed = crate::domain::hardware::fast_decay_m0(
            self.membrane_potentials[i],
            (self.config.decay_factor.0 as u32 & 0xFFFF) as u16,
        );
        #[cfg(not(feature = "cortex-m0"))]
        let decayed = self.membrane_potentials[i] * self.config.decay_factor;
        let v = decayed + input;

        if v >= self.config.threshold {
            self.membrane_potentials[i] = match self.config.reset_mode {
                ResetMode::Hard => self.config.reset_potential,
                ResetMode::Subtract => v - self.config.threshold,
            };
            *refractory = self.config.refractory_period as u8;
            true
        } else {
            self.membrane_potentials[i] = v;
            false
        }
    }
}

impl<const N: usize> SpikingPopulation<Fix32> for CompactLifPopulation<N> {
    #[inline]
    fn len(&self) -> usize {
        N
    }

    #[inline]
    fn receive_input(&mut self, index: usize, input: Fix32) {
        self.receive_input(index, input);
    }

    fn reset(&mut self) {
        for v in &mut self.membrane_potentials {
            *v = self.config.reset_potential;
        }
        for refr in &mut self.refractory_times_left {
            *refr = 0;
        }
        for acc in &mut self.input_accumulators {
            *acc = Fix32::ZERO;
        }
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
            let spike = self.step_neuron(i);
            current_spikes[i] = spike;
            if spike {
                active_spikes_buf[*active_spikes_count] = i as u32;
                *active_spikes_count += 1;
            }
        }
    }

    #[inline]
    fn tick_bitset(&mut self, bitset_words: &mut [u32]) {
        for w in bitset_words.iter_mut() {
            *w = 0;
        }
        for i in 0..N {
            let spike = self.step_neuron(i);
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
    fn input_accumulators_mut(&mut self) -> &mut [Fix32] {
        &mut self.input_accumulators
    }

    #[inline]
    fn input_accumulators(&self) -> &[Fix32] {
        &self.input_accumulators
    }
}

/// Homogeneous population of $N$ Leaky Integrate-and-Fire neurons in 16-bit fixed-point (`Fix16`).
/// Record-low SRAM footprint: exactly 5 bytes per neuron (instead of 24 bytes in the initial AoS design).
///
/// # Example
/// ```rust
/// use snn_core::domain::compact_population::CompactLif16Population;
/// use snn_core::models::lif::Lif16Config;
/// use snn_core::Fix16;
///
/// let config = Lif16Config::new(Fix16::ONE, Fix16::from_num(0.9), Fix16::ZERO, 1);
/// let mut pop = CompactLif16Population::<4>::new(config);
///
/// pop.receive_input(0, Fix16::from_num(1.5));
/// let fired = pop.step_neuron(0);
/// assert!(fired);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CompactLif16Population<const N: usize> {
    /// Shared 16-bit physical configuration.
    pub config: Lif16Config,
    /// Membrane potentials (2 bytes per neuron).
    pub membrane_potentials: [Fix16; N],
    /// Synaptic input accumulators (2 bytes per neuron).
    pub input_accumulators: [Fix16; N],
    /// Remaining refractory time (1 byte per neuron).
    pub refractory_times_left: [u8; N],
    /// Optional constant static bias in Flash ROM (2 bytes per neuron in Flash).
    pub bias: Option<&'static [Fix16; N]>,
}

impl<const N: usize> CompactLif16Population<N> {
    /// Creates a new compact 16-bit population initialized at rest.
    pub const fn new(config: Lif16Config) -> Self {
        Self {
            config,
            membrane_potentials: [config.reset_potential; N],
            input_accumulators: [Fix16::ZERO; N],
            refractory_times_left: [0; N],
            bias: None,
        }
    }

    /// Attaches a constant static bias vector stored in Flash ROM.
    #[must_use]
    pub const fn with_bias(mut self, bias: &'static [Fix16; N]) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Number of neurons in the population.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        N
    }

    /// Indicates whether the population is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Injects synaptic current into neuron `index`.
    #[inline(always)]
    pub fn receive_input(&mut self, index: usize, input: Fix16) {
        if index < N {
            self.input_accumulators[index] += input;
        }
    }

    /// Evaluates one tick for individual neuron `i` in saturating 16-bit arithmetic.
    #[inline(always)]
    pub fn step_neuron(&mut self, i: usize) -> bool {
        if i >= N {
            return false;
        }
        let refractory = &mut self.refractory_times_left[i];
        let bias = self
            .bias
            .map_or(self.config.bias, |b| b[i] + self.config.bias);
        let input = self.input_accumulators[i] + bias;
        self.input_accumulators[i] = Fix16::ZERO;

        if *refractory > 0 {
            *refractory -= 1;
            self.membrane_potentials[i] = self.config.reset_potential;
            return false;
        }

        let decayed = self.membrane_potentials[i] * self.config.decay_factor;
        #[cfg(all(feature = "cortex-m-dsp", target_arch = "arm"))]
        let v = {
            let sum_raw = (decayed.0 as i32) + (input.0 as i32);
            Fix16(crate::domain::hardware::ssat_16(sum_raw) as i16)
        };
        #[cfg(not(all(feature = "cortex-m-dsp", target_arch = "arm")))]
        let v = decayed + input;

        if v >= self.config.threshold {
            self.membrane_potentials[i] = match self.config.reset_mode {
                ResetMode::Hard => self.config.reset_potential,
                ResetMode::Subtract => v - self.config.threshold,
            };
            *refractory = self.config.refractory_period as u8;
            true
        } else {
            self.membrane_potentials[i] = v;
            false
        }
    }
}

impl<const N: usize> SpikingPopulation<Fix16> for CompactLif16Population<N> {
    #[inline]
    fn len(&self) -> usize {
        N
    }

    #[inline]
    fn receive_input(&mut self, index: usize, input: Fix16) {
        self.receive_input(index, input);
    }

    fn reset(&mut self) {
        for v in &mut self.membrane_potentials {
            *v = self.config.reset_potential;
        }
        for refr in &mut self.refractory_times_left {
            *refr = 0;
        }
        for acc in &mut self.input_accumulators {
            *acc = Fix16::ZERO;
        }
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
            let spike = self.step_neuron(i);
            current_spikes[i] = spike;
            if spike {
                active_spikes_buf[*active_spikes_count] = i as u32;
                *active_spikes_count += 1;
            }
        }
    }

    #[inline]
    fn tick_bitset(&mut self, bitset_words: &mut [u32]) {
        for w in bitset_words.iter_mut() {
            *w = 0;
        }
        for i in 0..N {
            let spike = self.step_neuron(i);
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
    fn input_accumulators_mut(&mut self) -> &mut [Fix16] {
        &mut self.input_accumulators
    }

    #[inline]
    fn input_accumulators(&self) -> &[Fix16] {
        &self.input_accumulators
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 0.6 en Q16.16 = round(0.6 * 65536) = 39322
    // 0.1 en Q16.16 = round(0.1 * 65536) = 6554
    static BIAS: [Fix32; 2] = [Fix32::from_bits(39322), Fix32::from_bits(6554)];

    #[test]
    fn test_compact_lif_population_with_bias() {
        let cfg = LifConfig::new(Fix32::from_num(1.0), Fix32::from_num(0.5), Fix32::ZERO, 2);

        let mut pop = CompactLifPopulation::<2>::new(cfg).with_bias(&BIAS);
        let mut spikes = [false; 2];
        let mut active_buf = [0u32; 2];
        let mut active_count = 0;

        // Tick 1: input 0.5 sur n0 -> total input = 0.5 + 0.6 = 1.1 >= 1.0 (spike!)
        // input 0.2 sur n1 -> total input = 0.2 + 0.1 = 0.3 < 1.0
        pop.receive_input(0, Fix32::from_num(0.5));
        pop.receive_input(1, Fix32::from_num(0.2));
        pop.tick_active(&mut spikes, &mut active_buf, &mut active_count);

        assert!(spikes[0]);
        assert!(!spikes[1]);
        assert_eq!(active_count, 1);
        assert_eq!(active_buf[0], 0);

        // Tick 2: n0 is in refractory period (refr=2 -> decrements to 1)
        pop.receive_input(0, Fix32::from_num(2.0));
        pop.tick_active(&mut spikes, &mut active_buf, &mut active_count);
        assert!(!spikes[0]);
        assert_eq!(pop.refractory_times_left[0], 1);
    }

    #[test]
    fn test_compact_lif16_population_and_memory_footprint() {
        let cfg = Lif16Config::new(Fix16::from_num(1.0), Fix16::from_num(0.5), Fix16::ZERO, 1);

        let mut pop = CompactLif16Population::<4>::new(cfg);
        let mut words = [0u32; 1];

        pop.receive_input(2, Fix16::from_num(1.5));
        pop.tick_bitset(&mut words);

        // Neuron 2 fired: bit 2 set to 1 (value 0b0100 = 4)
        assert_eq!(words[0], 4);

        // Verify memory footprint for 100 neurons
        let size_100_fix16 = core::mem::size_of::<CompactLif16Population<100>>();
        // 100 * 2 (V) + 100 * 2 (Acc) + 100 * 1 (Refr) = exactly 500 bytes (+ config and bias pointer)
        assert!(
            size_100_fix16 <= 550,
            "Fix16 footprint {} exceeds 550 bytes (expected ~500 B + metadata)",
            size_100_fix16
        );

        let size_100_fix32 = core::mem::size_of::<CompactLifPopulation<100>>();
        // 100 * 4 (V) + 100 * 4 (Acc) + 100 * 1 (Refr) = 900 octets (+ config et bias pointer)
        assert!(
            size_100_fix32 <= 950,
            "Fix32 footprint {} exceeds 950 bytes (expected ~900 B + metadata)",
            size_100_fix32
        );
    }
}
