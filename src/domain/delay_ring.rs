//! ⏳ Static zero-allocation synaptic / axonal delay line (`no_std`).
//!
//! Enables injecting discrete time delays ($1 \dots D$ ticks) between neurons,
//! essential for bio-acoustic (SHD), radar, and DVS neuromorphic sensor temporal processing.

/// Static circular delay line for $N$ neurons and a maximum delay of $D$ ticks.
#[derive(Clone, Copy, Debug)]
pub struct StaticSpikeDelayLine<const N: usize, const D: usize> {
    buffer: [[bool; N]; D],
    head: usize,
}

impl<const N: usize, const D: usize> Default for StaticSpikeDelayLine<N, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const D: usize> StaticSpikeDelayLine<N, D> {
    /// Constructs a delay line initialized to silence.
    pub const fn new() -> Self {
        Self {
            buffer: [[false; N]; D],
            head: 0,
        }
    }

    /// Resets the delay history.
    pub fn reset(&mut self) {
        self.buffer = [[false; N]; D];
        self.head = 0;
    }

    /// Records current tick spikes and advances the circular pointer.
    pub fn push(&mut self, current_spikes: &[bool; N]) {
        if D > 0 {
            self.buffer[self.head] = *current_spikes;
            self.head = (self.head + 1) % D;
        }
    }

    /// Retrieves spikes emitted exactly `delay_ticks` time steps ago ($0 \le \text{delay\_ticks} < D$).
    /// - `delay_ticks == 0`: spikes recorded at current tick $t$.
    /// - `delay_ticks == 1`: spikes from 1 tick ago ($t - 1$).
    #[inline]
    pub fn get_delayed(&self, delay_ticks: usize) -> &[bool; N] {
        if D == 0 {
            &self.buffer[0]
        } else {
            let offset = delay_ticks.min(D - 1);
            let idx = (self.head + D - 1 - offset) % D;
            &self.buffer[idx]
        }
    }
}

/// Compact static circular delay line operating directly on bitset words (`[u32; WORDS]`).
/// SRAM footprint reduced by a factor of 8 compared to `StaticSpikeDelayLine` (1 bit per neuron instead of 1 byte).
#[derive(Clone, Copy, Debug)]
pub struct StaticSpikeWordDelayLine<const WORDS: usize, const D: usize> {
    buffer: [[u32; WORDS]; D],
    head: usize,
}

impl<const WORDS: usize, const D: usize> Default for StaticSpikeWordDelayLine<WORDS, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WORDS: usize, const D: usize> StaticSpikeWordDelayLine<WORDS, D> {
    /// Constructs a word delay line initialized to 0 (absolute silence).
    pub const fn new() -> Self {
        Self {
            buffer: [[0u32; WORDS]; D],
            head: 0,
        }
    }

    /// Resets the delay history.
    pub fn reset(&mut self) {
        self.buffer = [[0u32; WORDS]; D];
        self.head = 0;
    }

    /// Records current tick spike words and advances the circular pointer.
    pub fn push(&mut self, current_spikes_words: &[u32; WORDS]) {
        if D > 0 {
            self.buffer[self.head] = *current_spikes_words;
            self.head = (self.head + 1) % D;
        }
    }

    /// Retrieves spike words emitted exactly `delay_ticks` time steps ago ($0 \le \text{delay\_ticks} < D$).
    #[inline]
    pub fn get_delayed(&self, delay_ticks: usize) -> &[u32; WORDS] {
        if D == 0 {
            &self.buffer[0]
        } else {
            let offset = delay_ticks.min(D - 1);
            let idx = (self.head + D - 1 - offset) % D;
            &self.buffer[idx]
        }
    }
}

use crate::domain::numeric::{Fix32, NumericScalar};
use core::cell::RefCell;
use core::marker::PhantomData;

/// Synaptic wrapper applying a discrete delay to spikes before propagation.
#[deprecated(
    since = "0.2.0",
    note = "Use `DelayedSpikingLayer` instead (without RefCell)"
)]
#[derive(Debug)]
pub struct DelayedSynapseMatrix<
    S: crate::traits::SynapseMatrix<T>,
    const N: usize,
    const D: usize,
    T: NumericScalar = Fix32,
> {
    pub synapses: S,
    pub delay_line: RefCell<StaticSpikeDelayLine<N, D>>,
    pub delay_ticks: usize,
    _marker: PhantomData<T>,
}

#[allow(deprecated)]
impl<S: crate::traits::SynapseMatrix<T>, const N: usize, const D: usize, T: NumericScalar>
    DelayedSynapseMatrix<S, N, D, T>
{
    pub const fn new(synapses: S, delay_ticks: usize) -> Self {
        Self {
            synapses,
            delay_line: RefCell::new(StaticSpikeDelayLine::new()),
            delay_ticks,
            _marker: PhantomData,
        }
    }

    /// Records current tick spikes and propagates delayed spikes to accumulators.
    pub fn step_and_propagate(&self, current_spikes: &[bool; N], input_accumulators: &mut [T]) {
        let mut dl = self.delay_line.borrow_mut();
        dl.push(current_spikes);
        let delayed = dl.get_delayed(self.delay_ticks);
        let mut active = [0u32; N];
        let mut count = 0;
        for (i, &s) in delayed.iter().enumerate() {
            if s {
                active[count] = i as u32;
                count += 1;
            }
        }
        drop(dl);
        self.synapses
            .propagate_active_spikes(&active[..count], input_accumulators);
    }

    pub fn reset(&self) {
        self.delay_line.borrow_mut().reset();
    }
}

#[allow(deprecated)]
impl<S: crate::traits::SynapseMatrix<T>, const N: usize, const D: usize, T: NumericScalar>
    crate::traits::SynapseMatrix<T> for DelayedSynapseMatrix<S, N, D, T>
{
    #[inline]
    fn len(&self) -> usize {
        self.synapses.len()
    }

    #[inline]
    fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [T]) {
        let mut spikes = [false; N];
        for &spiker in active_spikes {
            if (spiker as usize) < N {
                spikes[spiker as usize] = true;
            }
        }
        self.step_and_propagate(&spikes, input_accumulators);
    }

    #[inline]
    fn propagate_weighted_inputs(&self, inputs: &[T], input_accumulators: &mut [T]) {
        self.synapses
            .propagate_weighted_inputs(inputs, input_accumulators);
    }

    #[inline]
    fn propagate_spikes_words(&self, words: &[u32], input_accumulators: &mut [T]) {
        let mut spikes = [false; N];
        for (w_idx, &word) in words.iter().enumerate() {
            let mut remaining = word;
            let base_idx = w_idx * 32;
            while remaining != 0 {
                let tz = remaining.trailing_zeros() as usize;
                let src_idx = base_idx + tz;
                if src_idx < N {
                    spikes[src_idx] = true;
                }
                remaining &= remaining - 1;
            }
        }
        self.step_and_propagate(&spikes, input_accumulators);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_line() {
        let mut delay_line = StaticSpikeDelayLine::<2, 3>::new();

        // Tick 0: spike on neuron 0
        delay_line.push(&[true, false]);

        // Tick 1: spike on neuron 1
        delay_line.push(&[false, true]);

        // Tick 2: no spike
        delay_line.push(&[false, false]);

        // Verify delays:
        // Delay of 0 ticks corresponds to tick 2 (false, false)
        assert_eq!(delay_line.get_delayed(0), &[false, false]);
        // Delay of 1 tick corresponds to tick 1 (false, true)
        assert_eq!(delay_line.get_delayed(1), &[false, true]);
        // Delay of 2 ticks corresponds to tick 0 (true, false)
        assert_eq!(delay_line.get_delayed(2), &[true, false]);
    }

    #[test]
    #[allow(deprecated)]
    fn test_delayed_synapse_matrix_trait() {
        use crate::Fix32;
        use crate::domain::rom_csr::RomCsrSynapses;
        use crate::traits::SynapseMatrix;

        static ROW_PTRS: [u16; 3] = [0, 1, 2];
        static TARGETS: [u16; 2] = [1, 0];
        static WEIGHTS: [Fix32; 2] = [Fix32::ONE, Fix32::from_int(2)];
        let base_synapses = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS);

        // Delay of 1 tick
        let delayed_matrix = DelayedSynapseMatrix::<_, 2, 4>::new(base_synapses, 1);
        assert_eq!(delayed_matrix.len(), 2);

        let mut acc = [Fix32::ZERO; 2];

        // Tick 0: Neuron 0 fires -> recorded in delay line (delayed)
        delayed_matrix.propagate_active_spikes(&[0], &mut acc);
        assert_eq!(acc[0], Fix32::ZERO);
        assert_eq!(acc[1], Fix32::ZERO); // delayed, not yet propagated at tick 0

        // Tick 1: 1 tick elapsed -> tick 0 spike is delivered
        delayed_matrix.propagate_active_spikes(&[], &mut acc);
        assert_eq!(acc[1], Fix32::ONE);

        // Tick 2: No new delayed spike
        delayed_matrix.propagate_active_spikes(&[], &mut acc);
        assert_eq!(acc[1], Fix32::ONE); // accumulator unchanged (no new spike)
    }

    #[test]
    #[allow(deprecated)]
    fn test_delayed_synapse_matrix_propagate_spikes_words_single_step() {
        use crate::Fix32;
        use crate::domain::rom_csr::RomCsrSynapses;
        use crate::traits::SynapseMatrix;

        static ROW_PTRS: [u16; 3] = [0, 1, 2];
        static TARGETS: [u16; 2] = [1, 0];
        static WEIGHTS: [Fix32; 2] = [Fix32::ONE, Fix32::from_int(2)];
        let base_synapses = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS);

        // Delay of 1 tick
        let delayed_matrix = DelayedSynapseMatrix::<_, 2, 4>::new(base_synapses, 1);
        let mut acc = [Fix32::ZERO; 2];

        // Tick 0: Both neurons 0 and 1 fire at the same time (bits 0 and 1 = 3u32)
        // Must not advance buffer twice, but exactly once!
        delayed_matrix.propagate_spikes_words(&[0b11], &mut acc);
        assert_eq!(acc[0], Fix32::ZERO);
        assert_eq!(acc[1], Fix32::ZERO);

        // Tick 1: no spike
        delayed_matrix.propagate_spikes_words(&[0], &mut acc);
        // With delay_ticks = 1, Tick 0 spikes are delivered at Tick 1
        // Neuron 0 fires to target 1 (weight 1), neuron 1 fires to target 0 (weight 2)
        assert_eq!(acc[0], Fix32::from_int(2));
        assert_eq!(acc[1], Fix32::ONE);
    }
}
