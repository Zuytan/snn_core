use crate::domain::numeric::{Fix32, NumericScalar};
use crate::domain::synapse::Synapse;

/// Static zero-allocation CSR (Compressed Sparse Row) synaptic structure (pure `no_std`).
/// `N_ROW_PTRS` is the number of row pointers (num_sources + 1).
/// `MAX_SYNAPSES` is the maximum capacity of synapses allocated on the stack / in ROM.
#[derive(Clone, Copy, Debug)]
pub struct StaticCsrSynapses<
    const N_ROW_PTRS: usize,
    const MAX_SYNAPSES: usize,
    T: NumericScalar = Fix32,
> {
    /// Start/end row pointers for each source neuron. Length: N_ROW_PTRS (num_sources + 1).
    pub row_ptrs: [u32; N_ROW_PTRS],
    /// Target neuron indices.
    pub targets: [u32; MAX_SYNAPSES],
    /// Fixed-point synaptic weights (Fix32 by default or Fix16).
    pub weights: [T; MAX_SYNAPSES],
    /// Actual number of stored synapses.
    pub total_synapses: usize,
}

impl<const N_ROW_PTRS: usize, const MAX_SYNAPSES: usize, T: NumericScalar> Default
    for StaticCsrSynapses<N_ROW_PTRS, MAX_SYNAPSES, T>
{
    fn default() -> Self {
        Self::empty()
    }
}

impl<const N_ROW_PTRS: usize, const MAX_SYNAPSES: usize, T: NumericScalar>
    StaticCsrSynapses<N_ROW_PTRS, MAX_SYNAPSES, T>
{
    /// Creates an empty static CSR structure.
    pub const fn empty() -> Self {
        StaticCsrSynapses {
            row_ptrs: [0; N_ROW_PTRS],
            targets: [0; MAX_SYNAPSES],
            weights: [T::ZERO; MAX_SYNAPSES],
            total_synapses: 0,
        }
    }

    /// Effective number of stored synapses.
    pub const fn len(&self) -> usize {
        self.total_synapses
    }

    pub const fn is_empty(&self) -> bool {
        self.total_synapses == 0
    }

    /// Constructs a static CSR structure from a slice of `Synapse<T>`.
    /// Performs ZERO heap allocations.
    pub fn from_synapses(synapses: &[Synapse<T>]) -> Self {
        let num_sources = if N_ROW_PTRS > 0 { N_ROW_PTRS - 1 } else { 0 };
        let mut row_counts = [0u32; N_ROW_PTRS];
        for s in synapses {
            if s.source_idx < num_sources {
                row_counts[s.source_idx] += 1;
            }
        }

        let mut row_ptrs = [0u32; N_ROW_PTRS];
        for i in 0..num_sources {
            if i + 1 < N_ROW_PTRS {
                row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
            }
        }

        let total = synapses.len().min(MAX_SYNAPSES);
        let mut targets = [0u32; MAX_SYNAPSES];
        let mut weights = [T::ZERO; MAX_SYNAPSES];
        let mut current_offsets = row_ptrs;

        for s in synapses {
            if s.source_idx < num_sources {
                let dest_idx = current_offsets[s.source_idx] as usize;
                if dest_idx < MAX_SYNAPSES {
                    targets[dest_idx] = s.target_idx as u32;
                    weights[dest_idx] = s.weight;
                    current_offsets[s.source_idx] += 1;
                }
            }
        }

        StaticCsrSynapses {
            row_ptrs,
            targets,
            weights,
            total_synapses: total,
        }
    }

    /// Initializes the CSR structure in-place from a slice of synapses.
    pub fn init_from_synapses(&mut self, synapses: &[Synapse<T>]) {
        *self = Self::from_synapses(synapses);
    }

    /// Contiguous event-driven propagator for active neurons.
    #[inline(always)]
    pub fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [T]) {
        for &src_idx in active_spikes {
            let src_idx = src_idx as usize;
            if src_idx + 1 < self.row_ptrs.len() {
                let start = self.row_ptrs[src_idx] as usize;
                let end = (self.row_ptrs[src_idx + 1] as usize)
                    .min(MAX_SYNAPSES)
                    .min(self.total_synapses);

                for i in start..end {
                    let target = self.targets[i] as usize;
                    let weight = self.weights[i];
                    if let Some(acc) = input_accumulators.get_mut(target) {
                        *acc += weight;
                    }
                }
            }
        }
    }

    /// Propagates external weighted inputs (only non-zero inputs are propagated).
    #[inline(always)]
    pub fn propagate_weighted_inputs(&self, inputs: &[T], input_accumulators: &mut [T]) {
        for (src_idx, &input_val) in inputs.iter().enumerate() {
            if input_val != T::ZERO && src_idx + 1 < self.row_ptrs.len() {
                let start = self.row_ptrs[src_idx] as usize;
                let end = (self.row_ptrs[src_idx + 1] as usize)
                    .min(MAX_SYNAPSES)
                    .min(self.total_synapses);

                for i in start..end {
                    let target = self.targets[i] as usize;
                    let weight = self.weights[i];
                    if let Some(acc) = input_accumulators.get_mut(target) {
                        *acc += input_val * weight;
                    }
                }
            }
        }
    }
}

impl<const N_ROW_PTRS: usize, const MAX_SYNAPSES: usize, T: NumericScalar>
    crate::traits::SynapseMatrix<T> for StaticCsrSynapses<N_ROW_PTRS, MAX_SYNAPSES, T>
{
    #[inline]
    fn len(&self) -> usize {
        self.total_synapses
    }

    #[inline]
    fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [T]) {
        self.propagate_active_spikes(active_spikes, input_accumulators);
    }

    #[inline]
    fn propagate_weighted_inputs(&self, inputs: &[T], input_accumulators: &mut [T]) {
        self.propagate_weighted_inputs(inputs, input_accumulators);
    }

    #[inline]
    fn propagate_spikes_words(&self, words: &[u32], input_accumulators: &mut [T]) {
        for (w_idx, &word) in words.iter().enumerate() {
            let mut remaining = word;
            let base_idx = w_idx * 32;
            while remaining != 0 {
                let tz = remaining.trailing_zeros() as usize;
                let src_idx = base_idx + tz;
                if src_idx + 1 < self.row_ptrs.len() {
                    let start = self.row_ptrs[src_idx] as usize;
                    let end = (self.row_ptrs[src_idx + 1] as usize)
                        .min(MAX_SYNAPSES)
                        .min(self.total_synapses);
                    for i in start..end {
                        let target = self.targets[i] as usize;
                        if let Some(acc) = input_accumulators.get_mut(target) {
                            *acc += self.weights[i];
                        }
                    }
                }
                remaining &= remaining - 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_csr_conversion_and_propagation() {
        let synapses = [
            Synapse::new(0, 1, Fix32::from_num(0.5)),
            Synapse::new(0, 2, Fix32::from_num(1.0)),
            Synapse::new(1, 2, Fix32::from_num(2.0)),
        ];

        // 3 sources -> 4 row_ptrs
        let csr: StaticCsrSynapses<4, 10> = StaticCsrSynapses::from_synapses(&synapses);
        assert_eq!(csr.row_ptrs[0], 0);
        assert_eq!(csr.row_ptrs[1], 2);
        assert_eq!(csr.row_ptrs[2], 3);
        assert_eq!(csr.row_ptrs[3], 3);
        assert_eq!(csr.targets[0], 1);
        assert_eq!(csr.targets[1], 2);
        assert_eq!(csr.targets[2], 2);
        assert_eq!(csr.len(), 3);

        let mut acc = [Fix32::ZERO; 3];
        csr.propagate_active_spikes(&[0], &mut acc);

        assert_eq!(acc[0], Fix32::ZERO);
        assert_eq!(acc[1], Fix32::from_num(0.5));
        assert_eq!(acc[2], Fix32::from_num(1.0));
    }
}
