use crate::domain::numeric::Fix32;
use crate::domain::synapse::Synapse;
use alloc::vec;
use alloc::vec::Vec;

/// Contiguous and vectorizable CSR (Compressed Sparse Row) synaptic structure.
#[derive(Clone, Debug, Default)]
pub struct CsrSynapses {
    /// Start/end row pointers for each source neuron. Length: num_sources + 1.
    pub row_ptrs: Vec<u32>,
    /// Target neuron indices (contiguous in memory).
    pub targets: Vec<u32>,
    /// Synaptic weights (contiguous in memory in Q16.16 fixed-point).
    pub weights: Vec<Fix32>,
}

impl CsrSynapses {
    /// Creates an empty CSR structure.
    pub fn empty() -> Self {
        CsrSynapses {
            row_ptrs: vec![0],
            targets: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Total number of stored synapses.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Constructs a CSR structure from a classical AoS Synapses slice.
    pub fn from_synapses(num_sources: usize, synapses: &[Synapse]) -> Self {
        let mut row_counts = vec![0u32; num_sources];
        for s in synapses {
            if s.source_idx < num_sources {
                row_counts[s.source_idx] += 1;
            }
        }

        let mut row_ptrs = vec![0u32; num_sources + 1];
        for i in 0..num_sources {
            row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
        }

        let total_synapses = synapses.len();
        let mut targets = vec![0u32; total_synapses];
        let mut weights = vec![Fix32::ZERO; total_synapses];
        let mut current_offsets = row_ptrs.clone();

        for s in synapses {
            if s.source_idx < num_sources {
                let dest_idx = current_offsets[s.source_idx] as usize;
                targets[dest_idx] = s.target_idx as u32;
                weights[dest_idx] = s.weight;
                current_offsets[s.source_idx] += 1;
            }
        }

        CsrSynapses {
            row_ptrs,
            targets,
            weights,
        }
    }

    /// Exports the CSR structure to a classical `Vec<Synapse>`.
    pub fn to_synapses(&self) -> Vec<Synapse> {
        let mut synapses = Vec::with_capacity(self.targets.len());
        let num_sources = if self.row_ptrs.len() > 1 {
            self.row_ptrs.len() - 1
        } else {
            0
        };

        for src in 0..num_sources {
            let start = self.row_ptrs[src] as usize;
            let end = self.row_ptrs[src + 1] as usize;
            for i in start..end {
                synapses.push(Synapse::new(src, self.targets[i] as usize, self.weights[i]));
            }
        }
        synapses
    }

    /// Validates the internal consistency of the CSR structure.
    pub fn is_valid(&self) -> bool {
        if self.row_ptrs.is_empty() || self.targets.len() != self.weights.len() {
            return false;
        }
        for w in self.row_ptrs.windows(2) {
            if w[0] > w[1] || (w[1] as usize) > self.targets.len() {
                return false;
            }
        }
        true
    }

    /// Contiguous event-driven propagator with memory safety guarantees.
    #[inline]
    pub fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [Fix32]) {
        for &src_idx in active_spikes {
            let src_idx = src_idx as usize;
            if src_idx + 1 < self.row_ptrs.len() {
                let start = self.row_ptrs[src_idx] as usize;
                let end = (self.row_ptrs[src_idx + 1] as usize)
                    .min(self.targets.len())
                    .min(self.weights.len());

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
    #[inline]
    pub fn propagate_weighted_inputs(&self, inputs: &[Fix32], input_accumulators: &mut [Fix32]) {
        for (src_idx, &input_val) in inputs.iter().enumerate() {
            if input_val != Fix32::ZERO && src_idx + 1 < self.row_ptrs.len() {
                let start = self.row_ptrs[src_idx] as usize;
                let end = (self.row_ptrs[src_idx + 1] as usize)
                    .min(self.targets.len())
                    .min(self.weights.len());

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

impl crate::traits::SynapseMatrix for CsrSynapses {
    #[inline]
    fn len(&self) -> usize {
        self.targets.len()
    }

    #[inline]
    fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [Fix32]) {
        self.propagate_active_spikes(active_spikes, input_accumulators);
    }

    #[inline]
    fn propagate_weighted_inputs(&self, inputs: &[Fix32], input_accumulators: &mut [Fix32]) {
        self.propagate_weighted_inputs(inputs, input_accumulators);
    }

    #[inline]
    fn propagate_spikes_words(&self, words: &[u32], input_accumulators: &mut [Fix32]) {
        for (w_idx, &word) in words.iter().enumerate() {
            let mut remaining = word;
            let base_idx = w_idx * 32;
            while remaining != 0 {
                let tz = remaining.trailing_zeros() as usize;
                let src_idx = base_idx + tz;
                if src_idx + 1 < self.row_ptrs.len() {
                    let start = self.row_ptrs[src_idx] as usize;
                    let end = (self.row_ptrs[src_idx + 1] as usize)
                        .min(self.targets.len())
                        .min(self.weights.len());
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
    fn test_csr_conversion_and_propagation() {
        let synapses = vec![
            Synapse::new(0, 1, Fix32::from_num(0.5)),
            Synapse::new(0, 2, Fix32::from_num(1.0)),
            Synapse::new(1, 2, Fix32::from_num(2.0)),
        ];

        let csr = CsrSynapses::from_synapses(3, &synapses);
        assert_eq!(csr.row_ptrs, vec![0, 2, 3, 3]);
        assert_eq!(csr.targets, vec![1, 2, 2]);
        assert_eq!(csr.len(), 3);

        let recovered = csr.to_synapses();
        assert_eq!(recovered.len(), 3);

        let mut acc = vec![Fix32::ZERO; 3];
        csr.propagate_active_spikes(&[0], &mut acc);

        assert_eq!(acc[0], Fix32::ZERO);
        assert_eq!(acc[1], Fix32::from_num(0.5));
        assert_eq!(acc[2], Fix32::from_num(1.0));
    }
}
