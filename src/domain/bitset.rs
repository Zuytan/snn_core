//! ⚡ Compact and ultra-fast spike representation via bitsets (`no_std`).
//!
//! Stores action potentials (spikes) using 1 bit per neuron instead of a 1-byte `bool`.
//! Enables ultra-fast event-driven iteration over active neurons via
//! hardware instructions `CLZ` / `CTZ` (`u32::trailing_zeros`).

/// Static bitset for $N \le \text{WORDS} \times 32$ neurons.
///
/// Stores action potentials (spikes) using 1 bit per neuron instead of a 1-byte `bool`.
/// Enables ultra-fast event-driven iteration over active neurons via
/// native hardware `CTZ` (`u32::trailing_zeros`).
///
/// # Example
/// ```rust
/// use snn_core::SpikeBitset;
///
/// let mut bitset = SpikeBitset::<2>::new(); // 64 neurons max
/// bitset.set(5);
/// bitset.set(42);
///
/// assert!(bitset.get(5));
/// assert!(!bitset.get(6));
/// assert!(bitset.get(42));
/// assert_eq!(bitset.count_ones(), 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpikeBitset<const WORDS: usize> {
    pub words: [u32; WORDS],
}

impl<const WORDS: usize> Default for SpikeBitset<WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WORDS: usize> SpikeBitset<WORDS> {
    pub const fn new() -> Self {
        Self { words: [0; WORDS] }
    }

    /// Returns a reference to the underlying 32-bit words.
    #[inline(always)]
    pub const fn words(&self) -> &[u32; WORDS] {
        &self.words
    }

    /// Returns a mutable reference to the underlying 32-bit words.
    #[inline(always)]
    pub fn words_mut(&mut self) -> &mut [u32; WORDS] {
        &mut self.words
    }

    /// Constructs a bitset directly from a slice of booleans.
    pub fn from_bools(spikes: &[bool]) -> Self {
        let mut bitset = Self::new();
        for (i, &s) in spikes.iter().enumerate() {
            if s {
                bitset.set(i);
            }
        }
        bitset
    }

    /// Clears all bits to 0.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.words = [0; WORDS];
    }

    /// Sets the bit for neuron `index`.
    #[inline(always)]
    pub fn set(&mut self, index: usize) {
        let word_idx = index / 32;
        let bit_idx = index % 32;
        if word_idx < WORDS {
            self.words[word_idx] |= 1 << bit_idx;
        }
    }

    /// Returns `true` if neuron `index` has fired a spike.
    #[inline(always)]
    pub fn get(&self, index: usize) -> bool {
        let word_idx = index / 32;
        let bit_idx = index % 32;
        if word_idx < WORDS {
            (self.words[word_idx] & (1 << bit_idx)) != 0
        } else {
            false
        }
    }

    /// Total number of active spikes (hardware population count).
    #[inline]
    pub fn count_ones(&self) -> usize {
        let mut total = 0;
        for &w in &self.words {
            total += w.count_ones() as usize;
        }
        total
    }

    /// Extracts compact list of active neuron indices into `out_buf`.
    /// Returns the number of written indices.
    #[inline]
    pub fn extract_active_indices(&self, out_buf: &mut [u32]) -> usize {
        let mut count = 0;
        for (w_idx, &word) in self.words.iter().enumerate() {
            let mut remaining = word;
            let base_idx = (w_idx * 32) as u32;
            while remaining != 0 && count < out_buf.len() {
                let tz = remaining.trailing_zeros();
                out_buf[count] = base_idx + tz;
                count += 1;
                remaining &= remaining - 1; // Clear lowest active bit
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_bitset_operations() {
        let mut bitset = SpikeBitset::<2>::new(); // up to 64 neurons
        assert_eq!(bitset.count_ones(), 0);

        bitset.set(0);
        bitset.set(5);
        bitset.set(33); // second word

        assert!(bitset.get(0));
        assert!(bitset.get(5));
        assert!(bitset.get(33));
        assert!(!bitset.get(1));
        assert!(!bitset.get(32));

        assert_eq!(bitset.count_ones(), 3);

        let mut active = [0u32; 10];
        let n = bitset.extract_active_indices(&mut active);
        assert_eq!(n, 3);
        assert_eq!(&active[..3], &[0, 5, 33]);

        bitset.clear();
        assert_eq!(bitset.count_ones(), 0);
    }
}
