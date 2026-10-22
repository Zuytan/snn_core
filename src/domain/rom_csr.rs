//! 💾 Immutable CSR synaptic matrix stored directly in Flash ROM (`no_std`).
//!
//! Unlike heap- or stack-allocated CSR structures,
//! [`RomCsrSynapses`] maintains only static references `&'a [...]`
//! pointing directly to tables stored in Flash memory (`.rodata`).
//! The RAM (SRAM) footprint is strictly reduced to pointer storage (3 references).

use crate::domain::numeric::{Fix16, Fix32, NumericScalar};
use crate::traits::SynapseMatrix;

/// Trait defining conversion from a synaptic weight stored in ROM to an accumulable value `T`.
pub trait SynapticWeight<T: NumericScalar = Fix32>: Copy {
    fn to_scalar(self) -> T;

    #[inline(always)]
    fn to_fix32(self) -> Fix32 {
        self.to_scalar().to_fix32()
    }
}

impl SynapticWeight<Fix32> for Fix32 {
    #[inline(always)]
    fn to_scalar(self) -> Fix32 {
        self
    }
}

impl SynapticWeight<Fix16> for Fix32 {
    #[inline(always)]
    fn to_scalar(self) -> Fix16 {
        Fix16::from_fix32(self)
    }
}

impl SynapticWeight<Fix32> for Fix16 {
    #[inline(always)]
    fn to_scalar(self) -> Fix32 {
        self.to_fix32()
    }
}

impl SynapticWeight<Fix16> for Fix16 {
    #[inline(always)]
    fn to_scalar(self) -> Fix16 {
        self
    }
}

/// 8-bit quantized integer weight with a shared scale factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct QuantizedI8<const SCALE_SHIFT: u32 = 8>(pub i8);

impl<const SCALE_SHIFT: u32> SynapticWeight<Fix32> for QuantizedI8<SCALE_SHIFT> {
    #[inline(always)]
    fn to_scalar(self) -> Fix32 {
        // Sign-extend and convert to Fix32 Q16.16
        // If SCALE_SHIFT = 8 (format Q0.7 or Q1.7), the weight -128..127 is shifted
        Fix32::from_bits((i32::from(self.0)) << (16 - SCALE_SHIFT.min(16)))
    }
}

impl<const SCALE_SHIFT: u32> SynapticWeight<Fix16> for QuantizedI8<SCALE_SHIFT> {
    #[inline(always)]
    fn to_scalar(self) -> Fix16 {
        Fix16::from_fix32(Fix32::from_bits(
            (i32::from(self.0)) << (16 - SCALE_SHIFT.min(16)),
        ))
    }
}

/// 16-bit quantized integer weight in Q8.8 fixed-point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct QuantizedI16Q8(pub i16);

impl SynapticWeight<Fix32> for QuantizedI16Q8 {
    #[inline(always)]
    fn to_scalar(self) -> Fix32 {
        // Q8.8 to Q16.16: shift 8 bits left
        Fix32::from_bits((i32::from(self.0)) << 8)
    }
}

impl SynapticWeight<Fix16> for QuantizedI16Q8 {
    #[inline(always)]
    fn to_scalar(self) -> Fix16 {
        Fix16(self.0)
    }
}

/// Trait defining integer types usable to index synapses in Flash ROM (`u8`, `u16`, `u32`, `usize`).
pub trait RomIndex: Copy {
    fn to_usize(self) -> usize;
}

impl RomIndex for u8 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl RomIndex for u16 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl RomIndex for u32 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl RomIndex for usize {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self
    }
}

/// CSR synaptic matrix stored in Flash ROM.
/// `Idx` can be `u8`, `u16` (up to 65,535 neurons), `u32`, or `usize`.
/// `Weight` can be `Fix32`, `QuantizedI8`, or `QuantizedI16Q8`.
///
/// # Example
/// ```rust
/// use snn_core::domain::rom_csr::RomCsrSynapses;
/// use snn_core::traits::SynapseMatrix;
/// use snn_core::Fix32;
///
/// static ROW_PTRS: [u16; 3] = [0, 1, 2];
/// static TARGETS: [u16; 2] = [0, 1];
/// static WEIGHTS: [Fix32; 2] = [Fix32::from_bits(32768), Fix32::from_bits(52429)]; // 0.5 and 0.8 in Q16.16
///
/// let synapses = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS);
/// assert_eq!(synapses.len(), 2);
///
/// let mut accumulators = [Fix32::ZERO; 2];
/// synapses.propagate_active_spikes(&[0], &mut accumulators);
/// assert_eq!(accumulators[0], Fix32::from_bits(32768));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RomCsrSynapses<'a, Idx: RomIndex = u16, Weight: SynapticWeight<Fix32> = Fix32> {
    pub row_ptrs: &'a [Idx],
    pub targets: &'a [Idx],
    pub weights: &'a [Weight],
}

impl<'a, Idx: RomIndex, Weight: SynapticWeight<Fix32>> RomCsrSynapses<'a, Idx, Weight> {
    /// Constructs a synaptic matrix pointing directly to Flash ROM.
    #[inline]
    pub const fn new(row_ptrs: &'a [Idx], targets: &'a [Idx], weights: &'a [Weight]) -> Self {
        Self {
            row_ptrs,
            targets,
            weights,
        }
    }

    /// Number of synapses stored in ROM.
    #[inline]
    pub const fn len(&self) -> usize {
        self.targets.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

impl<T: NumericScalar, Idx: RomIndex, Weight: SynapticWeight<T> + SynapticWeight<Fix32>>
    SynapseMatrix<T> for RomCsrSynapses<'_, Idx, Weight>
{
    #[inline]
    fn len(&self) -> usize {
        self.targets.len()
    }

    #[inline]
    fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [T]) {
        for &src_idx in active_spikes {
            let src_idx = src_idx as usize;
            if src_idx + 1 < self.row_ptrs.len() {
                let start: usize = self.row_ptrs[src_idx].to_usize();
                let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                let end = end.min(self.targets.len()).min(self.weights.len());

                for i in start..end {
                    let target: usize = self.targets[i].to_usize();
                    if target < input_accumulators.len() {
                        let weight = self.weights[i].to_scalar();
                        input_accumulators[target] += weight;
                    }
                }
            }
        }
    }

    #[inline]
    fn propagate_weighted_inputs(&self, inputs: &[T], input_accumulators: &mut [T]) {
        for (src_idx, &input_val) in inputs.iter().enumerate() {
            if input_val != T::ZERO && src_idx + 1 < self.row_ptrs.len() {
                let start: usize = self.row_ptrs[src_idx].to_usize();
                let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                let end = end.min(self.targets.len()).min(self.weights.len());

                for i in start..end {
                    let target: usize = self.targets[i].to_usize();
                    if target < input_accumulators.len() {
                        let weight = self.weights[i].to_scalar();
                        input_accumulators[target] += input_val * weight;
                    }
                }
            }
        }
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
                    let start: usize = self.row_ptrs[src_idx].to_usize();
                    let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                    let end = end.min(self.targets.len()).min(self.weights.len());

                    for i in start..end {
                        let target: usize = self.targets[i].to_usize();
                        if target < input_accumulators.len() {
                            let weight = self.weights[i].to_scalar();
                            input_accumulators[target] += weight;
                        }
                    }
                }
                remaining &= remaining - 1;
            }
        }
    }
}

/// INT8 quantized synaptic matrix stored in Flash ROM with a shared scale factor (`scale: Fix32`).
/// Eliminates overflow and interfaces directly with the NIR AOT compiler (`emit_rust_static_quantized_i8`).
#[derive(Clone, Copy, Debug)]
pub struct QuantizedRomCsrSynapses<'a, Idx: RomIndex = u16> {
    pub row_ptrs: &'a [Idx],
    pub targets: &'a [Idx],
    pub weights: &'a [i8],
    pub scale: Fix32,
}

impl<'a, Idx: RomIndex> QuantizedRomCsrSynapses<'a, Idx> {
    #[inline]
    pub const fn new(
        row_ptrs: &'a [Idx],
        targets: &'a [Idx],
        weights: &'a [i8],
        scale: Fix32,
    ) -> Self {
        Self {
            row_ptrs,
            targets,
            weights,
            scale,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.targets.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    #[inline(always)]
    fn scaled_weight<T: NumericScalar>(&self, idx: usize) -> T {
        // 32-bit optimization without division or redundant clamp for the vast majority of cases
        let scale_bits = self.scale.0;
        let w = self.weights[idx] as i32;
        // If scale_bits and w are within standard 32-bit limits (scale <= 128.0 in Q16.16)
        let bits = if scale_bits.unsigned_abs() < (1 << 23) {
            scale_bits * w
        } else {
            let raw_prod = (scale_bits as i64) * (w as i64);
            raw_prod.clamp(i32::MIN as i64, i32::MAX as i64) as i32
        };
        T::from_fix32(Fix32::from_bits(bits))
    }

    /// Directly accumulates integer i8 weights into raw i32 accumulators
    /// at 1 CPU cycle per synapse (ZERO multiplications).
    /// The scale factor is applied globally afterwards.
    #[inline]
    pub fn propagate_active_spikes_i32(&self, active_spikes: &[u32], int_accumulators: &mut [i32]) {
        for &src_idx in active_spikes {
            let src_idx = src_idx as usize;
            if src_idx + 1 < self.row_ptrs.len() {
                let start: usize = self.row_ptrs[src_idx].to_usize();
                let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                let end = end.min(self.targets.len()).min(self.weights.len());

                for i in start..end {
                    let target: usize = self.targets[i].to_usize();
                    if target < int_accumulators.len() {
                        int_accumulators[target] =
                            int_accumulators[target].saturating_add(self.weights[i] as i32);
                    }
                }
            }
        }
    }

    /// Applies the shared scale factor from integer accumulators to network accumulators.
    #[inline]
    pub fn scale_accumulators<T: NumericScalar>(
        &self,
        int_accumulators: &[i32],
        out_accumulators: &mut [T],
    ) {
        let n = int_accumulators.len().min(out_accumulators.len());
        for i in 0..n {
            if int_accumulators[i] != 0 {
                let raw_prod = (self.scale.0 as i64) * (int_accumulators[i] as i64);
                let bits = raw_prod.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                out_accumulators[i] += T::from_fix32(Fix32::from_bits(bits));
            }
        }
    }

    /// Directly accumulates bitset word spikes (CTZ) into raw `i32` accumulators
    /// at 1 CPU cycle per synapse (ZERO multiplications).
    #[inline]
    pub fn propagate_spikes_words_i32(&self, words: &[u32], int_accumulators: &mut [i32]) {
        for (w_idx, &word) in words.iter().enumerate() {
            let mut remaining = word;
            let base_idx = w_idx * 32;
            while remaining != 0 {
                let tz = remaining.trailing_zeros() as usize;
                let src_idx = base_idx + tz;
                if src_idx + 1 < self.row_ptrs.len() {
                    let start: usize = self.row_ptrs[src_idx].to_usize();
                    let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                    let end = end.min(self.targets.len()).min(self.weights.len());

                    for i in start..end {
                        let target: usize = self.targets[i].to_usize();
                        if target < int_accumulators.len() {
                            int_accumulators[target] =
                                int_accumulators[target].saturating_add(self.weights[i] as i32);
                        }
                    }
                }
                remaining &= remaining - 1;
            }
        }
    }
}

impl<T: NumericScalar, Idx: RomIndex> SynapseMatrix<T> for QuantizedRomCsrSynapses<'_, Idx> {
    #[inline]
    fn len(&self) -> usize {
        self.targets.len()
    }

    #[inline]
    fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [T]) {
        for &src_idx in active_spikes {
            let src_idx = src_idx as usize;
            if src_idx + 1 < self.row_ptrs.len() {
                let start: usize = self.row_ptrs[src_idx].to_usize();
                let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                let end = end.min(self.targets.len()).min(self.weights.len());

                for i in start..end {
                    let target: usize = self.targets[i].to_usize();
                    if target < input_accumulators.len() {
                        input_accumulators[target] += self.scaled_weight(i);
                    }
                }
            }
        }
    }

    #[inline]
    fn propagate_weighted_inputs(&self, inputs: &[T], input_accumulators: &mut [T]) {
        for (src_idx, &input_val) in inputs.iter().enumerate() {
            if input_val != T::ZERO && src_idx + 1 < self.row_ptrs.len() {
                let start: usize = self.row_ptrs[src_idx].to_usize();
                let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                let end = end.min(self.targets.len()).min(self.weights.len());

                for i in start..end {
                    let target: usize = self.targets[i].to_usize();
                    if target < input_accumulators.len() {
                        let w: T = self.scaled_weight(i);
                        input_accumulators[target] += input_val * w;
                    }
                }
            }
        }
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
                    let start: usize = self.row_ptrs[src_idx].to_usize();
                    let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                    let end = end.min(self.targets.len()).min(self.weights.len());

                    for i in start..end {
                        let target: usize = self.targets[i].to_usize();
                        if target < input_accumulators.len() {
                            input_accumulators[target] += self.scaled_weight(i);
                        }
                    }
                }
                remaining &= remaining - 1;
            }
        }
    }
}

/// BSR (Block Compressed Sparse Row) synaptic matrix stored in Flash ROM.
/// Groups synaptic connections by blocks of contiguous target neurons (`BLOCK_SIZE = 2` by default).
/// On ARM Cortex-M with DSP, leverages the SIMD instruction `qadd16` to update 2 neurons in 1 CPU cycle.
#[derive(Clone, Copy, Debug)]
pub struct BlockRomCsrSynapses<
    'a,
    Idx: RomIndex = u16,
    Weight: SynapticWeight<T> = Fix16,
    const BLOCK_SIZE: usize = 2,
    T: NumericScalar = Fix16,
> {
    pub row_ptrs: &'a [Idx],
    pub block_targets: &'a [Idx],
    pub weights: &'a [Weight],
    _marker: core::marker::PhantomData<T>,
}

impl<'a, Idx: RomIndex, Weight: SynapticWeight<T>, const BLOCK_SIZE: usize, T: NumericScalar>
    BlockRomCsrSynapses<'a, Idx, Weight, BLOCK_SIZE, T>
{
    #[inline]
    pub const fn new(row_ptrs: &'a [Idx], block_targets: &'a [Idx], weights: &'a [Weight]) -> Self {
        Self {
            row_ptrs,
            block_targets,
            weights,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.weights.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

impl<Idx: RomIndex, Weight: SynapticWeight<T>, const BLOCK_SIZE: usize, T: NumericScalar>
    SynapseMatrix<T> for BlockRomCsrSynapses<'_, Idx, Weight, BLOCK_SIZE, T>
{
    #[inline]
    fn len(&self) -> usize {
        self.weights.len()
    }

    #[inline]
    fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [T]) {
        for &src_idx in active_spikes {
            let src_idx = src_idx as usize;
            if src_idx + 1 < self.row_ptrs.len() {
                let start: usize = self.row_ptrs[src_idx].to_usize();
                let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                let end = end.min(self.block_targets.len());

                for b in start..end {
                    let base_target: usize = self.block_targets[b].to_usize();
                    let w_base = b * BLOCK_SIZE;

                    for k in 0..BLOCK_SIZE {
                        let target = base_target + k;
                        let w_idx = w_base + k;
                        if target < input_accumulators.len() && w_idx < self.weights.len() {
                            input_accumulators[target] = input_accumulators[target]
                                .saturating_add(self.weights[w_idx].to_scalar());
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn propagate_weighted_inputs(&self, inputs: &[T], input_accumulators: &mut [T]) {
        for (src_idx, &input_val) in inputs.iter().enumerate() {
            if input_val != T::ZERO && src_idx + 1 < self.row_ptrs.len() {
                let start: usize = self.row_ptrs[src_idx].to_usize();
                let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                let end = end.min(self.block_targets.len());

                for b in start..end {
                    let base_target: usize = self.block_targets[b].to_usize();
                    let w_base = b * BLOCK_SIZE;

                    for k in 0..BLOCK_SIZE {
                        let target = base_target + k;
                        let w_idx = w_base + k;
                        if target < input_accumulators.len() && w_idx < self.weights.len() {
                            let w = self.weights[w_idx].to_scalar();
                            input_accumulators[target] =
                                input_accumulators[target].saturating_add(input_val * w);
                        }
                    }
                }
            }
        }
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
                    let start: usize = self.row_ptrs[src_idx].to_usize();
                    let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                    let end = end.min(self.block_targets.len());

                    for b in start..end {
                        let base_target: usize = self.block_targets[b].to_usize();
                        let w_base = b * BLOCK_SIZE;

                        for k in 0..BLOCK_SIZE {
                            let target = base_target + k;
                            let w_idx = w_base + k;
                            if target < input_accumulators.len() && w_idx < self.weights.len() {
                                input_accumulators[target] = input_accumulators[target]
                                    .saturating_add(self.weights[w_idx].to_scalar());
                            }
                        }
                    }
                }
                remaining &= remaining - 1;
            }
        }
    }
}

impl<Idx: RomIndex> BlockRomCsrSynapses<'_, Idx, Fix16, 2, Fix16> {
    /// Hardware-accelerated synaptic inference via the SIMD QADD16 instruction.
    /// Updates 2 target neurons in a single CPU cycle on ARM Cortex-M DSP.
    #[inline]
    pub fn propagate_active_spikes_simd(
        &self,
        active_spikes: &[u32],
        input_accumulators: &mut [Fix16],
    ) {
        use crate::domain::hardware::qadd16;
        for &src_idx in active_spikes {
            let src_idx = src_idx as usize;
            if src_idx + 1 < self.row_ptrs.len() {
                let start: usize = self.row_ptrs[src_idx].to_usize();
                let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                let end = end.min(self.block_targets.len());

                for b in start..end {
                    let base_target: usize = self.block_targets[b].to_usize();
                    let w_base = b * 2;
                    if base_target + 1 < input_accumulators.len() && w_base + 1 < self.weights.len()
                    {
                        let w0 = self.weights[w_base].0 as u16 as u32;
                        let w1 = self.weights[w_base + 1].0 as u16 as u32;
                        let w_packed = w0 | (w1 << 16);

                        let acc0 = input_accumulators[base_target].0 as u16 as u32;
                        let acc1 = input_accumulators[base_target + 1].0 as u16 as u32;
                        let acc_packed = acc0 | (acc1 << 16);

                        let res_packed = qadd16(acc_packed, w_packed);
                        input_accumulators[base_target] = Fix16(res_packed as i16);
                        input_accumulators[base_target + 1] = Fix16((res_packed >> 16) as i16);
                    }
                }
            }
        }
    }

    /// Hardware-accelerated synaptic inference via SIMD (`QADD16` / `KADD16`) for bitset word spike streams (`SpikeBitset`).
    /// Decodes spikes via hardware CTZ and updates neuron pairs in 1 CPU cycle.
    #[inline]
    pub fn propagate_spikes_words_simd(&self, words: &[u32], input_accumulators: &mut [Fix16]) {
        use crate::domain::hardware::qadd16;
        for (w_idx, &word) in words.iter().enumerate() {
            let mut remaining = word;
            let base_idx = w_idx * 32;
            while remaining != 0 {
                let tz = remaining.trailing_zeros() as usize;
                let src_idx = base_idx + tz;
                if src_idx + 1 < self.row_ptrs.len() {
                    let start: usize = self.row_ptrs[src_idx].to_usize();
                    let end: usize = self.row_ptrs[src_idx + 1].to_usize();
                    let end = end.min(self.block_targets.len());

                    for b in start..end {
                        let base_target: usize = self.block_targets[b].to_usize();
                        let w_base = b * 2;
                        if base_target + 1 < input_accumulators.len()
                            && w_base + 1 < self.weights.len()
                        {
                            let w0 = self.weights[w_base].0 as u16 as u32;
                            let w1 = self.weights[w_base + 1].0 as u16 as u32;
                            let w_packed = w0 | (w1 << 16);

                            let acc0 = input_accumulators[base_target].0 as u16 as u32;
                            let acc1 = input_accumulators[base_target + 1].0 as u16 as u32;
                            let acc_packed = acc0 | (acc1 << 16);

                            let res_packed = qadd16(acc_packed, w_packed);
                            input_accumulators[base_target] = Fix16(res_packed as i16);
                            input_accumulators[base_target + 1] = Fix16((res_packed >> 16) as i16);
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
    fn test_rom_csr_flash_zero_heap() {
        static ROW_PTRS: [u16; 4] = [0, 2, 3, 3];
        static TARGETS: [u16; 3] = [1, 2, 0];
        static WEIGHTS: [Fix32; 3] = [
            Fix32::from_bits(65536),  // 1.0
            Fix32::from_bits(32768),  // 0.5
            Fix32::from_bits(-32768), // -0.5
        ];

        let rom_csr = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS);
        assert_eq!(rom_csr.len(), 3);

        let mut acc = [Fix32::ZERO; 3];
        rom_csr.propagate_active_spikes(&[0], &mut acc);

        assert_eq!(acc[0], Fix32::ZERO);
        assert_eq!(acc[1], Fix32::ONE);
        assert_eq!(acc[2], Fix32::from_num(0.5));
    }

    #[test]
    fn test_quantized_i8_weights() {
        static ROW_PTRS: [u8; 3] = [0, 1, 2];
        static TARGETS: [u8; 2] = [1, 0];
        // 127 scaled with shift 7 corresponds to ~0.992
        static WEIGHTS_I8: [QuantizedI8<7>; 2] = [QuantizedI8(127), QuantizedI8(-64)];

        let rom_csr = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS_I8);
        let mut acc = [Fix32::ZERO; 2];
        rom_csr.propagate_active_spikes(&[0], &mut acc);

        assert!((acc[1].to_float() - 0.992).abs() < 0.01);
    }

    #[test]
    fn test_quantized_rom_csr_synapses() {
        static ROW_PTRS: [u16; 3] = [0, 2, 2];
        static TARGETS: [u16; 2] = [1, 2];
        static WEIGHTS_I8: [i8; 2] = [100, -50];
        // scale = 0.01 (100 * 0.01 = 1.0, -50 * 0.01 = -0.5)
        let scale = Fix32::from_num(0.01);

        let q_synapses = QuantizedRomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS_I8, scale);
        assert_eq!(q_synapses.len(), 2);

        let mut acc = [Fix32::ZERO; 3];
        q_synapses.propagate_active_spikes(&[0], &mut acc);

        assert_eq!(acc[0], Fix32::ZERO);
        assert!((acc[1].to_float() - 1.0).abs() < 0.01);
        assert!((acc[2].to_float() - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn test_quantized_rom_csr_synapses_int_accumulation() {
        static ROW_PTRS: [u16; 3] = [0, 2, 2];
        static TARGETS: [u16; 2] = [1, 2];
        static WEIGHTS_I8: [i8; 2] = [100, -50];
        let scale = Fix32::from_num(0.01);

        let q_synapses = QuantizedRomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS_I8, scale);

        // 1. Pure integer accumulation 1 cycle/synapse without multiplication
        let mut int_acc = [0i32; 3];
        q_synapses.propagate_active_spikes_i32(&[0], &mut int_acc);
        assert_eq!(int_acc[0], 0);
        assert_eq!(int_acc[1], 100);
        assert_eq!(int_acc[2], -50);

        // 2. Application of global scale factor to Fix32
        let mut fix_acc = [Fix32::ZERO; 3];
        q_synapses.scale_accumulators(&int_acc, &mut fix_acc);
        assert_eq!(fix_acc[0], Fix32::ZERO);
        assert!((fix_acc[1].to_float() - 1.0).abs() < 0.01);
        assert!((fix_acc[2].to_float() - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn test_quantized_rom_csr_words_int_accumulation() {
        static ROW_PTRS: [u16; 3] = [0, 2, 2];
        static TARGETS: [u16; 2] = [1, 2];
        static WEIGHTS_I8: [i8; 2] = [40, -20];
        let scale = Fix32::from_num(0.05);

        let q_synapses = QuantizedRomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS_I8, scale);
        let mut int_acc = [0i32; 3];
        // Bit 0 active = neuron 0
        q_synapses.propagate_spikes_words_i32(&[1u32], &mut int_acc);
        assert_eq!(int_acc[1], 40);
        assert_eq!(int_acc[2], -20);
    }

    #[test]
    fn test_block_rom_csr_synapses() {
        // 2 source neurons.
        // Neuron 0 targets block starting at 0 (neurons 0 and 1) with weights [1.5, -0.5].
        static ROW_PTRS: [u16; 3] = [0, 1, 1];
        static BLOCK_TARGETS: [u16; 1] = [0];
        static WEIGHTS: [Fix16; 2] = [Fix16::from_bits(384), Fix16::from_bits(-128)];

        let bsr =
            BlockRomCsrSynapses::<u16, Fix16, 2, Fix16>::new(&ROW_PTRS, &BLOCK_TARGETS, &WEIGHTS);

        let mut acc = [Fix16::ZERO; 2];
        bsr.propagate_active_spikes(&[0], &mut acc);
        assert_eq!(acc[0], Fix16::from_num(1.5));
        assert_eq!(acc[1], Fix16::from_num(-0.5));

        // Test SIMD active spikes
        let mut acc_simd = [Fix16::ZERO; 2];
        bsr.propagate_active_spikes_simd(&[0], &mut acc_simd);
        assert_eq!(acc_simd[0], Fix16::from_num(1.5));
        assert_eq!(acc_simd[1], Fix16::from_num(-0.5));

        // Test SIMD spikes words
        let mut acc_words = [Fix16::ZERO; 2];
        bsr.propagate_spikes_words_simd(&[1u32], &mut acc_words);
        assert_eq!(acc_words[0], Fix16::from_num(1.5));
        assert_eq!(acc_words[1], Fix16::from_num(-0.5));
    }
}
