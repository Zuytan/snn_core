use crate::domain::numeric::{Fix32, NumericScalar};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::marker::PhantomData;

/// A classification readout layer optimized for microcontroller inference (`no_std`).
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct ReadoutLayer<T: NumericScalar = Fix32> {
    pub class_weights: Vec<Vec<T>>,
}

#[cfg(feature = "alloc")]
impl<T: NumericScalar> ReadoutLayer<T> {
    pub fn new(class_weights: Vec<Vec<T>>) -> Self {
        ReadoutLayer { class_weights }
    }

    pub fn predict(&self, state: &[T]) -> usize {
        let mut best_class = 0;
        let mut max_score = T::MIN;

        for (c, weights) in self.class_weights.iter().enumerate() {
            let mut score = weights[0];
            for (i, &x) in state.iter().enumerate() {
                if i + 1 < weights.len() {
                    score = score.saturating_add(weights[i + 1] * x);
                }
            }

            if score > max_score {
                max_score = score;
                best_class = c;
            }
        }
        best_class
    }
}

#[cfg(feature = "alloc")]
impl<T: NumericScalar> crate::traits::Readout<T> for ReadoutLayer<T> {
    #[inline]
    fn predict(&self, state: &[T]) -> usize {
        self.predict(state)
    }
}

/// Static Readout layer stored in flash ROM (zero heap allocation, pure `no_std`).
#[derive(Clone, Copy, Debug)]
pub struct StaticReadoutLayer<const C: usize, const D: usize, T: NumericScalar + 'static = Fix32> {
    pub weights: &'static [[T; D]; C],
    _marker: PhantomData<T>,
}

impl<const C: usize, const D: usize, T: NumericScalar + 'static> StaticReadoutLayer<C, D, T> {
    pub const fn new(weights: &'static [[T; D]; C]) -> Self {
        Self {
            weights,
            _marker: PhantomData,
        }
    }

    pub fn predict(&self, state: &[T]) -> usize {
        let mut best_class = 0;
        let mut max_score = T::MIN;

        for c in 0..C {
            let mut score = self.weights[c][0];
            for (i, &x) in state.iter().enumerate() {
                if i + 1 < D {
                    score = score.saturating_add(self.weights[c][i + 1] * x);
                }
            }
            if score > max_score {
                max_score = score;
                best_class = c;
            }
        }
        best_class
    }

    /// Ultra-fast multiplication-free inference from a `u64` spike bitmask.
    ///
    /// Each bit set to 1 in `mask` corresponds to a firing neuron.
    /// Uses native hardware `CTZ` (`trailing_zeros`) to instantly skip inactive neurons
    /// and accumulate only the weights of active neurons.
    /// Zero multiplication required!
    #[inline]
    pub fn predict_spikes_mask(&self, mask: u64) -> usize {
        let mut best_class = 0;
        let mut max_score = T::MIN;

        for c in 0..C {
            let mut score = self.weights[c][0]; // Initial bias
            let mut remaining = mask;

            while remaining != 0 {
                let tz = remaining.trailing_zeros() as usize;
                let feat_idx = tz + 1;
                if feat_idx < D {
                    score = score.saturating_add(self.weights[c][feat_idx]);
                }
                remaining &= remaining - 1; // Clear active bit in 1 cycle (BLSR)
            }

            if score > max_score {
                max_score = score;
                best_class = c;
            }
        }

        best_class
    }

    /// Direct event-driven inference from a bitset [`crate::domain::bitset::SpikeBitset`].
    #[inline]
    pub fn predict_bitset<const WORDS: usize>(
        &self,
        bitset: &crate::domain::bitset::SpikeBitset<WORDS>,
    ) -> usize {
        let mut best_class = 0;
        let mut max_score = T::MIN;

        for c in 0..C {
            let mut score = self.weights[c][0];

            for (w_idx, &word) in bitset.words().iter().enumerate() {
                let base_idx = w_idx * 32;
                let mut remaining = word;
                while remaining != 0 {
                    let tz = remaining.trailing_zeros() as usize;
                    let feat_idx = base_idx + tz + 1;
                    if feat_idx < D {
                        score = score.saturating_add(self.weights[c][feat_idx]);
                    }
                    remaining &= remaining - 1;
                }
            }

            if score > max_score {
                max_score = score;
                best_class = c;
            }
        }

        best_class
    }
}

impl<const C: usize, const D: usize> StaticReadoutLayer<C, D, crate::domain::numeric::Fix16> {
    /// Hardware-accelerated classification inference for Fix16 (Q8.8).
    /// On ARM Cortex-M4/M7/M33 microcontrollers with DSP, leverages the SMLAD instruction (2 MACs in 1 cycle).
    pub fn predict_simd(&self, state: &[crate::domain::numeric::Fix16]) -> usize {
        use crate::domain::hardware::smlad;
        use crate::domain::numeric::Fix16;

        let mut best_class = 0;
        let mut max_score = Fix16::MIN;

        for c in 0..C {
            let mut acc_raw = (self.weights[c][0].0 as i32) << 8; // Aligned to Q8.16 for accumulation
            let count = state.len().min(D.saturating_sub(1));
            let mut i = 0;

            // Unrolled loop processing 2 weights/features in parallel
            while i + 1 < count {
                let w0 = self.weights[c][i + 1].0 as u16 as u32;
                let w1 = self.weights[c][i + 2].0 as u16 as u32;
                let w_packed = w0 | (w1 << 16);

                let x0 = state[i].0 as u16 as u32;
                let x1 = state[i + 1].0 as u16 as u32;
                let x_packed = x0 | (x1 << 16);

                acc_raw = smlad(w_packed, x_packed, acc_raw);
                i += 2;
            }

            // Odd residual element
            if i < count {
                let p = (self.weights[c][i + 1].0 as i32) * (state[i].0 as i32);
                acc_raw = acc_raw.saturating_add(p);
            }

            // Re-normalization Q8.16 -> Q8.8 (divide by 256 / shift by 8 bits)
            let score_bits = (acc_raw >> 8).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            let score = Fix16(score_bits);

            if score > max_score {
                max_score = score;
                best_class = c;
            }
        }
        best_class
    }
}

impl<const C: usize, const D: usize, T: NumericScalar + 'static> crate::traits::Readout<T>
    for StaticReadoutLayer<C, D, T>
{
    #[inline]
    fn predict(&self, state: &[T]) -> usize {
        self.predict(state)
    }
}

/// Trainable online Readout layer in pure `no_std` (zero heap allocation).
///
/// Implements the supervised Delta learning rule (LMS / Widrow-Hoff) with
/// L2 regularization (Ridge / Weight Decay).
#[derive(Clone, Copy, Debug)]
pub struct TrainableReadoutLayer<const C: usize, const D: usize, T: NumericScalar = Fix32> {
    pub weights: [[T; D]; C],
    pub learning_rate: T,
    pub l2_reg: T,
}

impl<const C: usize, const D: usize, T: NumericScalar> TrainableReadoutLayer<C, D, T> {
    /// Constructs a Readout layer ready for training.
    pub const fn new(learning_rate: T, l2_reg: T) -> Self {
        Self {
            weights: [[T::ZERO; D]; C],
            learning_rate,
            l2_reg,
        }
    }

    pub fn predict(&self, state: &[T]) -> usize {
        let mut best_class = 0;
        let mut max_score = T::MIN;

        for c in 0..C {
            let mut score = self.weights[c][0];
            for (i, &x) in state.iter().enumerate() {
                if i + 1 < D {
                    score = score.saturating_add(self.weights[c][i + 1] * x);
                }
            }
            if score > max_score {
                max_score = score;
                best_class = c;
            }
        }
        best_class
    }

    /// Online supervised learning step (regularized Delta / LMS rule).
    /// `target_class` is the index of the ground truth class (0..C).
    pub fn train_step(&mut self, state: &[T], target_class: usize) {
        if target_class >= C {
            return;
        }

        for c in 0..C {
            let target_val = if c == target_class { T::ONE } else { T::ZERO };

            let mut pred_score = self.weights[c][0];
            for (i, &x) in state.iter().enumerate() {
                if i + 1 < D {
                    pred_score = pred_score.saturating_add(self.weights[c][i + 1] * x);
                }
            }

            let err = target_val.saturating_sub(pred_score);
            let step = self.learning_rate * err;

            // Bias update
            self.weights[c][0] = self.weights[c][0].saturating_add(step);

            // Weights update with L2 regularization
            for (i, &x) in state.iter().enumerate() {
                if i + 1 < D {
                    let w = &mut self.weights[c][i + 1];
                    let decay = *w * self.l2_reg;
                    *w = w.saturating_sub(decay).saturating_add(step * x);
                }
            }
        }
    }
}

impl<const C: usize, const D: usize, T: NumericScalar> crate::traits::Readout<T>
    for TrainableReadoutLayer<C, D, T>
{
    #[inline]
    fn predict(&self, state: &[T]) -> usize {
        self.predict(state)
    }
}

/// Continuous regression Readout layer trainable online in pure `no_std` (zero heap allocation).
///
/// Implements LMS / Delta rule with L2 regularization (Weight Decay) to predict a continuous
/// scalar value (essential for NARMA-10, motor control, adaptive signal filtering).
#[derive(Clone, Copy, Debug)]
pub struct TrainableRegressionReadout<const D: usize, T: NumericScalar = Fix32> {
    /// Model weights: `weights[0]` is bias, `weights[1..D]` are feature weights.
    pub weights: [T; D],
    pub learning_rate: T,
    pub l2_reg: T,
}

impl<const D: usize, T: NumericScalar> TrainableRegressionReadout<D, T> {
    /// Constructs a regression layer initialized to zero.
    pub const fn new(learning_rate: T, l2_reg: T) -> Self {
        Self {
            weights: [T::ZERO; D],
            learning_rate,
            l2_reg,
        }
    }

    /// Predicts the continuous scalar value for a given state vector.
    pub fn predict(&self, state: &[T]) -> T {
        let mut pred = if D > 0 { self.weights[0] } else { T::ZERO };
        for (i, &x) in state.iter().enumerate() {
            if i + 1 < D {
                pred = pred.saturating_add(self.weights[i + 1] * x);
            }
        }
        pred
    }

    /// Online supervised learning step (Delta / LMS rule with L2 regularization).
    /// Returns the residual error `target - prediction`.
    pub fn train_step(&mut self, state: &[T], target: T) -> T {
        let pred = self.predict(state);
        let err = target.saturating_sub(pred);
        let step = self.learning_rate * err;

        if D > 0 {
            // Update bias without weight regularization
            self.weights[0] = self.weights[0].saturating_add(step);
        }

        for (i, &x) in state.iter().enumerate() {
            if i + 1 < D {
                let w = &mut self.weights[i + 1];
                let decay = *w * self.l2_reg;
                *w = w.saturating_sub(decay).saturating_add(step * x);
            }
        }

        err
    }

    /// NLMS (Normalized Least Mean Squares) supervised learning step.
    ///
    /// Adapts the step size normalized by input vector energy:
    /// $$\Delta w_i = \frac{\eta}{1 + \|x\|^2} (y - \hat{y}) x_i$$
    /// Guarantees fast, unconditionally stable convergence in Fix32 arithmetic,
    /// eliminating divergence risks on neuromorphic reservoir states.
    pub fn train_step_nlms(&mut self, state: &[T], target: T) -> T {
        let pred = self.predict(state);
        let err = target.saturating_sub(pred);

        let mut norm_sq = T::ONE;
        for &x in state {
            norm_sq = norm_sq.saturating_add(x * x);
        }

        let normalized_lr = if norm_sq > T::ZERO {
            self.learning_rate / norm_sq
        } else {
            T::ZERO
        };
        let step = normalized_lr * err;

        if D > 0 {
            self.weights[0] = self.weights[0].saturating_add(step);
        }

        for (i, &x) in state.iter().enumerate() {
            if i + 1 < D {
                let w = &mut self.weights[i + 1];
                let decay = *w * self.l2_reg;
                *w = w.saturating_sub(decay).saturating_add(step * x);
            }
        }

        err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainable_readout_convergence() {
        // 2 classes, 3 dimensions (bias + 2 features)
        let mut readout = TrainableReadoutLayer::<2, 3>::new(
            Fix32::from_num(0.1),   // learning rate
            Fix32::from_num(0.001), // l2 reg
        );

        let sample_class_0 = [Fix32::ONE, Fix32::ZERO];
        let sample_class_1 = [Fix32::ZERO, Fix32::ONE];

        // Train for 30 epochs
        for _ in 0..30 {
            readout.train_step(&sample_class_0, 0);
            readout.train_step(&sample_class_1, 1);
        }

        assert_eq!(readout.predict(&sample_class_0), 0);
        assert_eq!(readout.predict(&sample_class_1), 1);
    }

    #[test]
    fn test_trainable_regression_readout_convergence() {
        // D = 3 : bias + 2 inputs x0, x1
        // Target relationship: y = 2 * x0 - x1 + 0.5
        let mut readout =
            TrainableRegressionReadout::<3>::new(Fix32::from_num(0.05), Fix32::from_num(0.0001));

        let data = [
            (
                [Fix32::from_num(1.0), Fix32::from_num(0.0)],
                Fix32::from_num(2.5),
            ),
            (
                [Fix32::from_num(0.0), Fix32::from_num(1.0)],
                Fix32::from_num(-0.5),
            ),
            (
                [Fix32::from_num(1.0), Fix32::from_num(1.0)],
                Fix32::from_num(1.5),
            ),
            (
                [Fix32::from_num(0.5), Fix32::from_num(0.5)],
                Fix32::from_num(1.0),
            ),
        ];

        for _ in 0..100 {
            for (x, y) in &data {
                readout.train_step(x, *y);
            }
        }

        let pred = readout.predict(&[Fix32::from_num(1.0), Fix32::from_num(0.0)]);
        let err = (pred - Fix32::from_num(2.5)).abs();
        assert!(
            err < Fix32::from_num(0.2),
            "Regression error should be small: got {:?}",
            pred
        );
    }

    #[test]
    fn test_static_readout_layer_simd_parity() {
        use crate::domain::numeric::Fix16;
        static WEIGHTS: [[Fix16; 3]; 2] = [
            [
                Fix16::from_bits(256),
                Fix16::from_bits(512),
                Fix16::from_bits(-256),
            ], // c0: bias 1.0, w1 2.0, w2 -1.0
            [
                Fix16::from_bits(0),
                Fix16::from_bits(-512),
                Fix16::from_bits(768),
            ], // c1: bias 0.0, w1 -2.0, w2 3.0
        ];
        let layer = StaticReadoutLayer::<2, 3, Fix16>::new(&WEIGHTS);

        // state: [0.5, 1.0]
        let state = [Fix16::from_num(0.5), Fix16::from_num(1.0)];
        let pred_std = layer.predict(&state);
        let pred_simd = layer.predict_simd(&state);

        // c0 score: 1.0 + (0.5 * 2.0) + (1.0 * -1.0) = 1.0 + 1.0 - 1.0 = 1.0
        // c1 score: 0.0 + (0.5 * -2.0) + (1.0 * 3.0) = 0.0 - 1.0 + 3.0 = 2.0 -> c1 wins
        assert_eq!(pred_std, 1);
        assert_eq!(pred_simd, 1);
        assert_eq!(pred_std, pred_simd);
    }

    #[test]
    fn test_static_readout_predict_spikes_mask_and_bitset() {
        use crate::domain::bitset::SpikeBitset;

        // D = 4 : Bias + 3 features (neurons 0, 1, 2)
        static WEIGHTS: [[Fix32; 4]; 2] = [
            [
                Fix32::from_bits(0),
                Fix32::from_bits(2 << 16),  // n0: +2.0
                Fix32::from_bits(-1 << 16), // n1: -1.0
                Fix32::from_bits(1 << 15),  // n2: +0.5
            ],
            [
                Fix32::from_bits(0),
                Fix32::from_bits(-1 << 16), // n0: -1.0
                Fix32::from_bits(3 << 16),  // n1: +3.0
                Fix32::from_bits(0),        // n2: 0.0
            ],
        ];

        let layer = StaticReadoutLayer::<2, 4, Fix32>::new(&WEIGHTS);

        // Case 1: Only neuron 0 spikes (mask = 0b001 = 1)
        let mask_0 = 1u64;
        let pred_mask_0 = layer.predict_spikes_mask(mask_0);
        let mut bitset_0 = SpikeBitset::<1>::new();
        bitset_0.set(0);
        let pred_bitset_0 = layer.predict_bitset(&bitset_0);
        let state_0 = [Fix32::ONE, Fix32::ZERO, Fix32::ZERO];
        let pred_dense_0 = layer.predict(&state_0);

        assert_eq!(pred_mask_0, 0);
        assert_eq!(pred_bitset_0, 0);
        assert_eq!(pred_dense_0, 0);

        // Case 2: Only neuron 1 spikes (mask = 0b010 = 2)
        let mask_1 = 2u64;
        let pred_mask_1 = layer.predict_spikes_mask(mask_1);
        let mut bitset_1 = SpikeBitset::<1>::new();
        bitset_1.set(1);
        let pred_bitset_1 = layer.predict_bitset(&bitset_1);
        let state_1 = [Fix32::ZERO, Fix32::ONE, Fix32::ZERO];
        let pred_dense_1 = layer.predict(&state_1);

        assert_eq!(pred_mask_1, 1);
        assert_eq!(pred_bitset_1, 1);
        assert_eq!(pred_dense_1, 1);
    }
}
