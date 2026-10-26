//! 📈 Canonical NARMA-10 Benchmark (Nonlinear Auto-Regressive Moving Average).
//!
//! International benchmark task (Jaeger, Maass) to evaluate:
//! 1. **Temporal memory capacity** (*Memory Capacity* $MC$) of a recurrent reservoir.
//! 2. **Nonlinear computational richness** without catastrophic forgetting.
//! 3. Normalized Mean Squared Error (**NMSE**).
//!
//! Standard NARMA-10 equation:
//! $$y[t+1] = 0.3 y[t] + 0.05 y[t] \left(\sum_{i=0}^9 y[t-i]\right) + 1.5 u[t-9] u[t] + 0.1$$

use snn_core::Fix32;

/// Generator and simulator for the NARMA-10 chaotic nonlinear time series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Narma10System {
    /// Sliding history of the last 10 target values $y[t-i]$.
    pub y_history: [Fix32; 10],
    /// Sliding history of the last 10 inputs $u[t-i]$.
    pub u_history: [Fix32; 10],
    /// Circular cursor for the sliding buffer.
    cursor: usize,
}

impl Default for Narma10System {
    fn default() -> Self {
        Self::new()
    }
}

impl Narma10System {
    pub const fn new() -> Self {
        Self {
            y_history: [Fix32::ZERO; 10],
            u_history: [Fix32::ZERO; 10],
            cursor: 0,
        }
    }

    /// Resets the NARMA-10 system state.
    pub fn reset(&mut self) {
        self.y_history = [Fix32::ZERO; 10];
        self.u_history = [Fix32::ZERO; 10];
        self.cursor = 0;
    }

    /// Advances by one time step by injecting input $u[t]$ (typically in $[0, 0.5]$).
    /// Returns the target value $y[t+1]$.
    pub fn step(&mut self, u_t: Fix32) -> Fix32 {
        let c_03 = Fix32::from_bits(19661); // 0.3
        let c_005 = Fix32::from_bits(3277); // 0.05
        let c_15 = Fix32::from_bits(98304); // 1.5
        let c_01 = Fix32::from_bits(6554); // 0.1

        // u[t-9] corresponds to oldest element in circular buffer of size 10
        let oldest_idx = (self.cursor + 1) % 10;
        let u_t_minus_9 = self.u_history[oldest_idx];

        // y[t] corresponds to most recent value
        let y_t = self.y_history[self.cursor];

        // Sum of past 10 values of y
        let sum_y: Fix32 = self.y_history.iter().copied().sum();

        // y[t+1] = 0.3*y[t] + 0.05*y[t]*(sum) + 1.5*u[t-9]*u[t] + 0.1
        let term1 = c_03 * y_t;
        let term2 = c_005 * y_t * sum_y;
        let term3 = c_15 * u_t_minus_9 * u_t;
        let y_next = term1 + term2 + term3 + c_01;

        // Circular update
        self.cursor = oldest_idx;
        self.u_history[self.cursor] = u_t;
        self.y_history[self.cursor] = y_next;

        y_next
    }

    /// Computes Normalized Mean Squared Error (NMSE) between targets and predictions without overflow.
    ///
    /// $$NMSE = \frac{\sum (y[t] - \hat{y}[t])^2}{\sum (y[t] - \bar{y})^2}$$
    pub fn compute_nmse(targets: &[Fix32], predictions: &[Fix32]) -> Fix32 {
        if targets.is_empty() || targets.len() != predictions.len() {
            return Fix32::MAX;
        }

        let count = targets.len() as i64;
        let mut sum_target_i64: i64 = 0;
        for &t in targets {
            sum_target_i64 += i64::from(t.0);
        }
        let mean_target_i64 = sum_target_i64 / count;

        let mut mse_sum_i128: i128 = 0;
        let mut var_sum_i128: i128 = 0;

        for (&y, &y_hat) in targets.iter().zip(predictions.iter()) {
            let err = i64::from(y.0) - i64::from(y_hat.0);
            mse_sum_i128 += i128::from(err) * i128::from(err);

            let diff = i64::from(y.0) - mean_target_i64;
            var_sum_i128 += i128::from(diff) * i128::from(diff);
        }

        if var_sum_i128 == 0 {
            return Fix32::MAX;
        }

        let ratio = (mse_sum_i128 << 16) / var_sum_i128;
        if ratio > i128::from(i32::MAX) {
            Fix32::MAX
        } else {
            Fix32::from_bits(ratio as i32)
        }
    }

    /// Computes linear correlation coefficient $r^2$ between two signals without overflow.
    pub fn compute_r_squared(x: &[Fix32], y: &[Fix32]) -> Fix32 {
        if x.len() != y.len() || x.is_empty() {
            return Fix32::ZERO;
        }

        let count = x.len() as i64;
        let mut sum_x: i64 = 0;
        let mut sum_y: i64 = 0;
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            sum_x += i64::from(xi.0);
            sum_y += i64::from(yi.0);
        }
        let mean_x = sum_x / count;
        let mean_y = sum_y / count;

        let mut cov_i128: i128 = 0;
        let mut var_x_i128: i128 = 0;
        let mut var_y_i128: i128 = 0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = i128::from(i64::from(xi.0) - mean_x);
            let dy = i128::from(i64::from(yi.0) - mean_y);
            cov_i128 += dx * dy;
            var_x_i128 += dx * dx;
            var_y_i128 += dy * dy;
        }

        let denom = var_x_i128 * var_y_i128;
        if denom == 0 {
            return Fix32::ZERO;
        }

        let cov_sq = cov_i128 * cov_i128;
        let r2_raw = (cov_sq << 16) / denom;
        if r2_raw > i128::from(i32::MAX) {
            Fix32::ONE
        } else {
            Fix32::from_bits(r2_raw as i32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_narma10_dynamics_bounded() {
        let mut narma = Narma10System::new();

        // Generate 100 NARMA-10 steps with pseudo-random u[t] in [0, 0.5]
        let mut rng = fastrand::Rng::with_seed(42);
        for _ in 0..100 {
            let rand_val = Fix32::from_bits((rng.u32(0..32768)) as i32); // 0.0 to 0.5
            let y = narma.step(rand_val);

            // Verify physical bounds: y remains positive and < 1.8
            assert!(y >= Fix32::ZERO, "NARMA-10 output must remain non-negative");
            assert!(y < Fix32::from_num(1.8), "NARMA-10 output must not diverge");
        }
    }

    #[test]
    fn test_nmse_perfect_prediction() {
        let targets = [
            Fix32::from_num(0.5),
            Fix32::from_num(0.8),
            Fix32::from_num(0.2),
        ];
        let preds = targets;
        assert_eq!(Narma10System::compute_nmse(&targets, &preds), Fix32::ZERO);
    }
}
