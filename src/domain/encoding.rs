//! 📡 Signal-to-spike encoders (Spike Encoders) in pure `no_std` Fix32.
//!
//! Converts continuous sensor streams (audio, accelerometer, temperature, ECG)
//! into discrete spike trains usable by neuromorphic networks:
//! - [`DeltaModulationEncoder`]: Differential events via Asynchronous Delta Modulation (UP/DOWN spikes).
//! - [`RateEncoder`]: Frequency encoding using a deterministic Sigma-Delta accumulator (no FPU or RNG).
//! - [`LatencyEncoder`]: Temporal latency encoding (Time-to-First-Spike - TTFS).

use crate::domain::numeric::Fix32;

/// Differential delta modulation encoder (Asynchronous Delta Modulation - ADM).
///
/// Generates increase (+delta) or decrease (-delta) spikes when
/// the signal crosses a variation threshold $\theta$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaModulationEncoder {
    pub threshold: Fix32,
    pub last_val: Fix32,
}

impl DeltaModulationEncoder {
    pub const fn new(threshold: Fix32, initial_value: Fix32) -> Self {
        Self {
            threshold,
            last_val: initial_value,
        }
    }

    /// Resets the baseline state.
    pub fn reset(&mut self, val: Fix32) {
        self.last_val = val;
    }

    /// Processes a new sample value.
    /// Returns `(spike_up, spike_down)`.
    #[inline]
    pub fn step(&mut self, current_val: Fix32) -> (bool, bool) {
        let diff = current_val - self.last_val;
        if diff >= self.threshold {
            self.last_val += self.threshold;
            (true, false)
        } else if diff <= -self.threshold {
            self.last_val -= self.threshold;
            (false, true)
        } else {
            (false, false)
        }
    }
}

/// Frequency modulation encoder via deterministic Sigma-Delta accumulation.
///
/// Integrates the input value into an accumulator; when reaching `Fix32::ONE`,
/// a spike is emitted and the accumulator is decremented.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RateEncoder {
    pub accumulator: Fix32,
}

impl RateEncoder {
    pub const fn new() -> Self {
        Self {
            accumulator: Fix32::ZERO,
        }
    }

    pub fn reset(&mut self) {
        self.accumulator = Fix32::ZERO;
    }

    /// Advances one step with input intensity `input` (positive).
    /// Returns `true` if a spike is emitted this tick.
    #[inline]
    pub fn step(&mut self, input: Fix32) -> bool {
        self.accumulator += input;
        if self.accumulator >= Fix32::ONE {
            self.accumulator -= Fix32::ONE;
            true
        } else {
            false
        }
    }
}

/// Latency encoder (Time-To-First-Spike - TTFS).
///
/// The more intense the signal, the earlier the first spike is emitted across a window of $T$ ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatencyEncoder {
    pub time_window: usize,
}

impl LatencyEncoder {
    pub const fn new(time_window: usize) -> Self {
        Self { time_window }
    }

    /// Computes the spike emission tick for a normalized input value in $[0, 1]$.
    /// Returns `None` if value is zero or too small.
    pub fn encode_tick(&self, normalized_val: Fix32) -> Option<usize> {
        if normalized_val <= Fix32::ZERO || self.time_window == 0 {
            return None;
        }

        // tick = (1.0 - val) * (time_window - 1)
        let inverted = (Fix32::ONE - normalized_val).max(Fix32::ZERO);
        let max_t = Fix32::from_int((self.time_window - 1) as i32);
        let tick_fix = inverted * max_t;
        let tick_idx = (tick_fix.0 >> 16) as usize;
        Some(tick_idx.min(self.time_window - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_modulation_encoder() {
        let mut encoder = DeltaModulationEncoder::new(Fix32::from_num(0.5), Fix32::ZERO);

        // Increase of 0.6 >= 0.5 -> spike UP
        assert_eq!(encoder.step(Fix32::from_num(0.6)), (true, false));
        // Small variation < 0.5 -> no spike
        assert_eq!(encoder.step(Fix32::from_num(0.8)), (false, false));
        // Sudden drop to 0.0 -> spike DOWN
        assert_eq!(encoder.step(Fix32::ZERO), (false, true));
    }

    #[test]
    fn test_rate_encoder_deterministic() {
        let mut rate = RateEncoder::new();
        let input = Fix32::from_num(0.5); // Should spike 1 tick out of 2

        assert!(!rate.step(input)); // 0.5 < 1.0
        assert!(rate.step(input)); // 1.0 >= 1.0 (Spike!)
        assert!(!rate.step(input));
        assert!(rate.step(input)); // Spike!
    }

    #[test]
    fn test_latency_encoder() {
        let encoder = LatencyEncoder::new(10);

        // Maximum value 1.0 -> spike at tick 0 (fastest)
        assert_eq!(encoder.encode_tick(Fix32::ONE), Some(0));

        // Intermediate value 0.5 -> spike around tick 4 or 5
        let mid_tick = encoder.encode_tick(Fix32::from_num(0.5)).unwrap();
        assert!((4..=5).contains(&mid_tick));
    }
}
