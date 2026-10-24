//! ⚡ Leaky Integrate-and-Fire (LIF) model in `Fix32` arithmetic.

use crate::domain::numeric::{Fix16, Fix32};
use crate::traits::NeuronDynamics;

/// Post-spike membrane reset mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ResetMode {
    /// Strict reset to resting potential (Hard Reset: V = V_reset).
    #[default]
    Hard,
    /// Reset by subtracting threshold (Soft Reset: V = V - V_threshold).
    /// Preserves residual integration and avoids loss of temporal information.
    Subtract,
}

/// Configuration parameters for a LIF neuron.
///
/// # Example
/// ```rust
/// use snn_core::models::lif::{LifConfig, ResetMode};
/// use snn_core::Fix32;
///
/// let config = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 2)
///     .with_reset_mode(ResetMode::Subtract);
/// assert_eq!(config.threshold, Fix32::ONE);
/// assert_eq!(config.reset_mode, ResetMode::Subtract);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LifConfig {
    /// Action potential threshold (V_thresh).
    pub threshold: Fix32,
    /// Membrane decay factor (decay = exp(-dt/tau)).
    pub decay_factor: Fix32,
    /// Reset potential after spike emission (V_reset).
    pub reset_potential: Fix32,
    /// Absolute refractory period duration in ticks.
    pub refractory_period: u16,
    /// Bias or permanent tonic current injected every tick.
    pub bias: Fix32,
    /// Membrane reset mode (Hard ou Subtract).
    pub reset_mode: ResetMode,
}

impl LifConfig {
    /// Constructs a new LIF configuration without permanent bias.
    pub const fn new(
        threshold: Fix32,
        decay_factor: Fix32,
        reset_potential: Fix32,
        refractory_period: u16,
    ) -> Self {
        Self {
            threshold,
            decay_factor,
            reset_potential,
            refractory_period,
            bias: Fix32::ZERO,
            reset_mode: ResetMode::Hard,
        }
    }

    /// Constructs a new LIF configuration with injected continuous bias.
    pub const fn with_bias(
        threshold: Fix32,
        decay_factor: Fix32,
        reset_potential: Fix32,
        refractory_period: u16,
        bias: Fix32,
    ) -> Self {
        Self {
            threshold,
            decay_factor,
            reset_potential,
            refractory_period,
            bias,
            reset_mode: ResetMode::Hard,
        }
    }

    /// Specifies post-spike reset mode (Hard or Subtract).
    #[must_use]
    pub const fn with_reset_mode(mut self, reset_mode: ResetMode) -> Self {
        self.reset_mode = reset_mode;
        self
    }
}

impl Default for LifConfig {
    fn default() -> Self {
        Self {
            threshold: Fix32::ONE,
            decay_factor: Fix32::from_bits(58982), // ~0.90
            reset_potential: Fix32::ZERO,
            refractory_period: 2,
            bias: Fix32::ZERO,
            reset_mode: ResetMode::Hard,
        }
    }
}

/// Dynamic state variables of a LIF neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct LifState {
    /// Instantaneous membrane potential (V).
    pub membrane_potential: Fix32,
    /// Remaining ticks in the absolute refractory period.
    pub refractory_time_left: u16,
}

impl LifState {
    /// Creates a state initialized at a given resting potential.
    pub const fn new(resting_potential: Fix32) -> Self {
        Self {
            membrane_potential: resting_potential,
            refractory_time_left: 0,
        }
    }
}

/// Leaky Integrate-and-Fire neuron dynamics.
pub struct LifDynamics;

impl NeuronDynamics<Fix32> for LifDynamics {
    type Config = LifConfig;
    type State = LifState;

    #[inline]
    fn step(config: &Self::Config, state: &mut Self::State, input: Fix32) -> bool {
        if state.refractory_time_left > 0 {
            state.refractory_time_left -= 1;
            state.membrane_potential = config.reset_potential;
            return false;
        }

        #[cfg(feature = "cortex-m0")]
        let decayed = crate::domain::hardware::fast_decay_m0(
            state.membrane_potential,
            config.decay_factor.0.max(0) as u32,
        );
        #[cfg(not(feature = "cortex-m0"))]
        let decayed = state.membrane_potential * config.decay_factor;

        state.membrane_potential = decayed + input + config.bias;

        if state.membrane_potential >= config.threshold {
            state.membrane_potential = match config.reset_mode {
                ResetMode::Hard => config.reset_potential,
                ResetMode::Subtract => state.membrane_potential - config.threshold,
            };
            state.refractory_time_left = config.refractory_period;
            true
        } else {
            false
        }
    }
}

/// Configuration parameters for a LIF neuron in 16-bit precision (`Fix16` Q8.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lif16Config {
    pub threshold: Fix16,
    pub decay_factor: Fix16,
    pub reset_potential: Fix16,
    pub refractory_period: u16,
    pub bias: Fix16,
    pub reset_mode: ResetMode,
}

impl Lif16Config {
    pub const fn new(
        threshold: Fix16,
        decay_factor: Fix16,
        reset_potential: Fix16,
        refractory_period: u16,
    ) -> Self {
        Self {
            threshold,
            decay_factor,
            reset_potential,
            refractory_period,
            bias: Fix16::ZERO,
            reset_mode: ResetMode::Hard,
        }
    }

    pub const fn with_bias(
        threshold: Fix16,
        decay_factor: Fix16,
        reset_potential: Fix16,
        refractory_period: u16,
        bias: Fix16,
    ) -> Self {
        Self {
            threshold,
            decay_factor,
            reset_potential,
            refractory_period,
            bias,
            reset_mode: ResetMode::Hard,
        }
    }

    #[must_use]
    pub const fn with_reset_mode(mut self, reset_mode: ResetMode) -> Self {
        self.reset_mode = reset_mode;
        self
    }
}

impl Default for Lif16Config {
    fn default() -> Self {
        Self {
            threshold: Fix16::ONE,
            decay_factor: Fix16::from_bits(230), // ~0.90 in Q8.8
            reset_potential: Fix16::ZERO,
            refractory_period: 2,
            bias: Fix16::ZERO,
            reset_mode: ResetMode::Hard,
        }
    }
}

/// Dynamic state variables of a LIF neuron in 16-bit precision.
/// SRAM footprint: exactly 4 bytes (2 bytes V + 2 bytes refractory, zero padding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Lif16State {
    pub membrane_potential: Fix16,
    pub refractory_time_left: u16,
}

impl Lif16State {
    pub const fn new(resting_potential: Fix16) -> Self {
        Self {
            membrane_potential: resting_potential,
            refractory_time_left: 0,
        }
    }
}

/// LIF neuron dynamics in 16-bit fixed-point Q8.8 (`no_std`).
pub struct Lif16Dynamics;

impl NeuronDynamics<Fix16> for Lif16Dynamics {
    type Config = Lif16Config;
    type State = Lif16State;

    #[inline]
    fn step(config: &Self::Config, state: &mut Self::State, input: Fix16) -> bool {
        if state.refractory_time_left > 0 {
            state.refractory_time_left -= 1;
            state.membrane_potential = config.reset_potential;
            return false;
        }

        state.membrane_potential =
            state.membrane_potential * config.decay_factor + input + config.bias;

        if state.membrane_potential >= config.threshold {
            state.membrane_potential = match config.reset_mode {
                ResetMode::Hard => config.reset_potential,
                ResetMode::Subtract => state.membrane_potential - config.threshold,
            };
            state.refractory_time_left = config.refractory_period;
            true
        } else {
            false
        }
    }
}

impl NeuronDynamics<Fix32> for Lif16Dynamics {
    type Config = Lif16Config;
    type State = Lif16State;

    #[inline]
    fn step(config: &Self::Config, state: &mut Self::State, input: Fix32) -> bool {
        let input_16 = Fix16::from_fix32(input);
        <Self as NeuronDynamics<Fix16>>::step(config, state, input_16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_dynamics_bias() {
        let config = LifConfig::with_bias(
            Fix32::ONE,
            Fix32::from_num(0.5),
            Fix32::ZERO,
            1,
            Fix32::from_num(0.6), // bias = 0.6
        );
        let mut state = LifState::default();
        // With input = 0.5, total injection = 0.5 + 0.6 = 1.1 >= 1.0 (spike!)
        assert!(LifDynamics::step(&config, &mut state, Fix32::from_num(0.5)));
    }

    #[test]
    fn test_lif_dynamics_spike_and_refractory() {
        let config = LifConfig::new(Fix32::from_num(1.0), Fix32::from_num(0.9), Fix32::ZERO, 2);
        let mut state = LifState::default();

        // Tick 1: input 0.5 -> V = 0.5 (no spike)
        assert!(!LifDynamics::step(
            &config,
            &mut state,
            Fix32::from_num(0.5)
        ));
        assert_eq!(state.membrane_potential, Fix32::from_num(0.5));

        // Tick 2: input 0.6 -> V = 0.5 * 0.9 + 0.6 = 1.05 >= 1.0 (spike!)
        assert!(LifDynamics::step(&config, &mut state, Fix32::from_num(0.6)));
        assert_eq!(state.membrane_potential, Fix32::ZERO);
        assert_eq!(state.refractory_time_left, 2);

        // Tick 3: in refractory period (input ignored)
        assert!(!LifDynamics::step(
            &config,
            &mut state,
            Fix32::from_num(5.0)
        ));
        assert_eq!(state.membrane_potential, Fix32::ZERO);
        assert_eq!(state.refractory_time_left, 1);

        // Tick 4: last refractory tick
        assert!(!LifDynamics::step(
            &config,
            &mut state,
            Fix32::from_num(5.0)
        ));
        assert_eq!(state.membrane_potential, Fix32::ZERO);
        assert_eq!(state.refractory_time_left, 0);

        // Tick 5: exiting refractory period
        assert!(LifDynamics::step(&config, &mut state, Fix32::from_num(1.2)));
    }

    #[test]
    fn test_lif_dynamics_subtract_reset() {
        let config = LifConfig::new(
            Fix32::from_num(1.0),
            Fix32::from_num(1.0), // no decay
            Fix32::ZERO,
            0, // no refractory period to observe immediately
        )
        .with_reset_mode(ResetMode::Subtract);
        let mut state = LifState::default();

        // Input 1.25 -> V = 1.25 >= 1.0 (spike!)
        // V must become 1.25 - 1.0 = 0.25 (exact preservation of surplus)
        assert!(LifDynamics::step(
            &config,
            &mut state,
            Fix32::from_num(1.25)
        ));
        assert_eq!(state.membrane_potential, Fix32::from_num(0.25));

        // Next tick with input 0.875 -> V = 0.25 + 0.875 = 1.125 >= 1.0 (spike!)
        // V becomes 1.125 - 1.0 = 0.125
        assert!(LifDynamics::step(
            &config,
            &mut state,
            Fix32::from_num(0.875)
        ));
        assert_eq!(state.membrane_potential, Fix32::from_num(0.125));
    }

    #[test]
    fn test_lif_dynamics_refractory_clamping_with_negative_reset() {
        let v_reset = Fix32::from_num(-0.7);
        let config = LifConfig::new(
            Fix32::from_num(1.0),
            Fix32::from_num(0.5), // non-zero decay
            v_reset,
            2,
        );
        let mut state = LifState::default();

        // Trigger spike
        assert!(LifDynamics::step(&config, &mut state, Fix32::from_num(1.5)));
        assert_eq!(state.membrane_potential, v_reset);
        assert_eq!(state.refractory_time_left, 2);

        // Refractory tick 1: V MUST remain clamped at v_reset (-0.7) and NOT decay towards 0 (-0.35)
        assert!(!LifDynamics::step(
            &config,
            &mut state,
            Fix32::from_num(10.0)
        ));
        assert_eq!(state.membrane_potential, v_reset);
        assert_eq!(state.refractory_time_left, 1);

        // Refractory tick 2
        assert!(!LifDynamics::step(
            &config,
            &mut state,
            Fix32::from_num(10.0)
        ));
        assert_eq!(state.membrane_potential, v_reset);
        assert_eq!(state.refractory_time_left, 0);
    }
}
