//! ⚡ Current-Based Leaky Integrate-and-Fire (CUBA-LIF) model in `Fix32` arithmetic.
//!
//! Dual-time-constant model conforming to the standard specifications
//! of **NIR (Neuromorphic Intermediate Representation)**, `snnTorch` (`snn.Synaptic`), and `Norse`:
//! 1. Exponential decay of synaptic current:
//!    $$I[t] = I[t-1] \cdot \alpha_{\text{syn}} + \text{input}$$
//! 2. Membrane decay and integration:
//!    $$V[t] = V[t-1] \cdot \alpha_{\text{mem}} + I[t]$$
//! 3. Spike emission if $V[t] \ge V_{\text{th}}$, with reset to $V_{\text{reset}}$.

use super::lif::ResetMode;
use crate::domain::numeric::Fix32;
use crate::traits::NeuronDynamics;

/// CUBA-LIF model configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubaLifConfig {
    /// Action potential threshold.
    pub threshold: Fix32,
    /// Membrane decay factor (alpha_mem = exp(-dt/tau_mem)).
    pub decay_factor_mem: Fix32,
    /// Synaptic current decay factor (alpha_syn = exp(-dt/tau_syn)).
    pub decay_factor_syn: Fix32,
    /// Reset potential after spike emission.
    pub reset_potential: Fix32,
    /// Absolute refractory period duration in ticks.
    pub refractory_period: u16,
    /// Bias or permanent tonic injected current.
    pub bias: Fix32,
    /// Membrane reset mode (Hard or Subtract).
    pub reset_mode: ResetMode,
}

impl CubaLifConfig {
    pub const fn new(
        threshold: Fix32,
        decay_factor_mem: Fix32,
        decay_factor_syn: Fix32,
        reset_potential: Fix32,
        refractory_period: u16,
    ) -> Self {
        Self {
            threshold,
            decay_factor_mem,
            decay_factor_syn,
            reset_potential,
            refractory_period,
            bias: Fix32::ZERO,
            reset_mode: ResetMode::Hard,
        }
    }

    pub const fn with_bias(
        threshold: Fix32,
        decay_factor_mem: Fix32,
        decay_factor_syn: Fix32,
        reset_potential: Fix32,
        refractory_period: u16,
        bias: Fix32,
    ) -> Self {
        Self {
            threshold,
            decay_factor_mem,
            decay_factor_syn,
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

impl Default for CubaLifConfig {
    fn default() -> Self {
        Self {
            threshold: Fix32::ONE,
            decay_factor_mem: Fix32::from_bits(58982), // ~0.90
            decay_factor_syn: Fix32::from_bits(52429), // ~0.80
            reset_potential: Fix32::ZERO,
            refractory_period: 2,
            bias: Fix32::ZERO,
            reset_mode: ResetMode::Hard,
        }
    }
}

/// State variables of a CUBA-LIF neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CubaLifState {
    /// Instantaneous membrane potential (V).
    pub membrane_potential: Fix32,
    /// Instantaneous synaptic current (I).
    pub synaptic_current: Fix32,
    /// Remaining ticks in the absolute refractory period.
    pub refractory_time_left: u16,
}

impl CubaLifState {
    pub const fn new(resting_potential: Fix32) -> Self {
        Self {
            membrane_potential: resting_potential,
            synaptic_current: Fix32::ZERO,
            refractory_time_left: 0,
        }
    }
}

/// Current-Based Leaky Integrate-and-Fire neuron dynamics.
pub struct CubaLifDynamics;

impl NeuronDynamics for CubaLifDynamics {
    type Config = CubaLifConfig;
    type State = CubaLifState;

    #[inline]
    fn step(config: &Self::Config, state: &mut Self::State, input: Fix32) -> bool {
        // 1. Decay and integration of synaptic current
        state.synaptic_current = (state.synaptic_current * config.decay_factor_syn) + input;

        // 2. Refractory period management
        if state.refractory_time_left > 0 {
            state.refractory_time_left -= 1;
            state.membrane_potential = config.reset_potential;
            return false;
        }

        // 3. Membrane integration under synaptic current and bias injection
        state.membrane_potential = (state.membrane_potential * config.decay_factor_mem)
            + state.synaptic_current
            + config.bias;

        // 4. Threshold and reset
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuba_lif_synaptic_filtering_and_spike() {
        let config = CubaLifConfig::new(
            Fix32::from_num(1.0),
            Fix32::from_num(0.9),
            Fix32::from_num(0.8),
            Fix32::ZERO,
            1,
        );
        let mut state = CubaLifState::default();

        // Brief pulse injected at tick 1
        assert!(!CubaLifDynamics::step(
            &config,
            &mut state,
            Fix32::from_num(0.8)
        ));
        assert_eq!(state.synaptic_current, Fix32::from_num(0.8));
        assert_eq!(state.membrane_potential, Fix32::from_num(0.8));

        // At tick 2, without new input, synaptic current persists (0.8 * 0.8 = 0.64)
        // and continues feeding the membrane: V = 0.8 * 0.9 + 0.64 = 0.72 + 0.64 = 1.36 >= 1.0 (Spike!)
        assert!(CubaLifDynamics::step(&config, &mut state, Fix32::ZERO));
        assert_eq!(state.membrane_potential, Fix32::ZERO);
        assert_eq!(state.refractory_time_left, 1);
    }
}
