//! 🧬 Adaptive Leaky Integrate-and-Fire (ALIF) model in `Fix32` arithmetic.
//!
//! Threshold adaptation allows the neuron to adjust its temporal sensitivity:
//! each spike temporarily raises the effective threshold ($V_{\text{eff}} = V_{\text{base}} + A$),
//! modeling biological accommodation and robust transient detection.

use crate::domain::numeric::Fix32;
use crate::traits::NeuronDynamics;

/// Configuration parameters for an ALIF neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlifConfig {
    /// Base threshold without adaptation (V_base).
    pub base_threshold: Fix32,
    /// Membrane decay factor (decay V).
    pub decay_factor_v: Fix32,
    /// Adaptation temporal decay factor (decay A, generally slower than V).
    pub decay_factor_adapt: Fix32,
    /// Adaptation increment added to threshold after each spike (beta).
    pub adaptation_step: Fix32,
    /// Reset potential after spike emission (V_reset).
    pub reset_potential: Fix32,
    /// Absolute refractory period duration in ticks.
    pub refractory_period: u16,
    /// Bias or permanent tonic injected current.
    pub bias: Fix32,
}

impl AlifConfig {
    pub const fn new(
        base_threshold: Fix32,
        decay_factor_v: Fix32,
        decay_factor_adapt: Fix32,
        adaptation_step: Fix32,
        reset_potential: Fix32,
        refractory_period: u16,
    ) -> Self {
        Self {
            base_threshold,
            decay_factor_v,
            decay_factor_adapt,
            adaptation_step,
            reset_potential,
            refractory_period,
            bias: Fix32::ZERO,
        }
    }

    pub const fn with_bias(
        base_threshold: Fix32,
        decay_factor_v: Fix32,
        decay_factor_adapt: Fix32,
        adaptation_step: Fix32,
        reset_potential: Fix32,
        refractory_period: u16,
        bias: Fix32,
    ) -> Self {
        Self {
            base_threshold,
            decay_factor_v,
            decay_factor_adapt,
            adaptation_step,
            reset_potential,
            refractory_period,
            bias,
        }
    }
}

impl Default for AlifConfig {
    fn default() -> Self {
        Self {
            base_threshold: Fix32::ONE,
            decay_factor_v: Fix32::from_bits(58982), // ~0.90
            decay_factor_adapt: Fix32::from_bits(63570), // ~0.97 (slow decay of adaptation)
            adaptation_step: Fix32::from_bits(19661), // ~0.30
            reset_potential: Fix32::ZERO,
            refractory_period: 2,
            bias: Fix32::ZERO,
        }
    }
}

/// Dynamic state variables of an ALIF neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AlifState {
    /// Instantaneous membrane potential (V).
    pub membrane_potential: Fix32,
    /// Current value of threshold adaptation (A).
    pub adaptation: Fix32,
    /// Remaining ticks in the absolute refractory period.
    pub refractory_time_left: u16,
}

impl AlifState {
    pub const fn new(resting_potential: Fix32) -> Self {
        Self {
            membrane_potential: resting_potential,
            adaptation: Fix32::ZERO,
            refractory_time_left: 0,
        }
    }
}

/// Adaptive Leaky Integrate-and-Fire neuron dynamics.
pub struct AlifDynamics;

impl NeuronDynamics for AlifDynamics {
    type Config = AlifConfig;
    type State = AlifState;

    #[inline]
    fn step(config: &Self::Config, state: &mut Self::State, input: Fix32) -> bool {
        // Decay of adaptation variable
        state.adaptation *= config.decay_factor_adapt;

        if state.refractory_time_left > 0 {
            state.refractory_time_left -= 1;
            state.membrane_potential *= config.decay_factor_v;
            return false;
        }

        state.membrane_potential =
            state.membrane_potential * config.decay_factor_v + input + config.bias;
        let effective_threshold = config.base_threshold + state.adaptation;

        if state.membrane_potential >= effective_threshold {
            state.membrane_potential = config.reset_potential;
            state.refractory_time_left = config.refractory_period;
            state.adaptation += config.adaptation_step;
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
    fn test_alif_threshold_adaptation() {
        let config = AlifConfig::new(
            Fix32::from_num(1.0),
            Fix32::from_num(0.9),
            Fix32::from_num(0.95),
            Fix32::from_num(0.5), // adaptation step = 0.5
            Fix32::ZERO,
            1,
        );
        let mut state = AlifState::default();

        // 1. First spike with input 1.1 >= 1.0
        assert!(AlifDynamics::step(
            &config,
            &mut state,
            Fix32::from_num(1.1)
        ));
        assert_eq!(state.adaptation, Fix32::from_num(0.5));

        // 2. Refractory tick
        assert!(!AlifDynamics::step(&config, &mut state, Fix32::ZERO));

        // 3. New pulse 1.1 -> effective threshold is 1.0 + adaptation (~0.5*0.95 = 0.475) = ~1.475
        // So 1.1 does NOT trigger a spike this time!
        assert!(!AlifDynamics::step(
            &config,
            &mut state,
            Fix32::from_num(1.1)
        ));
    }
}
