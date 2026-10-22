//! 🧠 Izhikevich model in `Fix32` arithmetic.
//!
//! Bio-inspired model using two coupled differential equations, capable of reproducing
//! a rich variety of neuronal firing patterns (tonic spiking, bursting, chattering, spike frequency adaptation):
//!
//! $$v' = 0.04 v^2 + 5 v + 140 - u + I$$
//! $$u' = a (b v - u)$$
//!
//! If $v \ge v_{\text{peak}}$:
//! $$v \leftarrow c$$
//! $$u \leftarrow u + d$$

use crate::domain::numeric::Fix32;
use crate::traits::NeuronDynamics;

/// Dimensionless parameters of the Izhikevich model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IzhikevichConfig {
    /// Time scale of the recovery variable u (a).
    pub a: Fix32,
    /// Sensitivity of u to subthreshold fluctuations of v (b).
    pub b: Fix32,
    /// Reset potential after spike emission (c, typically -65 mV).
    pub c: Fix32,
    /// Post-spike recovery increment (d).
    pub d: Fix32,
    /// Peak potential threshold (v_peak, typically +30 mV).
    pub v_peak: Fix32,
}

impl IzhikevichConfig {
    /// Configuration for "Regular Spiking" regime (RS - standard excitatory cortical neurons).
    pub fn regular_spiking() -> Self {
        Self {
            a: Fix32::from_num(0.02),
            b: Fix32::from_num(0.2),
            c: Fix32::from_num(-65.0),
            d: Fix32::from_num(8.0),
            v_peak: Fix32::from_num(30.0),
        }
    }

    /// Configuration for "Fast Spiking" regime (FS - inhibitory cortical interneurons).
    pub fn fast_spiking() -> Self {
        Self {
            a: Fix32::from_num(0.1),
            b: Fix32::from_num(0.2),
            c: Fix32::from_num(-65.0),
            d: Fix32::from_num(2.0),
            v_peak: Fix32::from_num(30.0),
        }
    }
}

impl Default for IzhikevichConfig {
    fn default() -> Self {
        Self::regular_spiking()
    }
}

/// State variables of the Izhikevich model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IzhikevichState {
    /// Instantaneous membrane potential (v).
    pub v: Fix32,
    /// Membrane recovery variable (u).
    pub u: Fix32,
}

impl Default for IzhikevichState {
    fn default() -> Self {
        Self {
            v: Fix32::from_num(-65.0),
            u: Fix32::from_num(-13.0), // u = b * v = 0.2 * -65 = -13
        }
    }
}

/// Izhikevich neuron dynamics in fixed-point arithmetic.
pub struct IzhikevichDynamics;

impl NeuronDynamics for IzhikevichDynamics {
    type Config = IzhikevichConfig;
    type State = IzhikevichState;

    #[inline]
    fn step(config: &Self::Config, state: &mut Self::State, input: Fix32) -> bool {
        let half = Fix32::from_bits(32768); // 0.5
        let factor_lin = Fix32::from_int(5);
        let constant_term = Fix32::from_int(140);

        // Euler half-step 1 on v (dt = 0.5 ms)
        // 0.04 * v^2 = v^2 / 25 computed in 64-bit to prevent overflow and truncation
        let v_raw = i64::from(state.v.0);
        let v_sq_div_25 = Fix32::from_bits(((v_raw * v_raw) / (25 * 65536)) as i32);
        let dv1 = half * (v_sq_div_25 + (factor_lin * state.v) + constant_term - state.u + input);
        state.v += dv1;

        // Euler half-step 2 on v (dt = 0.5 ms)
        let v_raw = i64::from(state.v.0);
        let v_sq_div_25 = Fix32::from_bits(((v_raw * v_raw) / (25 * 65536)) as i32);
        let dv2 = half * (v_sq_div_25 + (factor_lin * state.v) + constant_term - state.u + input);
        state.v += dv2;

        // Euler step on u (dt = 1.0 ms)
        let du = config.a * ((config.b * state.v) - state.u);
        state.u += du;

        if state.v >= config.v_peak {
            state.v = config.c;
            state.u += config.d;
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
    fn test_izhikevich_regular_spiking() {
        let config = IzhikevichConfig::regular_spiking();
        let mut state = IzhikevichState::default();

        // Without current, neuron remains at rest ~ -65 mV
        let spiked = IzhikevichDynamics::step(&config, &mut state, Fix32::ZERO);
        assert!(!spiked);
        assert!(state.v < Fix32::from_num(-60.0));

        // With strong injected current (I = 15.0), the neuron fires spikes
        let mut spikes = 0;
        for _ in 0..50 {
            if IzhikevichDynamics::step(&config, &mut state, Fix32::from_num(15.0)) {
                spikes += 1;
            }
        }
        assert!(
            spikes > 0,
            "Izhikevich model should spike under constant current"
        );
    }
}
