//! ⚡ Example 3: Izhikevich Model Firing Regimes
//!
//! Demonstrates Izhikevich neuronal regimes (Regular Spiking)
//! with real-time ASCII membrane voltmeter rendering.
//!
//! Execution:
//! ```bash
//! cargo run --example 03_izhikevich_regimes -p snn_core
//! ```

use snn_core::Fix32;
use snn_core::models::{IzhikevichConfig, IzhikevichDynamics, IzhikevichState};
use snn_core::traits::NeuronDynamics;

fn main() {
    println!("=== ⚡ snn_core: Izhikevich Model Demonstration (Regular Spiking) ===");
    println!("Step current injection I = 10.0 Fix32.\n");

    let config = IzhikevichConfig::regular_spiking();
    let mut state = IzhikevichState::default();

    println!("Tick |     v [mV] |     u      | Membrane Voltmeter [-70 mV to +30 mV]");
    println!("-----+------------+------------+------------------------------------------");

    for t in 0..40 {
        let input = if t >= 5 {
            Fix32::from_num(10.0)
        } else {
            Fix32::ZERO
        };

        let spiked = IzhikevichDynamics::step(&config, &mut state, input);

        let v_val = state.v.to_float();
        let u_val = state.u.to_float();

        // Normalization for ASCII rendering between -70 and +30 mV (100 mV span over 35 columns)
        let normalized = ((v_val + 70.0) / 100.0).clamp(0.0, 1.0);
        let bar_len = (normalized * 35.0) as usize;
        let mut bar = String::from("|");
        for _ in 0..bar_len {
            bar.push('█');
        }
        if spiked {
            bar.push_str(" ⚡ SPIKE!");
        }

        println!("{:4} | {:8.2}   | {:8.2}   | {}", t, v_val, u_val, bar);
    }
    println!("-----+------------+------------+------------------------------------------");
}
