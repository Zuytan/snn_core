//! 🧬 Example 2: Spike Frequency Adaptation (ALIF vs LIF)
//!
//! Compares the behavior of a classical LIF neuron with an ALIF (Adaptive LIF) neuron
//! subjected to a constant continuous current injection.
//!
//! Execution:
//! ```bash
//! cargo run -p snn_core_demos --example 02_alif_adaptation
//! ```

use snn_core::Fix32;
use snn_core::models::{AlifConfig, AlifDynamics, AlifState, LifConfig, LifDynamics, LifState};
use snn_core::traits::NeuronDynamics;

fn main() {
    println!("=== 🧬 snn_core: Temporal Adaptation Demonstration (LIF vs ALIF) ===");
    println!("Simulation under constant current injection I = 0.5 over 40 ticks.\n");

    let lif_cfg = LifConfig::new(Fix32::from_num(1.0), Fix32::from_num(0.90), Fix32::ZERO, 1);
    let mut lif_state = LifState::default();

    let alif_cfg = AlifConfig::new(
        Fix32::from_num(1.0),  // base threshold
        Fix32::from_num(0.90), // membrane decay V
        Fix32::from_num(0.96), // slow adaptation decay A
        Fix32::from_num(0.40), // threshold increment per spike (beta)
        Fix32::ZERO,
        1,
    );
    let mut alif_state = AlifState::default();

    let input_current = Fix32::from_num(0.5);

    println!("Tick | LIF (V_mem, Spike)       | ALIF (V_mem, Eff. Thresh, Spike)");
    println!("-----+--------------------------+--------------------------------");

    let mut lif_spikes = 0;
    let mut alif_spikes = 0;

    for t in 0..40 {
        let lif_spiked = LifDynamics::step(&lif_cfg, &mut lif_state, input_current);
        let alif_spiked = AlifDynamics::step(&alif_cfg, &mut alif_state, input_current);

        if lif_spiked {
            lif_spikes += 1;
        }
        if alif_spiked {
            alif_spikes += 1;
        }

        let eff_thresh = alif_cfg.base_threshold + alif_state.adaptation;

        println!(
            "{:4} | V: {:5.2} Spike: {}        | V: {:5.2} Thr: {:5.2} Spike: {}",
            t,
            lif_state.membrane_potential.to_float(),
            if lif_spiked { "●" } else { "·" },
            alif_state.membrane_potential.to_float(),
            eff_thresh.to_float(),
            if alif_spiked { "●" } else { "·" },
        );
    }

    println!("-----+--------------------------+--------------------------------");
    println!(
        "Total LIF Spikes :  {} (maximal steady firing without fatigue)",
        lif_spikes
    );
    println!(
        "Total ALIF Spikes:  {} (biological accommodation via dynamic threshold elevation)",
        alif_spikes
    );
}
