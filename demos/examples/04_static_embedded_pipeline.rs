//! 🌲 Example 4: 100% Static Zero-Allocation Inference Pipeline (`no_std`)
//!
//! Demonstrates using `StaticNetwork` and `StaticReadoutLayer`
//! for ultra-lightweight execution without dynamic heap allocation (bare-metal Cortex-M ready).
//!
//! Execution:
//! ```bash
//! cargo run -p snn_core_demos --example 04_static_embedded_pipeline
//! ```

use snn_core::traits::{Readout, SpikingNetwork};
use snn_core::{
    CompactLifPopulation, Fix32, GenericStaticNetwork, LifConfig, ResetMode, StaticCsrSynapses,
    StaticReadoutLayer, Synapse,
};

// Static architecture:
// 8 reservoir neurons, 2 input channels, 2 output classes (0: Normal, 1: Alert)
const N_NEURONS: usize = 8;
const N_NEURONS_P1: usize = N_NEURONS + 1; // 9
const MAX_INTERNAL: usize = 32;
const MAX_INPUT: usize = 16;
const N_INPUTS: usize = 2;
const N_INPUTS_P1: usize = N_INPUTS + 1; // 3

// Static Flash ROM readout weights: [[Bias, W0, W1, ... W7]; NumClasses]
static ROM_READOUT_WEIGHTS: [[Fix32; 9]; 2] = [
    // Class 0 (Normal): strong correlation with neurons 0..3
    [
        Fix32::from_bits(6554),
        Fix32::from_bits(65536),
        Fix32::from_bits(65536),
        Fix32::from_bits(65536),
        Fix32::from_bits(65536),
        Fix32::ZERO,
        Fix32::ZERO,
        Fix32::ZERO,
        Fix32::ZERO,
    ],
    // Class 1 (Alert): strong correlation with neurons 4..7
    [
        Fix32::from_bits(6554),
        Fix32::ZERO,
        Fix32::ZERO,
        Fix32::ZERO,
        Fix32::ZERO,
        Fix32::from_bits(98304),
        Fix32::from_bits(98304),
        Fix32::from_bits(98304),
        Fix32::from_bits(98304),
    ],
];

fn main() {
    println!("=== 🌲 snn_core: Zero-Heap Static Inference Pipeline (`no_std`) ===");

    // 1. Define neuron population in pure Fix32 arithmetic (Flash ROM)
    let config = LifConfig::new(
        Fix32::ONE,
        Fix32::from_bits(58982), // decay = 0.90
        Fix32::ZERO,
        1,
    )
    .with_reset_mode(ResetMode::Subtract);
    let neurons = CompactLifPopulation::<N_NEURONS>::new(config);

    // 2. Input synapses (Input 0 drives neurons 0..3, Input 1 drives neurons 4..7)
    let input_synapses = [
        Synapse::new(0, 0, Fix32::from_bits(78643)), // 1.2
        Synapse::new(0, 1, Fix32::from_bits(78643)),
        Synapse::new(1, 4, Fix32::from_bits(98304)), // 1.5
        Synapse::new(1, 5, Fix32::from_bits(98304)),
    ];
    let input_csr = StaticCsrSynapses::<N_INPUTS_P1, MAX_INPUT>::from_synapses(&input_synapses);

    // 3. Recurrent internal synapses
    let internal_synapses = [
        Synapse::new(0, 2, Fix32::from_bits(32768)),
        Synapse::new(4, 6, Fix32::from_bits(45875)),
    ];
    let internal_csr =
        StaticCsrSynapses::<N_NEURONS_P1, MAX_INTERNAL>::from_synapses(&internal_synapses);

    // 4. Instantiate generic static network
    let mut network =
        GenericStaticNetwork::<_, _, _, N_NEURONS, N_INPUTS>::new(neurons, internal_csr, input_csr);

    // 5. Static Flash ROM Readout layer
    let readout = StaticReadoutLayer::<2, 9>::new(&ROM_READOUT_WEIGHTS);

    println!(
        "Network size allocated on stack: {} bytes",
        core::mem::size_of_val(&network)
    );
    println!(
        "Flash ROM Readout layer size   : {} bytes\n",
        core::mem::size_of_val(&readout)
    );

    // Scenario A: Normal stimulus on channel 0
    let normal_input = [Fix32::from_bits(65536), Fix32::ZERO];
    let mut state = [Fix32::ZERO; 8];
    for _ in 0..5 {
        let spikes = network.tick(&normal_input);
        for i in 0..8 {
            state[i] = (state[i] * Fix32::from_bits(52429)) // decay 0.8
                + if spikes[i] { Fix32::ONE } else { Fix32::ZERO };
        }
    }
    let class_pred_a = Readout::predict(&readout, &state);
    println!(
        "Scenario A (Normal signal channel 0) -> Predicted Class: {} (0 = Normal, 1 = Alert)",
        class_pred_a
    );
    assert_eq!(class_pred_a, 0);

    // Reset via SpikingNetwork trait
    SpikingNetwork::reset(&mut network);
    state = [Fix32::ZERO; 8];

    // Scenario B: Alert stimulus on channel 1
    let alert_input = [Fix32::ZERO, Fix32::from_bits(131072)]; // 2.0
    for _ in 0..5 {
        let spikes = network.tick(&alert_input);
        for i in 0..8 {
            state[i] = (state[i] * Fix32::from_bits(52429)) // decay 0.8
                + if spikes[i] { Fix32::ONE } else { Fix32::ZERO };
        }
    }
    let class_pred_b = Readout::predict(&readout, &state);
    println!(
        "Scenario B (Alert signal channel 1)  -> Predicted Class: {} (0 = Normal, 1 = Alert)",
        class_pred_b
    );
    assert_eq!(class_pred_b, 1);

    println!("\n✅ Static pipeline executed successfully with ZERO dynamic heap allocations!");
}
