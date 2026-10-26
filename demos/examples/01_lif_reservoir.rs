//! 🧠 Example 1: Simple LIF Neuromorphic Reservoir & ASCII Raster Plot
//!
//! Demonstrates creating a population of LIF neurons, propagating spike events,
//! and rendering a real-time spike raster plot in the terminal.
//!
//! Execution:
//! ```bash
//! cargo run -p snn_core_demos --example 01_lif_reservoir
//! ```

use snn_core::domain::network::Network;
use snn_core::domain::neuron::NeuronPopulation;
use snn_core::domain::numeric::Fix32;
use snn_core::domain::synapse::Synapse;

fn main() {
    println!("=== 🧠 snn_core: LIF Reservoir Demonstration (Spiking Neural Network) ===");

    const NUM_NEURONS: usize = 16;
    const NUM_INPUTS: usize = 4;
    const SIM_STEPS: usize = 30;

    // 1. Create a population of 16 LIF neurons
    let threshold = Fix32::from_num(1.0);
    let decay_factor = Fix32::from_num(0.85); // ~15% membrane leak per tick
    let reset_potential = Fix32::ZERO;
    let refractory_period = 2; // 2 ticks post-spike refractory dormancy

    let neurons = NeuronPopulation::new(
        NUM_NEURONS,
        threshold,
        decay_factor,
        reset_potential,
        refractory_period,
    );

    // 2. Input synapses (4 channels mapped into 16 neurons)
    let mut input_synapses = Vec::new();
    for i in 0..NUM_INPUTS {
        for n in 0..4 {
            let target = (i * 4 + n) % NUM_NEURONS;
            input_synapses.push(Synapse::new(i, target, Fix32::from_num(0.6)));
        }
    }

    // 3. Recurrent internal synapses (~25% sparsity with 80/20 Dale balance)
    let mut internal_synapses = Vec::new();
    for i in 0..NUM_NEURONS {
        for j in 0..NUM_NEURONS {
            if i != j && (i + j * 3) % 4 == 0 {
                let is_inhibitory = i % 5 == 0;
                let weight = if is_inhibitory {
                    Fix32::from_num(-0.4)
                } else {
                    Fix32::from_num(0.35)
                };
                internal_synapses.push(Synapse::new(i, j, weight));
            }
        }
    }

    let mut network = Network::new(neurons, &internal_synapses, &input_synapses);

    println!(
        "Network initialized: {} neurons, {} internal synapses.",
        NUM_NEURONS,
        network.internal_synapses.len()
    );
    println!(
        "\nSimulation across {} time-steps (Raster Plot):\n",
        SIM_STEPS
    );
    println!("Tick | Neurons 0..15 (● = Spike, · = Silent)");
    println!("-----+-----------------------------------------");

    let mut total_spikes = 0;
    for t in 0..SIM_STEPS {
        // Spike stimulus: periodic burst injection on inputs
        let mut inputs = [Fix32::ZERO; NUM_INPUTS];
        if t % 5 == 0 {
            inputs[0] = Fix32::from_num(1.2);
        }
        if t % 7 == 0 {
            inputs[1] = Fix32::from_num(1.5);
        }

        let spikes = network.tick(&inputs);

        print!("{:4} | ", t);
        for &spiked in spikes {
            if spiked {
                print!("● ");
                total_spikes += 1;
            } else {
                print!("· ");
            }
        }
        println!();
    }

    println!("-----+-----------------------------------------");
    println!("Total recorded spikes: {}", total_spikes);
}
