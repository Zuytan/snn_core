//! 🧪 Core Trait Integration Tests for `snn_core`
#![allow(deprecated)]

use snn_core::models::{
    AlifConfig, AlifDynamics, AlifState, IzhikevichConfig, IzhikevichDynamics, IzhikevichState,
    LifConfig, LifDynamics, LifState,
};
use snn_core::traits::{Readout, SpikingNetwork, SpikingPopulation, SynapseMatrix};
use snn_core::{
    Fix32, GenericStaticNeuronPopulation, StaticCsrSynapses, StaticNetwork, StaticNeuronPopulation,
    StaticReadoutLayer, Synapse,
};

#[test]
#[allow(deprecated)]
fn test_polymorphic_spiking_population() {
    let lif_cfg = LifConfig::new(Fix32::from_num(1.0), Fix32::from_num(0.9), Fix32::ZERO, 2);
    let mut generic_lif =
        GenericStaticNeuronPopulation::<LifDynamics, 2>::new(lif_cfg, LifState::default());

    let mut static_soa = StaticNeuronPopulation::<2>::new(
        Fix32::from_num(1.0),
        Fix32::from_num(0.9),
        Fix32::ZERO,
        2,
    );

    // Function accepting any population via the SpikingPopulation trait
    fn exercise_population(pop: &mut dyn SpikingPopulation) {
        assert_eq!(pop.len(), 2);
        assert!(!pop.is_empty());

        let mut spikes = [false; 2];
        let mut active_buf = [0u32; 2];
        let mut active_count = 0;

        // No spike without input
        pop.tick_active(&mut spikes, &mut active_buf, &mut active_count);
        assert_eq!(active_count, 0);

        // Current injection triggering a spike on neuron 0
        pop.receive_input(0, Fix32::from_num(1.5));
        pop.tick_active(&mut spikes, &mut active_buf, &mut active_count);
        assert_eq!(active_count, 1);
        assert_eq!(active_buf[0], 0);
        assert!(spikes[0]);
        assert!(!spikes[1]);

        // Reset
        pop.reset();
    }

    let mut compact_lif = snn_core::CompactLifPopulation::<2>::new(lif_cfg);

    exercise_population(&mut generic_lif);
    exercise_population(&mut static_soa);
    exercise_population(&mut compact_lif);
}

#[test]
fn test_alif_and_izhikevich_generic_populations() {
    // 1. Test ALIF with adaptive threshold
    let alif_cfg = AlifConfig::new(
        Fix32::from_num(1.0),
        Fix32::from_num(0.9),
        Fix32::from_num(0.95),
        Fix32::from_num(0.5),
        Fix32::ZERO,
        1,
    );
    let mut alif_pop =
        GenericStaticNeuronPopulation::<AlifDynamics, 1>::new(alif_cfg, AlifState::default());

    let mut spikes = [false; 1];
    let mut active_buf = [0u32; 1];
    let mut active_count = 0;

    // First firing
    alif_pop.receive_input(0, Fix32::from_num(1.2));
    alif_pop.tick_active(&mut spikes, &mut active_buf, &mut active_count);
    assert_eq!(active_count, 1);
    assert!(spikes[0]);

    // Next tick (refractory state + elevated adaptive threshold)
    alif_pop.receive_input(0, Fix32::from_num(1.2));
    alif_pop.tick_active(&mut spikes, &mut active_buf, &mut active_count);
    assert_eq!(active_count, 0);
    assert!(!spikes[0]);

    // 2. Test Izhikevich Regular Spiking
    let izh_cfg = IzhikevichConfig::regular_spiking();
    let mut izh_pop = GenericStaticNeuronPopulation::<IzhikevichDynamics, 1>::new(
        izh_cfg,
        IzhikevichState::default(),
    );

    // Continuous current injection
    let mut total_spikes = 0;
    for _ in 0..100 {
        izh_pop.receive_input(0, Fix32::from_num(12.0));
        izh_pop.tick_active(&mut spikes, &mut active_buf, &mut active_count);
        if spikes[0] {
            total_spikes += 1;
        }
    }
    assert!(
        total_spikes > 0,
        "Izhikevich neuron must spike under continuous current"
    );
}

#[test]
fn test_synapse_matrix_trait() {
    let synapses = [
        Synapse::new(0, 1, Fix32::from_num(0.5)),
        Synapse::new(0, 2, Fix32::from_num(1.0)),
    ];
    let csr: StaticCsrSynapses<2, 10> = StaticCsrSynapses::from_synapses(&synapses);

    fn check_synapse_matrix(matrix: &dyn SynapseMatrix) {
        assert_eq!(matrix.len(), 2);
        assert!(!matrix.is_empty());

        let mut acc = [Fix32::ZERO; 3];
        matrix.propagate_active_spikes(&[0], &mut acc);
        assert_eq!(acc[0], Fix32::ZERO);
        assert_eq!(acc[1], Fix32::from_num(0.5));
        assert_eq!(acc[2], Fix32::from_num(1.0));
    }

    check_synapse_matrix(&csr);
}

#[test]
fn test_readout_trait() {
    static WEIGHTS: [[Fix32; 3]; 2] = [
        [
            Fix32::from_bits(6554),
            Fix32::from_bits(52429),
            Fix32::from_bits(6554),
        ],
        [
            Fix32::from_bits(6554),
            Fix32::from_bits(6554),
            Fix32::from_bits(58982),
        ],
    ];
    let static_readout = StaticReadoutLayer::<2, 3>::new(&WEIGHTS);

    fn evaluate_readout(readout: &dyn Readout, state: &[Fix32]) -> usize {
        readout.predict(state)
    }

    // If x[0] dominates, class 0 wins
    let state_0 = [Fix32::from_num(2.0), Fix32::ZERO];
    assert_eq!(evaluate_readout(&static_readout, &state_0), 0);

    // If x[1] dominates, class 1 wins
    let state_1 = [Fix32::ZERO, Fix32::from_num(2.0)];
    assert_eq!(evaluate_readout(&static_readout, &state_1), 1);
}

#[test]
#[allow(deprecated)]
fn test_spiking_network_trait() {
    let neurons =
        StaticNeuronPopulation::<2>::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
    let mut net = StaticNetwork::<2, 3, 5, 5, 2, 3>::new(neurons, &[], &[]);

    fn tick_any_network(net: &mut dyn SpikingNetwork, input: &[Fix32]) {
        let spikes = net.tick(input);
        assert_eq!(spikes.len(), 2);
        net.reset();
    }

    tick_any_network(&mut net, &[Fix32::from_num(0.5), Fix32::ZERO]);

    static EMPTY_ROW: [u16; 3] = [0, 0, 0];
    static EMPTY_WEIGHTS: [Fix32; 0] = [];
    static EMPTY_TARGETS: [u16; 0] = [];
    let in_syn = snn_core::RomCsrSynapses::new(&EMPTY_ROW[..2], &EMPTY_TARGETS, &EMPTY_WEIGHTS);
    let int_syn = snn_core::RomCsrSynapses::new(&EMPTY_ROW, &EMPTY_TARGETS, &EMPTY_WEIGHTS);
    let compact_pop = snn_core::CompactLifPopulation::<2>::new(snn_core::LifConfig::new(
        Fix32::ONE,
        Fix32::from_num(0.9),
        Fix32::ZERO,
        1,
    ));
    let mut generic_net: snn_core::GenericStaticNetwork<_, _, _, 2, 2> =
        snn_core::GenericStaticNetwork::new(compact_pop, int_syn, in_syn);
    tick_any_network(&mut generic_net, &[Fix32::from_num(0.5), Fix32::ZERO]);
}

#[test]
fn test_csr_memory_safety_out_of_bounds_targets() {
    // Synapse pointing to an out-of-bounds destination index
    let bad_synapses = [
        Synapse::new(0, 99_999, Fix32::ONE),
        Synapse::new(0, 1, Fix32::from_num(0.5)),
    ];
    let csr = StaticCsrSynapses::<2, 5>::from_synapses(&bad_synapses);
    let mut acc = [Fix32::ZERO; 2];

    // Must not panic or write out of bounds
    csr.propagate_active_spikes(&[0], &mut acc);
    assert_eq!(acc[0], Fix32::ZERO);
    assert_eq!(acc[1], Fix32::from_num(0.5));

    let inputs = [Fix32::from_num(2.0)];
    csr.propagate_weighted_inputs(&inputs, &mut acc);
    assert_eq!(acc[1], Fix32::from_num(1.5)); // 0.5 + 2.0 * 0.5
}

#[cfg(feature = "alloc")]
#[test]
fn test_dynamic_csr_memory_safety_out_of_bounds_targets() {
    use snn_core::domain::csr::CsrSynapses;

    let bad_synapses = vec![
        Synapse::new(0, 99_999, Fix32::ONE),
        Synapse::new(0, 1, Fix32::from_num(0.5)),
    ];
    let csr = CsrSynapses::from_synapses(1, &bad_synapses);
    let mut acc = vec![Fix32::ZERO; 2];

    csr.propagate_active_spikes(&[0], &mut acc);
    assert_eq!(acc[0], Fix32::ZERO);
    assert_eq!(acc[1], Fix32::from_num(0.5));

    let inputs = [Fix32::from_num(2.0)];
    csr.propagate_weighted_inputs(&inputs, &mut acc);
    assert_eq!(acc[1], Fix32::from_num(1.5));
}

#[test]
#[allow(deprecated)]
fn test_population_short_buffer_safety() {
    let mut pop =
        StaticNeuronPopulation::<3>::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
    pop.receive_input(0, Fix32::from_num(2.0));

    // Buffer shorter than the population capacity
    let mut short_spikes = [false; 1];
    let mut short_active = [0u32; 1];
    let mut active_count = 0;

    // Must not exceed short_spikes.len()
    pop.tick_active(&mut short_spikes, &mut short_active, &mut active_count);
    assert!(short_spikes[0]);
    assert_eq!(active_count, 1);
}

#[test]
fn test_generic_static_network_with_alif_and_rom_csr() {
    use snn_core::GenericStaticNetwork;
    use snn_core::RomCsrSynapses;

    static IN_ROW_PTRS: [u16; 2] = [0, 1];
    static IN_TARGETS: [u16; 1] = [0];
    static IN_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(98304)]; // 1.5

    static INT_ROW_PTRS: [u16; 3] = [0, 1, 1];
    static INT_TARGETS: [u16; 1] = [1];
    static INT_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(98304)]; // 1.5 (> threshold 1.0)

    let input_synapses = RomCsrSynapses::new(&IN_ROW_PTRS, &IN_TARGETS, &IN_WEIGHTS);
    let internal_synapses = RomCsrSynapses::new(&INT_ROW_PTRS, &INT_TARGETS, &INT_WEIGHTS);

    let alif_cfg = AlifConfig::new(
        Fix32::ONE,
        Fix32::from_num(0.9),
        Fix32::from_num(0.95),
        Fix32::from_num(0.5),
        Fix32::ZERO,
        1,
    );
    let neurons =
        GenericStaticNeuronPopulation::<AlifDynamics, 2>::new(alif_cfg, AlifState::default());

    let mut net =
        GenericStaticNetwork::<_, _, _, 2, 1>::new(neurons, internal_synapses, input_synapses);

    // Tick 1: Active input to neuron 0
    let spikes = net.tick(&[Fix32::ONE]);
    assert!(spikes[0], "Neuron 0 (ALIF) must emit a spike");
    assert!(!spikes[1]);

    // Tick 2: Recurrent internal synapse triggers neuron 1
    let spikes = net.tick(&[Fix32::ZERO]);
    assert!(!spikes[0], "Neuron 0 must be refractory / elevated threshold");
    assert!(
        spikes[1],
        "Neuron 1 must have received recurrent synaptic pulse"
    );
}

#[test]
fn test_native_fix16_network_and_deep_pipeline() {
    use snn_core::application::readout::StaticReadoutLayer;
    use snn_core::domain::generic_static_neuron::GenericStaticNeuronPopulation;
    use snn_core::domain::layers::{LayerChain, SpikingFeedforwardLayer, SpikingLayer};
    use snn_core::domain::models::lif::{Lif16Config, Lif16Dynamics, Lif16State};
    use snn_core::domain::numeric::Fix16;
    use snn_core::domain::rom_csr::RomCsrSynapses;

    // 16-bit LIF config
    let cfg16 = Lif16Config::new(Fix16::ONE, Fix16::from_num(0.8), Fix16::ZERO, 1);

    // Layer 1: 2 Fix16 neurons
    let l1_pop =
        GenericStaticNeuronPopulation::<Lif16Dynamics, 2, Fix16>::new(cfg16, Lif16State::default());
    static L1_ROW_PTRS: [u16; 2] = [0, 1];
    static L1_TARGETS: [u16; 1] = [0];
    static L1_WEIGHTS: [Fix16; 1] = [Fix16::from_bits(384)]; // 1.5 in Q8.8 (256 * 1.5)
    let l1_syn = RomCsrSynapses::new(&L1_ROW_PTRS, &L1_TARGETS, &L1_WEIGHTS);
    let layer1 = SpikingFeedforwardLayer::<_, _, 2, 1, Fix16>::new(l1_pop, l1_syn);

    // Layer 2: 2 Fix16 neurons
    let l2_pop =
        GenericStaticNeuronPopulation::<Lif16Dynamics, 2, Fix16>::new(cfg16, Lif16State::default());
    static L2_ROW_PTRS: [u16; 3] = [0, 1, 1];
    static L2_TARGETS: [u16; 1] = [1];
    static L2_WEIGHTS: [Fix16; 1] = [Fix16::from_bits(384)]; // 1.5
    let l2_syn = RomCsrSynapses::new(&L2_ROW_PTRS, &L2_TARGETS, &L2_WEIGHTS);
    let layer2 = SpikingFeedforwardLayer::<_, _, 2, 1, Fix16>::new(l2_pop, l2_syn);

    // Composition via intra-tick LayerChain
    let mut chain = LayerChain::new(layer1, layer2);

    // At tick 0: Fix16::ONE input -> Layer 1 emits spike (0) -> Layer 2 emits spike (1)
    let inputs = [Fix16::ONE];
    let out_words = chain.tick_inputs_words(&inputs);
    assert_ne!(
        out_words[0] & (1 << 1),
        0,
        "Neuron 1 of layer 2 must emit a spike in Fix16"
    );
    assert!(chain.get_spike(1));

    // Test static Readout in Fix16
    static READOUT_WEIGHTS: [[Fix16; 3]; 2] = [
        [Fix16::ZERO, Fix16::ZERO, Fix16::ONE], // Class 0: high weight on neuron 1
        [Fix16::ZERO, Fix16::ONE, Fix16::ZERO], // Class 1: high weight on neuron 0
    ];
    let readout = StaticReadoutLayer::<2, 3, Fix16>::new(&READOUT_WEIGHTS);
    let class = readout.predict(&[Fix16::ZERO, Fix16::ONE]);
    assert_eq!(class, 0);
}
