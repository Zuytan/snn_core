use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use snn_core::domain::compact_population::CompactLifPopulation;
use snn_core::domain::csr::CsrSynapses;
use snn_core::domain::engine::{LayerDescriptor, ModelDescriptor};
use snn_core::domain::network::Network;
use snn_core::domain::neuron::NeuronPopulation;
use snn_core::domain::numeric::Fix32;
use snn_core::domain::rom_csr::{QuantizedRomCsrSynapses, RomCsrSynapses};
use snn_core::domain::static_csr::StaticCsrSynapses;
use snn_core::domain::static_network::{GenericStaticNetwork, StaticNetwork};
#[allow(deprecated)]
use snn_core::domain::static_neuron::StaticNeuronPopulation;
use snn_core::domain::synapse::Synapse;
use snn_core::models::{
    AlifConfig, AlifDynamics, AlifState, IzhikevichConfig, IzhikevichDynamics, IzhikevichState,
    LifConfig, LifDynamics, LifState,
};
use snn_core::traits::NeuronDynamics;

/// Benchmark 1: Spatial Stress Test (No synapses, massive vectorization)
fn bench_spatial_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spatial Stress Test (Fix32)");
    for &size in &[10_000, 50_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            let neurons =
                NeuronPopulation::new(n, Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 2);
            let mut network = Network::new(neurons, &[], &[]);

            let mut inputs = Vec::with_capacity(n);
            let mut rng = fastrand::Rng::with_seed(42);
            for _ in 0..n {
                inputs.push(if rng.f32() < 0.1 {
                    Fix32::from_num(2.0)
                } else {
                    Fix32::ZERO
                });
            }

            b.iter(|| {
                black_box(network.tick(black_box(&inputs)));
            });
        });
    }
    group.finish();
}

/// Benchmark 2: Fully Connected Avalanche (Cache Locality)
fn bench_avalanche(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fully Connected Avalanche");
    let size = 1_000;

    group.bench_function("1K neurons (1M synapses)", |b| {
        let neurons = NeuronPopulation::new(size, Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 2);

        let mut internal_synapses = Vec::with_capacity(size * size);
        for i in 0..size {
            for j in 0..size {
                internal_synapses.push(Synapse::new(i, j, Fix32::from_num(0.5)));
            }
        }

        let mut network = Network::new(neurons, &internal_synapses, &[]);
        let inputs = vec![Fix32::from_num(5.0); size];

        b.iter(|| {
            black_box(network.tick(black_box(&inputs)));
        });
    });
    group.finish();
}

/// Benchmark 3: Liquid State Machine (Balanced Network 500 neurons)
fn bench_liquid_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("Liquid State Machine");
    let size = 500;
    let connectivity = 0.1;

    group.bench_function("500 neurons (sparse, 80E/20I)", |b| {
        let neurons = NeuronPopulation::new(size, Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 2);

        let mut internal_synapses = Vec::new();
        let mut rng = fastrand::Rng::with_seed(1234);

        for i in 0..size {
            for j in 0..size {
                if rng.f32() < connectivity {
                    let weight = if rng.f32() < 0.8 {
                        Fix32::from_num(0.5)
                    } else {
                        Fix32::from_num(-0.5)
                    };
                    internal_synapses.push(Synapse::new(i, j, weight));
                }
            }
        }

        let mut network = Network::new(neurons, &internal_synapses, &[]);
        let mut inputs = vec![Fix32::ZERO; size];
        inputs[..size / 10].fill(Fix32::from_num(2.0));

        b.iter(|| {
            black_box(network.tick(black_box(&inputs)));
        });
    });
    group.finish();
}

/// Benchmark 4: Bare-Metal Static Reservoir (100 neurons, 2,500 synapses, embedded profile)
#[allow(deprecated)]
fn bench_static_reservoir(c: &mut Criterion) {
    let mut group = c.benchmark_group("Embedded Static Reservoir (no_std)");

    const N: usize = 100;
    const N_P1: usize = 101;
    const MAX_INTERNAL: usize = 2500;
    const MAX_INPUT: usize = 600;
    const N_INPUTS: usize = 8;
    const N_INPUTS_P1: usize = 9;

    let neurons =
        StaticNeuronPopulation::<N>::new(Fix32::ONE, Fix32::from_bits(58982), Fix32::ZERO, 2);

    let compact_neurons = CompactLifPopulation::<N>::new(LifConfig::new(
        Fix32::ONE,
        Fix32::from_bits(58982),
        Fix32::ZERO,
        2,
    ));

    let mut internal_synapses = Vec::with_capacity(MAX_INTERNAL);
    let mut rng = fastrand::Rng::with_seed(777);
    for i in 0..N {
        for j in 0..N {
            if i != j && rng.f32() < 0.25 && internal_synapses.len() < MAX_INTERNAL {
                let weight = if rng.f32() < 0.8 {
                    Fix32::from_num(0.4)
                } else {
                    Fix32::from_num(-0.6)
                };
                internal_synapses.push(Synapse::new(i, j, weight));
            }
        }
    }

    let mut input_synapses = Vec::with_capacity(MAX_INPUT);
    for in_ch in 0..N_INPUTS {
        for target in 0..N {
            if rng.f32() < 0.3 && input_synapses.len() < MAX_INPUT {
                input_synapses.push(Synapse::new(in_ch, target, Fix32::from_num(1.2)));
            }
        }
    }

    let internal_csr = StaticCsrSynapses::<N_P1, MAX_INTERNAL>::from_synapses(&internal_synapses);
    let input_csr = StaticCsrSynapses::<N_INPUTS_P1, MAX_INPUT>::from_synapses(&input_synapses);

    let mut static_net =
        StaticNetwork::<N, N_P1, MAX_INTERNAL, MAX_INPUT, N_INPUTS, N_INPUTS_P1>::new_csr(
            neurons,
            internal_csr,
            input_csr,
        );

    let mut compact_net: GenericStaticNetwork<_, _, _, N, N_INPUTS> =
        GenericStaticNetwork::new(compact_neurons, internal_csr, input_csr);

    let reservoir_neurons = CompactLifPopulation::<N>::new(LifConfig::new(
        Fix32::ONE,
        Fix32::from_bits(58982),
        Fix32::ZERO,
        2,
    ));
    let mut static_reservoir = snn_core::domain::static_network::StaticReservoir::<
        _,
        _,
        _,
        { N.div_ceil(32) },
    >::new(reservoir_neurons, internal_csr, input_csr);

    let inputs = [
        Fix32::from_num(1.5),
        Fix32::ZERO,
        Fix32::from_num(0.8),
        Fix32::ZERO,
        Fix32::from_num(2.0),
        Fix32::ZERO,
        Fix32::ZERO,
        Fix32::from_num(1.0),
    ];

    group.bench_function("Legacy StaticNetwork 100N (2.5K synapses CSR)", |b| {
        b.iter(|| {
            black_box(static_net.tick(black_box(&inputs)));
        });
    });

    group.bench_function(
        "GenericStaticNetwork + CompactLIF 100N (2.5K synapses CSR)",
        |b| {
            b.iter(|| {
                black_box(compact_net.tick(black_box(&inputs)));
            });
        },
    );

    group.bench_function(
        "StaticReservoir (SpikeBitset CTZ Vectorized) 100N (2.5K synapses CSR)",
        |b| {
            b.iter(|| {
                black_box(static_reservoir.tick(black_box(&inputs)));
            });
        },
    );

    group.finish();
}

/// Benchmark 5: Comparative micro-benchmark of neuron dynamics (LIF vs ALIF vs Izhikevich)
fn bench_neuron_dynamics_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("Neuron Dynamics (1000 Steps)");

    let lif_cfg = LifConfig::default();
    let alif_cfg = AlifConfig::default();
    let izh_cfg = IzhikevichConfig::regular_spiking();
    let input = Fix32::from_num(1.2);

    group.bench_function("LifDynamics::step", |b| {
        let mut state = LifState::default();
        b.iter(|| {
            for _ in 0..1000 {
                black_box(LifDynamics::step(&lif_cfg, &mut state, black_box(input)));
            }
        });
    });

    group.bench_function("AlifDynamics::step", |b| {
        let mut state = AlifState::default();
        b.iter(|| {
            for _ in 0..1000 {
                black_box(AlifDynamics::step(&alif_cfg, &mut state, black_box(input)));
            }
        });
    });

    group.bench_function("IzhikevichDynamics::step", |b| {
        let mut state = IzhikevichState::default();
        b.iter(|| {
            for _ in 0..1000 {
                black_box(IzhikevichDynamics::step(
                    &izh_cfg,
                    &mut state,
                    black_box(input),
                ));
            }
        });
    });

    group.finish();
}

/// Benchmark 6: Synaptic propagation Static vs Dynamic CSR
fn bench_csr_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("CSR Propagation (50 active spikes / 2500 synapses)");

    let mut synapses = Vec::new();
    let mut rng = fastrand::Rng::with_seed(999);
    for src in 0..100 {
        for tgt in 0..100 {
            if rng.f32() < 0.25 && synapses.len() < 2500 {
                synapses.push(Synapse::new(src, tgt, Fix32::from_num(0.5)));
            }
        }
    }

    let static_csr: StaticCsrSynapses<101, 2500> = StaticCsrSynapses::from_synapses(&synapses);
    let dynamic_csr = CsrSynapses::from_synapses(100, &synapses);

    let active_spikes: Vec<u32> = (0..50).map(|i| i * 2).collect();

    group.bench_function("StaticCsrSynapses::propagate", |b| {
        let mut acc = [Fix32::ZERO; 100];
        b.iter(|| {
            static_csr.propagate_active_spikes(black_box(&active_spikes), black_box(&mut acc));
        });
    });

    group.bench_function("CsrSynapses (Dynamic)::propagate", |b| {
        let mut acc = vec![Fix32::ZERO; 100];
        b.iter(|| {
            dynamic_csr.propagate_active_spikes(black_box(&active_spikes), black_box(&mut acc));
        });
    });

    group.finish();
}

/// Benchmark 7: Event-driven 2D convolutions (Spike Propagation)
fn bench_conv2d_propagation(c: &mut Criterion) {
    use snn_core::traits::SynapseMatrix;

    let mut group = c.benchmark_group("Conv2d Propagation (Zero-Division)");

    // Kernel 3x3, 4 in_channels -> 8 out_channels
    // Image 32x32 -> 32*32*4 = 4096 input neurons
    static KERNEL: [Fix32; 8 * 4 * 3 * 3] = [Fix32::from_bits(32768); 8 * 4 * 3 * 3];
    let conv_chw = snn_core::Conv2dSynapses::with_padding(&KERNEL, 4, 8, 3, 3, 1, 1, 1, 1, 32, 32);
    let conv_hwc = snn_core::Conv2dSynapses::with_padding(&KERNEL, 4, 8, 3, 3, 1, 1, 1, 1, 32, 32)
        .with_layout(snn_core::ConvDataLayout::Hwc);

    let active_spikes: Vec<u32> = (0..200).map(|i| i * 20).collect();

    let mut acc_chw = vec![Fix32::ZERO; 8 * 32 * 32];
    group.bench_function("Conv2d CHW (200 spikes, 32x32x4)", |b| {
        b.iter(|| {
            conv_chw.propagate_active_spikes(black_box(&active_spikes), black_box(&mut acc_chw));
        });
    });

    let mut acc_hwc = vec![Fix32::ZERO; 8 * 32 * 32];
    group.bench_function("Conv2d HWC (200 spikes, 32x32x4)", |b| {
        b.iter(|| {
            conv_hwc.propagate_active_spikes(black_box(&active_spikes), black_box(&mut acc_hwc));
        });
    });

    group.finish();
}

/// Benchmark 8: Activation Arena vs Compact Spiking Layers (Zero-Alloc Ping-Pong)
fn bench_arena_vs_compact(c: &mut Criterion) {
    use snn_core::domain::arena::{ActivationArena, ArenaFeedforwardLayer, ArenaLifPopulation};
    use snn_core::domain::layers::{SpikingFeedforwardLayer, SpikingLayer};
    use snn_core::domain::models::lif::LifConfig;
    use snn_core::domain::rom_csr::RomCsrSynapses;

    let mut group = c.benchmark_group("Arena vs Compact (128 neurons)");

    const N: usize = 128;
    const WORDS: usize = N.div_ceil(32);

    static ROW_PTRS: [u16; N + 1] = {
        let mut arr = [0u16; N + 1];
        let mut i = 0;
        while i <= N {
            arr[i] = i as u16;
            i += 1;
        }
        arr
    };
    static TARGETS: [u16; N] = {
        let mut arr = [0u16; N];
        let mut i = 0;
        while i < N {
            arr[i] = i as u16;
            i += 1;
        }
        arr
    };
    static WEIGHTS: [Fix32; N] = [Fix32::from_bits(32768); N];
    let synapses = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS);

    let config = LifConfig::new(Fix32::from_num(1.0), Fix32::from_num(0.9), Fix32::ZERO, 0);

    // Compact Feedforward Layer
    let compact_pop = CompactLifPopulation::<N>::new(config);
    let mut compact_layer =
        SpikingFeedforwardLayer::<CompactLifPopulation<N>, _, N, WORDS>::new(compact_pop, synapses);
    let inputs = [Fix32::from_num(1.5); N];

    group.bench_function("Compact Layer (128 neurons, tick)", |b| {
        b.iter(|| {
            black_box(compact_layer.tick_inputs_words(black_box(&inputs)));
        });
    });

    // Arena Feedforward Layer
    let arena_pop = ArenaLifPopulation::<N>::new(config);
    let arena_layer =
        ArenaFeedforwardLayer::<ArenaLifPopulation<N>, _, N, Fix32>::new(arena_pop, synapses);
    let arena: ActivationArena<N, WORDS, Fix32> = ActivationArena::new();
    let mut arena_net = snn_core::domain::arena::ArenaSpikingNetwork::<_, N, WORDS, N, Fix32>::new(
        arena_layer,
        arena,
    );

    group.bench_function("Arena Network (128 neurons, ping-pong step)", |b| {
        b.iter(|| {
            black_box(arena_net.tick(black_box(&inputs)));
        });
    });

    group.finish();
}

/// Benchmark 9: Unified ModelDescriptor (Fix32 vs Quantized INT8)
fn bench_unified_model_descriptor(c: &mut Criterion) {
    let mut group = c.benchmark_group("Unified ModelDescriptor Engine");
    const N: usize = 128;

    static ROW_PTRS: [u16; N + 1] = {
        let mut arr = [0u16; N + 1];
        let mut i = 0;
        while i <= N {
            arr[i] = i as u16;
            i += 1;
        }
        arr
    };
    static TARGETS: [u16; N] = {
        let mut arr = [0u16; N];
        let mut i = 0;
        while i < N {
            arr[i] = i as u16;
            i += 1;
        }
        arr
    };
    static WEIGHTS_FIX32: [Fix32; N] = [Fix32::from_bits(32768); N];
    static WEIGHTS_I8: [i8; N] = [64; N];
    let scale = Fix32::from_num(0.015625);

    let syn_fix32 = RomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS_FIX32);
    let syn_i8 = QuantizedRomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS_I8, scale);

    let config = LifConfig::new(Fix32::from_num(1.0), Fix32::from_num(0.9), Fix32::ZERO, 0);

    let l_fix32 = LayerDescriptor::new(N, N, syn_fix32, config, None);
    let l_i8 = LayerDescriptor::new_quantized_i8(N, N, syn_i8, config, None);

    let layers_fix32 = [l_fix32];
    let model_fix32 = ModelDescriptor::new(&layers_fix32);
    let layers_i8 = [l_i8];
    let model_i8 = ModelDescriptor::new(&layers_i8);

    let mut sram_fix32 = [0u8; 2048];
    let mut sram_i8 = [0u8; 2048];

    let mut ctx_fix32 = model_fix32
        .init_context(&mut sram_fix32)
        .expect("Context valid");
    let mut ctx_i8 = model_i8.init_context(&mut sram_i8).expect("Context valid");

    let inputs = [Fix32::from_num(1.5); N];
    let mut out_spikes = [0u32; N / 32];

    group.bench_function("ModelDescriptor Fix32 (128 neurons, step)", |b| {
        b.iter(|| {
            model_fix32
                .step(
                    &mut ctx_fix32,
                    black_box(&inputs),
                    black_box(&mut out_spikes),
                )
                .unwrap();
            black_box(&out_spikes);
        });
    });

    group.bench_function("ModelDescriptor Quantized INT8 (128 neurons, step)", |b| {
        b.iter(|| {
            model_i8
                .step(&mut ctx_i8, black_box(&inputs), black_box(&mut out_spikes))
                .unwrap();
            black_box(&out_spikes);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_spatial_stress,
    bench_avalanche,
    bench_liquid_state,
    bench_static_reservoir,
    bench_neuron_dynamics_models,
    bench_csr_propagation,
    bench_conv2d_propagation,
    bench_arena_vs_compact,
    bench_unified_model_descriptor
);
criterion_main!(benches);
