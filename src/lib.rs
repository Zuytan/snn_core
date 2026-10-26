//! # `snn_core`
//!
//! Spiking-neural-network runtime for microcontrollers (`#![no_std]`, Cortex-M0+ to M7, RISC-V), with
//! saturating fixed-point arithmetic ([`Fix32`] Q16.16, [`Fix16`] Q8.8) and no heap allocation on the
//! inference path.
//!
//! ## Which API to use
//!
//! | Need | Use |
//! | :--- | :--- |
//! | Run a model on a device | [`ModelDescriptor`] + [`ExecutionContext`] (this page, first example) |
//! | Simulate on a host, learn online (STDP), or build custom dynamics | [`GenericStaticNetwork`] and the [`SpikingNetwork`] trait |
//! | Deploy a trained NIR / ONNX graph | `tools/snn_compiler`, which emits a [`ModelDescriptor`] |
//! | Call from C / C++ | the `snn_core_c` crate and `include/snn_core.h` |
//!
//! `ChainedSpikingNetwork`, `ArenaSpikingNetwork` and `arena_spiking_network!` are deprecated: they
//! duplicate what [`ModelDescriptor`] does.
//!
//! ## Memory model
//!
//! - **Flash (`.rodata`)**: the [`ModelDescriptor`], its [`LayerDescriptor`]s, the CSR synapse tables
//!   ([`RomCsrSynapses`]) and the neuron configurations. All are `const`/`static` and never written.
//! - **SRAM**: one buffer you provide, sized by [`ModelDescriptor::sram_bytes`] (computed at compile time
//!   from the layer sizes). It holds the accumulators, the activation bitsets shared between layers,
//!   the membrane potentials, refractory counters and recurrent spikes.
//! - **Alignment**: the buffer must be 4-byte aligned; use [`AlignedBuffer`]. Otherwise
//!   `init_context` returns [`EngineError::MisalignedBuffer`]. A buffer that is too small returns
//!   [`EngineError::BufferTooSmall`].
//!
//! ## Inference
//!
//! ```rust
//! use snn_core::{
//!     AlignedBuffer, Fix32, LayerDescriptor, LifConfig, ModelDescriptor, RomCsrSynapses,
//! };
//!
//! static ROW: [u16; 3] = [0, 1, 2];
//! static TARGETS: [u16; 2] = [0, 1];
//! static WEIGHTS: [Fix32; 2] = [Fix32::from_bits(98304), Fix32::from_bits(65536)]; // 1.5 and 1.0
//!
//! static SYNAPSES: RomCsrSynapses<'static> = RomCsrSynapses::new(&ROW, &TARGETS, &WEIGHTS);
//! static LAYERS: [LayerDescriptor<'static>; 1] = [LayerDescriptor::new(
//!     2,
//!     2,
//!     SYNAPSES,
//!     LifConfig::new(Fix32::ONE, Fix32::from_bits(58982), Fix32::ZERO, 1),
//!     None,
//! )];
//! static MODEL: ModelDescriptor<'static> = ModelDescriptor::new(&LAYERS);
//!
//! let mut sram = AlignedBuffer::<256>::new(); // >= MODEL.sram_bytes()
//! let mut ctx = MODEL.init_context(sram.as_mut_bytes()).expect("buffer large enough and aligned");
//!
//! let mut out_spikes = [0u32; 1];
//! MODEL.step(&mut ctx, &[Fix32::ONE, Fix32::ZERO], &mut out_spikes).expect("step succeeds");
//! assert_eq!(out_spikes[0] & 1, 1); // neuron 0 received 1.5 >= threshold 1.0
//! ```
//!
//! ## Host simulation and online learning
//!
//! ```rust
//! use snn_core::domain::compact_population::CompactLifPopulation;
//! use snn_core::models::lif::{LifConfig, ResetMode};
//! use snn_core::{Fix32, GenericStaticNetwork, RomCsrSynapses, SpikingNetwork};
//!
//! static IN_ROW_PTR: [u16; 2] = [0, 1];
//! static IN_TARGETS: [u16; 1] = [0];
//! static IN_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(98304)]; // 1.5 in Q16.16
//!
//! static INT_ROW_PTR: [u16; 3] = [0, 1, 1];
//! static INT_TARGETS: [u16; 1] = [1];
//! static INT_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(65536)]; // 1.0 in Q16.16
//!
//! let in_synapses = RomCsrSynapses::new(&IN_ROW_PTR, &IN_TARGETS, &IN_WEIGHTS);
//! let int_synapses = RomCsrSynapses::new(&INT_ROW_PTR, &INT_TARGETS, &INT_WEIGHTS);
//!
//! let config = LifConfig::new(Fix32::ONE, Fix32::from_bits(58982), Fix32::ZERO, 1)
//!     .with_reset_mode(ResetMode::Subtract);
//! let neurons = CompactLifPopulation::<2>::new(config);
//!
//! let mut network: GenericStaticNetwork<_, _, _, 2, 1> =
//!     GenericStaticNetwork::new(neurons, int_synapses, in_synapses);
//! let spikes = network.tick(&[Fix32::ONE]);
//! assert!(spikes[0]);
//! ```

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod application;
pub mod domain;
pub mod traits;

// Neuron dynamics models
pub mod models {
    pub use crate::domain::models::*;
}
pub use domain::models::{AlifConfig, CubaLifConfig, IzhikevichConfig, LifConfig, ResetMode};

// Neuromorphic metrics and benchmarks
pub mod metrics {
    pub use crate::domain::metrics::*;
}

// Silicon hardware acceleration primitives (ARM Cortex-M, RISC-V)
pub mod hardware {
    pub use crate::domain::hardware::*;
}

// Re-exports of fundamental traits
pub use traits::{NeuronDynamics, Readout, SpikingNetwork, SpikingPopulation, SynapseMatrix};

pub use application::nir_import::{
    NirAffineNode, NirConv1dNode, NirConv2dNode, NirCubaLifNode, NirLifNode, NirLinearNode,
    pade_exp_neg,
};
#[cfg(feature = "alloc")]
pub use application::nir_import::{emit_rust_static_csr, emit_rust_static_quantized_i8};
#[cfg(feature = "alloc")]
pub use application::readout::ReadoutLayer;
pub use application::readout::{
    StaticReadoutLayer, TrainableReadoutLayer, TrainableRegressionReadout,
};

pub use domain::aer::{AerBitsetDecoder, AerDmaRingBuffer, AerSpike1d, AerSpike2d};
#[allow(deprecated)]
pub use domain::arena::ArenaSpikingNetwork;
pub use domain::arena::{
    ActivationArena, ArenaFeedforwardLayer, ArenaLayerChain, ArenaLif16Population,
    ArenaLifPopulation, ArenaSpikingLayer,
};
pub use domain::bitset::SpikeBitset;
pub use domain::blob::{BlobError, ModelBlob};
pub use domain::compact_population::{CompactLif16Population, CompactLifPopulation};
#[allow(deprecated)]
pub use domain::delay_ring::DelayedSynapseMatrix;
pub use domain::delay_ring::{StaticSpikeDelayLine, StaticSpikeWordDelayLine};
pub use domain::encoding::{DeltaModulationEncoder, LatencyEncoder, RateEncoder};
pub use domain::engine::{
    AlignedBuffer, EngineError, ExecutionContext, LayerDescriptor, MaxPool2dConfig,
    MemoryFootprint, ModelDescriptor, SynapseMatrixKind,
};
pub use domain::generic_static_neuron::{
    GenericStaticNeuronPopulation, HeterogeneousStaticNeuronPopulation,
};
pub use domain::hardware::{FastDivider, fast_decay_m0, qadd16, smlad, ssat_16};
#[allow(deprecated)]
pub use domain::layers::ChainedSpikingNetwork;
pub use domain::layers::{
    Conv1dSynapses, Conv2dSynapses, ConvDataLayout, DelayedSpikingLayer, LayerChain,
    QuantizedFeedforwardLayer, ResidualSpikingLayer, SpikeMaxPool2d, SpikingFeedforwardLayer,
    SpikingLayer, SpikingRecurrentLayer,
};
#[allow(deprecated)]
pub use domain::layers::{DeepSpikingNetwork2, DeepSpikingNetwork3, DeepSpikingNetwork4};
pub use domain::lazy_population::LazyDecayLifPopulation;
#[allow(deprecated)]
pub use domain::metrics::HardwareProfile;
pub use domain::metrics::{EnergyEstimator, LatencyTracker, SparsityMeter, SynOpsTracker};

#[cfg(feature = "alloc")]
pub use domain::network::Network;
#[cfg(feature = "alloc")]
pub use domain::neuron::NeuronPopulation;
pub use domain::numeric::{Fix16, Fix32, NumericScalar};

pub use domain::rom_csr::{
    BlockRomCsrSynapses, QuantizedI8, QuantizedI16Q8, QuantizedRomCsrSynapses, RomCsrSynapses,
    RomIndex, SynapticWeight,
};
pub use domain::static_csr::StaticCsrSynapses;
pub use domain::static_network::{GenericStaticNetwork, StaticNetwork, StaticReservoir};
#[allow(deprecated)]
pub use domain::static_neuron::StaticNeuronPopulation;
#[cfg(feature = "alloc")]
pub use domain::stdp::{CscIndex, StdpLearner};
pub use domain::stdp::{
    HomeostaticThreshold, RewardModulatedStdp, StaticCscIndex, StaticStdpLearner, StdpConfig,
};
pub use domain::synapse::Synapse;

/// Compiles and runs the Rust code blocks of `README.md` as doctests, so the README cannot drift from the API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
