//! # 🧠 `snn_core`
//!
//! Ultra-low power neuromorphic engine in pure Rust for bare-metal embedded architectures (`#![no_std]`).
//!
//! Designed specifically for sparse neuromorphic inference on microcontrollers (ARM Cortex-M0+/M3/M4 without FPU, RISC-V):
//! - **Zero dynamic heap allocation** on the critical path (`no_std`, Flash ROM `.rodata` layout).
//! - **Saturating fixed-point arithmetic**: [`Fix32`] (Q16.16) and [`Fix16`] (Q8.8).
//! - **Ultra-dense memory layouts**: [`CompactLifPopulation`] (9 bytes/neuron), [`CompactLif16Population`] (5 bytes/neuron), [`ActivationArena`].
//! - **Vectorized spike bus**: [`SpikeBitset`] with hardware native `CTZ` (`trailing_zeros`) decoding.
//!
//! ## Minimal zero-allocation inference example
//!
//! ```rust
//! use snn_core::domain::compact_population::CompactLifPopulation;
//! use snn_core::models::lif::{LifConfig, ResetMode};
//! use snn_core::{Fix32, GenericStaticNetwork, RomCsrSynapses, SpikingNetwork};
//!
//! // 1. CSR matrices stored in Flash ROM (.rodata)
//! static IN_ROW_PTR: [u16; 2] = [0, 1];
//! static IN_TARGETS: [u16; 1] = [0];
//! static IN_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(98304)]; // 1.5 in Q16.16 fixed-point
//!
//! static INT_ROW_PTR: [u16; 3] = [0, 1, 1];
//! static INT_TARGETS: [u16; 1] = [1];
//! static INT_WEIGHTS: [Fix32; 1] = [Fix32::from_bits(65536)]; // 1.0 in Q16.16 fixed-point
//!
//! let in_synapses = RomCsrSynapses::new(&IN_ROW_PTR, &IN_TARGETS, &IN_WEIGHTS);
//! let int_synapses = RomCsrSynapses::new(&INT_ROW_PTR, &INT_TARGETS, &INT_WEIGHTS);
//!
//! // 2. Compact population of 2 LIF neurons (9 bytes/neuron in Fix32)
//! let config = LifConfig::new(Fix32::ONE, Fix32::from_bits(58982), Fix32::ZERO, 1)
//!     .with_reset_mode(ResetMode::Subtract);
//! let neurons = CompactLifPopulation::<2>::new(config);
//!
//! // 3. Generic static network (2 neurons, 1 input)
//! let mut network: GenericStaticNetwork<_, _, _, 2, 1> =
//!     GenericStaticNetwork::new(neurons, int_synapses, in_synapses);
//!
//! let inputs = [Fix32::ONE];
//! let spikes = network.tick(&inputs);
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

pub mod narma {
    pub use crate::domain::narma::*;
}

// Silicon hardware acceleration primitives (ARM Cortex-M, RISC-V)
pub mod hardware {
    pub use crate::domain::hardware::*;
}

// C FFI interface for microcontroller and RTOS C/C++ integration
pub mod ffi {
    pub use crate::application::ffi::*;
}

// Re-exports of fundamental traits
pub use traits::{NeuronDynamics, Readout, SpikingNetwork, SpikingPopulation, SynapseMatrix};

pub use application::can_parser::{
    ParsedCanFrame, parse_any_can_line, parse_candump_compact_line, parse_candump_standard_line,
    parse_hcrl_csv_line, parse_timestamp_us,
};
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

// Silico-native BPTN (Bit-Parallel Temporal Neural Network) computing model - Optional feature
#[cfg(feature = "bptn")]
pub mod bptn {
    pub use crate::domain::bptn::*;
}

pub use domain::aer::{AerBitsetDecoder, AerDmaRingBuffer, AerSpike1d, AerSpike2d};
pub use domain::arena::{
    ActivationArena, ArenaFeedforwardLayer, ArenaLayerChain, ArenaLif16Population,
    ArenaLifPopulation, ArenaSpikingLayer, ArenaSpikingNetwork,
};
pub use domain::bitset::SpikeBitset;
#[cfg(feature = "bptn")]
pub use domain::bptn::{
    BPTN_WINDOW_SIZE, BptnBistableCell, BptnFeedforwardLayer, BptnGatedCell, BptnHysteresisLatch,
    BptnLatchMode, BptnNeuron, BptnRobustToggleCell, BptnSpikingNetwork, BptnSynapse,
    BptnToggleCell, CompactBptnNeuron, MultiScaleBptnNeuron, MultiScalePyramidBuffer, RBptnLayer,
    RBptnSpikingNetwork, TemporalFramingBuffer64, bptn_parallel_smith_step_u64,
    dilate_temporal_jitter, erode_temporal_noise, open_temporal_filter, shift_temporal_age,
    shift_temporal_delay, swar_compress_dyadic_u64, swar_majority_pool_8x8,
};
pub use domain::compact_population::{CompactLif16Population, CompactLifPopulation};
#[allow(deprecated)]
pub use domain::delay_ring::DelayedSynapseMatrix;
pub use domain::delay_ring::{StaticSpikeDelayLine, StaticSpikeWordDelayLine};
pub use domain::encoding::{DeltaModulationEncoder, LatencyEncoder, RateEncoder};
pub use domain::engine::{
    EngineError, ExecutionContext, LayerDescriptor, MaxPool2dConfig, MemoryFootprint,
    ModelDescriptor, SynapseMatrixKind,
};
pub use domain::generic_static_neuron::{
    GenericStaticNeuronPopulation, HeterogeneousStaticNeuronPopulation,
};
pub use domain::hardware::{FastDivider, fast_decay_m0, qadd16, smlad, ssat_16};
pub use domain::layers::{
    ChainedSpikingNetwork, Conv1dSynapses, Conv2dSynapses, ConvDataLayout, DelayedSpikingLayer,
    LayerChain, QuantizedFeedforwardLayer, ResidualSpikingLayer, SpikeMaxPool2d,
    SpikingFeedforwardLayer, SpikingLayer, SpikingRecurrentLayer,
};
#[allow(deprecated)]
pub use domain::layers::{DeepSpikingNetwork2, DeepSpikingNetwork3, DeepSpikingNetwork4};
pub use domain::lazy_population::LazyDecayLifPopulation;
pub use domain::metrics::{
    EnergyEstimator, HardwareProfile, LatencyTracker, SparsityMeter, SynOpsTracker,
};

pub use domain::narma::Narma10System;
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
