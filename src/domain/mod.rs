pub mod aer;
pub mod arena;
pub mod bitset;
#[cfg(feature = "bptn")]
pub mod bptn;
pub mod compact_population;
#[cfg(feature = "alloc")]
pub mod csr;
pub mod delay_ring;
pub mod encoding;
pub mod engine;
pub mod generic_static_neuron;
pub mod hardware;
pub mod layers;
pub mod lazy_population;
pub mod metrics;
pub mod models;
pub mod narma;
#[cfg(feature = "alloc")]
pub mod network;
#[cfg(feature = "alloc")]
pub mod neuron;
pub mod numeric;
pub mod rom_csr;
pub mod static_csr;
pub mod static_network;
pub mod static_neuron;
pub mod stdp;
pub mod synapse;

pub use aer::{AerBitsetDecoder, AerDmaRingBuffer, AerSpike1d, AerSpike2d};
pub use arena::{
    ActivationArena, ArenaFeedforwardLayer, ArenaLayerChain, ArenaLif16Population,
    ArenaLifPopulation, ArenaSpikingLayer, ArenaSpikingNetwork,
};
pub use bitset::SpikeBitset;
#[cfg(feature = "bptn")]
pub use bptn::{
    BPTN_WINDOW_SIZE, BptnBistableCell, BptnFeedforwardLayer, BptnGatedCell, BptnLatchMode,
    BptnNeuron, BptnSpikingNetwork, BptnSynapse, BptnToggleCell, CompactBptnNeuron,
    MultiScaleBptnNeuron, MultiScalePyramidBuffer, RBptnLayer, RBptnSpikingNetwork,
    TemporalFramingBuffer64, dilate_temporal_jitter, shift_temporal_age, shift_temporal_delay,
    swar_compress_dyadic_u64,
};
pub use compact_population::{CompactLif16Population, CompactLifPopulation};
#[cfg(feature = "alloc")]
pub use csr::CsrSynapses;
#[allow(deprecated)]
pub use delay_ring::DelayedSynapseMatrix;
pub use delay_ring::{StaticSpikeDelayLine, StaticSpikeWordDelayLine};
pub use encoding::{DeltaModulationEncoder, LatencyEncoder, RateEncoder};
pub use engine::{
    EngineError, ExecutionContext, LayerDescriptor, MaxPool2dConfig, MemoryFootprint,
    ModelDescriptor, SynapseMatrixKind,
};
pub use hardware::{FastDivider, fast_decay_m0, qadd16, smlad, ssat_16};
pub use layers::{
    ChainedSpikingNetwork, Conv1dSynapses, Conv2dSynapses, ConvDataLayout, DelayedSpikingLayer,
    LayerChain, QuantizedFeedforwardLayer, ResidualSpikingLayer, SpikeMaxPool2d,
    SpikingFeedforwardLayer, SpikingLayer, SpikingRecurrentLayer,
};
#[allow(deprecated)]
pub use layers::{DeepSpikingNetwork2, DeepSpikingNetwork3, DeepSpikingNetwork4};
pub use lazy_population::LazyDecayLifPopulation;

pub use generic_static_neuron::{
    GenericStaticNeuronPopulation, HeterogeneousStaticNeuronPopulation,
};
pub use metrics::{EnergyEstimator, HardwareProfile, LatencyTracker, SparsityMeter, SynOpsTracker};
pub use models::{
    AlifConfig, AlifDynamics, AlifState, CubaLifConfig, CubaLifDynamics, CubaLifState,
    IzhikevichConfig, IzhikevichDynamics, IzhikevichState, LifConfig, LifDynamics, LifState,
};
pub use narma::Narma10System;
#[cfg(feature = "alloc")]
pub use network::Network;
#[cfg(feature = "alloc")]
pub use neuron::NeuronPopulation;
pub use numeric::{Fix16, Fix32};

pub use rom_csr::{
    BlockRomCsrSynapses, QuantizedI8, QuantizedI16Q8, QuantizedRomCsrSynapses, RomCsrSynapses,
    RomIndex, SynapticWeight,
};
pub use static_csr::StaticCsrSynapses;
pub use static_network::{GenericStaticNetwork, StaticNetwork, StaticReservoir};
#[allow(deprecated)]
pub use static_neuron::StaticNeuronPopulation;
#[cfg(feature = "alloc")]
pub use stdp::{CscIndex, StdpLearner};
pub use stdp::{
    HomeostaticThreshold, RewardModulatedStdp, StaticCscIndex, StaticStdpLearner, StdpConfig,
};
pub use synapse::Synapse;
