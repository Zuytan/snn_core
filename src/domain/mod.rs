pub mod aer;
pub mod arena;
pub mod bitset;
pub mod blob;
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
#[allow(deprecated)]
pub use arena::ArenaSpikingNetwork;
pub use arena::{
    ActivationArena, ArenaFeedforwardLayer, ArenaLayerChain, ArenaLif16Population,
    ArenaLifPopulation, ArenaSpikingLayer,
};
pub use bitset::SpikeBitset;
pub use compact_population::{CompactLif16Population, CompactLifPopulation};
#[cfg(feature = "alloc")]
pub use csr::CsrSynapses;
#[allow(deprecated)]
pub use delay_ring::DelayedSynapseMatrix;
pub use delay_ring::{StaticSpikeDelayLine, StaticSpikeWordDelayLine};
pub use encoding::{DeltaModulationEncoder, LatencyEncoder, RateEncoder};
pub use engine::{
    AlignedBuffer, EngineError, ExecutionContext, LayerDescriptor, MaxPool2dConfig,
    MemoryFootprint, ModelDescriptor, SynapseMatrixKind,
};
pub use hardware::{FastDivider, fast_decay_m0, qadd16, smlad, ssat_16};
#[allow(deprecated)]
pub use layers::ChainedSpikingNetwork;
pub use layers::{
    Conv1dSynapses, Conv2dSynapses, ConvDataLayout, DelayedSpikingLayer, LayerChain,
    QuantizedFeedforwardLayer, ResidualSpikingLayer, SpikeMaxPool2d, SpikingFeedforwardLayer,
    SpikingLayer, SpikingRecurrentLayer,
};
#[allow(deprecated)]
pub use layers::{DeepSpikingNetwork2, DeepSpikingNetwork3, DeepSpikingNetwork4};
pub use lazy_population::LazyDecayLifPopulation;

pub use generic_static_neuron::{
    GenericStaticNeuronPopulation, HeterogeneousStaticNeuronPopulation,
};
#[allow(deprecated)]
pub use metrics::HardwareProfile;
pub use metrics::{EnergyEstimator, LatencyTracker, SparsityMeter, SynOpsTracker};
pub use models::{
    AlifConfig, AlifDynamics, AlifState, CubaLifConfig, CubaLifDynamics, CubaLifState,
    IzhikevichConfig, IzhikevichDynamics, IzhikevichState, LifConfig, LifDynamics, LifState,
};
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
