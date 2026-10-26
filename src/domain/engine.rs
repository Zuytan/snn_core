//! 🚀 Unified Model Runner & Execution Context
//!
//! Provides a standardized high-level interface without complex generic types,
//! compliant with embedded industry standards (TFLite Micro / CMSIS-NN paradigm):
//! - [`ModelDescriptor`]: Static metadata and weights residing in Flash ROM (`.rodata`).
//! - [`ExecutionContext`]: Dynamic state memory partitioned in SRAM without allocation.

use crate::domain::layers::{Conv1dSynapses, Conv2dSynapses};
use crate::domain::models::lif::{LifConfig, ResetMode};
use crate::domain::numeric::Fix32;
use crate::domain::rom_csr::{QuantizedRomCsrSynapses, RomCsrSynapses};
use crate::traits::SynapseMatrix;

/// Potential execution errors in the unified engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineError {
    /// The provided SRAM arena is too small to hold the model's dynamic state.
    BufferTooSmall { required: usize, provided: usize },
    /// The buffer is not properly aligned (4-byte alignment required for Fix32/u32).
    MisalignedBuffer,
    /// Input or output dimensions mismatch the model topology.
    DimensionMismatch,
}

/// Type of synaptic matrix stored in immutable Flash ROM.
#[derive(Clone, Copy, Debug)]
pub enum SynapseMatrixKind<'a> {
    /// Standard CSR matrix with Fix32 Q16.16 weights and u16 indices (compact microcontroller).
    Fix32(RomCsrSynapses<'a>),
    /// Standard CSR matrix with Fix32 Q16.16 weights and u32 indices (large models / NIR compatibility).
    Fix32U32(RomCsrSynapses<'a, u32>),
    /// Quantized INT8 CSR matrix with Fix32 scaling factor (4x Flash footprint reduction).
    QuantizedInt8(QuantizedRomCsrSynapses<'a>),
    /// 1D convolution synaptic matrix with kernel in Flash ROM.
    Conv1d(Conv1dSynapses<'a, Fix32>),
    /// Spatial 2D convolution synaptic matrix (DVS neuromorphic vision) with kernel in Flash ROM.
    Conv2d(Conv2dSynapses<'a, Fix32>),
}

impl<'a> From<RomCsrSynapses<'a>> for SynapseMatrixKind<'a> {
    #[inline(always)]
    fn from(syn: RomCsrSynapses<'a>) -> Self {
        Self::Fix32(syn)
    }
}

impl<'a> From<RomCsrSynapses<'a, u32>> for SynapseMatrixKind<'a> {
    #[inline(always)]
    fn from(syn: RomCsrSynapses<'a, u32>) -> Self {
        Self::Fix32U32(syn)
    }
}

impl<'a> From<QuantizedRomCsrSynapses<'a>> for SynapseMatrixKind<'a> {
    #[inline(always)]
    fn from(syn: QuantizedRomCsrSynapses<'a>) -> Self {
        Self::QuantizedInt8(syn)
    }
}

impl<'a> From<Conv1dSynapses<'a, Fix32>> for SynapseMatrixKind<'a> {
    #[inline(always)]
    fn from(syn: Conv1dSynapses<'a, Fix32>) -> Self {
        Self::Conv1d(syn)
    }
}

impl<'a> From<Conv2dSynapses<'a, Fix32>> for SynapseMatrixKind<'a> {
    #[inline(always)]
    fn from(syn: Conv2dSynapses<'a, Fix32>) -> Self {
        Self::Conv2d(syn)
    }
}

impl SynapseMatrix<Fix32> for SynapseMatrixKind<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            Self::Fix32(s) => s.len(),
            Self::Fix32U32(s) => s.len(),
            Self::QuantizedInt8(s) => s.len(),
            Self::Conv1d(s) => s.len(),
            Self::Conv2d(s) => s.len(),
        }
    }

    #[inline(always)]
    fn propagate_active_spikes(&self, active_spikes: &[u32], input_accumulators: &mut [Fix32]) {
        match self {
            Self::Fix32(s) => s.propagate_active_spikes(active_spikes, input_accumulators),
            Self::Fix32U32(s) => s.propagate_active_spikes(active_spikes, input_accumulators),
            Self::QuantizedInt8(s) => s.propagate_active_spikes(active_spikes, input_accumulators),
            Self::Conv1d(s) => s.propagate_active_spikes(active_spikes, input_accumulators),
            Self::Conv2d(s) => s.propagate_active_spikes(active_spikes, input_accumulators),
        }
    }

    #[inline(always)]
    fn propagate_weighted_inputs(&self, inputs: &[Fix32], input_accumulators: &mut [Fix32]) {
        match self {
            Self::Fix32(s) => s.propagate_weighted_inputs(inputs, input_accumulators),
            Self::Fix32U32(s) => s.propagate_weighted_inputs(inputs, input_accumulators),
            Self::QuantizedInt8(s) => s.propagate_weighted_inputs(inputs, input_accumulators),
            Self::Conv1d(s) => s.propagate_weighted_inputs(inputs, input_accumulators),
            Self::Conv2d(s) => s.propagate_weighted_inputs(inputs, input_accumulators),
        }
    }

    #[inline(always)]
    fn propagate_spikes_words(&self, words: &[u32], input_accumulators: &mut [Fix32]) {
        match self {
            Self::Fix32(s) => s.propagate_spikes_words(words, input_accumulators),
            Self::Fix32U32(s) => s.propagate_spikes_words(words, input_accumulators),
            Self::QuantizedInt8(s) => s.propagate_spikes_words(words, input_accumulators),
            Self::Conv1d(s) => s.propagate_spikes_words(words, input_accumulators),
            Self::Conv2d(s) => s.propagate_spikes_words(words, input_accumulators),
        }
    }
}

/// Static configuration for a 2D spatial Max-Pooling spike reduction layer.
/// Applies logical OR reduction over receptive fields without arithmetic computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxPool2dConfig {
    pub channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
}

impl MaxPool2dConfig {
    /// Creates a new 2D Max-Pooling configuration.
    #[inline(always)]
    pub const fn new(
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Self {
        Self {
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        }
    }

    /// Computed output height.
    #[inline(always)]
    pub const fn out_h(&self) -> usize {
        if self.in_h >= self.kernel_h && self.stride_h > 0 {
            (self.in_h - self.kernel_h) / self.stride_h + 1
        } else {
            0
        }
    }

    /// Computed output width.
    #[inline(always)]
    pub const fn out_w(&self) -> usize {
        if self.in_w >= self.kernel_w && self.stride_w > 0 {
            (self.in_w - self.kernel_w) / self.stride_w + 1
        } else {
            0
        }
    }

    /// Total number of input features.
    #[inline(always)]
    pub const fn in_features(&self) -> usize {
        self.channels * self.in_h * self.in_w
    }

    /// Total number of output neurons/spikes after pooling.
    #[inline(always)]
    pub const fn out_features(&self) -> usize {
        self.channels * self.out_h() * self.out_w()
    }

    /// Executes bitwise logical OR pooling operation from input words to output words.
    pub fn pool_spikes_words(&self, in_spikes_words: &[u32], out_spikes_words: &mut [u32]) {
        out_spikes_words.fill(0);
        let out_h = self.out_h();
        let out_w = self.out_w();
        let mut out_idx = 0;

        for c in 0..self.channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut pooled = false;
                    for kh in 0..self.kernel_h {
                        for kw in 0..self.kernel_w {
                            let ih = oh * self.stride_h + kh;
                            let iw = ow * self.stride_w + kw;
                            let in_idx = (c * self.in_h + ih) * self.in_w + iw;
                            let w = in_idx / 32;
                            let b = in_idx % 32;
                            if w < in_spikes_words.len() && ((in_spikes_words[w] >> b) & 1) != 0 {
                                pooled = true;
                                break;
                            }
                        }
                        if pooled {
                            break;
                        }
                    }
                    if pooled {
                        let ow_idx = out_idx / 32;
                        let ob_idx = out_idx % 32;
                        if ow_idx < out_spikes_words.len() {
                            out_spikes_words[ow_idx] |= 1 << ob_idx;
                        }
                    }
                    out_idx += 1;
                }
            }
        }
    }
}

/// Static descriptor of a LIF layer (dense, convolutional, or pooling) in Flash ROM.
#[derive(Clone, Copy, Debug)]
pub struct LayerDescriptor<'a> {
    pub in_features: usize,
    pub out_features: usize,
    pub synapses: SynapseMatrixKind<'a>,
    /// Optional recurrent synapses W_rec for intra-layer self-feedback on the next tick.
    pub recurrent_synapses: Option<SynapseMatrixKind<'a>>,
    pub lif_config: LifConfig,
    pub bias: Option<&'a [Fix32]>,
    /// Optional source layer index for DAG connections.
    /// `None` indicates the preceding sequential layer (`l_idx - 1`, or external inputs if `l_idx == 0`).
    pub source_layer: Option<usize>,
    /// Optional layer index for a residual connection (skip-connection).
    pub residual_source: Option<usize>,
    /// Optional scaling factor applied to residual spikes (`Fix32::ONE` by default).
    pub residual_weight: Option<Fix32>,
    /// Offset in 32-bit words within the activation arena for scheduled DAG topologies.
    pub activation_word_offset: Option<usize>,
    /// Optional spatial 2D Max-Pooling configuration (pure logical reduction without synapses).
    pub pool_config: Option<MaxPool2dConfig>,
}

impl LayerDescriptor<'static> {
    /// Placeholder layer (no synapses) used to initialise the scratch array that
    /// [`ModelBlob::load_layers`](crate::domain::blob::ModelBlob::load_layers) fills.
    pub const EMPTY: Self = Self {
        in_features: 0,
        out_features: 0,
        synapses: SynapseMatrixKind::Fix32(RomCsrSynapses::new(&[0u16], &[], &[])),
        recurrent_synapses: None,
        lif_config: LifConfig::new(Fix32::ZERO, Fix32::ZERO, Fix32::ZERO, 0),
        bias: None,
        source_layer: None,
        residual_source: None,
        residual_weight: None,
        activation_word_offset: None,
        pool_config: None,
    };
}

impl<'a> LayerDescriptor<'a> {
    /// Creates a Dense layer with Fix32 CSR synapses.
    pub const fn new(
        in_features: usize,
        out_features: usize,
        synapses: RomCsrSynapses<'a>,
        lif_config: LifConfig,
        bias: Option<&'a [Fix32]>,
    ) -> Self {
        Self {
            in_features,
            out_features,
            synapses: SynapseMatrixKind::Fix32(synapses),
            recurrent_synapses: None,
            lif_config,
            bias,
            source_layer: None,
            residual_source: None,
            residual_weight: None,
            activation_word_offset: None,
            pool_config: None,
        }
    }

    /// Creates a Dense layer with Fix32 CSR synapses and 32-bit indices (large models / NIR).
    pub const fn new_u32(
        in_features: usize,
        out_features: usize,
        synapses: RomCsrSynapses<'a, u32>,
        lif_config: LifConfig,
        bias: Option<&'a [Fix32]>,
    ) -> Self {
        Self {
            in_features,
            out_features,
            synapses: SynapseMatrixKind::Fix32U32(synapses),
            recurrent_synapses: None,
            lif_config,
            bias,
            source_layer: None,
            residual_source: None,
            residual_weight: None,
            activation_word_offset: None,
            pool_config: None,
        }
    }

    /// Creates a Dense layer with quantized INT8 CSR synapses.
    pub const fn new_quantized_i8(
        in_features: usize,
        out_features: usize,
        synapses: QuantizedRomCsrSynapses<'a>,
        lif_config: LifConfig,
        bias: Option<&'a [Fix32]>,
    ) -> Self {
        Self {
            in_features,
            out_features,
            synapses: SynapseMatrixKind::QuantizedInt8(synapses),
            recurrent_synapses: None,
            lif_config,
            bias,
            source_layer: None,
            residual_source: None,
            residual_weight: None,
            activation_word_offset: None,
            pool_config: None,
        }
    }

    /// Creates a Conv1D layer with shared kernel in Flash ROM.
    pub const fn new_conv1d(
        in_features: usize,
        out_features: usize,
        synapses: Conv1dSynapses<'a, Fix32>,
        lif_config: LifConfig,
        bias: Option<&'a [Fix32]>,
    ) -> Self {
        Self {
            in_features,
            out_features,
            synapses: SynapseMatrixKind::Conv1d(synapses),
            recurrent_synapses: None,
            lif_config,
            bias,
            source_layer: None,
            residual_source: None,
            residual_weight: None,
            activation_word_offset: None,
            pool_config: None,
        }
    }

    /// Creates a Conv2D layer (Neuromorphic vision / DVS) with kernel in Flash ROM.
    pub const fn new_conv2d(
        in_features: usize,
        out_features: usize,
        synapses: Conv2dSynapses<'a, Fix32>,
        lif_config: LifConfig,
        bias: Option<&'a [Fix32]>,
    ) -> Self {
        Self {
            in_features,
            out_features,
            synapses: SynapseMatrixKind::Conv2d(synapses),
            recurrent_synapses: None,
            lif_config,
            bias,
            source_layer: None,
            residual_source: None,
            residual_weight: None,
            activation_word_offset: None,
            pool_config: None,
        }
    }

    /// Creates a dedicated 2D spatial Max-Pooling layer (without synapses or membrane dynamics).
    pub const fn new_max_pool_2d(pool_config: MaxPool2dConfig) -> Self {
        Self {
            in_features: pool_config.in_features(),
            out_features: pool_config.out_features(),
            synapses: SynapseMatrixKind::Fix32(RomCsrSynapses::new(&[], &[], &[])),
            recurrent_synapses: None,
            lif_config: LifConfig::new(Fix32::ONE, Fix32::ZERO, Fix32::ZERO, 0),
            bias: None,
            source_layer: None,
            residual_source: None,
            residual_weight: None,
            activation_word_offset: None,
            pool_config: Some(pool_config),
        }
    }

    /// Associates a recurrent synaptic matrix W_rec with the layer for intra-layer feedback.
    #[must_use]
    pub const fn with_recurrent(mut self, recurrent_synapses: SynapseMatrixKind<'a>) -> Self {
        self.recurrent_synapses = Some(recurrent_synapses);
        self
    }

    /// Associates a spatial 2D Max-Pooling stage with the layer output.
    #[must_use]
    pub const fn with_pool(mut self, pool_config: MaxPool2dConfig) -> Self {
        self.pool_config = Some(pool_config);
        self
    }

    /// Specifies the upstream source layer for DAG topologies.
    #[must_use]
    pub const fn with_source_layer(mut self, source: usize) -> Self {
        self.source_layer = Some(source);
        self
    }

    /// Specifies a residual skip connection with an optional scalar weight.
    #[must_use]
    pub const fn with_residual(mut self, residual_source: usize, weight: Option<Fix32>) -> Self {
        self.residual_source = Some(residual_source);
        self.residual_weight = weight;
        self
    }

    /// Specifies the memory offset in the activation arena.
    #[must_use]
    pub const fn with_activation_offset(mut self, offset: usize) -> Self {
        self.activation_word_offset = Some(offset);
        self
    }
}

/// Static memory allocation metadata computed at compile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MemoryFootprint {
    pub max_layer_neurons: usize,
    pub max_layer_words: usize,
    pub total_activation_words: usize,
    pub total_state_neurons: usize,
    pub total_recurrent_words: usize,
    pub total_sram_bytes: usize,
}

/// SRAM buffer with the 4-byte alignment that [`ModelDescriptor::init_context`] requires.
///
/// Declare it as a `static mut` or a stack variable sized with
/// [`ModelDescriptor::sram_bytes`] (or any upper bound of it); a plain `[u8; N]` has alignment 1
/// and is rejected with [`EngineError::MisalignedBuffer`] when its address is not a multiple of 4.
#[repr(align(4))]
#[derive(Clone, Copy, Debug)]
pub struct AlignedBuffer<const N: usize>([u8; N]);

impl<const N: usize> AlignedBuffer<N> {
    /// Zero-filled buffer.
    pub const fn new() -> Self {
        Self([0u8; N])
    }

    /// The bytes to pass to [`ModelDescriptor::init_context`].
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl<const N: usize> Default for AlignedBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete static descriptor of a deep SNN in Flash ROM (`.rodata`).
#[derive(Clone, Copy, Debug)]
pub struct ModelDescriptor<'a> {
    pub layers: &'a [LayerDescriptor<'a>],
    pub memory_footprint: MemoryFootprint,
}

impl<'a> ModelDescriptor<'a> {
    /// Computes the exact SRAM memory footprint required to instantiate the model's dynamic state.
    pub const fn compute_footprint(layers: &'a [LayerDescriptor<'a>]) -> MemoryFootprint {
        let mut max_neurons = 0;
        let mut total_neurons = 0;
        let mut total_recurrent_words = 0;
        let mut max_out_words = 0;
        let mut max_in_words = 0;
        let mut max_explicit_activation_word = 0;
        let mut has_explicit_offsets = false;

        let mut i = 0;
        while i < layers.len() {
            let l = &layers[i];
            let n = l.out_features;
            let out_w = n.div_ceil(32);

            // Only layers with membrane dynamics consume state neurons
            if l.pool_config.is_none() {
                if n > max_neurons {
                    max_neurons = n;
                }
                total_neurons += n;
            }

            if l.recurrent_synapses.is_some() {
                total_recurrent_words += out_w;
            }

            if out_w > max_out_words {
                max_out_words = out_w;
            }

            if i > 0 {
                let in_w = l.in_features.div_ceil(32);
                if in_w > max_in_words {
                    max_in_words = in_w;
                }
            }

            if l.activation_word_offset.is_some()
                || l.source_layer.is_some()
                || l.residual_source.is_some()
            {
                has_explicit_offsets = true;
            }

            let offset_val = match l.activation_word_offset {
                Some(off) => off,
                None => 0,
            };
            let end_word = offset_val + out_w;
            if end_word > max_explicit_activation_word {
                max_explicit_activation_word = end_word;
            }

            i += 1;
        }

        let max_words = if max_out_words > max_in_words {
            max_out_words
        } else {
            max_in_words
        };

        let total_activation_words = if has_explicit_offsets {
            if max_explicit_activation_word > 2 * max_words {
                max_explicit_activation_word
            } else {
                2 * max_words
            }
        } else {
            2 * max_words
        };

        // 1. Shared accumulators: max_neurons * sizeof(Fix32)
        let acc_bytes = max_neurons * core::mem::size_of::<Fix32>();
        // 2. Activation arena: total_activation_words * sizeof(u32)
        let act_bytes = total_activation_words * core::mem::size_of::<u32>();
        // 3. Persistent membrane potentials: total_neurons * sizeof(Fix32)
        let mem_bytes = total_neurons * core::mem::size_of::<Fix32>();
        // 4. Persistent refractory delays: total_neurons * sizeof(u8) (aligned to 4 bytes)
        let refr_bytes = (total_neurons * core::mem::size_of::<u8>() + 3) & !3;
        // 5. Persistent recurrent spike words: total_recurrent_words * sizeof(u32)
        let rec_bytes = total_recurrent_words * core::mem::size_of::<u32>();

        let total_sram_bytes = acc_bytes + act_bytes + mem_bytes + refr_bytes + rec_bytes;

        MemoryFootprint {
            max_layer_neurons: max_neurons,
            max_layer_words: max_words,
            total_activation_words,
            total_state_neurons: total_neurons,
            total_recurrent_words,
            total_sram_bytes,
        }
    }

    /// Creates a new static model descriptor.
    pub const fn new(layers: &'a [LayerDescriptor<'a>]) -> Self {
        let memory_footprint = Self::compute_footprint(layers);
        Self {
            layers,
            memory_footprint,
        }
    }

    /// Bytes of SRAM the model needs for its dynamic state (`const`, usable to size a buffer).
    #[inline(always)]
    pub const fn sram_bytes(&self) -> usize {
        self.memory_footprint.total_sram_bytes
    }

    /// Number of input features of the first layer.
    #[inline(always)]
    pub fn input_features(&self) -> usize {
        self.layers.first().map_or(0, |l| l.in_features)
    }

    /// Number of output neurons of the last layer.
    #[inline(always)]
    pub fn output_neurons(&self) -> usize {
        self.layers.last().map_or(0, |l| l.out_features)
    }

    /// Number of 32-bit words required to receive output spikes.
    #[inline(always)]
    pub fn output_words(&self) -> usize {
        self.output_neurons().div_ceil(32)
    }

    /// Initializes the dynamic execution context within a provided SRAM buffer (zero allocation).
    #[allow(clippy::cast_ptr_alignment)]
    pub fn init_context<'ctx>(
        &self,
        sram_buffer: &'ctx mut [u8],
    ) -> Result<ExecutionContext<'ctx>, EngineError> {
        let required = self.memory_footprint.total_sram_bytes;
        if sram_buffer.len() < required {
            return Err(EngineError::BufferTooSmall {
                required,
                provided: sram_buffer.len(),
            });
        }
        if !(sram_buffer.as_ptr() as usize).is_multiple_of(4) {
            return Err(EngineError::MisalignedBuffer);
        }

        let max_n = self.memory_footprint.max_layer_neurons;
        let total_act_w = self.memory_footprint.total_activation_words;
        let total_n = self.memory_footprint.total_state_neurons;
        let total_rec_w = self.memory_footprint.total_recurrent_words;

        let (acc_slice, rest) = sram_buffer.split_at_mut(max_n * core::mem::size_of::<Fix32>());
        let (act_slice, rest2) = rest.split_at_mut(total_act_w * core::mem::size_of::<u32>());
        let (mem_slice, rest3) = rest2.split_at_mut(total_n * core::mem::size_of::<Fix32>());
        let (refr_slice, rec_slice) =
            rest3.split_at_mut((total_n * core::mem::size_of::<u8>() + 3) & !3);

        // SAFETY (the four casts below): `sram_buffer` was checked 4-byte aligned (`MisalignedBuffer`) and
        // large enough for `memory_footprint`; each sub-slice starts at a multiple of 4 bytes (all preceding
        // sizes are multiples of 4, the refractory part is rounded up to 4), has exactly `len * size_of::<T>()`
        // bytes, and the slices are disjoint (`split_at_mut`). `Fix32`/`u32` have no invalid bit patterns.
        let accumulators: &mut [Fix32] = unsafe {
            core::slice::from_raw_parts_mut(acc_slice.as_mut_ptr().cast::<Fix32>(), max_n)
        };
        let activation_arena: &mut [u32] = unsafe {
            core::slice::from_raw_parts_mut(act_slice.as_mut_ptr().cast::<u32>(), total_act_w)
        };
        let membrane_potentials: &mut [Fix32] = unsafe {
            core::slice::from_raw_parts_mut(mem_slice.as_mut_ptr().cast::<Fix32>(), total_n)
        };
        let refractory_times: &mut [u8] = &mut refr_slice[..total_n];
        let recurrent_spikes: &mut [u32] = unsafe {
            core::slice::from_raw_parts_mut(rec_slice.as_mut_ptr().cast::<u32>(), total_rec_w)
        };

        // State initialization
        accumulators.fill(Fix32::ZERO);
        activation_arena.fill(0);
        refractory_times.fill(0);
        recurrent_spikes.fill(0);

        // Initialize membrane potentials to their respective reset potentials
        let mut offset = 0;
        for layer in self.layers {
            if layer.pool_config.is_none() {
                let n = layer.out_features;
                membrane_potentials[offset..offset + n].fill(layer.lif_config.reset_potential);
                offset += n;
            }
        }

        Ok(ExecutionContext {
            accumulators,
            activation_arena,
            membrane_potentials,
            refractory_times,
            recurrent_spikes,
            last_output_offset: 0,
        })
    }

    /// Executes a complete inference time step (tick) across all model layers.
    pub fn step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        inputs: &[Fix32],
        out_spikes_words: &mut [u32],
    ) -> Result<(), EngineError> {
        if self.layers.is_empty() {
            return Ok(());
        }
        if inputs.len() < self.input_features() {
            return Err(EngineError::DimensionMismatch);
        }
        if out_spikes_words.len() < self.output_words() {
            return Err(EngineError::DimensionMismatch);
        }

        let max_w = self.memory_footprint.max_layer_words;
        let mut mem_offset = 0;
        let mut rec_offset = 0;
        let mut last_out_offset = 0;
        let mut last_out_words = 0;

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let n = layer.out_features;
            let out_words = n.div_ceil(32);

            let out_offset = match layer.activation_word_offset {
                Some(offset) => offset,
                None => {
                    if l_idx % 2 == 0 {
                        0
                    } else {
                        max_w
                    }
                }
            };

            // Case of a pure 2D Max-Pooling layer
            if let Some(pool_cfg) = &layer.pool_config {
                let in_offset = if let Some(s_idx) = layer.source_layer {
                    let src_l = &self.layers[s_idx];
                    match src_l.activation_word_offset {
                        Some(offset) => offset,
                        None => {
                            if s_idx % 2 == 0 {
                                0
                            } else {
                                max_w
                            }
                        }
                    }
                } else if l_idx > 0 {
                    let prev_idx = l_idx - 1;
                    let prev_l = &self.layers[prev_idx];
                    match prev_l.activation_word_offset {
                        Some(offset) => offset,
                        None => {
                            if prev_idx % 2 == 0 {
                                0
                            } else {
                                max_w
                            }
                        }
                    }
                } else {
                    0
                };

                let in_words_count = layer.in_features.div_ceil(32);
                if in_offset + in_words_count <= out_offset {
                    let (first, second) = ctx.activation_arena.split_at_mut(out_offset);
                    let in_words = &first[in_offset..in_offset + in_words_count];
                    let out_target = &mut second[..out_words];
                    pool_cfg.pool_spikes_words(in_words, out_target);
                } else if out_offset + out_words <= in_offset {
                    let (first, second) = ctx.activation_arena.split_at_mut(in_offset);
                    let out_target = &mut first[out_offset..out_offset + out_words];
                    let in_words = &second[..in_words_count];
                    pool_cfg.pool_spikes_words(in_words, out_target);
                } else {
                    // SAFETY: `accumulators` is 4-byte aligned with `max_n` (>= any layer's word count)
                    // `Fix32` (4 bytes) elements, so reinterpreting its head as `u32` words is in-bounds
                    // and aligned; it is a different field from `activation_arena`, so no aliasing.
                    let temp_in: &mut [u32] = unsafe {
                        core::slice::from_raw_parts_mut(
                            ctx.accumulators.as_mut_ptr().cast::<u32>(),
                            in_words_count,
                        )
                    };
                    temp_in.copy_from_slice(
                        &ctx.activation_arena[in_offset..in_offset + in_words_count],
                    );
                    let out_target = &mut ctx.activation_arena[out_offset..out_offset + out_words];
                    pool_cfg.pool_spikes_words(temp_in, out_target);
                }

                last_out_offset = out_offset;
                last_out_words = out_words;
                continue;
            }

            let acc = &mut ctx.accumulators[..n];
            acc.fill(Fix32::ZERO);

            // 1. Determine input source and perform synaptic propagation (feedforward)
            if l_idx == 0 && layer.source_layer.is_none() {
                layer.synapses.propagate_weighted_inputs(inputs, acc);
            } else {
                let in_offset = if let Some(s_idx) = layer.source_layer {
                    let src_l = &self.layers[s_idx];
                    match src_l.activation_word_offset {
                        Some(offset) => offset,
                        None => {
                            if s_idx % 2 == 0 {
                                0
                            } else {
                                max_w
                            }
                        }
                    }
                } else {
                    let prev_idx = l_idx - 1;
                    let prev_l = &self.layers[prev_idx];
                    match prev_l.activation_word_offset {
                        Some(offset) => offset,
                        None => {
                            if prev_idx % 2 == 0 {
                                0
                            } else {
                                max_w
                            }
                        }
                    }
                };

                let in_words_count = layer.in_features.div_ceil(32);
                let in_words = &ctx.activation_arena[in_offset..in_offset + in_words_count];
                layer.synapses.propagate_spikes_words(in_words, acc);
            }

            // 1b. Optional residual connection (skip connection)
            if let Some(res_idx) = layer.residual_source {
                let res_l = &self.layers[res_idx];
                let res_offset = match res_l.activation_word_offset {
                    Some(offset) => offset,
                    None => {
                        if res_idx % 2 == 0 {
                            0
                        } else {
                            max_w
                        }
                    }
                };
                let res_words_count = res_l.out_features.div_ceil(32);
                let res_words = &ctx.activation_arena[res_offset..res_offset + res_words_count];
                let weight = layer.residual_weight.unwrap_or(Fix32::ONE);

                for (w_idx, &word) in res_words.iter().enumerate() {
                    let mut active = word;
                    while active != 0 {
                        let bit_idx = active.trailing_zeros() as usize;
                        let neuron_idx = w_idx * 32 + bit_idx;
                        if neuron_idx < n {
                            acc[neuron_idx] = acc[neuron_idx].saturating_add(weight);
                        }
                        active &= active - 1;
                    }
                }
            }

            // 1c. Intra-layer recurrent connection (W_rec) from spikes of previous tick (t-1)
            if let Some(rec_syn) = &layer.recurrent_synapses {
                let prev_rec = &ctx.recurrent_spikes[rec_offset..rec_offset + out_words];
                rec_syn.propagate_spikes_words(prev_rec, acc);
            }

            let out_target = &mut ctx.activation_arena[out_offset..out_offset + out_words];
            out_target.fill(0);

            // 3. Dynamic LIF membrane evaluation
            let mem = &mut ctx.membrane_potentials[mem_offset..mem_offset + n];
            let refr = &mut ctx.refractory_times[mem_offset..mem_offset + n];
            let cfg = &layer.lif_config;

            for i in 0..n {
                let b = layer.bias.map_or(Fix32::ZERO, |biases| biases[i]);
                let inp = acc[i] + cfg.bias + b;

                if refr[i] > 0 {
                    refr[i] -= 1;
                    mem[i] = cfg.reset_potential;
                    continue;
                }

                #[cfg(feature = "cortex-m0")]
                let decayed = crate::domain::hardware::fast_decay_m0(
                    mem[i],
                    cfg.decay_factor.to_bits().max(0) as u32,
                );
                #[cfg(not(feature = "cortex-m0"))]
                let decayed = mem[i] * cfg.decay_factor;

                let v = decayed + inp;

                if v >= cfg.threshold {
                    mem[i] = match cfg.reset_mode {
                        ResetMode::Hard => cfg.reset_potential,
                        ResetMode::Subtract => v - cfg.threshold,
                    };
                    refr[i] = cfg.refractory_period.min(255) as u8;

                    let w_idx = i / 32;
                    let bit_idx = i % 32;
                    out_target[w_idx] |= 1 << bit_idx;
                } else {
                    mem[i] = v;
                }
            }

            // 4. Memorize spikes for recurrence in the next time step
            if layer.recurrent_synapses.is_some() {
                ctx.recurrent_spikes[rec_offset..rec_offset + out_words]
                    .copy_from_slice(out_target);
                rec_offset += out_words;
            }

            last_out_offset = out_offset;
            last_out_words = out_words;
            mem_offset += n;
        }

        ctx.last_output_offset = last_out_offset;

        // Copy final layer spikes into output buffer
        let final_words = self.output_words().min(last_out_words);
        out_spikes_words[..final_words]
            .copy_from_slice(&ctx.activation_arena[last_out_offset..last_out_offset + final_words]);

        Ok(())
    }

    /// Resets dynamic state (membrane potentials, refractory delays, and recurrent spikes).
    pub fn reset(&self, ctx: &mut ExecutionContext<'_>) {
        ctx.accumulators.fill(Fix32::ZERO);
        ctx.activation_arena.fill(0);
        ctx.refractory_times.fill(0);
        ctx.recurrent_spikes.fill(0);

        let mut offset = 0;
        for layer in self.layers {
            if layer.pool_config.is_none() {
                let n = layer.out_features;
                ctx.membrane_potentials[offset..offset + n].fill(layer.lif_config.reset_potential);
                offset += n;
            }
        }
        ctx.last_output_offset = 0;
    }
}

/// Dynamic execution context allocated in SRAM (zero heap allocation).
#[derive(Debug)]
pub struct ExecutionContext<'a> {
    pub accumulators: &'a mut [Fix32],
    pub activation_arena: &'a mut [u32],
    pub membrane_potentials: &'a mut [Fix32],
    pub refractory_times: &'a mut [u8],
    pub recurrent_spikes: &'a mut [u32],
    pub last_output_offset: usize,
}

impl ExecutionContext<'_> {
    /// Returns the slice of spike words from the last executed layer.
    #[inline(always)]
    pub fn last_output_spikes(&self, output_words: usize) -> &[u32] {
        let start = self.last_output_offset;
        &self.activation_arena[start..start + output_words]
    }

    /// Resets dynamic state (membrane potentials, accumulators, refractory periods, and recurrent spikes).
    pub fn reset(&mut self, model: &ModelDescriptor<'_>) {
        model.reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_model_descriptor_and_context() {
        // Definition of 2 layers: 4 inputs -> 3 neurons -> 2 neurons
        static L1_ROW_PTRS: [u16; 5] = [0, 1, 2, 3, 4];
        static L1_TARGETS: [u16; 4] = [0, 1, 2, 0];
        static L1_WEIGHTS: [Fix32; 4] = [
            Fix32::from_bits(98304),  // 1.5
            Fix32::from_bits(52429),  // 0.8
            Fix32::from_bits(131072), // 2.0
            Fix32::from_bits(32768),  // 0.5
        ];
        let l1_synapses = RomCsrSynapses::new(&L1_ROW_PTRS, &L1_TARGETS, &L1_WEIGHTS);
        let l1_cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
        let l1_desc = LayerDescriptor::new(4, 3, l1_synapses, l1_cfg, None);

        static L2_ROW_PTRS: [u16; 4] = [0, 1, 2, 3];
        static L2_TARGETS: [u16; 3] = [0, 1, 0];
        static L2_WEIGHTS: [Fix32; 3] = [
            Fix32::from_bits(78643), // 1.2
            Fix32::from_bits(98304), // 1.5
            Fix32::from_bits(45875), // 0.7
        ];
        let l2_synapses = RomCsrSynapses::new(&L2_ROW_PTRS, &L2_TARGETS, &L2_WEIGHTS);
        let l2_cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.85), Fix32::ZERO, 1);
        let l2_desc = LayerDescriptor::new(3, 2, l2_synapses, l2_cfg, None);

        let layers = [l1_desc, l2_desc];
        let model = ModelDescriptor::new(&layers);

        assert_eq!(model.input_features(), 4);
        assert_eq!(model.output_neurons(), 2);
        assert_eq!(model.output_words(), 1);

        // Allocation of raw aligned SRAM arena
        let footprint = model.memory_footprint;
        assert!(footprint.total_sram_bytes > 0);

        let mut sram_buffer = [0u8; 256];
        let mut ctx = model.init_context(&mut sram_buffer).expect("Buffer valid");

        let inputs = [
            Fix32::from_num(2.0),
            Fix32::ZERO,
            Fix32::from_num(1.5),
            Fix32::ZERO,
        ];
        let mut out_spikes = [0u32; 1];

        // Inference of one time step
        model
            .step(&mut ctx, &inputs, &mut out_spikes)
            .expect("Step succeeded");
        assert!(out_spikes[0] > 0);

        // Reset
        model.reset(&mut ctx);
        assert_eq!(ctx.refractory_times[0], 0);
    }

    #[test]
    fn test_quantized_i8_model_descriptor() {
        // Test quantized INT8 model with Fix32 scale factor
        static ROW_PTRS: [u16; 3] = [0, 1, 2];
        static TARGETS: [u16; 2] = [0, 1];
        static WEIGHTS_I8: [i8; 2] = [100, 120];
        let scale = Fix32::from_num(0.02);

        let q_synapses = QuantizedRomCsrSynapses::new(&ROW_PTRS, &TARGETS, &WEIGHTS_I8, scale);
        let cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
        let layer = LayerDescriptor::new_quantized_i8(2, 2, q_synapses, cfg, None);

        let layers = [layer];
        let model = ModelDescriptor::new(&layers);

        let mut sram = [0u8; 128];
        let mut ctx = model.init_context(&mut sram).expect("Context valid");

        let inputs = [Fix32::ONE, Fix32::ONE];
        let mut out_spikes = [0u32; 1];
        model
            .step(&mut ctx, &inputs, &mut out_spikes)
            .expect("Step success");
    }

    #[test]
    fn test_conv1d_model_descriptor() {
        // Conv1D kernel: 1 in_channel, 1 out_channel, kernel_size 2, stride 1, in_len 3 -> out_len 2
        static KERNEL: [Fix32; 2] = [Fix32::ONE, Fix32::from_bits(32768)]; // 1.0 and 0.5
        let conv = Conv1dSynapses::new(&KERNEL, 1, 1, 2, 1, 3);
        let cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
        let layer = LayerDescriptor::new_conv1d(3, 2, conv, cfg, None);

        let layers = [layer];
        let model = ModelDescriptor::new(&layers);

        let mut sram = [0u8; 128];
        let mut ctx = model.init_context(&mut sram).expect("Context valid");

        let inputs = [Fix32::from_num(1.5), Fix32::ONE, Fix32::ZERO];
        let mut out_spikes = [0u32; 1];
        model
            .step(&mut ctx, &inputs, &mut out_spikes)
            .expect("Conv1D step success");
    }

    #[test]
    fn test_conv2d_model_descriptor() {
        // Conv2D kernel: 1 in_channel, 1 out_channel, kernel 1x1, 2x2 image -> 2x2 output = 4 neurons
        static KERNEL_2D: [Fix32; 1] = [Fix32::from_bits(131072)]; // 2.0
        let conv = Conv2dSynapses::new(&KERNEL_2D, 1, 1, 1, 1, 1, 1, 2, 2);
        let cfg = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
        let layer = LayerDescriptor::new_conv2d(4, 4, conv, cfg, None);

        let layers = [layer];
        let model = ModelDescriptor::new(&layers);

        let mut sram = [0u8; 128];
        let mut ctx = model.init_context(&mut sram).expect("Context valid");

        let inputs = [Fix32::ONE, Fix32::ZERO, Fix32::ONE, Fix32::ZERO];
        let mut out_spikes = [0u32; 1];
        model
            .step(&mut ctx, &inputs, &mut out_spikes)
            .expect("Conv2D step success");
        assert!(out_spikes[0] > 0);
    }

    #[test]
    fn test_dag_residual_skip_connection_model_descriptor() {
        // 3-layer DAG network with residual skip connection
        // L0: 2 in -> 2 out
        // L1: 2 in -> 2 out (takes L0)
        // L2: 2 in -> 2 out (takes L0 directly, with residual connection from L1!)
        static L0_ROWS: [u16; 3] = [0, 1, 2];
        static L0_TGTS: [u16; 2] = [0, 1];
        static L0_W: [Fix32; 2] = [Fix32::from_bits(98304), Fix32::from_bits(98304)]; // 1.5
        let syn0 = RomCsrSynapses::new(&L0_ROWS, &L0_TGTS, &L0_W);
        let cfg0 = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
        let l0 = LayerDescriptor::new(2, 2, syn0, cfg0, None).with_activation_offset(0);

        static L1_ROWS: [u16; 3] = [0, 1, 2];
        static L1_TGTS: [u16; 2] = [0, 1];
        static L1_W: [Fix32; 2] = [Fix32::from_bits(98304), Fix32::from_bits(98304)]; // 1.5
        let syn1 = RomCsrSynapses::new(&L1_ROWS, &L1_TGTS, &L1_W);
        let cfg1 = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
        let l1 = LayerDescriptor::new(2, 2, syn1, cfg1, None)
            .with_source_layer(0)
            .with_activation_offset(1);

        static L2_ROWS: [u16; 3] = [0, 1, 2];
        static L2_TGTS: [u16; 2] = [0, 1];
        static L2_W: [Fix32; 2] = [Fix32::from_bits(98304), Fix32::from_bits(98304)]; // 1.5
        let syn2 = RomCsrSynapses::new(&L2_ROWS, &L2_TGTS, &L2_W);
        let cfg2 = LifConfig::new(Fix32::ONE, Fix32::from_num(0.9), Fix32::ZERO, 1);
        let l2 = LayerDescriptor::new(2, 2, syn2, cfg2, None)
            .with_source_layer(0)
            .with_residual(1, Some(Fix32::from_bits(32768))) // 0.5
            .with_activation_offset(2);

        let layers = [l0, l1, l2];
        let model = ModelDescriptor::new(&layers);

        let mut sram = [0u8; 512];
        let mut ctx = model.init_context(&mut sram).expect("Context valid");

        let inputs = [Fix32::from_num(2.0), Fix32::from_num(2.0)];
        let mut out_spikes = [0u32; 1];
        model
            .step(&mut ctx, &inputs, &mut out_spikes)
            .expect("DAG step success");
        assert!(out_spikes[0] > 0);
    }

    #[test]
    fn test_recurrent_layer_descriptor_memory_retention() {
        // Recurrent layer of 2 neurons:
        // External input -> N0 only.
        // N0 has recurrent synapse W_rec to N1 with weight 2.0.
        // Tick 0: Input [2.0, 0.0] -> N0 emits spike (out = [1, 0]).
        // Tick 1: Input [0.0, 0.0] -> N1 receives recurrent spike from N0 and emits spike!
        static IN_ROWS: [u16; 3] = [0, 1, 1];
        static IN_TGTS: [u16; 1] = [0];
        static IN_W: [Fix32; 1] = [Fix32::from_bits(131072)]; // 2.0
        let in_syn = RomCsrSynapses::new(&IN_ROWS, &IN_TGTS, &IN_W);

        static REC_ROWS: [u16; 3] = [0, 1, 1];
        static REC_TGTS: [u16; 1] = [1];
        static REC_W: [Fix32; 1] = [Fix32::from_bits(131072)]; // 2.0 to N1
        let rec_syn = RomCsrSynapses::new(&REC_ROWS, &REC_TGTS, &REC_W);

        let cfg = LifConfig::new(Fix32::ONE, Fix32::ZERO, Fix32::ZERO, 0);
        let layer = LayerDescriptor::new(2, 2, in_syn, cfg, None)
            .with_recurrent(SynapseMatrixKind::Fix32(rec_syn));

        let layers = [layer];
        let model = ModelDescriptor::new(&layers);

        assert_eq!(model.memory_footprint.total_recurrent_words, 1);

        let mut sram = [0u8; 256];
        let mut ctx = model.init_context(&mut sram).expect("Valid SRAM");

        // Tick 0: stimulate N0
        let mut out_spikes = [0u32; 1];
        model
            .step(
                &mut ctx,
                &[Fix32::from_num(2.0), Fix32::ZERO],
                &mut out_spikes,
            )
            .expect("Tick 0 success");
        assert_eq!(out_spikes[0] & 1, 1, "N0 should fire at tick 0");
        assert_eq!(out_spikes[0] & 2, 0, "N1 should not fire at tick 0");

        // Tick 1: zero input, recurrent propagation
        model
            .step(&mut ctx, &[Fix32::ZERO, Fix32::ZERO], &mut out_spikes)
            .expect("Tick 1 success");
        assert_eq!(
            out_spikes[0] & 2,
            2,
            "N1 should fire at tick 1 due to recurrence from N0!"
        );

        // Reset
        ctx.reset(&model);
        assert_eq!(ctx.recurrent_spikes[0], 0);
    }

    #[test]
    fn test_max_pool_2d_model_descriptor() {
        // Conv2D -> MaxPool2D pipeline:
        // 1 in_ch, 1 out_ch, 4x4 image (16 pixels).
        // Conv2D 1x1 identity (weight 2.0) -> 16 LIF neurons.
        // MaxPool2D 2x2 stride 2x2 -> reduced 2x2 image = 4 output neurons.
        static K_IDENT: [Fix32; 1] = [Fix32::from_bits(131072)]; // 2.0
        let conv = Conv2dSynapses::new(&K_IDENT, 1, 1, 1, 1, 1, 1, 4, 4);
        let cfg = LifConfig::new(Fix32::ONE, Fix32::ZERO, Fix32::ZERO, 0);
        let l0 = LayerDescriptor::new_conv2d(16, 16, conv, cfg, None).with_activation_offset(0);

        let pool_cfg = MaxPool2dConfig::new(1, 4, 4, 2, 2, 2, 2);
        let l1 = LayerDescriptor::new_max_pool_2d(pool_cfg)
            .with_source_layer(0)
            .with_activation_offset(1);

        let layers = [l0, l1];
        let model = ModelDescriptor::new(&layers);

        assert_eq!(model.input_features(), 16);
        assert_eq!(model.output_neurons(), 4);
        assert_eq!(model.output_words(), 1);

        let mut sram = [0u8; 512];
        let mut ctx = model.init_context(&mut sram).expect("Context valid");

        // Stimulate only pixel (row 0, col 1) -> window (0..2, 0..2)
        // This pixel is at index 1. Pooling should activate top-left quadrant (output bit 0).
        let mut inputs = [Fix32::ZERO; 16];
        inputs[1] = Fix32::from_num(2.0);

        let mut out_spikes = [0u32; 1];
        model
            .step(&mut ctx, &inputs, &mut out_spikes)
            .expect("Step success");
        assert_eq!(
            out_spikes[0], 1,
            "Top-left quadrant (bit 0) should be activated by max-pooling"
        );
    }
}
