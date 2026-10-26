/**
 * @file snn_core.h
 * @brief 🧠 snn_core Industrial C FFI Interface for Bare-Metal Microcontrollers & RTOS
 *
 * Ultra-low-power neuromorphic spiking neural network engine in pure no_std Rust.
 * Compatible with ARM Cortex-M0+/M4/M7/M33, RISC-V, FreeRTOS, Zephyr RTOS, STM32Cube, and ESP-IDF.
 */

#ifndef SNN_CORE_H
#define SNN_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief AER (Address-Event Representation) 1D event structure.
 */
typedef struct {
    uint16_t address;   /**< Source channel/neuron address */
    uint16_t timestamp; /**< Hardware microsecond timestamp */
} SnnAerSpike1d;

/**
 * @brief AER 2D DVS (Dynamic Vision Sensor) bipolar event structure.
 */
typedef struct {
    uint16_t x;         /**< Spatial X coordinate */
    uint16_t y;         /**< Spatial Y coordinate */
    bool polarity;      /**< Polarity (true: ON spike, false: OFF spike) */
    uint32_t timestamp; /**< Hardware microsecond timestamp */
} SnnAerSpike2d;

/**
 * @brief Converts a standard IEEE-754 float into a 32-bit Q16.16 fixed-point integer (Fix32).
 */
int32_t snn_fix32_from_float(float val);

/**
 * @brief Converts a 32-bit Q16.16 fixed-point integer (Fix32) into an IEEE-754 float.
 */
float snn_fix32_to_float(int32_t bits);

/**
 * @brief Decodes a stream of 1D AER events directly into a compact 1-bit/neuron bitset buffer.
 *
 * @param events Pointer to array of SnnAerSpike1d events.
 * @param event_count Number of events in the array.
 * @param out_words Pointer to output 32-bit word array (SpikeBitset).
 * @param words_count Number of 32-bit words in out_words.
 * @return 0 on success, negative error code on invalid pointers or buffer overflow.
 */
int32_t snn_aer_decode_1d(
    const SnnAerSpike1d* events,
    size_t event_count,
    uint32_t* out_words,
    size_t words_count
);

/**
 * @brief Decodes a stream of 2D DVS bipolar vision events directly into a compact bitset buffer.
 *
 * @param events Pointer to array of SnnAerSpike2d events.
 * @param event_count Number of events in the array.
 * @param width Spatial width of the neuromorphic sensor array.
 * @param height Spatial height of the neuromorphic sensor array.
 * @param out_words Pointer to output 32-bit word array.
 * @param words_count Number of 32-bit words in out_words.
 * @return 0 on success, negative error code on invalid parameters.
 */
int32_t snn_aer_decode_2d_bipolar(
    const SnnAerSpike2d* events,
    size_t event_count,
    size_t width,
    size_t height,
    uint32_t* out_words,
    size_t words_count
);

/**
 * @brief Queries whether a specific neuron fired in the bitset (zero-allocation branchless query).
 *
 * @param words Pointer to bitset 32-bit word array.
 * @param words_count Number of words in the bitset.
 * @param neuron_index Linear index of the neuron.
 * @return true if the neuron fired, false otherwise.
 */
bool snn_bitset_get_spike(const uint32_t* words, size_t words_count, size_t neuron_index);

/**
 * @brief Counts the total number of active spikes across the bitset using native hardware POPCNT.
 *
 * @param words Pointer to bitset 32-bit word array.
 * @param words_count Number of words in the bitset.
 * @return Total number of active spikes (Hamming weight).
 */
uint32_t snn_bitset_count_spikes(const uint32_t* words, size_t words_count);

/**
 * @brief Configuration parameters for a population of Leaky Integrate-and-Fire (LIF) neurons.
 */
typedef struct {
    int32_t threshold;          /**< Action potential threshold V_th (Q16.16 Fix32) */
    int32_t decay_factor;       /**< Membrane potential decay factor exp(-dt/tau) (Q16.16 Fix32) */
    int32_t reset_potential;    /**< Post-spike resting potential V_reset (Q16.16 Fix32) */
    uint16_t refractory_period; /**< Absolute refractory duration in discrete ticks */
    int32_t bias;               /**< Constant tonic current injected every tick (Q16.16 Fix32) */
    uint8_t reset_mode;         /**< Reset mode (0: Hard reset V=V_reset, 1: Soft reset V=V-V_th) */
} SnnLifConfig;

/**
 * @brief Immutable Compressed Sparse Row (CSR) synaptic matrix stored in Flash ROM (.rodata).
 */
typedef struct {
    const uint16_t* row_ptrs;   /**< CSR row offsets array of length (in_features + 1) */
    const uint16_t* targets;    /**< Target post-synaptic neuron indices array of length total_synapses */
    const int32_t* weights;     /**< Synaptic weights array in Q16.16 Fix32 of length total_synapses */
    size_t in_features;         /**< Number of input pre-synaptic channels */
    size_t out_features;        /**< Number of output post-synaptic neurons */
    size_t total_synapses;      /**< Total non-zero synaptic connections */
} SnnCsrMatrix;

/**
 * @brief Static network descriptor representing an entire SNN layer or reservoir in Flash ROM.
 */
typedef struct {
    size_t num_neurons;                 /**< Number of neurons in the population */
    size_t in_features;                 /**< Number of input channels */
    SnnLifConfig config;                /**< LIF biophysical dynamics parameters */
    SnnCsrMatrix input_synapses;        /**< Input projection synaptic matrix */
    SnnCsrMatrix recurrent_synapses;    /**< Recurrent feedback synaptic matrix (row_ptrs=NULL if none) */
    const int32_t* bias;                /**< Per-neuron static bias in ROM (NULL if none) */
} SnnNetworkDesc;

/**
 * @brief Mutable dynamic state of the SNN population residing in SRAM (zero heap allocation).
 */
typedef struct {
    int32_t* membrane_potentials;   /**< Array of instantaneous membrane potentials V_mem */
    int32_t* input_accumulators;    /**< Array of synaptic input currents I_in */
    uint8_t* refractory_times;      /**< Array of remaining refractory periods */
    uint32_t* last_spikes;          /**< Bitset array of active spikes emitted at previous tick */
    size_t num_neurons;             /**< Total number of neurons */
    size_t words_count;             /**< Number of 32-bit words in last_spikes bitset */
} SnnNetworkState;

/**
 * @brief Macro to statically calculate required SRAM state arena size in bytes at compile-time.
 */
#define SNN_NETWORK_STATE_SIZE(n) \
    (((n) * 4) + ((n) * 4) + ((((n) + 3) & ~3)) + ((((n) + 31) / 32) * 4))

/**
 * @brief Calculates exact SRAM state arena size in bytes needed for a population of num_neurons.
 */
size_t snn_network_state_size(size_t num_neurons);

/**
 * @brief Initializes the network's dynamic mutable state inside a user-provided SRAM buffer.
 *
 * @param desc Pointer to immutable network descriptor in Flash ROM.
 * @param arena Pointer to 4-byte aligned SRAM buffer.
 * @param arena_size Size of arena buffer in bytes (must be >= snn_network_state_size(desc->num_neurons)).
 * @param out_state Pointer to SnnNetworkState structure populated upon success.
 * @return 0 on success, negative error code on invalid parameters or insufficient buffer.
 */
int32_t snn_network_init_state(
    const SnnNetworkDesc* desc,
    uint8_t* arena,
    size_t arena_size,
    SnnNetworkState* out_state
);

/**
 * @brief Executes one discrete simulation tick of the spiking neural network.
 *
 * @param desc Pointer to immutable network descriptor.
 * @param state Pointer to mutable dynamic state in SRAM.
 * @param inputs Pointer to array of weighted input values in Q16.16 Fix32 (length desc->in_features).
 * @param inputs_len Number of elements in inputs array.
 * @param out_spikes_words Pointer to output bitset buffer receiving fired spikes.
 * @param words_count Number of 32-bit words in out_spikes_words buffer (must be >= state->words_count).
 * @return 0 on success, negative error code on failure.
 */
int32_t snn_network_step(
    const SnnNetworkDesc* desc,
    SnnNetworkState* state,
    const int32_t* inputs,
    size_t inputs_len,
    uint32_t* out_spikes_words,
    size_t words_count
);

/**
 * @brief Resets all dynamic neuron membrane potentials and refractory states to resting values.
 *
 * @param desc Pointer to immutable network descriptor.
 * @param state Pointer to mutable dynamic state in SRAM.
 * @return 0 on success, negative error code on failure.
 */
int32_t snn_network_reset(
    const SnnNetworkDesc* desc,
    SnnNetworkState* state
);

/**
 * @brief Dense linear classifier readout layer computing class logits from emitted spike bitset.
 *
 * @param weights_q16 Row-major weight matrix of shape (num_classes, num_neurons) in Q16.16 Fix32.
 * @param bias_q16 Optional bias vector of length num_classes in Q16.16 Fix32 (NULL if none).
 * @param spike_words Pointer to input spike bitset words emitted by the spiking population.
 * @param num_neurons Total number of spiking neurons driving the readout.
 * @param num_classes Number of output target classes.
 * @param out_logits_q16 Output array of length num_classes receiving computed logits in Q16.16 Fix32.
 * @return 0 on success, negative error code on failure.
 */
int32_t snn_readout_dense(
    const int32_t* weights_q16,
    const int32_t* bias_q16,
    const uint32_t* spike_words,
    size_t num_neurons,
    size_t num_classes,
    int32_t* out_logits_q16
);

/**
 * @brief Layer type within a multilayer spiking neural network.
 */
typedef enum {
    SNN_LAYER_FEEDFORWARD    = 0, /**< Standard Fix32 CSR feedforward layer */
    SNN_LAYER_QUANTIZED_INT8 = 1, /**< Quantized INT8 weights with Fix32 common scale */
    SNN_LAYER_CONV2D         = 2, /**< Event-driven 2D convolution */
    SNN_LAYER_MAX_POOL_2D    = 3  /**< Spatial max-pooling (bitwise OR) */
} SnnLayerType;

/** @brief 2D convolution layer descriptor (kernel weights in Q16.16, in Flash ROM). */
typedef struct {
    const int32_t* kernel;
    size_t in_channels;
    size_t out_channels;
    size_t kernel_h;
    size_t kernel_w;
    size_t stride_h;
    size_t stride_w;
    size_t padding_h;
    size_t padding_w;
    size_t in_h;
    size_t in_w;
    size_t out_h;
    size_t out_w;
} SnnConv2dDesc;

/** @brief Spatial 2D max-pooling descriptor. */
typedef struct {
    size_t channels;
    size_t in_h;
    size_t in_w;
    size_t kernel_h;
    size_t kernel_w;
    size_t stride_h;
    size_t stride_w;
    size_t out_h;
    size_t out_w;
} SnnMaxPool2dDesc;

/**
 * @brief Descriptor of a single layer within a deep spiking network.
 */
typedef struct {
    SnnLayerType layer_type;     /**< Type of layer (Feedforward / Quantized INT8) */
    size_t in_features;          /**< Number of input connections/neurons */
    size_t num_neurons;          /**< Number of neurons in this layer */
    SnnLifConfig config;         /**< Biophysical LIF dynamics configuration */
    SnnCsrMatrix synapses;       /**< Sparse synaptic connectivity matrix in Flash ROM */
    SnnConv2dDesc conv2d;        /**< Used when layer_type == SNN_LAYER_CONV2D */
    SnnMaxPool2dDesc pool2d;     /**< Used when layer_type == SNN_LAYER_MAX_POOL_2D */
    const SnnCsrMatrix* recurrent_synapses; /**< Optional recurrent CSR synapses (NULL if none) */
    const int32_t* bias;         /**< Optional bias vector in Q16.16 (NULL if none) */
    int32_t quant_scale;         /**< Fix32 common scale factor (if QuantizedInt8) */
} SnnLayerDesc;

/**
 * @brief Complete descriptor for a deep multilayer network executing on ActivationArena.
 */
typedef struct {
    const SnnLayerDesc* layers;  /**< Pointer to array of layer descriptors in Flash ROM */
    size_t num_layers;           /**< Total number of layers in the chain */
    size_t in_features;          /**< Input dimensionality of the first layer */
} SnnDeepNetworkDesc;

/**
 * @brief Dynamic mutable state of a deep network residing in user-provided SRAM arena.
 */
typedef struct {
    int32_t* membrane_potentials; /**< Pointer to all layer membrane potentials in SRAM */
    uint8_t* refractory_times;    /**< Pointer to refractory counters in SRAM */
    int32_t* arena_accumulators;  /**< Shared synaptic accumulator buffer (size max_neurons) */
    uint32_t* ping_words;         /**< Ping spike bitset buffer (size max_words) */
    uint32_t* pong_words;         /**< Pong spike bitset buffer (size max_words) */
    uint32_t* recurrent_spikes;   /**< Previous-step spikes of recurrent layers */
    size_t total_neurons;         /**< Total persistent neurons across all layers */
    size_t max_neurons;           /**< Largest single layer neuron count */
    size_t max_words;             /**< Largest single layer word count */
    size_t total_recurrent_words; /**< Total words of recurrent_spikes */
} SnnDeepNetworkState;

/**
 * @brief Calculates exact SRAM state arena size in bytes needed for a deep multilayer network.
 *
 * @param desc Pointer to deep network descriptor.
 * @return Size in bytes required to allocate the full network state.
 */
size_t snn_deep_network_state_size(const SnnDeepNetworkDesc* desc);

/**
 * @brief Initializes the deep network dynamic mutable state inside user SRAM arena.
 *
 * @param desc Pointer to deep network descriptor in Flash ROM.
 * @param arena Pointer to 4-byte aligned SRAM buffer.
 * @param arena_size Size of arena buffer in bytes.
 * @param out_state Pointer to state struct populated upon return.
 * @return 0 on success, negative error code on invalid parameters or insufficient buffer.
 */
int32_t snn_deep_network_init_state(
    const SnnDeepNetworkDesc* desc,
    uint8_t* arena,
    size_t arena_size,
    SnnDeepNetworkState* out_state
);

/**
 * @brief Executes one discrete simulation tick across the entire deep network via zero-alloc ping-pong.
 *
 * @param desc Pointer to deep network descriptor.
 * @param state Pointer to mutable dynamic state in SRAM.
 * @param inputs Pointer to external input vector in Q16.16 Fix32 for Layer 0.
 * @param inputs_len Number of elements in inputs array.
 * @param out_spikes_words Pointer to output bitset buffer receiving last layer spikes.
 * @param words_count Number of 32-bit words in out_spikes_words buffer.
 * @return 0 on success, negative error code on failure.
 */
int32_t snn_deep_network_step(
    const SnnDeepNetworkDesc* desc,
    SnnDeepNetworkState* state,
    const int32_t* inputs,
    size_t inputs_len,
    uint32_t* out_spikes_words,
    size_t words_count
);

/**
 * @brief Resets all membrane potentials and refractory states across all layers.
 *
 * @param desc Pointer to deep network descriptor.
 * @param state Pointer to mutable dynamic state.
 * @return 0 on success, negative error code on failure.
 */
int32_t snn_deep_network_reset(
    const SnnDeepNetworkDesc* desc,
    SnnDeepNetworkState* state
);

/**
 * @brief Propagates active spikes through a 2D spatial convolution in contiguous HWC layout without divisions.
 *
 * @param kernel_weights Pointer to 4D kernel weights in Q16.16 Fix32 (in_c * out_c * kh * kw).
 * @param in_channels Number of input channels.
 * @param out_channels Number of output channels.
 * @param kernel_h Spatial kernel height.
 * @param kernel_w Spatial kernel width.
 * @param stride_h Vertical convolution stride.
 * @param stride_w Horizontal convolution stride.
 * @param padding_h Vertical zero-padding.
 * @param padding_w Horizontal zero-padding.
 * @param in_h Spatial input height.
 * @param in_w Spatial input width.
 * @param active_spikes Pointer to active input linear spike addresses.
 * @param active_spikes_count Number of active spikes.
 * @param accumulators Output accumulators in Q16.16 Fix32.
 * @param accumulators_len Length of output accumulators buffer (must be >= out_h * out_w * out_channels).
 * @return 0 on success, negative error code on failure.
 */
int32_t snn_conv2d_propagate_hwc(
    const int32_t* kernel_weights,
    size_t in_channels,
    size_t out_channels,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t padding_h,
    size_t padding_w,
    size_t in_h,
    size_t in_w,
    const uint32_t* active_spikes,
    size_t active_spikes_count,
    int32_t* accumulators,
    size_t accumulators_len
);

/* ============================================================================
 * BPTN (Bit-Parallel Temporal Neural Network) Word-RAM C FFI API
 * ============================================================================ */

/**
 * @brief Pushes a binary spike into a 64-bit temporal framing buffer (shift-register).
 *
 * @param buffer Pointer to 64-bit framing buffer state.
 * @param spike true if an input event/spike occurred at current tick, false otherwise.
 * @return Updated 64-bit history window (LSB = current tick).
 */
uint64_t snn_bptn_framing_push(uint64_t* buffer, bool spike);

/**
 * @brief Evaluates one tick of a synchronous bistable latch cell (R-BPTN Bistable).
 *
 * Provides infinite memory retention without drift or membrane decay.
 *
 * @param latch_state Pointer to 64-bit state history (LSB = active state).
 * @param mode Priority mode (0: ResetDominant, 1: SetDominant).
 * @param set Set trigger signal.
 * @param reset Reset trigger signal.
 * @return New active state (true: 1, false: 0).
 */
bool snn_bptn_latch_step(uint64_t* latch_state, int32_t mode, bool set, bool reset);

/**
 * @brief Evaluates one tick of a synchronous parity inversion T-Latch.
 *
 * Computes non-linear temporal parity: q(t) = q(t-1) ^ toggle.
 *
 * @param toggle_state Pointer to 64-bit state history.
 * @param toggle Toggle trigger signal.
 * @return New active state.
 */
bool snn_bptn_toggle_step(uint64_t* toggle_state, bool toggle);

/**
 * @brief Evaluates a single vectorized Word-RAM BPTN neuron across M input channels.
 *
 * @param histories Pointer to array of M 64-bit history words.
 * @param kernels Pointer to array of M 64-bit kernel bitmasks.
 * @param polarities Pointer to array of M 64-bit polarity bitmasks (XOR invert).
 * @param channels Number of input channels M.
 * @param threshold Coincidence threshold theta.
 * @return true if neuron fired (potential >= threshold), false otherwise.
 */
bool snn_bptn_neuron_eval(
    const uint64_t* histories,
    const uint64_t* kernels,
    const uint64_t* polarities,
    size_t channels,
    uint32_t threshold
);

/**
 * @brief Evaluates an entire layer of up to 64 BPTN neurons receiving M 64-bit channels.
 *
 * @param histories Pointer to array of M 64-bit input history words.
 * @param channels Number of input channels M.
 * @param kernels Pointer to array of (num_neurons * channels) 64-bit kernel words.
 * @param polarities Pointer to array of (num_neurons * channels) 64-bit polarity words.
 * @param thresholds Pointer to array of num_neurons threshold values.
 * @param num_neurons Number of neurons in the layer (must be <= 64).
 * @param out_mask Pointer to uint64_t receiving the emitted spikes mask (1 bit per neuron).
 * @return 0 on success, negative error code on invalid parameters.
 */
int32_t snn_bptn_layer_eval(
    const uint64_t* histories,
    size_t channels,
    const uint64_t* kernels,
    const uint64_t* polarities,
    const uint32_t* thresholds,
    size_t num_neurons,
    uint64_t* out_mask
);

/**
 * @brief Decodes the predicted class from a 64-bit spike mask using hardware CTZ (trailing zeros).
 *
 * Completely eliminates multiplications during classification readout.
 *
 * @param weights_q16 Pointer to dense weights array (num_classes * num_features) in Q16.16 Fix32.
 * @param bias_q16 Pointer to class biases in Q16.16 Fix32 (NULL if no bias).
 * @param spikes_mask 64-bit spike mask emitted by the BPTN layer.
 * @param num_features Number of features (neurons) in the layer.
 * @param num_classes Number of classification target classes.
 * @return Best predicted class index, or negative error code on invalid parameters.
 */
int32_t snn_readout_predict_spikes_mask(
    const int32_t* weights_q16,
    const int32_t* bias_q16,
    uint64_t spikes_mask,
    size_t num_features,
    size_t num_classes
);

/** @brief Fix32 (Q16.16) saturating multiplication, bit-exact with the Rust runtime. */
int32_t snn_fix32_mul(int32_t a, int32_t b);
/** @brief Saturating 32-bit addition. */
int32_t snn_saturating_add_i32(int32_t a, int32_t b);
/** @brief Saturating 32-bit subtraction. */
int32_t snn_saturating_sub_i32(int32_t a, int32_t b);
/** @brief Pade (2,2) approximation of exp(-x); input and output in Q16.16. */
int32_t snn_pade_exp_neg(int32_t x_q16);

/** @brief Temporal morphological erosion of a 64-tick spike history. */
uint64_t snn_bptn_erode_64(uint64_t x, uint8_t kappa);
/** @brief Temporal morphological opening (erosion then dilation) of a 64-tick spike history. */
uint64_t snn_bptn_open_filter_64(uint64_t x, uint8_t kappa);
/** @brief SWAR 8x8 majority pooling, one output bit per byte lane. */
uint8_t snn_bptn_majority_pool_8x8(uint64_t x, uint8_t threshold);
/** @brief 64-lane parallel Smith counter step; writes next (c1, c0) planes. */
void snn_bptn_parallel_smith_step_u64(
    uint64_t c1, uint64_t c0, uint64_t inc, uint64_t dec, uint64_t* out_c1, uint64_t* out_c0
);
/** @brief 2-bit hysteresis latch step (state 0..3, active when >= 2). */
bool snn_bptn_hysteresis_step(uint8_t* state, bool set, bool reset);
/** @brief Robust toggle step: a trigger must hold 2 consecutive ticks to flip the state. */
bool snn_bptn_robust_toggle_step(uint64_t* history, bool* prev_trigger, bool trigger);

/**
 * @brief Validates a versioned model blob (.snnb) and builds a deep-network descriptor pointing into it.
 *
 * The blob must be 4-byte aligned, little-endian host, and stay valid and unmodified while the
 * descriptor is used. Supports Fix32 and INT8 CSR layers.
 *
 * @return 0 on success; a positive error code when the blob is refused
 *         (1 too short, 2 bad magic, 3 unsupported major version, 4 unknown flags, 5 length mismatch,
 *          6 CRC mismatch, 7 misaligned blob, 8 big-endian host, 9 bad layout, 10 unknown layer kind,
 *          11 bad table offset, 12 invalid structure, 13 not enough layer slots); -1 on null arguments.
 */
int32_t snn_blob_load(
    const uint8_t* blob,
    size_t blob_len,
    SnnLayerDesc* out_layers,
    size_t capacity,
    size_t* out_num_layers,
    SnnDeepNetworkDesc* out_desc
);

/* ---- Model-level aliases (same semantics as the snn_deep_network_* functions) ---- */

/** @brief Alias of snn_deep_network_state_size(). */
size_t snn_model_memory_footprint(const SnnDeepNetworkDesc* desc);

/** @brief Alias of snn_deep_network_init_state(). */
int32_t snn_model_init_context(
    const SnnDeepNetworkDesc* desc,
    uint8_t* arena,
    size_t arena_size,
    SnnDeepNetworkState* out_context
);

/** @brief Alias of snn_deep_network_step(). */
int32_t snn_model_step(
    const SnnDeepNetworkDesc* desc,
    SnnDeepNetworkState* context,
    const int32_t* inputs,
    size_t inputs_len,
    uint32_t* out_spikes_words,
    size_t words_count
);

/** @brief Alias of snn_deep_network_reset(). */
int32_t snn_model_reset(const SnnDeepNetworkDesc* desc, SnnDeepNetworkState* context);

#ifdef __cplusplus
}
#endif

#endif /* SNN_CORE_H */

