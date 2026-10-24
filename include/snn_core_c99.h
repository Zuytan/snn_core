/**
 * @file snn_core_c99.h
 * @brief 🧠 snn_core Standalone Autonomous C99 Header-Only Runtime
 *
 * ⚡ Zero Dynamic Allocation (100% SRAM Arena / Flash ROM .rodata)
 * ⚡ Pure Fixed-Point Arithmetic (Fix32 Q16.16 & Fix16 Q8.8, No FPU required)
 * ⚡ Zero Cargo / Rust Dependency: Drop into STM32CubeIDE, Keil MDK, IAR, ESP-IDF, Zephyr
 * ⚡ Native Hardware Acceleration: __builtin_ctz (CTZ), CMSIS-DSP (__SSAT, __QADD, __SMLAD)
 *
 * Conforming to snn_core standard specifications and AGENTS.md rules.
 */

#ifndef SNN_CORE_C99_H
#define SNN_CORE_C99_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Compiler & Hardware Portability Layer (CMSIS-DSP / CTZ / POPCNT)
 * ============================================================================ */

#if defined(__GNUC__) || defined(__clang__)
    #define SNN_INLINE static inline __attribute__((always_inline))
    #define SNN_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
    #define SNN_INLINE static __inline
    #define SNN_UNUSED
#else
    #define SNN_INLINE static inline
    #define SNN_UNUSED
#endif

/**
 * @brief Hardware-accelerated or branchless Count Trailing Zeros (CTZ).
 * Emits native `CTZ` / `RBIT+CLZ` on ARM Cortex-M4/M33/M7.
 */
SNN_INLINE uint32_t snn_ctz32(uint32_t val) {
#if defined(__GNUC__) || defined(__clang__)
    return (val == 0) ? 32U : (uint32_t)__builtin_ctz(val);
#elif defined(_MSC_VER)
    unsigned long idx;
    return _BitScanForward(&idx, val) ? (uint32_t)idx : 32U;
#else
    if (val == 0) return 32U;
    uint32_t c = 0;
    if ((val & 0x0000FFFF) == 0) { val >>= 16; c += 16; }
    if ((val & 0x000000FF) == 0) { val >>= 8;  c += 8;  }
    if ((val & 0x0000000F) == 0) { val >>= 4;  c += 4;  }
    if ((val & 0x00000003) == 0) { val >>= 2;  c += 2;  }
    if ((val & 0x00000001) == 0) { c += 1; }
    return c;
#endif
}

/**
 * @brief Hardware-accelerated or bit-parallel Population Count (Hamming weight).
 */
SNN_INLINE uint32_t snn_popcount32(uint32_t val) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_popcount(val);
#else
    val = val - ((val >> 1) & 0x55555555U);
    val = (val & 0x33333333U) + ((val >> 2) & 0x33333333U);
    return (((val + (val >> 4)) & 0x0F0F0F0FU) * 0x01010101U) >> 24;
#endif
}

/* ============================================================================
 * Fixed-Point Arithmetic (Fix32 Q16.16)
 * ============================================================================ */

typedef int32_t snn_fix32_t;
typedef int16_t snn_fix16_t;

#define SNN_FIX32_SCALE     (65536.0f)
#define SNN_FIX32_ONE       ((snn_fix32_t)65536)
#define SNN_FIX32_ZERO      ((snn_fix32_t)0)
#define SNN_FIX32_MAX       ((snn_fix32_t)2147483647)
#define SNN_FIX32_MIN       ((snn_fix32_t)(-2147483647 - 1))

/**
 * @brief Saturating 32-bit signed addition.
 * Exploits ARM Cortex-M DSP `__QADD` / `qadd` instruction when available.
 */
SNN_INLINE int32_t snn_saturating_add_i32(int32_t a, int32_t b) {
#if (defined(__ARM_FEATURE_DSP) && __ARM_FEATURE_DSP == 1) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_arm_qadd(a, b);
#else
    int64_t res = (int64_t)a + (int64_t)b;
    if (res > (int64_t)SNN_FIX32_MAX) return SNN_FIX32_MAX;
    if (res < (int64_t)SNN_FIX32_MIN) return SNN_FIX32_MIN;
    return (int32_t)res;
#endif
}

/**
 * @brief Saturating 32-bit signed subtraction.
 * Exploits ARM Cortex-M DSP `__QSUB` / `qsub` instruction when available.
 */
SNN_INLINE int32_t snn_saturating_sub_i32(int32_t a, int32_t b) {
#if (defined(__ARM_FEATURE_DSP) && __ARM_FEATURE_DSP == 1) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_arm_qsub(a, b);
#else
    int64_t res = (int64_t)a - (int64_t)b;
    if (res > (int64_t)SNN_FIX32_MAX) return SNN_FIX32_MAX;
    if (res < (int64_t)SNN_FIX32_MIN) return SNN_FIX32_MIN;
    return (int32_t)res;
#endif
}

/**
 * @brief Fixed-point Q16.16 multiplication with saturation.
 */
SNN_INLINE snn_fix32_t snn_fix32_mul(snn_fix32_t a, snn_fix32_t b) {
    int64_t prod = ((int64_t)a * (int64_t)b) >> 16;
    if (prod > (int64_t)SNN_FIX32_MAX) return SNN_FIX32_MAX;
    if (prod < (int64_t)SNN_FIX32_MIN) return SNN_FIX32_MIN;
    return (snn_fix32_t)prod;
}

/**
 * @brief Converts float to Fix32 Q16.16 with saturation.
 */
SNN_INLINE snn_fix32_t snn_fix32_from_float(float val) {
    float scaled = val * SNN_FIX32_SCALE;
    if (scaled >= 2147483647.0f) return SNN_FIX32_MAX;
    if (scaled <= -2147483648.0f) return SNN_FIX32_MIN;
    return (snn_fix32_t)(scaled >= 0.0f ? (scaled + 0.5f) : (scaled - 0.5f));
}

/**
 * @brief Converts Fix32 Q16.16 to float.
 */
SNN_INLINE float snn_fix32_to_float(snn_fix32_t bits) {
    return (float)bits / SNN_FIX32_SCALE;
}

/**
 * @brief Integer to Fix32 Q16.16.
 */
SNN_INLINE snn_fix32_t snn_fix32_from_int(int32_t val) {
    int64_t scaled = (int64_t)val << 16;
    if (scaled > (int64_t)SNN_FIX32_MAX) return SNN_FIX32_MAX;
    if (scaled < (int64_t)SNN_FIX32_MIN) return SNN_FIX32_MIN;
    return (snn_fix32_t)scaled;
}

/**
 * @brief Padé (2,2) approximant of exp(-x) for decaying membrane dynamics in pure C99.
 */
SNN_INLINE snn_fix32_t snn_pade_exp_neg(float x) {
    if (x <= 0.0f) return SNN_FIX32_ONE;
    if (x >= 10.0f) return SNN_FIX32_ZERO;
    float num = (12.0f - 6.0f * x) + (x * x);
    float den = (12.0f + 6.0f * x) + (x * x);
    if (num <= 0.0f) return SNN_FIX32_ZERO;
    return snn_fix32_from_float(num / den);
}

/* ============================================================================
 * Address-Event Representation (AER) & Bitset Manipulation
 * ============================================================================ */

typedef struct {
    uint16_t address;   /**< Source channel address */
    uint16_t timestamp; /**< Microsecond timestamp */
} SnnAerSpike1d;

typedef struct {
    uint16_t x;         /**< Spatial X coordinate */
    uint16_t y;         /**< Spatial Y coordinate */
    bool polarity;      /**< Polarity (true: ON, false: OFF) */
    uint32_t timestamp; /**< Microsecond timestamp */
} SnnAerSpike2d;

SNN_INLINE int32_t snn_aer_decode_1d(
    const SnnAerSpike1d* events,
    size_t event_count,
    uint32_t* out_words,
    size_t words_count
) {
    if (out_words == NULL || words_count == 0) return -1;
    if (event_count > 0 && events == NULL) return -1;

    for (size_t i = 0; i < words_count; ++i) {
        out_words[i] = 0;
    }
    for (size_t i = 0; i < event_count; ++i) {
        size_t addr = (size_t)events[i].address;
        size_t w_idx = addr / 32;
        size_t b_idx = addr % 32;
        if (w_idx < words_count) {
            out_words[w_idx] |= (1U << b_idx);
        }
    }
    return 0;
}

SNN_INLINE int32_t snn_aer_decode_2d_bipolar(
    const SnnAerSpike2d* events,
    size_t event_count,
    size_t width,
    size_t height,
    uint32_t* out_words,
    size_t words_count
) {
    if (out_words == NULL || words_count == 0 || width == 0 || height == 0) return -1;
    if (event_count > 0 && events == NULL) return -1;

    for (size_t i = 0; i < words_count; ++i) {
        out_words[i] = 0;
    }
    for (size_t i = 0; i < event_count; ++i) {
        size_t channel_offset = events[i].polarity ? 0 : (width * height);
        size_t linear_idx = channel_offset + (((size_t)events[i].y * width) + (size_t)events[i].x);
        size_t w_idx = linear_idx / 32;
        size_t b_idx = linear_idx % 32;
        if (w_idx < words_count) {
            out_words[w_idx] |= (1U << b_idx);
        }
    }
    return 0;
}

SNN_INLINE bool snn_bitset_get_spike(const uint32_t* words, size_t words_count, size_t neuron_index) {
    if (words == NULL) return false;
    size_t w_idx = neuron_index / 32;
    size_t b_idx = neuron_index % 32;
    if (w_idx < words_count) {
        return (words[w_idx] & (1U << b_idx)) != 0;
    }
    return false;
}

SNN_INLINE uint32_t snn_bitset_count_spikes(const uint32_t* words, size_t words_count) {
    if (words == NULL) return 0;
    uint32_t total = 0;
    for (size_t i = 0; i < words_count; ++i) {
        total += snn_popcount32(words[i]);
    }
    return total;
}

/* ============================================================================
 * Core Data Structures (Matching include/snn_core.h)
 * ============================================================================ */

typedef struct {
    int32_t threshold;          /**< Action potential threshold V_th (Q16.16 Fix32) */
    int32_t decay_factor;       /**< Membrane potential decay factor exp(-dt/tau) (Q16.16 Fix32) */
    int32_t reset_potential;    /**< Post-spike resting potential V_reset (Q16.16 Fix32) */
    uint16_t refractory_period; /**< Absolute refractory duration in discrete ticks */
    int32_t bias;               /**< Constant tonic current injected every tick (Q16.16 Fix32) */
    uint8_t reset_mode;         /**< Reset mode (0: Hard reset V=V_reset, 1: Soft reset V=V-V_th) */
} SnnLifConfig;

typedef struct {
    const uint16_t* row_ptrs;   /**< CSR row offsets array of length (in_features + 1) */
    const uint16_t* targets;    /**< Target post-synaptic neuron indices of length total_synapses */
    const int32_t* weights;     /**< Synaptic weights array in Q16.16 Fix32 (or int8_t* cast) */
    size_t in_features;         /**< Number of input pre-synaptic channels */
    size_t out_features;        /**< Number of output post-synaptic neurons */
    size_t total_synapses;      /**< Total non-zero synaptic connections */
} SnnCsrMatrix;

typedef struct {
    size_t num_neurons;                 /**< Number of neurons in the population */
    size_t in_features;                 /**< Number of input channels */
    SnnLifConfig config;                /**< LIF biophysical dynamics parameters */
    SnnCsrMatrix input_synapses;        /**< Input projection synaptic matrix */
    SnnCsrMatrix recurrent_synapses;    /**< Recurrent feedback synaptic matrix (row_ptrs=NULL if none) */
    const int32_t* bias;                /**< Per-neuron static bias in Flash ROM (NULL if none) */
} SnnNetworkDesc;

typedef struct {
    int32_t* membrane_potentials;   /**< Array of instantaneous membrane potentials V_mem */
    int32_t* input_accumulators;    /**< Array of synaptic input currents I_in */
    uint8_t* refractory_times;      /**< Array of remaining refractory periods */
    uint32_t* last_spikes;          /**< Bitset array of active spikes emitted at previous tick */
    size_t num_neurons;             /**< Total number of neurons */
    size_t words_count;             /**< Number of 32-bit words in last_spikes bitset */
} SnnNetworkState;

#define SNN_NETWORK_STATE_SIZE(n) \
    (((n) * 4) + ((n) * 4) + ((((n) + 3) & ~3)) + ((((n) + 31) / 32) * 4))

SNN_INLINE size_t snn_network_state_size(size_t num_neurons) {
    if (num_neurons == 0) return 0;
    size_t words = (num_neurons + 31) / 32;
    size_t mem_bytes = num_neurons * sizeof(int32_t);
    size_t acc_bytes = num_neurons * sizeof(int32_t);
    size_t refr_bytes = (num_neurons * sizeof(uint8_t) + 3) & ~3;
    size_t spikes_bytes = words * sizeof(uint32_t);
    return mem_bytes + acc_bytes + refr_bytes + spikes_bytes;
}

SNN_INLINE int32_t snn_network_init_state(
    const SnnNetworkDesc* desc,
    uint8_t* arena,
    size_t arena_size,
    SnnNetworkState* out_state
) {
    if (desc == NULL || arena == NULL || out_state == NULL) return -1;
    size_t n = desc->num_neurons;
    if (n == 0) return -2;
    size_t required = snn_network_state_size(n);
    if (arena_size < required) return -3;
    if (((uintptr_t)arena % 4) != 0) return -4;

    size_t words = (n + 31) / 32;
    int32_t* mem_ptr = (int32_t*)arena;
    int32_t* acc_ptr = mem_ptr + n;
    uint8_t* refr_ptr = (uint8_t*)(acc_ptr + n);
    size_t refr_aligned = (n + 3) & ~3;
    uint32_t* spikes_ptr = (uint32_t*)((uintptr_t)refr_ptr + refr_aligned);

    int32_t reset_v = desc->config.reset_potential;
    for (size_t i = 0; i < n; ++i) {
        mem_ptr[i] = reset_v;
        acc_ptr[i] = 0;
        refr_ptr[i] = 0;
    }
    for (size_t w = 0; w < words; ++w) {
        spikes_ptr[w] = 0;
    }

    out_state->membrane_potentials = mem_ptr;
    out_state->input_accumulators = acc_ptr;
    out_state->refractory_times = refr_ptr;
    out_state->last_spikes = spikes_ptr;
    out_state->num_neurons = n;
    out_state->words_count = words;
    return 0;
}

SNN_INLINE int32_t snn_network_step(
    const SnnNetworkDesc* desc,
    SnnNetworkState* state,
    const int32_t* inputs,
    size_t inputs_len,
    uint32_t* out_spikes_words,
    size_t words_count
) {
    if (desc == NULL || state == NULL || out_spikes_words == NULL) return -1;
    size_t n = desc->num_neurons;
    size_t words = state->words_count;
    if (words_count < words) return -2;

    for (size_t w = 0; w < words_count; ++w) {
        out_spikes_words[w] = 0;
    }

    /* 1. Input CSR synaptic propagation */
    if (inputs != NULL && inputs_len > 0
        && desc->input_synapses.row_ptrs != NULL
        && desc->input_synapses.targets != NULL
        && desc->input_synapses.weights != NULL
        && desc->input_synapses.total_synapses > 0)
    {
        const SnnCsrMatrix* syn = &desc->input_synapses;
        size_t active_inputs = inputs_len < syn->in_features ? inputs_len : syn->in_features;

        for (size_t in_idx = 0; in_idx < active_inputs; ++in_idx) {
            int32_t in_val = inputs[in_idx];
            if (in_val != 0) {
                size_t start = syn->row_ptrs[in_idx];
                size_t end = syn->row_ptrs[in_idx + 1];
                if (start > syn->total_synapses) start = syn->total_synapses;
                if (end > syn->total_synapses) end = syn->total_synapses;

                for (size_t s = start; s < end; ++s) {
                    size_t target = syn->targets[s];
                    if (target < n) {
                        int32_t weighted = snn_fix32_mul(in_val, syn->weights[s]);
                        state->input_accumulators[target] =
                            snn_saturating_add_i32(state->input_accumulators[target], weighted);
                    }
                }
            }
        }
    }

    /* 2. Recurrent CSR synaptic propagation */
    if (desc->recurrent_synapses.row_ptrs != NULL
        && desc->recurrent_synapses.targets != NULL
        && desc->recurrent_synapses.weights != NULL
        && desc->recurrent_synapses.total_synapses > 0)
    {
        const SnnCsrMatrix* rec = &desc->recurrent_synapses;
        for (size_t w_idx = 0; w_idx < words; ++w_idx) {
            uint32_t active = state->last_spikes[w_idx];
            while (active != 0) {
                uint32_t bit_idx = snn_ctz32(active);
                size_t pre_neuron = w_idx * 32 + bit_idx;
                if (pre_neuron < rec->in_features) {
                    size_t start = rec->row_ptrs[pre_neuron];
                    size_t end = rec->row_ptrs[pre_neuron + 1];
                    if (start > rec->total_synapses) start = rec->total_synapses;
                    if (end > rec->total_synapses) end = rec->total_synapses;

                    for (size_t s = start; s < end; ++s) {
                        size_t target = rec->targets[s];
                        if (target < n) {
                            state->input_accumulators[target] =
                                snn_saturating_add_i32(state->input_accumulators[target], rec->weights[s]);
                        }
                    }
                }
                active &= active - 1;
            }
        }
    }

    /* 3. LIF Biophysical dynamics evaluation */
    int32_t cfg_decay = desc->config.decay_factor;
    int32_t cfg_thresh = desc->config.threshold;
    int32_t cfg_reset = desc->config.reset_potential;
    int32_t cfg_bias = desc->config.bias;
    bool is_soft_reset = (desc->config.reset_mode == 1);

    for (size_t i = 0; i < n; ++i) {
        uint8_t refr = state->refractory_times[i];
        int32_t static_b = (desc->bias != NULL) ? desc->bias[i] : 0;
        int32_t total_bias = snn_saturating_add_i32(static_b, cfg_bias);
        int32_t inp = snn_saturating_add_i32(state->input_accumulators[i], total_bias);
        state->input_accumulators[i] = 0;

        if (refr > 0) {
            state->refractory_times[i] = refr - 1;
            state->membrane_potentials[i] = cfg_reset;
            continue;
        }

        int32_t current_v = state->membrane_potentials[i];
        int32_t decayed = snn_fix32_mul(current_v, cfg_decay);
        int32_t v = snn_saturating_add_i32(decayed, inp);

        if (v >= cfg_thresh) {
            int32_t next_v = is_soft_reset ? snn_saturating_sub_i32(v, cfg_thresh) : cfg_reset;
            state->membrane_potentials[i] = next_v;
            state->refractory_times[i] = (uint8_t)desc->config.refractory_period;

            size_t w_idx = i / 32;
            size_t b_idx = i % 32;
            if (w_idx < words) {
                out_spikes_words[w_idx] |= (1U << b_idx);
                state->last_spikes[w_idx] |= (1U << b_idx);
            }
        } else {
            state->membrane_potentials[i] = v;
            size_t w_idx = i / 32;
            size_t b_idx = i % 32;
            if (w_idx < words) {
                state->last_spikes[w_idx] &= ~(1U << b_idx);
            }
        }
    }

    return 0;
}

SNN_INLINE int32_t snn_network_reset(const SnnNetworkDesc* desc, SnnNetworkState* state) {
    if (desc == NULL || state == NULL) return -1;
    size_t n = desc->num_neurons;
    size_t words = state->words_count;
    int32_t reset_v = desc->config.reset_potential;

    for (size_t i = 0; i < n; ++i) {
        state->membrane_potentials[i] = reset_v;
        state->input_accumulators[i] = 0;
        state->refractory_times[i] = 0;
    }
    for (size_t w = 0; w < words; ++w) {
        state->last_spikes[w] = 0;
    }
    return 0;
}

/* ============================================================================
 * Deep Multilayer Network API (ActivationArena Zero-Alloc Ping-Pong)
 * ============================================================================ */

typedef enum {
    SNN_LAYER_FEEDFORWARD    = 0, /**< Standard Fix32 CSR feedforward layer */
    SNN_LAYER_QUANTIZED_INT8 = 1, /**< Quantized INT8 weights with Fix32 common scale */
    SNN_LAYER_CONV2D         = 2, /**< Event-driven 2D Convolution layer */
    SNN_LAYER_MAX_POOL_2D    = 3  /**< Spatial Max-Pooling 2D (bitwise OR) */
} SnnLayerType;

typedef struct {
    const int32_t* kernel;       /**< Q16.16 Fix32 kernel tensor [out_c * in_c * kh * kw] */
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

typedef struct {
    SnnLayerType layer_type;     /**< Feedforward / Quantized INT8 / Conv2d / MaxPool2d */
    size_t in_features;          /**< Number of input connections */
    size_t num_neurons;          /**< Number of neurons in this layer */
    SnnLifConfig config;         /**< Biophysical LIF dynamics */
    SnnCsrMatrix synapses;       /**< CSR matrix in Flash ROM */
    SnnConv2dDesc conv2d;        /**< Conv2d descriptor if layer_type == SNN_LAYER_CONV2D */
    SnnMaxPool2dDesc pool2d;     /**< MaxPool2d descriptor if layer_type == SNN_LAYER_MAX_POOL_2D */
    const SnnCsrMatrix* recurrent_synapses; /**< Optional recurrent CSR synapses (NULL if none) */
    const int32_t* bias;         /**< Optional bias vector in Flash ROM */
    int32_t quant_scale;         /**< Fix32 common scale factor (if QuantizedInt8) */
} SnnLayerDesc;

typedef struct {
    const SnnLayerDesc* layers;  /**< Pointer to array of layer descriptors in Flash ROM */
    size_t num_layers;           /**< Total number of layers */
    size_t in_features;          /**< Input dimensionality of Layer 0 */
} SnnDeepNetworkDesc;

typedef struct {
    int32_t* membrane_potentials; /**< Persistent membrane potentials in SRAM */
    uint8_t* refractory_times;    /**< Refractory counters in SRAM */
    int32_t* arena_accumulators;  /**< Shared synaptic accumulator buffer (size max_neurons) */
    uint32_t* ping_words;         /**< Ping spike bitset buffer (size max_words) */
    uint32_t* pong_words;         /**< Pong spike bitset buffer (size max_words) */
    uint32_t* recurrent_spikes;   /**< Persistent previous-step spikes for recurrent layers */
    size_t total_neurons;         /**< Total persistent neurons */
    size_t max_neurons;           /**< Largest single layer neuron count */
    size_t max_words;             /**< Largest single layer word count */
    size_t total_recurrent_words; /**< Total words reserved for recurrent spikes */
} SnnDeepNetworkState;

SNN_INLINE size_t snn_deep_network_state_size(const SnnDeepNetworkDesc* desc) {
    if (desc == NULL || desc->num_layers == 0 || desc->layers == NULL) return 0;

    size_t total_neurons = 0;
    size_t max_neurons = 0;
    size_t max_words = 0;
    size_t total_rec_words = 0;

    for (size_t l = 0; l < desc->num_layers; ++l) {
        size_t n = desc->layers[l].num_neurons;
        total_neurons += n;
        if (n > max_neurons) max_neurons = n;
        size_t w = (n + 31) / 32;
        if (w > max_words) max_words = w;
        if (desc->layers[l].recurrent_synapses != NULL) {
            total_rec_words += w;
        }
    }

    if (total_neurons == 0) return 0;

    size_t mem_bytes = total_neurons * sizeof(int32_t);
    size_t refr_bytes = (total_neurons * sizeof(uint8_t) + 3) & ~3;
    size_t acc_bytes = max_neurons * sizeof(int32_t);
    size_t ping_bytes = max_words * sizeof(uint32_t);
    size_t pong_bytes = max_words * sizeof(uint32_t);
    size_t rec_bytes = total_rec_words * sizeof(uint32_t);

    return mem_bytes + refr_bytes + acc_bytes + ping_bytes + pong_bytes + rec_bytes;
}

SNN_INLINE int32_t snn_deep_network_init_state(
    const SnnDeepNetworkDesc* desc,
    uint8_t* arena,
    size_t arena_size,
    SnnDeepNetworkState* out_state
) {
    if (desc == NULL || arena == NULL || out_state == NULL) return -1;
    if (((uintptr_t)arena % 4) != 0) return -4;
    if (desc->num_layers == 0 || desc->layers == NULL) return -2;

    size_t required = snn_deep_network_state_size(desc);
    if (arena_size < required || required == 0) return -3;

    size_t total_neurons = 0;
    size_t max_neurons = 0;
    size_t max_words = 0;
    size_t total_rec_words = 0;

    for (size_t l = 0; l < desc->num_layers; ++l) {
        size_t n = desc->layers[l].num_neurons;
        total_neurons += n;
        if (n > max_neurons) max_neurons = n;
        size_t w = (n + 31) / 32;
        if (w > max_words) max_words = w;
        if (desc->layers[l].recurrent_synapses != NULL) {
            total_rec_words += w;
        }
    }

    int32_t* mem_ptr = (int32_t*)arena;
    uint8_t* refr_ptr = (uint8_t*)(mem_ptr + total_neurons);
    size_t refr_aligned = (total_neurons + 3) & ~3;
    int32_t* acc_ptr = (int32_t*)((uintptr_t)refr_ptr + refr_aligned);
    uint32_t* ping_ptr = (uint32_t*)(acc_ptr + max_neurons);
    uint32_t* pong_ptr = ping_ptr + max_words;
    uint32_t* rec_ptr = (total_rec_words > 0) ? (pong_ptr + max_words) : NULL;

    size_t offset = 0;
    for (size_t l = 0; l < desc->num_layers; ++l) {
        int32_t reset_v = desc->layers[l].config.reset_potential;
        size_t n = desc->layers[l].num_neurons;
        for (size_t i = 0; i < n; ++i) {
            mem_ptr[offset + i] = reset_v;
            refr_ptr[offset + i] = 0;
        }
        offset += n;
    }

    for (size_t i = 0; i < max_neurons; ++i) acc_ptr[i] = 0;
    for (size_t w = 0; w < max_words; ++w) {
        ping_ptr[w] = 0;
        pong_ptr[w] = 0;
    }
    if (rec_ptr != NULL) {
        for (size_t w = 0; w < total_rec_words; ++w) {
            rec_ptr[w] = 0;
        }
    }

    out_state->membrane_potentials = mem_ptr;
    out_state->refractory_times = refr_ptr;
    out_state->arena_accumulators = acc_ptr;
    out_state->ping_words = ping_ptr;
    out_state->pong_words = pong_ptr;
    out_state->recurrent_spikes = rec_ptr;
    out_state->total_neurons = total_neurons;
    out_state->max_neurons = max_neurons;
    out_state->max_words = max_words;
    out_state->total_recurrent_words = total_rec_words;
    return 0;
}

SNN_INLINE int32_t snn_deep_network_step(
    const SnnDeepNetworkDesc* desc,
    SnnDeepNetworkState* state,
    const int32_t* inputs,
    size_t inputs_len,
    uint32_t* out_spikes_words,
    size_t words_count
) {
    if (desc == NULL || state == NULL || out_spikes_words == NULL) return -1;
    if (desc->num_layers == 0 || desc->layers == NULL) return -2;

    const SnnLayerDesc* last_layer = &desc->layers[desc->num_layers - 1];
    size_t last_words = (last_layer->num_neurons + 31) / 32;
    if (words_count < last_words) return -3;

    size_t neuron_offset = 0;
    size_t rec_word_offset = 0;

    for (size_t layer_idx = 0; layer_idx < desc->num_layers; ++layer_idx) {
        const SnnLayerDesc* layer = &desc->layers[layer_idx];
        size_t n = layer->num_neurons;
        size_t words = (n + 31) / 32;
        bool is_even = ((layer_idx % 2) == 0);

        /* 1. Clear shared accumulators and active destination buffer */
        for (size_t i = 0; i < n; ++i) {
            state->arena_accumulators[i] = 0;
        }

        uint32_t* dst_spikes = is_even ? state->pong_words : state->ping_words;
        for (size_t w = 0; w < words; ++w) {
            dst_spikes[w] = 0;
        }

        /* 2. Spatial Pooling (MaxPool2D) bypass */
        if (layer->layer_type == SNN_LAYER_MAX_POOL_2D) {
            const SnnMaxPool2dDesc* p = &layer->pool2d;
            const uint32_t* src_spikes = is_even ? state->ping_words : state->pong_words;
            size_t prev_words = (layer_idx > 0) ? (desc->layers[layer_idx - 1].num_neurons + 31) / 32 : 0;
            size_t out_idx = 0;

            for (size_t c = 0; c < p->channels; ++c) {
                for (size_t oh = 0; oh < p->out_h; ++oh) {
                    for (size_t ow = 0; ow < p->out_w; ++ow) {
                        bool pooled = false;
                        for (size_t kh = 0; kh < p->kernel_h; ++kh) {
                            for (size_t kw = 0; kw < p->kernel_w; ++kw) {
                                size_t ih = oh * p->stride_h + kh;
                                size_t iw = ow * p->stride_w + kw;
                                size_t in_idx = (c * p->in_h + ih) * p->in_w + iw;
                                size_t w = in_idx / 32;
                                size_t b = in_idx % 32;
                                if (w < prev_words && ((src_spikes[w] >> b) & 1U) != 0) {
                                    pooled = true;
                                    break;
                                }
                            }
                            if (pooled) break;
                        }
                        if (pooled) {
                            size_t ow_idx = out_idx / 32;
                            size_t ob_idx = out_idx % 32;
                            if (ow_idx < words) {
                                dst_spikes[ow_idx] |= (1U << ob_idx);
                            }
                        }
                        out_idx++;
                    }
                }
            }
            neuron_offset += n;
            continue;
        }

        /* 3. Synaptic propagation */
        if (layer->layer_type == SNN_LAYER_CONV2D) {
            const SnnConv2dDesc* conv = &layer->conv2d;
            if (layer_idx == 0) {
                /* Conv2D with continuous inputs */
                if (inputs != NULL && inputs_len > 0 && conv->kernel != NULL) {
                    for (size_t ic = 0; ic < conv->in_channels; ++ic) {
                        for (size_t ih = 0; ih < conv->in_h; ++ih) {
                            for (size_t iw = 0; iw < conv->in_w; ++iw) {
                                size_t in_idx = (ic * conv->in_h + ih) * conv->in_w + iw;
                                if (in_idx < inputs_len) {
                                    int32_t in_val = inputs[in_idx];
                                    if (in_val != 0) {
                                        size_t virt_h = ih + conv->padding_h;
                                        size_t virt_w = iw + conv->padding_w;
                                        for (size_t kh = 0; kh < conv->kernel_h; ++kh) {
                                            if (virt_h >= kh && (virt_h - kh) % conv->stride_h == 0) {
                                                size_t oh = (virt_h - kh) / conv->stride_h;
                                                if (oh < conv->out_h) {
                                                    for (size_t kw = 0; kw < conv->kernel_w; ++kw) {
                                                        if (virt_w >= kw && (virt_w - kw) % conv->stride_w == 0) {
                                                            size_t ow = (virt_w - kw) / conv->stride_w;
                                                            if (ow < conv->out_w) {
                                                                for (size_t oc = 0; oc < conv->out_channels; ++oc) {
                                                                    size_t k_idx = ((oc * conv->in_channels + ic) * conv->kernel_h + kh) * conv->kernel_w + kw;
                                                                    int32_t w = conv->kernel[k_idx];
                                                                    size_t out_idx = (oc * conv->out_h + oh) * conv->out_w + ow;
                                                                    if (out_idx < n) {
                                                                        state->arena_accumulators[out_idx] =
                                                                            snn_saturating_add_i32(state->arena_accumulators[out_idx], snn_fix32_mul(in_val, w));
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                /* Conv2D with input spikes */
                const SnnLayerDesc* prev_layer = &desc->layers[layer_idx - 1];
                size_t prev_words = (prev_layer->num_neurons + 31) / 32;
                const uint32_t* src_spikes = is_even ? state->ping_words : state->pong_words;
                if (conv->kernel != NULL) {
                    for (size_t w_idx = 0; w_idx < prev_words; ++w_idx) {
                        uint32_t active = src_spikes[w_idx];
                        while (active != 0) {
                            uint32_t bit_idx = snn_ctz32(active);
                            size_t pre_neuron = w_idx * 32 + bit_idx;
                            if (pre_neuron < prev_layer->num_neurons) {
                                size_t iw = pre_neuron % conv->in_w;
                                size_t tmp = pre_neuron / conv->in_w;
                                size_t ih = tmp % conv->in_h;
                                size_t ic = tmp / conv->in_h;
                                if (ic < conv->in_channels) {
                                    size_t virt_h = ih + conv->padding_h;
                                    size_t virt_w = iw + conv->padding_w;
                                    for (size_t kh = 0; kh < conv->kernel_h; ++kh) {
                                        if (virt_h >= kh && (virt_h - kh) % conv->stride_h == 0) {
                                            size_t oh = (virt_h - kh) / conv->stride_h;
                                            if (oh < conv->out_h) {
                                                for (size_t kw = 0; kw < conv->kernel_w; ++kw) {
                                                    if (virt_w >= kw && (virt_w - kw) % conv->stride_w == 0) {
                                                        size_t ow = (virt_w - kw) / conv->stride_w;
                                                        if (ow < conv->out_w) {
                                                            for (size_t oc = 0; oc < conv->out_channels; ++oc) {
                                                                size_t k_idx = ((oc * conv->in_channels + ic) * conv->kernel_h + kh) * conv->kernel_w + kw;
                                                                int32_t w = conv->kernel[k_idx];
                                                                size_t out_idx = (oc * conv->out_h + oh) * conv->out_w + ow;
                                                                if (out_idx < n) {
                                                                    state->arena_accumulators[out_idx] =
                                                                        snn_saturating_add_i32(state->arena_accumulators[out_idx], w);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            active &= active - 1;
                        }
                    }
                }
            }
        } else {
            /* Standard Dense / Quantized CSR Synapses */
            const SnnCsrMatrix* syn = &layer->synapses;
            if (layer_idx == 0) {
                if (inputs != NULL && inputs_len > 0
                    && syn->row_ptrs != NULL && syn->targets != NULL
                    && syn->weights != NULL && syn->total_synapses > 0)
                {
                    size_t active_inputs = inputs_len < syn->in_features ? inputs_len : syn->in_features;
                    for (size_t in_idx = 0; in_idx < active_inputs; ++in_idx) {
                        int32_t in_val = inputs[in_idx];
                        if (in_val != 0) {
                            size_t start = syn->row_ptrs[in_idx];
                            size_t end = syn->row_ptrs[in_idx + 1];
                            if (start > syn->total_synapses) start = syn->total_synapses;
                            if (end > syn->total_synapses) end = syn->total_synapses;

                            for (size_t s = start; s < end; ++s) {
                                size_t target = syn->targets[s];
                                if (target < n) {
                                    int32_t weighted;
                                    if (layer->layer_type == SNN_LAYER_QUANTIZED_INT8) {
                                        const int8_t* i8_weights = (const int8_t*)syn->weights;
                                        int32_t syn_w = snn_fix32_mul(layer->quant_scale, snn_fix32_from_int(i8_weights[s]));
                                        weighted = snn_fix32_mul(in_val, syn_w);
                                    } else {
                                        weighted = snn_fix32_mul(in_val, syn->weights[s]);
                                    }
                                    state->arena_accumulators[target] =
                                        snn_saturating_add_i32(state->arena_accumulators[target], weighted);
                                }
                            }
                        }
                    }
                }
            } else {
                const SnnLayerDesc* prev_layer = &desc->layers[layer_idx - 1];
                size_t prev_words = (prev_layer->num_neurons + 31) / 32;
                const uint32_t* src_spikes = is_even ? state->ping_words : state->pong_words;

                if (syn->row_ptrs != NULL && syn->targets != NULL
                    && syn->weights != NULL && syn->total_synapses > 0)
                {
                    for (size_t w_idx = 0; w_idx < prev_words; ++w_idx) {
                        uint32_t active = src_spikes[w_idx];
                        while (active != 0) {
                            uint32_t bit_idx = snn_ctz32(active);
                            size_t pre_neuron = w_idx * 32 + bit_idx;
                            if (pre_neuron < prev_layer->num_neurons) {
                                size_t start = syn->row_ptrs[pre_neuron];
                                size_t end = syn->row_ptrs[pre_neuron + 1];
                                if (start > syn->total_synapses) start = syn->total_synapses;
                                if (end > syn->total_synapses) end = syn->total_synapses;

                                for (size_t s = start; s < end; ++s) {
                                    size_t target = syn->targets[s];
                                    if (target < n) {
                                        int32_t weight;
                                        if (layer->layer_type == SNN_LAYER_QUANTIZED_INT8) {
                                            const int8_t* i8_weights = (const int8_t*)syn->weights;
                                            weight = snn_fix32_mul(layer->quant_scale, snn_fix32_from_int(i8_weights[s]));
                                        } else {
                                            weight = syn->weights[s];
                                        }
                                        state->arena_accumulators[target] =
                                            snn_saturating_add_i32(state->arena_accumulators[target], weight);
                                    }
                                }
                            }
                            active &= active - 1;
                        }
                    }
                }
            }
        }

        /* 4. Recurrent synapses W_rec propagation */
        if (layer->recurrent_synapses != NULL && state->recurrent_spikes != NULL) {
            const SnnCsrMatrix* rec = layer->recurrent_synapses;
            if (rec->row_ptrs != NULL && rec->targets != NULL
                && rec->weights != NULL && rec->total_synapses > 0)
            {
                const uint32_t* r_spikes = state->recurrent_spikes + rec_word_offset;
                for (size_t w_idx = 0; w_idx < words; ++w_idx) {
                    uint32_t active = r_spikes[w_idx];
                    while (active != 0) {
                        uint32_t bit_idx = snn_ctz32(active);
                        size_t pre_neuron = w_idx * 32 + bit_idx;
                        if (pre_neuron < n) {
                            size_t start = rec->row_ptrs[pre_neuron];
                            size_t end = rec->row_ptrs[pre_neuron + 1];
                            if (start > rec->total_synapses) start = rec->total_synapses;
                            if (end > rec->total_synapses) end = rec->total_synapses;

                            for (size_t s = start; s < end; ++s) {
                                size_t target = rec->targets[s];
                                if (target < n) {
                                    state->arena_accumulators[target] =
                                        snn_saturating_add_i32(state->arena_accumulators[target], rec->weights[s]);
                                }
                            }
                        }
                        active &= active - 1;
                    }
                }
            }
        }

        /* 5. LIF Biophysical dynamics */
        int32_t cfg_decay = layer->config.decay_factor;
        int32_t cfg_thresh = layer->config.threshold;
        int32_t cfg_reset = layer->config.reset_potential;
        int32_t cfg_bias = layer->config.bias;
        bool is_soft_reset = (layer->config.reset_mode == 1);

        for (size_t i = 0; i < n; ++i) {
            size_t global_idx = neuron_offset + i;
            uint8_t refr = state->refractory_times[global_idx];
            int32_t syn_input = state->arena_accumulators[i];
            int32_t static_b = (layer->bias != NULL) ? layer->bias[i] : 0;
            int32_t total_b = snn_saturating_add_i32(static_b, cfg_bias);
            int32_t total_in = snn_saturating_add_i32(syn_input, total_b);

            if (refr > 0) {
                state->refractory_times[global_idx] = refr - 1;
                state->membrane_potentials[global_idx] = cfg_reset;
            } else {
                int32_t v_cur = state->membrane_potentials[global_idx];
                int32_t decayed = snn_fix32_mul(v_cur, cfg_decay);
                int32_t v_next = snn_saturating_add_i32(decayed, total_in);

                if (v_next >= cfg_thresh) {
                    size_t w_idx = i / 32;
                    size_t b_idx = i % 32;
                    dst_spikes[w_idx] |= (1U << b_idx);

                    state->membrane_potentials[global_idx] =
                        is_soft_reset ? snn_saturating_sub_i32(v_next, cfg_thresh) : cfg_reset;
                    state->refractory_times[global_idx] = (uint8_t)layer->config.refractory_period;
                } else {
                    state->membrane_potentials[global_idx] = v_next;
                }
            }
        }

        /* 6. Save current spikes for recurrent next-step */
        if (layer->recurrent_synapses != NULL && state->recurrent_spikes != NULL) {
            for (size_t w = 0; w < words; ++w) {
                state->recurrent_spikes[rec_word_offset + w] = dst_spikes[w];
            }
            rec_word_offset += words;
        }

        neuron_offset += n;
    }

    /* 7. Copy final layer spikes into output buffer */
    bool last_is_even = (((desc->num_layers - 1) % 2) == 0);
    const uint32_t* final_words = last_is_even ? state->pong_words : state->ping_words;

    for (size_t w = 0; w < last_words; ++w) {
        out_spikes_words[w] = final_words[w];
    }
    for (size_t w = last_words; w < words_count; ++w) {
        out_spikes_words[w] = 0;
    }

    return 0;
}

SNN_INLINE int32_t snn_deep_network_reset(const SnnDeepNetworkDesc* desc, SnnDeepNetworkState* state) {
    if (desc == NULL || state == NULL) return -1;
    size_t offset = 0;
    for (size_t l = 0; l < desc->num_layers; ++l) {
        int32_t reset_v = desc->layers[l].config.reset_potential;
        size_t n = desc->layers[l].num_neurons;
        for (size_t i = 0; i < n; ++i) {
            state->membrane_potentials[offset + i] = reset_v;
            state->refractory_times[offset + i] = 0;
        }
        offset += n;
    }
    for (size_t i = 0; i < state->max_neurons; ++i) state->arena_accumulators[i] = 0;
    for (size_t w = 0; w < state->max_words; ++w) {
        state->ping_words[w] = 0;
        state->pong_words[w] = 0;
    }
    if (state->recurrent_spikes != NULL) {
        for (size_t w = 0; w < state->total_recurrent_words; ++w) {
            state->recurrent_spikes[w] = 0;
        }
    }
    return 0;
}

/* ============================================================================
 * Unified Industrial ModelDescriptor C99 API (100% Rust snn_core Parity)
 * ============================================================================ */

typedef SnnDeepNetworkDesc SnnModelDescriptor;
typedef SnnDeepNetworkState SnnExecutionContext;

#define snn_model_memory_footprint(desc)             snn_deep_network_state_size(desc)
#define snn_model_init_context(desc, arena, sz, ctx) snn_deep_network_init_state((desc), (arena), (sz), (ctx))
#define snn_model_step(desc, ctx, inp, in_sz, out, out_sz) snn_deep_network_step((desc), (ctx), (inp), (in_sz), (out), (out_sz))
#define snn_model_reset(desc, ctx)                   snn_deep_network_reset((desc), (ctx))

/* ============================================================================
 * Readout Layer API (Dense Classifier with CTZ Bit-Skipping)
 * ============================================================================ */

SNN_INLINE int32_t snn_readout_dense(
    const int32_t* weights_q16,
    const int32_t* bias_q16,
    const uint32_t* spike_words,
    size_t num_neurons,
    size_t num_classes,
    int32_t* out_logits_q16
) {
    if (weights_q16 == NULL || spike_words == NULL || out_logits_q16 == NULL) return -1;
    if (num_neurons == 0 || num_classes == 0) return -2;

    size_t words_count = (num_neurons + 31) / 32;

    for (size_t c = 0; c < num_classes; ++c) {
        int32_t acc = (bias_q16 != NULL) ? bias_q16[c] : 0;
        size_t class_offset = c * num_neurons;

        for (size_t w_idx = 0; w_idx < words_count; ++w_idx) {
            uint32_t active = spike_words[w_idx];
            while (active != 0) {
                uint32_t bit_idx = snn_ctz32(active);
                size_t neuron_idx = w_idx * 32 + bit_idx;
                if (neuron_idx < num_neurons) {
                    acc = snn_saturating_add_i32(acc, weights_q16[class_offset + neuron_idx]);
                }
                active &= active - 1;
            }
        }
        out_logits_q16[c] = acc;
    }
    return 0;
}

/* ============================================================================
 * BPTN (Bit-Parallel Temporal Neural Network) Word-RAM C99 API
 * ============================================================================ */

SNN_INLINE uint64_t snn_bptn_framing_push(uint64_t* buffer, bool spike) {
    if (buffer == NULL) return 0;
    *buffer = (*buffer << 1) | (spike ? 1ULL : 0ULL);
    return *buffer;
}

SNN_INLINE bool snn_bptn_latch_step(uint64_t* latch_state, int32_t mode, bool set, bool reset) {
    if (latch_state == NULL) return false;
    bool prev = (*latch_state & 1ULL) != 0;
    bool q;
    if (mode == 1) { /* SetDominant */
        q = (set || (prev && !reset));
    } else {         /* ResetDominant */
        q = (!reset && (prev || set));
    }
    *latch_state = (*latch_state << 1) | (q ? 1ULL : 0ULL);
    return q;
}

SNN_INLINE bool snn_bptn_toggle_step(uint64_t* toggle_state, bool toggle) {
    if (toggle_state == NULL) return false;
    bool prev = (*toggle_state & 1ULL) != 0;
    bool q = prev ^ toggle;
    *toggle_state = (*toggle_state << 1) | (q ? 1ULL : 0ULL);
    return q;
}

SNN_INLINE bool snn_bptn_neuron_eval(
    const uint64_t* histories,
    const uint64_t* kernels,
    const uint64_t* polarities,
    size_t channels,
    uint32_t threshold
) {
    if (histories == NULL || kernels == NULL || polarities == NULL || channels == 0) return false;
    uint32_t total_potential = 0;

    for (size_t c = 0; c < channels; ++c) {
        uint64_t aligned = histories[c] ^ polarities[c];
        uint64_t coincidences = aligned & kernels[c];
        total_potential += snn_popcount32((uint32_t)(coincidences & 0xFFFFFFFFULL)) +
                           snn_popcount32((uint32_t)(coincidences >> 32));
    }
    return total_potential >= threshold;
}

SNN_INLINE int32_t snn_bptn_layer_eval(
    const uint64_t* histories,
    size_t channels,
    const uint64_t* kernels,
    const uint64_t* polarities,
    const uint32_t* thresholds,
    size_t num_neurons,
    uint64_t* out_mask
) {
    if (histories == NULL || kernels == NULL || polarities == NULL ||
        thresholds == NULL || out_mask == NULL || num_neurons == 0 || num_neurons > 64)
    {
        return -1;
    }

    uint64_t mask = 0;
    for (size_t n = 0; n < num_neurons; ++n) {
        size_t offset = n * channels;
        if (snn_bptn_neuron_eval(histories, &kernels[offset], &polarities[offset], channels, thresholds[n])) {
            mask |= (1ULL << n);
        }
    }
    *out_mask = mask;
    return 0;
}

/**
 * @brief 1D Morphological Temporal Erosion Operator (E_kappa).
 * Eliminates isolated noise glitches of duration < 2*kappa + 1.
 */
SNN_INLINE uint64_t snn_bptn_erode_64(uint64_t x, uint8_t kappa) {
    uint64_t res = x;
    uint8_t k_max = (kappa > 63U) ? 63U : kappa;
    for (uint8_t r = 1; r <= k_max; ++r) {
        res &= (x >> r) & (x << r);
    }
    return res;
}

/**
 * @brief 1D Morphological Temporal Dilation Operator (D_kappa).
 */
SNN_INLINE uint64_t snn_bptn_dilate_64(uint64_t x, uint8_t kappa) {
    uint64_t res = x;
    uint8_t k_max = (kappa > 63U) ? 63U : kappa;
    for (uint8_t r = 1; r <= k_max; ++r) {
        res |= (x >> r) | (x << r);
    }
    return res;
}

/**
 * @brief 1D Morphological Temporal Opening Operator (O_kappa = D_kappa o E_kappa).
 * Suppresses isolated shot-noise while preserving true temporal packet envelopes.
 */
SNN_INLINE uint64_t snn_bptn_open_filter_64(uint64_t x, uint8_t kappa) {
    return snn_bptn_dilate_64(snn_bptn_erode_64(x, kappa), kappa);
}

/**
 * @brief 2-Bit Saturating Hysteresis Latch Step (4-state Smith automaton).
 * Returns true if active (state >= 2).
 */
SNN_INLINE bool snn_bptn_hysteresis_step(uint8_t* state, bool set, bool reset) {
    if (state == NULL) return false;
    uint8_t s = *state;
    if (set && !reset && s < 3U) {
        s++;
    } else if (reset && !set && s > 0U) {
        s--;
    }
    *state = s;
    return s >= 2U;
}

/**
 * @brief Robust Anti-Glitch Toggle Latch Step.
 * Requires 2 consecutive ticks of trigger validation to toggle state.
 */
SNN_INLINE bool snn_bptn_robust_toggle_step(uint64_t* state_history, bool* prev_trigger, bool trigger) {
    if (state_history == NULL || prev_trigger == NULL) return false;
    bool confirmed = trigger && (*prev_trigger);
    *prev_trigger = confirmed ? false : trigger;
    bool q_prev = ((*state_history) & 1ULL) != 0ULL;
    bool new_q = confirmed ? (!q_prev) : q_prev;
    *state_history = ((*state_history) << 1) | (new_q ? 1ULL : 0ULL);
    return new_q;
}

/**
 * @brief SWAR 8x8 Majority-Vote Temporal Pooling Operator.
 * Compresses 64 bits into 8 decision bits based on slice spike count >= threshold.
 */
SNN_INLINE uint8_t snn_bptn_majority_pool_8x8(uint64_t x, uint8_t threshold) {
    uint8_t th = (threshold == 0U) ? 1U : ((threshold > 8U) ? 8U : threshold);

    uint64_t x0 = x - ((x >> 1) & 0x5555555555555555ULL);
    uint64_t x1 = (x0 & 0x3333333333333333ULL) + ((x0 >> 2) & 0x3333333333333333ULL);
    uint64_t byte_counts = (x1 + (x1 >> 4)) & 0x0F0F0F0F0F0F0F0FULL;

    uint64_t offset = (uint64_t)((128U - (uint32_t)th) & 0xFFU);
    uint64_t offset_word = offset * 0x0101010101010101ULL;

    uint64_t res = (byte_counts + offset_word) & 0x8080808080808080ULL;

    uint8_t b0 = (uint8_t)((res >> 7) & 1ULL);
    uint8_t b1 = (uint8_t)((res >> 14) & 2ULL);
    uint8_t b2 = (uint8_t)((res >> 21) & 4ULL);
    uint8_t b3 = (uint8_t)((res >> 28) & 8ULL);
    uint8_t b4 = (uint8_t)((res >> 35) & 16ULL);
    uint8_t b5 = (uint8_t)((res >> 42) & 32ULL);
    uint8_t b6 = (uint8_t)((res >> 49) & 64ULL);
    uint8_t b7 = (uint8_t)((res >> 56) & 128ULL);

    return (uint8_t)(b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7);
}

/**
 * @brief 64-Parallel 2-Bit Saturating Smith Counter Step (HS-BPTN).
 * Evaluates 64 four-state Smith automata in a single clock cycle without branches.
 */
SNN_INLINE void snn_bptn_parallel_smith_step_u64(
    uint64_t c1,
    uint64_t c0,
    uint64_t inc,
    uint64_t dec,
    uint64_t* out_next_c1,
    uint64_t* out_next_c0
) {
    uint64_t eff_inc = inc & (~dec);
    uint64_t eff_dec = dec & (~inc);
    uint64_t none = ~(eff_inc | eff_dec);
    if (out_next_c1 != NULL) {
        *out_next_c1 = (none & c1) | (eff_inc & (c1 | c0)) | (eff_dec & (c1 & c0));
    }
    if (out_next_c0 != NULL) {
        *out_next_c0 = (none & c0) | (eff_inc & ((~c0) | c1)) | (eff_dec & (c1 & (~c0)));
    }
}


SNN_INLINE int32_t snn_readout_predict_spikes_mask(
    const int32_t* weights_q16,
    const int32_t* bias_q16,
    uint64_t spikes_mask,
    size_t num_features,
    size_t num_classes
) {
    if (weights_q16 == NULL || num_features == 0 || num_classes == 0) return -1;

    int32_t best_class = 0;
    int32_t best_logit = SNN_FIX32_MIN;

    for (size_t c = 0; c < num_classes; ++c) {
        int32_t acc = (bias_q16 != NULL) ? bias_q16[c] : 0;
        size_t class_offset = c * num_features;
        uint64_t active = spikes_mask;

        while (active != 0) {
            uint32_t bit_idx;
            uint32_t low = (uint32_t)(active & 0xFFFFFFFFULL);
            if (low != 0) {
                bit_idx = snn_ctz32(low);
            } else {
                bit_idx = 32 + snn_ctz32((uint32_t)(active >> 32));
            }
            if (bit_idx < num_features) {
                acc = snn_saturating_add_i32(acc, weights_q16[class_offset + bit_idx]);
            }
            active &= active - 1;
        }

        if (acc > best_logit) {
            best_logit = acc;
            best_class = (int32_t)c;
        }
    }

    return best_class;
}

#ifdef __cplusplus
}
#endif

#endif /* SNN_CORE_C99_H */
