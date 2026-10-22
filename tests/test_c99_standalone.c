/**
 * @file test_c99_standalone.c
 * @brief Comprehensive verification suite for snn_core_c99.h autonomous runtime.
 *
 * Compiles with: gcc -std=c99 -Wall -Wextra -Werror -pedantic tests/test_c99_standalone.c -o test_c99_standalone
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "../include/snn_core_c99.h"

static void test_fixed_point(void) {
    printf("[TEST] Fixed-Point Q16.16 Arithmetic...\n");
    snn_fix32_t a = snn_fix32_from_float(1.5f);
    snn_fix32_t b = snn_fix32_from_float(2.0f);
    snn_fix32_t sum = snn_saturating_add_i32(a, b);
    assert(snn_fix32_to_float(sum) >= 3.49f && snn_fix32_to_float(sum) <= 3.51f);

    snn_fix32_t prod = snn_fix32_mul(a, b);
    assert(snn_fix32_to_float(prod) >= 2.99f && snn_fix32_to_float(prod) <= 3.01f);

    /* Saturation tests */
    snn_fix32_t max_val = SNN_FIX32_MAX;
    snn_fix32_t sat_sum = snn_saturating_add_i32(max_val, SNN_FIX32_ONE);
    assert(sat_sum == SNN_FIX32_MAX);

    snn_fix32_t min_val = SNN_FIX32_MIN;
    snn_fix32_t sat_sub = snn_saturating_sub_i32(min_val, SNN_FIX32_ONE);
    assert(sat_sub == SNN_FIX32_MIN);

    /* Padé exp(-x) test: exp(-1.0) approx 0.367879 */
    snn_fix32_t exp_neg1 = snn_pade_exp_neg(1.0f);
    float exp_f = snn_fix32_to_float(exp_neg1);
    assert(exp_f >= 0.36f && exp_f <= 0.37f);
    printf("  -> OK (1.5 + 2.0 = 3.5, 1.5 * 2.0 = 3.0, saturation and Pade verified)\n");
}

static void test_ctz_and_bitset(void) {
    printf("[TEST] CTZ & Spike Bitset Manipulation...\n");
    assert(snn_ctz32(1U) == 0);
    assert(snn_ctz32(2U) == 1);
    assert(snn_ctz32(4U) == 2);
    assert(snn_ctz32(0x80000000U) == 31);
    assert(snn_ctz32(0U) == 32);

    assert(snn_popcount32(0U) == 0);
    assert(snn_popcount32(1U) == 1);
    assert(snn_popcount32(0xFFFFFFFFU) == 32);
    assert(snn_popcount32(0x000F000FU) == 8);

    uint32_t words[2] = { 0x00000005U, 0x80000000U }; // Bits 0, 2, and 63 set
    assert(snn_bitset_get_spike(words, 2, 0) == true);
    assert(snn_bitset_get_spike(words, 2, 1) == false);
    assert(snn_bitset_get_spike(words, 2, 2) == true);
    assert(snn_bitset_get_spike(words, 2, 63) == true);
    assert(snn_bitset_get_spike(words, 2, 64) == false);
    assert(snn_bitset_count_spikes(words, 2) == 3);
    printf("  -> OK (CTZ, POPCNT, Bitset query verified)\n");
}

static void test_aer_decoding(void) {
    printf("[TEST] AER Event Decoding (1D and 2D DVS)...\n");
    SnnAerSpike1d ev1d[3] = {
        { .address = 2,  .timestamp = 10 },
        { .address = 33, .timestamp = 15 },
        { .address = 64, .timestamp = 20 }
    };
    uint32_t out_words_1d[3] = { 0, 0, 0 };
    int32_t res1 = snn_aer_decode_1d(ev1d, 3, out_words_1d, 3);
    assert(res1 == 0);
    assert(snn_bitset_get_spike(out_words_1d, 3, 2) == true);
    assert(snn_bitset_get_spike(out_words_1d, 3, 33) == true);
    assert(snn_bitset_get_spike(out_words_1d, 3, 64) == true);
    assert(snn_bitset_get_spike(out_words_1d, 3, 0) == false);
    assert(snn_bitset_count_spikes(out_words_1d, 3) == 3);

    SnnAerSpike2d ev2d[2] = {
        { .x = 1, .y = 0, .polarity = false, .timestamp = 100 }, // idx = (0*4+1)*2 + 0 = 2
        { .x = 2, .y = 1, .polarity = true,  .timestamp = 102 }  // idx = (1*4+2)*2 + 1 = 13
    };
    uint32_t out_words_2d[1] = { 0 };
    int32_t res2 = snn_aer_decode_2d_bipolar(ev2d, 2, 4, 4, out_words_2d, 1);
    assert(res2 == 0);
    assert(snn_bitset_get_spike(out_words_2d, 1, 2) == true);
    assert(snn_bitset_get_spike(out_words_2d, 1, 13) == true);
    assert(snn_bitset_count_spikes(out_words_2d, 1) == 2);
    printf("  -> OK (AER 1D & 2D DVS bit-exact decoding)\n");
}

static void test_single_layer_network(void) {
    printf("[TEST] Single Layer Network (CSR + LIF Dynamics)...\n");
    static const uint16_t row_ptrs[3] = { 0, 2, 4 };
    static const uint16_t targets[4] = { 0, 1, 2, 3 };
    static const int32_t weights[4] = {
        98304,   // 1.5 Fix32
        32768,   // 0.5 Fix32
        131072,  // 2.0 Fix32
        49152    // 0.75 Fix32
    };

    SnnCsrMatrix in_syn = {
        .row_ptrs = row_ptrs,
        .targets = targets,
        .weights = weights,
        .in_features = 2,
        .out_features = 4,
        .total_synapses = 4
    };
    SnnCsrMatrix no_rec = { NULL, NULL, NULL, 0, 0, 0 };
    SnnLifConfig cfg = {
        .threshold = snn_fix32_from_float(1.0f),
        .decay_factor = snn_fix32_from_float(0.8f),
        .reset_potential = 0,
        .refractory_period = 1,
        .bias = 0,
        .reset_mode = 1 // Soft reset
    };
    SnnNetworkDesc desc = {
        .num_neurons = 4,
        .in_features = 2,
        .config = cfg,
        .input_synapses = in_syn,
        .recurrent_synapses = no_rec,
        .bias = NULL
    };

    uint8_t arena[128] __attribute__((aligned(4)));
    SnnNetworkState state;
    assert(snn_network_init_state(&desc, arena, sizeof(arena), &state) == 0);

    int32_t inputs[2] = { snn_fix32_from_float(1.0f), 0 };
    uint32_t out_spikes = 0;

    /* Tick 1: input 0 fires. Neurons 0 and 1 receive input:
       N0 receives 1.5 -> threshold is 1.0 -> fires! Next V = 1.5 - 1.0 = 0.5
       N1 receives 0.5 -> threshold is 1.0 -> no spike, V = 0.5
       N2, N3 receive 0 -> V = 0 */
    assert(snn_network_step(&desc, &state, inputs, 2, &out_spikes, 1) == 0);
    assert(snn_bitset_get_spike(&out_spikes, 1, 0) == true);
    assert(snn_bitset_get_spike(&out_spikes, 1, 1) == false);
    assert(state.membrane_potentials[0] >= 32700 && state.membrane_potentials[0] <= 32800);

    /* Reset */
    assert(snn_network_reset(&desc, &state) == 0);
    assert(state.membrane_potentials[0] == 0);
    printf("  -> OK (Single layer execution, soft reset, and membrane tracking)\n");
}

static void test_deep_network_pingpong(void) {
    printf("[TEST] Deep Multi-Layer Network (ActivationArena Zero-Alloc Ping-Pong)...\n");
    static const uint16_t l0_rows[3] = { 0, 1, 2 };
    static const uint16_t l0_targets[2] = { 0, 1 };
    static const int32_t  l0_weights[2] = { 98304, 98304 }; // 1.5 Fix32

    static const uint16_t l1_rows[3] = { 0, 1, 2 };
    static const uint16_t l1_targets[2] = { 1, 0 };
    static const int32_t  l1_weights[2] = { 131072, 131072 }; // 2.0 Fix32

    SnnLifConfig cfg = {
        .threshold = snn_fix32_from_float(1.0f),
        .decay_factor = snn_fix32_from_float(0.85f),
        .reset_potential = 0,
        .refractory_period = 1,
        .bias = 0,
        .reset_mode = 1
    };

    SnnLayerDesc layers[2] = {
        {
            .layer_type = SNN_LAYER_FEEDFORWARD,
            .in_features = 2,
            .num_neurons = 2,
            .config = cfg,
            .synapses = { l0_rows, l0_targets, l0_weights, 2, 2, 2 },
            .bias = NULL,
            .quant_scale = 0
        },
        {
            .layer_type = SNN_LAYER_FEEDFORWARD,
            .in_features = 2,
            .num_neurons = 2,
            .config = cfg,
            .synapses = { l1_rows, l1_targets, l1_weights, 2, 2, 2 },
            .bias = NULL,
            .quant_scale = 0
        }
    };

    SnnDeepNetworkDesc deep_desc = {
        .layers = layers,
        .num_layers = 2,
        .in_features = 2
    };

    size_t req_bytes = snn_deep_network_state_size(&deep_desc);
    assert(req_bytes > 0 && req_bytes <= 128);

    uint8_t arena[128] __attribute__((aligned(4)));
    SnnDeepNetworkState deep_state;
    assert(snn_deep_network_init_state(&deep_desc, arena, sizeof(arena), &deep_state) == 0);

    /* Inject stimulus into Layer 0: input 0 = 1.0, input 1 = 0.0 */
    int32_t in[2] = { snn_fix32_from_float(1.0f), 0 };
    uint32_t out_spikes = 0;

    /* Tick 1: Layer 0 receives 1.5 at N0 -> N0 spikes!
       Layer 1 receives spike from Layer 0 N0 -> targets N1 with weight 2.0 -> N1 spikes! */
    assert(snn_deep_network_step(&deep_desc, &deep_state, in, 2, &out_spikes, 1) == 0);
    assert(snn_bitset_get_spike(&out_spikes, 1, 1) == true);
    assert(snn_bitset_get_spike(&out_spikes, 1, 0) == false);

    assert(snn_deep_network_reset(&deep_desc, &deep_state) == 0);

    /* Also verify SnnModelDescriptor & SnnExecutionContext parity aliases */
    const SnnModelDescriptor* model = &deep_desc;
    SnnExecutionContext ctx;
    assert(snn_model_memory_footprint(model) == req_bytes);
    assert(snn_model_init_context(model, arena, sizeof(arena), &ctx) == 0);
    assert(snn_model_step(model, &ctx, in, 2, &out_spikes, 1) == 0);
    assert(snn_bitset_get_spike(&out_spikes, 1, 1) == true);
    assert(snn_model_reset(model, &ctx) == 0);

    printf("  -> OK (2-layer ping-pong propagation through SRAM arena bit-exact, SnnModelDescriptor verified)\n");
}

static void test_quantized_int8_layer(void) {
    printf("[TEST] Quantized INT8 Synapses with Fix32 Scale...\n");
    static const uint16_t q_rows[3] = { 0, 1, 2 };
    static const uint16_t q_targets[2] = { 0, 1 };
    static const int8_t q_weights_i8[2] = { 100, -50 };

    SnnLifConfig cfg = {
        .threshold = snn_fix32_from_float(1.0f),
        .decay_factor = snn_fix32_from_float(0.9f),
        .reset_potential = 0,
        .refractory_period = 1,
        .bias = 0,
        .reset_mode = 0
    };

    SnnLayerDesc layer = {
        .layer_type = SNN_LAYER_QUANTIZED_INT8,
        .in_features = 2,
        .num_neurons = 2,
        .config = cfg,
        .synapses = { q_rows, q_targets, (const int32_t*)q_weights_i8, 2, 2, 2 },
        .bias = NULL,
        .quant_scale = snn_fix32_from_float(0.02f) // 100 * 0.02 = 2.0 Fix32
    };

    SnnDeepNetworkDesc desc = {
        .layers = &layer,
        .num_layers = 1,
        .in_features = 2
    };

    uint8_t arena[128] __attribute__((aligned(4)));
    SnnDeepNetworkState state;
    assert(snn_deep_network_init_state(&desc, arena, sizeof(arena), &state) == 0);

    int32_t in[2] = { snn_fix32_from_float(1.0f), 0 };
    uint32_t out_spikes = 0;
    assert(snn_deep_network_step(&desc, &state, in, 2, &out_spikes, 1) == 0);
    assert(snn_bitset_get_spike(&out_spikes, 1, 0) == true);
    printf("  -> OK (Quantized INT8 weights scaled correctly)\n");
}

static void test_dense_readout(void) {
    printf("[TEST] Dense Classifier Readout with CTZ...\n");
    static const int32_t w[8] = {
        65536,  -32768,  65536, -65536, // Class 0
        -65536,  65536, -32768,  65536  // Class 1
    };
    uint32_t spikes = 0x00000005U; // Neurons 0 and 2 fired
    int32_t logits[2] = { 0, 0 };
    assert(snn_readout_dense(w, NULL, &spikes, 4, 2, logits) == 0);
    /* Class 0: w[0] + w[2] = 1.0 + 1.0 = 2.0 (131072)
       Class 1: w[4] + w[6] = -1.0 + -0.5 = -1.5 (-98304) */
    assert(logits[0] == 131072);
    assert(logits[1] == -98304);
    printf("  -> OK (Readout logits bit-exact with zero multiplications)\n");
}

static void test_bptn_word_ram(void) {
    printf("[TEST] BPTN (Bit-Parallel Temporal Neural Network)...\n");
    uint64_t frame = 0;
    assert(snn_bptn_framing_push(&frame, true) == 1ULL);
    assert(snn_bptn_framing_push(&frame, false) == 2ULL);
    assert(snn_bptn_framing_push(&frame, true) == 5ULL);

    uint64_t latch = 0;
    assert(snn_bptn_latch_step(&latch, 1, true, false) == true);
    assert(snn_bptn_latch_step(&latch, 1, false, false) == true);
    assert(snn_bptn_latch_step(&latch, 1, false, true) == false);

    uint64_t toggle = 0;
    assert(snn_bptn_toggle_step(&toggle, true) == true);
    assert(snn_bptn_toggle_step(&toggle, true) == false);
    assert(snn_bptn_toggle_step(&toggle, false) == false);

    uint64_t history[1] = { 0x0000000000000007ULL }; // 3 spikes
    uint64_t kernel[1]  = { 0x0000000000000005ULL }; // matches 2 spikes
    uint64_t polar[1]   = { 0ULL };
    assert(snn_bptn_neuron_eval(history, kernel, polar, 1, 2) == true);
    assert(snn_bptn_neuron_eval(history, kernel, polar, 1, 3) == false);

    // Morphological operators
    uint64_t noisy = (1ULL << 2) | (1ULL << 10) | (1ULL << 11) | (1ULL << 12);
    assert(snn_bptn_erode_64(noisy, 1) == (1ULL << 11));
    assert(snn_bptn_open_filter_64(noisy, 1) == ((1ULL << 10) | (1ULL << 11) | (1ULL << 12)));

    // HS-BPTN Latch (4-state Smith automaton)
    uint8_t h_state = 0;
    assert(snn_bptn_hysteresis_step(&h_state, true, false) == false); // state 1 -> false
    assert(snn_bptn_hysteresis_step(&h_state, true, false) == true);  // state 2 -> true
    assert(snn_bptn_hysteresis_step(&h_state, false, true) == false); // state 1 -> false

    // Robust Toggle Cell (2-tick confirmation)
    uint64_t rob_hist = 0;
    bool prev_trig = false;
    assert(snn_bptn_robust_toggle_step(&rob_hist, &prev_trig, true) == false); // 1 tick candidate
    assert(snn_bptn_robust_toggle_step(&rob_hist, &prev_trig, true) == true);  // 2 ticks confirmed -> toggle
    assert(snn_bptn_robust_toggle_step(&rob_hist, &prev_trig, false) == true); // holds state

    // SWAR 8x8 Majority Pool
    uint64_t swar_x = 0x03ULL | (0x01ULL << 8) | (0x0FULL << 16);
    assert(snn_bptn_majority_pool_8x8(swar_x, 2) == 0x05U);
    assert(snn_bptn_majority_pool_8x8(swar_x, 3) == 0x04U);

    // 64-Parallel Smith Counter Step
    uint64_t c1 = 0, c0 = 0, n_c1 = 0, n_c0 = 0;
    snn_bptn_parallel_smith_step_u64(c1, c0, 1ULL, 0ULL, &n_c1, &n_c0);
    assert((n_c1 & 1ULL) == 0ULL && (n_c0 & 1ULL) == 1ULL); // state 1
    snn_bptn_parallel_smith_step_u64(n_c1, n_c0, 1ULL, 0ULL, &c1, &c0);
    assert((c1 & 1ULL) == 1ULL && (c0 & 1ULL) == 0ULL);     // state 2 (active!)

    printf("  -> OK (BPTN framing, morphological opening, 64-way Smith counter, SWAR majority pool)\n");
}

static void test_c99_conv2d_maxpool2d(void) {
    printf("[TEST] Conv2D & MaxPool2D Pipeline...\n");
    // Conv2D 1x1: 1 in_ch, 1 out_ch, 4x4 image, kernel weight = 2.0 (131072)
    static const int32_t k_weight[1] = { 131072 };
    SnnConv2dDesc conv_desc = {
        .kernel = k_weight,
        .in_channels = 1,
        .out_channels = 1,
        .kernel_h = 1,
        .kernel_w = 1,
        .stride_h = 1,
        .stride_w = 1,
        .padding_h = 0,
        .padding_w = 0,
        .in_h = 4,
        .in_w = 4,
        .out_h = 4,
        .out_w = 4
    };
    SnnLifConfig lif_cfg = {
        .threshold = snn_fix32_from_float(1.0f),
        .decay_factor = 0,
        .reset_potential = 0,
        .refractory_period = 0,
        .bias = 0,
        .reset_mode = 1
    };
    SnnCsrMatrix empty_syn = { NULL, NULL, NULL, 0, 0, 0 };

    SnnMaxPool2dDesc pool_desc = {
        .channels = 1,
        .in_h = 4,
        .in_w = 4,
        .kernel_h = 2,
        .kernel_w = 2,
        .stride_h = 2,
        .stride_w = 2,
        .out_h = 2,
        .out_w = 2
    };

    SnnLayerDesc layers[2] = {
        {
            .layer_type = SNN_LAYER_CONV2D,
            .in_features = 16,
            .num_neurons = 16,
            .config = lif_cfg,
            .synapses = empty_syn,
            .conv2d = conv_desc,
            .pool2d = {0},
            .recurrent_synapses = NULL,
            .bias = NULL,
            .quant_scale = 0
        },
        {
            .layer_type = SNN_LAYER_MAX_POOL_2D,
            .in_features = 16,
            .num_neurons = 4,
            .config = lif_cfg,
            .synapses = empty_syn,
            .conv2d = {0},
            .pool2d = pool_desc,
            .recurrent_synapses = NULL,
            .bias = NULL,
            .quant_scale = 0
        }
    };

    SnnDeepNetworkDesc net = {
        .layers = layers,
        .num_layers = 2,
        .in_features = 16
    };

    uint8_t arena[512] __attribute__((aligned(4)));
    SnnDeepNetworkState state;
    assert(snn_deep_network_init_state(&net, arena, sizeof(arena), &state) == 0);

    // Inject 1.5 Fix32 at position (0, 0)
    int32_t inputs[16] = {0};
    inputs[0] = snn_fix32_from_float(1.5f);
    uint32_t out_spikes = 0;

    assert(snn_deep_network_step(&net, &state, inputs, 16, &out_spikes, 1) == 0);
    // Pooled output neuron 0 covers (0,0)-(1,1) -> should spike!
    assert(snn_bitset_get_spike(&out_spikes, 1, 0) == true);
    // Other pooled regions (1, 2, 3) had no input -> should not spike
    assert(snn_bitset_get_spike(&out_spikes, 1, 1) == false);
    assert(snn_bitset_get_spike(&out_spikes, 1, 2) == false);
    assert(snn_bitset_get_spike(&out_spikes, 1, 3) == false);
    printf("  -> OK (Conv2D continuous injection -> LIF -> MaxPool2D bitwise OR)\n");
}

static void test_c99_recurrent_synapses(void) {
    printf("[TEST] Recurrent Synapses (W_rec Feedback)...\n");
    // Feedforward: input 0 -> neuron 0 (weight 1.5)
    static const uint16_t ff_row_ptrs[2] = { 0, 1 };
    static const uint16_t ff_targets[1]  = { 0 };
    static const int32_t  ff_weights[1]  = { 98304 }; // 1.5 Fix32
    SnnCsrMatrix ff_syn = {
        .row_ptrs = ff_row_ptrs,
        .targets = ff_targets,
        .weights = ff_weights,
        .in_features = 1,
        .out_features = 2,
        .total_synapses = 1
    };

    // Recurrent: neuron 0 -> neuron 1 (weight 1.5)
    static const uint16_t rec_row_ptrs[3] = { 0, 1, 1 };
    static const uint16_t rec_targets[1]  = { 1 };
    static const int32_t  rec_weights[1]  = { 98304 }; // 1.5 Fix32
    static const SnnCsrMatrix rec_syn = {
        .row_ptrs = rec_row_ptrs,
        .targets = rec_targets,
        .weights = rec_weights,
        .in_features = 2,
        .out_features = 2,
        .total_synapses = 1
    };

    SnnLifConfig cfg = {
        .threshold = snn_fix32_from_float(1.0f),
        .decay_factor = 0,
        .reset_potential = 0,
        .refractory_period = 0,
        .bias = 0,
        .reset_mode = 0 // Hard reset
    };

    SnnLayerDesc layer = {
        .layer_type = SNN_LAYER_FEEDFORWARD,
        .in_features = 1,
        .num_neurons = 2,
        .config = cfg,
        .synapses = ff_syn,
        .conv2d = {0},
        .pool2d = {0},
        .recurrent_synapses = &rec_syn,
        .bias = NULL,
        .quant_scale = 0
    };

    SnnDeepNetworkDesc net = {
        .layers = &layer,
        .num_layers = 1,
        .in_features = 1
    };

    uint8_t arena[256] __attribute__((aligned(4)));
    SnnDeepNetworkState state;
    assert(snn_deep_network_init_state(&net, arena, sizeof(arena), &state) == 0);

    // Tick 1: input = 1.0 -> neuron 0 receives 1.5 -> fires! Neuron 1 receives 0.
    int32_t in_tick1[1] = { snn_fix32_from_float(1.0f) };
    uint32_t out_spikes1 = 0;
    assert(snn_deep_network_step(&net, &state, in_tick1, 1, &out_spikes1, 1) == 0);
    assert(snn_bitset_get_spike(&out_spikes1, 1, 0) == true);
    assert(snn_bitset_get_spike(&out_spikes1, 1, 1) == false);

    // Tick 2: input = 0.0 -> neuron 0 receives 0 -> no spike.
    // Recurrent feedback from neuron 0 (which spiked at tick 1) arrives at neuron 1 with weight 1.5 -> neuron 1 fires!
    int32_t in_tick2[1] = { 0 };
    uint32_t out_spikes2 = 0;
    assert(snn_deep_network_step(&net, &state, in_tick2, 1, &out_spikes2, 1) == 0);
    assert(snn_bitset_get_spike(&out_spikes2, 1, 0) == false);
    assert(snn_bitset_get_spike(&out_spikes2, 1, 1) == true);

    printf("  -> OK (W_rec recurrent spike retention and 1-tick delayed activation verified)\n");
}

int main(void) {
    printf("====================================================================\n");
    printf("🚀 VALIDATION SUITE: snn_core_c99.h Standalone C99 Runtime\n");
    printf("====================================================================\n");

    test_fixed_point();
    test_ctz_and_bitset();
    test_aer_decoding();
    test_single_layer_network();
    test_deep_network_pingpong();
    test_quantized_int8_layer();
    test_c99_conv2d_maxpool2d();
    test_c99_recurrent_synapses();
    test_dense_readout();
    test_bptn_word_ram();

    printf("\n====================================================================\n");
    printf("✅ ALL C99 TESTS PASSED STRICTLY (100%% Bit-Exact, Zero-Alloc) !\n");
    printf("====================================================================\n");
    return 0;
}
