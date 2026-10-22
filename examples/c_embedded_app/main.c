/**
 * @file main.c
 * @brief 🚀 C/C++ Industrial Integration Example for snn_core
 *
 * Demonstrates how to use snn_core from standard C firmware (FreeRTOS / STM32Cube / Zephyr):
 * 1. Decoding a 1D AER event stream
 * 2. Manipulating a compact SpikeBitset (1 bit per neuron)
 * 3. Hardware spike counting (POPCNT) and sparsity calculation
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "../../include/snn_core.h"

int main(void) {
    printf("========================================================\n");
    printf("🧠 snn_core : C Embedded / RTOS Firmware Demo\n");
    printf("========================================================\n\n");

    // 1. Convert analog sensor voltage to Fix32 (Q16.16)
    float sensor_voltage = 1.65f;
    int32_t q16_val = snn_fix32_from_float(sensor_voltage);
    printf("[Analog] Sensor voltage converted to Fix32 Q16.16: %d (0x%08X)\n", q16_val, q16_val);
    printf("[Analog] Reconverted float value                  : %.4f V\n\n", snn_fix32_to_float(q16_val));

    // 2. Receive AER spike train (e.g. 64-channel cochlear audio sensor)
    // 3 neurons fire a spike at this timestep (channels 5, 23, 42)
    SnnAerSpike1d raw_aer_events[3] = {
        { .address = 5,  .timestamp = 1000 },
        { .address = 23, .timestamp = 1005 },
        { .address = 42, .timestamp = 1012 }
    };

    // Bitset output buffer: 64 neurons = 2x 32-bit words (8 bytes of SRAM only!)
    uint32_t bitset_words[2] = { 0, 0 };

    // Zero-allocation direct decoding
    int32_t status = snn_aer_decode_1d(raw_aer_events, 3, bitset_words, 2);
    if (status != 0) {
        printf("Error during AER decoding\n");
        return -1;
    }

    printf("[AER] Decoded events in SpikeBitset (8 bytes):\n");
    printf("      Word 0 (neurons  0..31): 0x%08X\n", bitset_words[0]);
    printf("      Word 1 (neurons 32..63): 0x%08X\n\n", bitset_words[1]);

    // 3. Query neuron firing state (O(1) branchless query)
    printf("[Query] Target neuron states:\n");
    printf("        - Neuron  5: %s (Expected: YES)\n", snn_bitset_get_spike(bitset_words, 2, 5) ? "ACTIVE" : "silent");
    printf("        - Neuron 10: %s (Expected: NO)\n", snn_bitset_get_spike(bitset_words, 2, 10) ? "ACTIVE" : "silent");
    printf("        - Neuron 23: %s (Expected: YES)\n", snn_bitset_get_spike(bitset_words, 2, 23) ? "ACTIVE" : "silent");
    printf("        - Neuron 42: %s (Expected: YES)\n", snn_bitset_get_spike(bitset_words, 2, 42) ? "ACTIVE" : "silent");

    // 4. Hardware POPCNT spike counting
    uint32_t total_spikes = snn_bitset_count_spikes(bitset_words, 2);
    float sparsity = (1.0f - ((float)total_spikes / 64.0f)) * 100.0f;
    printf("\n[Sparsity] Active spikes: %u / 64 neurons\n", total_spikes);
    printf("[Sparsity] Silence rate : %.1f %% (Neuromorphic energy advantage)\n\n", sparsity);

    // 5. Execute a complete SNN in pure C (Zero-Allocation SRAM Arena)
    printf("--------------------------------------------------------\n");
    printf("🧠 SNN Network Execution (4 LIF neurons, 2 inputs)\n");
    printf("--------------------------------------------------------\n");

    // Input synapses defined in Flash ROM (CSR format)
    static const uint16_t in_row_ptrs[3] = { 0, 2, 4 };
    static const uint16_t in_targets[4] = { 0, 1, 2, 3 };
    static const int32_t in_weights[4] = {
        98304,   // 1.5 in Q16.16 to neuron 0
        32768,   // 0.5 in Q16.16 to neuron 1
        131072,  // 2.0 in Q16.16 to neuron 2
        49152    // 0.75 in Q16.16 to neuron 3
    };

    SnnCsrMatrix in_synapses = {
        .row_ptrs = in_row_ptrs,
        .targets = in_targets,
        .weights = in_weights,
        .in_features = 2,
        .out_features = 4,
        .total_synapses = 4
    };

    SnnCsrMatrix no_recurrent = {
        .row_ptrs = NULL,
        .targets = NULL,
        .weights = NULL,
        .in_features = 0,
        .out_features = 0,
        .total_synapses = 0
    };

    SnnLifConfig lif_cfg = {
        .threshold = snn_fix32_from_float(1.0f),
        .decay_factor = snn_fix32_from_float(0.85f),
        .reset_potential = snn_fix32_from_float(0.0f),
        .refractory_period = 1,
        .bias = snn_fix32_from_float(0.0f),
        .reset_mode = 1 // Soft Reset (Subtract)
    };

    SnnNetworkDesc net_desc = {
        .num_neurons = 4,
        .in_features = 2,
        .config = lif_cfg,
        .input_synapses = in_synapses,
        .recurrent_synapses = no_recurrent,
        .bias = NULL
    };

    // Statically allocated SRAM state arena (zero heap allocation)
    uint8_t state_arena[SNN_NETWORK_STATE_SIZE(4)] __attribute__((aligned(4)));
    SnnNetworkState net_state;

    int32_t init_err = snn_network_init_state(&net_desc, state_arena, sizeof(state_arena), &net_state);
    if (init_err != 0) {
        printf("Error during network state initialization (code: %d)\n", init_err);
        return -1;
    }
    printf("[Init] Network state initialized in SRAM (%zu bytes statically allocated)\n", sizeof(state_arena));

    // Simulation of 3 timesteps (ticks)
    int32_t inputs[2] = {
        snn_fix32_from_float(1.0f), // Input 0 = 1.0 V
        snn_fix32_from_float(0.0f)  // Input 1 = 0.0 V
    };
    uint32_t out_spikes_word = 0;

    for (int tick = 1; tick <= 3; ++tick) {
        snn_network_step(&net_desc, &net_state, inputs, 2, &out_spikes_word, 1);
        printf("[Tick %d] Potentials: [N0: %.2fV, N1: %.2fV, N2: %.2fV, N3: %.2fV] | Spikes: 0x%X\n",
               tick,
               snn_fix32_to_float(net_state.membrane_potentials[0]),
               snn_fix32_to_float(net_state.membrane_potentials[1]),
               snn_fix32_to_float(net_state.membrane_potentials[2]),
               snn_fix32_to_float(net_state.membrane_potentials[3]),
               out_spikes_word);
    }

    // 6. Dense Readout Classification Layer (2 classes: Class A, Class B)
    static const int32_t readout_weights[8] = {
        65536,  -32768,  65536, -65536, // Class A
        -65536,  65536, -32768,  65536  // Class B
    };
    int32_t logits[2] = { 0, 0 };

    snn_readout_dense(readout_weights, NULL, &out_spikes_word, 4, 2, logits);
    printf("\n[Readout] Output logits: Class A = %.2f, Class B = %.2f\n",
           snn_fix32_to_float(logits[0]), snn_fix32_to_float(logits[1]));
    printf("[Decision] Predicted class: %s\n", logits[0] > logits[1] ? "CLASS A" : "CLASS B");

    // Network reset to resting potential
    snn_network_reset(&net_desc, &net_state);
    printf("[Reset] Network reset to resting potential.\n");

    // 7. Deep Multilayer Network with ActivationArena (Zero-Alloc Ping-Pong)
    printf("\n--------------------------------------------------------\n");
    printf("🏟️ Deep Multilayer Inference (ActivationArena C API)\n");
    printf("--------------------------------------------------------\n");

    // Define 2 layers: L0 (2 neurons) -> L1 (2 neurons)
    static const uint16_t l0_rows[3] = { 0, 1, 2 };
    static const uint16_t l0_tgts[2] = { 0, 1 };
    static const int32_t  l0_wgts[2] = { 98304, 98304 }; // 1.5 in Q16.16

    static const uint16_t l1_rows[3] = { 0, 1, 2 };
    static const uint16_t l1_tgts[2] = { 1, 0 };
    static const int32_t  l1_wgts[2] = { 131072, 131072 }; // 2.0 in Q16.16

    SnnLayerDesc deep_layers[2] = {
        {
            .layer_type = SNN_LAYER_FEEDFORWARD,
            .in_features = 2,
            .num_neurons = 2,
            .config = lif_cfg,
            .synapses = {
                .row_ptrs = l0_rows,
                .targets = l0_tgts,
                .weights = l0_wgts,
                .in_features = 2,
                .out_features = 2,
                .total_synapses = 2
            },
            .bias = NULL,
            .quant_scale = 0
        },
        {
            .layer_type = SNN_LAYER_FEEDFORWARD,
            .in_features = 2,
            .num_neurons = 2,
            .config = lif_cfg,
            .synapses = {
                .row_ptrs = l1_rows,
                .targets = l1_tgts,
                .weights = l1_wgts,
                .in_features = 2,
                .out_features = 2,
                .total_synapses = 2
            },
            .bias = NULL,
            .quant_scale = 0
        }
    };

    SnnDeepNetworkDesc deep_desc = {
        .layers = deep_layers,
        .num_layers = 2,
        .in_features = 2
    };

    size_t arena_bytes = snn_deep_network_state_size(&deep_desc);
    printf("[DeepArena] Required SRAM size for 2 shared layers: %zu bytes\n", arena_bytes);

    uint8_t deep_arena[128] __attribute__((aligned(4)));
    SnnDeepNetworkState deep_state;
    snn_deep_network_init_state(&deep_desc, deep_arena, sizeof(deep_arena), &deep_state);

    uint32_t deep_out_spikes = 0;
    int32_t deep_inputs[2] = { snn_fix32_from_float(1.2f), 0 };
    snn_deep_network_step(&deep_desc, &deep_state, deep_inputs, 2, &deep_out_spikes, 1);

    printf("[DeepArena] Step executed: Final layer output spikes = 0x%02X (N1 active: %s)\n",
           deep_out_spikes,
           snn_bitset_get_spike(&deep_out_spikes, 1, 1) ? "YES" : "NO");

    snn_deep_network_reset(&deep_desc, &deep_state);
    printf("[DeepArena] Multilayer arena successfully reset.\n");

    printf("\n========================================================\n");
    printf("✅ C firmware demo executed successfully!\n");
    printf("========================================================\n");

    return 0;
}
