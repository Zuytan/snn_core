/**
 * @file main.c
 * @brief Zephyr RTOS Spiking Neural Network (SNN) Inference Demonstration
 *
 * Demonstrates zero-allocation neuromorphic edge inference using `snn_core`.
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <snn_core.h>

#define NUM_INPUTS   4
#define NUM_NEURONS  8

/* 1. Synaptic Matrix in Flash ROM (.rodata) */
static const uint16_t ROW_PTRS[NUM_INPUTS + 1] = { 0, 2, 4, 6, 8 };
static const uint16_t TARGETS[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
static const int32_t WEIGHTS[8] = {
    65536,  /* 1.0 in Fix32 */
    -32768, /* -0.5 in Fix32 */
    49152,  /* 0.75 in Fix32 */
    98304,  /* 1.5 in Fix32 */
    -16384, /* -0.25 in Fix32 */
    65536,  /* 1.0 in Fix32 */
    32768,  /* 0.5 in Fix32 */
    81920   /* 1.25 in Fix32 */
};

static const SnnCsrMatrix INPUT_SYNAPSES = {
    .row_ptrs = ROW_PTRS,
    .targets = TARGETS,
    .weights = WEIGHTS,
    .in_features = NUM_INPUTS,
    .out_features = NUM_NEURONS,
    .total_synapses = 8,
};

/* 2. Network Descriptor in Flash ROM (.rodata) */
static const SnnNetworkDesc NETWORK_DESC = {
    .num_neurons = NUM_NEURONS,
    .in_features = NUM_INPUTS,
    .config = {
        .threshold = 65536,          /* 1.0 in Fix32 */
        .decay_factor = 58982,       /* exp(-1ms/20ms) = 0.9 in Fix32 */
        .reset_potential = 0,        /* 0.0 in Fix32 */
        .refractory_period = 2,      /* 2 discrete ticks */
        .bias = 0,
        .reset_mode = 1,             /* Soft / Subtract reset */
    },
    .input_synapses = INPUT_SYNAPSES,
    .recurrent_synapses = { 0 },
    .bias = NULL,
};

/* 3. Static SRAM Arena (Zero Heap Allocation) */
static uint8_t s_arena_buffer[SNN_NETWORK_STATE_SIZE(NUM_NEURONS)] __aligned(4);
static SnnNetworkState s_network_state;

int main(void)
{
    printk("=== 🧠 Zephyr RTOS : snn_core Edge Neuromorphic Inference ===\n");

    /* Initialize network state inside SRAM arena */
    int32_t rc = snn_network_init_state(&NETWORK_DESC, s_arena_buffer, sizeof(s_arena_buffer), &s_network_state);
    if (rc != 0) {
        printk("Error: Failed to initialize network state (rc=%d)\n", rc);
        return rc;
    }
    printk("Network state initialized successfully (%u bytes in SRAM)\n", (unsigned int)sizeof(s_arena_buffer));

    /* Continuous input vector in Fix32 (Q16.16) */
    int32_t inputs[NUM_INPUTS] = {
        snn_fix32_from_float(1.5f),
        snn_fix32_from_float(0.5f),
        snn_fix32_from_float(-0.2f),
        snn_fix32_from_float(2.0f),
    };

    /* Simulation loop over 10 discrete ticks (1ms each) */
    for (int tick = 0; tick < 10; tick++) {
        snn_network_tick_inputs(&NETWORK_DESC, &s_network_state, inputs, NUM_INPUTS);

        uint32_t active_spikes = snn_bitset_count_spikes(s_network_state.last_spikes, s_network_state.words_count);
        printk("Tick %d: %u active spikes emitted\n", tick, active_spikes);

        k_msleep(10);
    }

    printk("✅ snn_core Zephyr inference completed successfully.\n");
    return 0;
}
