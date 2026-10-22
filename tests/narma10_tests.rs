//! 🧪 Canonical NARMA-10 Integration Test & Neuromorphic Metrics
//!
//! Validates temporal memory and non-linearity of a recurrent SNN reservoir
//! on the standard NARMA-10 task (10th-order Non-linear Auto-Regressive Moving Average).
//!
//! Standard Reservoir Computing scientific workflow:
//! 1. Recurrent SNN reservoir stimulation by temporal input signals.
//! 2. Readout weights optimization via Ridge regression (using `nalgebra`).
//! 3. Deployment into `TrainableRegressionReadout` in pure `Fix32` fixed-point arithmetic (zero FPU).
//! 4. Evaluation on an independent test set and validation of NMSE and R^2 thresholds.

use nalgebra::{DMatrix, DVector};
use snn_core::metrics::{EnergyEstimator, SparsityMeter, SynOpsTracker};
use snn_core::models::LifConfig;
use snn_core::narma::Narma10System;
use snn_core::{
    CompactLifPopulation, Fix32, GenericStaticNetwork, StaticCsrSynapses, Synapse,
    TrainableRegressionReadout,
};

#[test]
fn test_narma10_reservoir_benchmark_and_metrics() {
    const N: usize = 96;
    const N_P1: usize = 97;
    const MAX_INTERNAL: usize = 768;
    const MAX_INPUT: usize = 256;
    const N_INPUTS: usize = 10;
    const N_INPUTS_P1: usize = 11;
    const TRAIN_TICKS: usize = 800;
    const TEST_TICKS: usize = 250;

    // 1. Initialize compact LIF neuron population
    let config = LifConfig::new(Fix32::from_num(0.35), Fix32::from_num(0.85), Fix32::ZERO, 1);
    let neurons = CompactLifPopulation::<N>::new(config);

    // 2. Multi-delay input synapses (10 temporal taps into the reservoir)
    let mut input_synapses = Vec::with_capacity(MAX_INPUT);
    let mut rng = fastrand::Rng::with_seed(42);
    for tap in 0..N_INPUTS {
        for target in 0..N {
            if (tap + target) % 5 == 0 && input_synapses.len() < MAX_INPUT {
                let weight = Fix32::from_num(rng.f32() * 1.5 + 0.3);
                input_synapses.push(Synapse::new(tap, target, weight));
            }
        }
    }

    // 3. Recurrent synapses with 20% sparsity and Dale's principle (80/20)
    let mut internal_synapses = Vec::with_capacity(MAX_INTERNAL);
    for src in 0..N {
        for tgt in 0..N {
            if src != tgt && rng.f32() < 0.20 && internal_synapses.len() < MAX_INTERNAL {
                let is_inhibitory = rng.f32() < 0.20;
                let mag = Fix32::from_num(rng.f32() * 0.4 + 0.1);
                let weight = if is_inhibitory { -mag } else { mag };
                internal_synapses.push(Synapse::new(src, tgt, weight));
            }
        }
    }

    let internal_csr = StaticCsrSynapses::<N_P1, MAX_INTERNAL>::from_synapses(&internal_synapses);
    let input_csr = StaticCsrSynapses::<N_INPUTS_P1, MAX_INPUT>::from_synapses(&input_synapses);

    let mut net: GenericStaticNetwork<_, _, _, N, N_INPUTS> =
        GenericStaticNetwork::new(neurons, internal_csr, input_csr);

    // 4. Initialize standardized metrics
    let mut synops_tracker = SynOpsTracker::new();
    let mut sparsity_meter = SparsityMeter::new();
    let mut narma_sys = Narma10System::new();

    let mut state_traces = [Fix32::ZERO; N];
    let trace_decay = Fix32::from_num(0.85);

    let mut train_features = Vec::with_capacity(TRAIN_TICKS);
    let mut y_train = Vec::with_capacity(TRAIN_TICKS);
    let mut u_delay_line = [Fix32::ZERO; N_INPUTS];

    // 5. Reservoir training phase across 800 time steps
    for _ in 0..TRAIN_TICKS {
        let u_val = Fix32::from_num(rng.f32() * 0.5); // u in [0, 0.5]
        let y_target = narma_sys.step(u_val);

        // Shift temporal delay line
        for i in (1..N_INPUTS).rev() {
            u_delay_line[i] = u_delay_line[i - 1];
        }
        u_delay_line[0] = u_val;

        let spikes = *net.tick(&u_delay_line);

        // Update filtered reservoir traces
        for i in 0..N {
            let spike_val = if spikes[i] { Fix32::ONE } else { Fix32::ZERO };
            state_traces[i] = state_traces[i] * trace_decay + spike_val;
        }

        // State features matrix: [1.0 (bias), u[t]...u[t-9], traces[0]...traces[N-1]]
        let mut row = Vec::with_capacity(1 + N_INPUTS + N);
        row.push(1.0f64);
        for &tap_val in &u_delay_line {
            row.push(tap_val.to_float() as f64);
        }
        for &trace in &state_traces {
            row.push(trace.to_float() as f64);
        }
        train_features.push(row);
        y_train.push(y_target.to_float() as f64);
    }

    // 6. Ridge Regression optimization W_out = (X^T X + λI)^(-1) X^T Y
    const N_FEATURES: usize = N_INPUTS + N;
    const READOUT_DIM: usize = 1 + N_FEATURES;
    let n_cols = READOUT_DIM;
    let n_rows = TRAIN_TICKS;
    let mut flat_x = Vec::with_capacity(n_rows * n_cols);
    for col in 0..n_cols {
        for row_feats in &train_features {
            flat_x.push(row_feats[col]);
        }
    }
    let x_mat = DMatrix::from_vec(n_rows, n_cols, flat_x);
    let y_vec = DVector::from_vec(y_train);

    let xt = x_mat.transpose();
    let xtx = &xt * &x_mat;
    let reg = DMatrix::identity(n_cols, n_cols) * 1e-3;
    let w_opt = (xtx + reg)
        .svd(true, true)
        .solve(&(&xt * &y_vec), 1e-5)
        .expect("Ridge matrix inversion succeeded");

    // 7. Deploy optimized weights into pure Fix32 embedded layer
    let mut readout =
        TrainableRegressionReadout::<READOUT_DIM>::new(Fix32::from_num(0.01), Fix32::ZERO);
    for i in 0..READOUT_DIM {
        readout.weights[i] = Fix32::from_num(w_opt[i] as f32);
    }

    // 8. Evaluation phase across 250 independent steps in pure Fix32 arithmetic
    let mut test_targets = Vec::with_capacity(TEST_TICKS);
    let mut test_preds = Vec::with_capacity(TEST_TICKS);

    for _ in 0..TEST_TICKS {
        let u_val = Fix32::from_num(rng.f32() * 0.5);
        let y_target = narma_sys.step(u_val);
        test_targets.push(y_target);

        for i in (1..N_INPUTS).rev() {
            u_delay_line[i] = u_delay_line[i - 1];
        }
        u_delay_line[0] = u_val;

        let spikes = *net.tick(&u_delay_line);

        // Track hardware energy metrics
        for &spiker_idx in &net.active_spikes_buf[..net.active_spikes_count] {
            let start = net.internal_synapses.row_ptrs[spiker_idx as usize] as usize;
            let end = net.internal_synapses.row_ptrs[spiker_idx as usize + 1] as usize;
            let fan_out = end.saturating_sub(start);
            synops_tracker.record_synaptic_spikes(fan_out);
        }

        sparsity_meter.record_tick(N, net.active_spikes_count);

        for i in 0..N {
            let spike_val = if spikes[i] { Fix32::ONE } else { Fix32::ZERO };
            state_traces[i] = state_traces[i] * trace_decay + spike_val;
        }

        let mut row = [Fix32::ZERO; N_FEATURES];
        row[..N_INPUTS].copy_from_slice(&u_delay_line);
        row[N_INPUTS..].copy_from_slice(&state_traces);

        let pred = readout.predict(&row);
        test_preds.push(pred);
    }

    // 9. Validation of neuromorphic and industrial metrics
    assert!(
        synops_tracker.accumulate_ops > 0,
        "Reservoir must generate SynOps"
    );
    assert!(
        sparsity_meter.sparsity_percent() > 50.0,
        "SNN network must remain sparse (> 50%)"
    );

    let energy_pj = EnergyEstimator::estimate_picojoules(&synops_tracker);
    assert!(energy_pj > 0);

    let nmse = Narma10System::compute_nmse(&test_targets, &test_preds);
    let r2 = Narma10System::compute_r_squared(&test_targets, &test_preds);

    println!("NARMA-10 Validated -> NMSE: {:?}, R2: {:?}", nmse, r2);

    // Formal validation of convergence criteria
    assert!(
        nmse < Fix32::from_num(0.65),
        "NMSE must be < 0.65 (convergent predictive model), obtained: {:?}",
        nmse
    );
    assert!(
        r2 > Fix32::from_num(0.35),
        "R2 must be > 0.35 (temporal variance explained), obtained: {:?}",
        r2
    );
}
