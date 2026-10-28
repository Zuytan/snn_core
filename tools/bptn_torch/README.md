# 🔬 `bptn_torch`: Differentiable Training and BPTN Crystallization

This package provides the complete tooling for **Track 2** formalized in [`docs/formal_specification_bptn.md`](../../docs/formal_specification_bptn.md):
1. **Continuous differentiable training** in the unit hypercube $[0, 1]^W$ via multilinear relaxation and surrogate gradients.
2. **Concave binarization penalty** $\mathcal{R}_{\text{bin}}$ driving weights to discrete vertices $\{0, 1\}$.
3. **Lossless compilation and crystallization** into static Rust constants and functions (`u64`, `no_std`, zero heap allocation).

---

## 📂 Package Overview

- **`bptn_module.py`**: PyTorch `BPTNLayer(torch.nn.Module)` for autograd, surrogate gradients, and QAT/STE training.
- **`bptn_numpy.py`**: Standalone vectorized differentiable engine in **pure NumPy** with built-in Adam optimizer (runs without PyTorch).
- **`crystallize.py`**: Compiler applying Lemma 7.1 ($\theta = \max(0, \lceil \tilde{\theta} \rceil)$) and packing continuous masks into `u64` integers.
- **`train_shd_comparison.py`**: BPTN (STE) vs binarized network vs LIF and recurrent LIF (snnTorch) of the same size on Spiking Heidelberg Digits; reports accuracy, Flash, SRAM and an estimated Wio-E5 cycle cost. See below.
- **`train_temporal_demo.py`**: Training demonstration on the non-linear temporal XOR task ($y = s(t-4) \oplus s(t-14)$), generating Rust code and validating Theorem 7.2.

---

## 🚀 Quick Start

### 1. Run the Training and Crystallization Demo
```bash
python3 tools/bptn_torch/train_temporal_demo.py
```

### 2. Embed Crystallized Code into Rust Firmware
The compiler automatically generates code structured as:
```rust
pub const MY_MODEL_N_NEURONS: usize = 2;
pub const MY_MODEL_M_CHANNELS: usize = 1;

pub static MY_MODEL_KERNELS: [[u64; 1]; 2] = [ ... ];
pub static MY_MODEL_POLARITIES: [[u64; 1]; 2] = [ ... ];
pub static MY_MODEL_THRESHOLDS: [u32; 2] = [ ... ];

pub const fn create_my_model_layer() -> snn_core::BptnFeedforwardLayer<2, 1> {
    ...
}
```
This firmware runs bare-metal using the native `POPCNT` instruction with zero dynamic heap allocation.

---

## SHD comparison (`train_shd_comparison.py`)

```bash
python3 -m venv ~/.cache/snn_core/venv
~/.cache/snn_core/venv/bin/pip install -r tools/bptn_torch/requirements-shd.txt
# Smoke test without data:
~/.cache/snn_core/venv/bin/python tools/bptn_torch/train_shd_comparison.py --synthetic --epochs 3
# Full run (downloads SHD, about 169 MB, into ~/.cache/snn_core/shd):
~/.cache/snn_core/venv/bin/python tools/bptn_torch/train_shd_comparison.py --download --seed 0 \
    --out tools/bptn_torch/results/shd_h64_seed0.json
```

Input: the 700 SHD channels grouped into 64, the first second split into 64 bins, a bin is 1 if it
contains a spike. Every model has 64 hidden neurons and the same int8 readout to the 20 classes.
The cycle, time and energy columns are **estimates** built from per-operation costs measured on the
Wio-E5 (`CLAIMS.md`), not measurements of these models on the board.

Since `V = Σ k·(x XOR p) = x·(k·(1 − 2p)) + Σ k·p`, a BPTN neuron is a perceptron with ternary
weights {−1, 0, +1} on the flattened window; the script trains it in that form and checks the
identity against the bitwise popcount form.

The temporal LUT model (`--models lut`) uses LUT neurons with k inputs at fixed random (or local)
positions of the binary window and a group-sum readout; `--lut-sizes 4000` or `4000,2000` sets the
layers. Its per-LUT cycle cost and Flash come from `examples/wio_e5_benchmark/src/bin/lut_benchmark.rs`
(measured for k = 6 only). The script checks the trained network against an integer bit-level
reimplementation.

`--lut-learn-mapping` also learns which input bit each LUT input reads: every LUT input has a score
for each of the 4,096 positions, the forward pass reads the best one and the backward pass goes
through a softmax over the scores (straight-through). On SHD this raised 2,000 LUTs from 54 % to
71 % test accuracy (see `CLAIMS.md`). Training needs n x k x 4,096 scores (98 M for 4,000 LUTs).

`lut_inference_analysis.py` simulates, on an exported LUT network, the early exit of the group-sum
readout (exact and margin rules, margin chosen on the validation split) and the fraction of LUTs
that change per tick in incremental streaming. The matching firmware binaries are
`lut_shd_early_exit` and `lut_stream_eval`, run with `tools/run_lut_shd_on_board.py --kind lut-ee`
or `--kind lut-stream --chunk 256`.

`--models bp` trains a forest of LUT decision trees (learned branching programs): `--bp-trees` trees
of depth `--bp-depth`, each node a LUT neuron with learned inputs whose output chooses the child,
int8 class scores in the leaves. `--export-bp DIR` writes it for `bp_shd_eval`
(`tools/run_lut_shd_on_board.py --kind bp --chunk 256`). On SHD, 64 trees of depth 4 reach about
74 % in about 30 k cycles on the Wio-E5 (see `CLAIMS.md`).

`tree_baselines.py` trains classic tree ensembles (scikit-learn random forest and
HistGradientBoosting) on the same input and split, reports accuracy of the integer model, nodes read
and Flash, and exports one model (`--export DIR --export-config hgb:50x4`) for the firmware
`tree_shd_eval` (`tools/run_lut_shd_on_board.py --kind tree --chunk 200`). Requires scikit-learn.

Other tasks and inputs, all written as `.npz` files readable with `--data-npz` (and by
`tree_baselines.py` and `cnn_int8_baseline.py`): `encode_speech_commands.py` (Google Speech
Commands v2) and `encode_autohack.py` (AutoHack CAN) encode a task as 64 x 64 bits;
`make_hybrid_features.py` adds per-channel counts over fixed windows; `make_multiscale_features.py`
adds a bank of 4,788 counts over channel groups and time windows (18,460 input bits; the
uncompressed output is about 2 GB on AutoHack).

`train_lut_automaton.py` trains LUT automata on the same `.npz` inputs: M automata of S states read
one frame per tick, a k-bit LUT of the frame chooses the next state, and the int8 scores of the
visited states are summed. It checks an integer reference against PyTorch, reports the accuracy
after each number of frames read and estimates the cycles per frame (not measured).

`train_lut_boosting.py` grows the same forest of LUT trees by gradient boosting instead of
gradient descent (greedy choice of the 6 bits and of the table of every node) and exports it in
the forest format (`--export DIR`, for `bp_shd_eval`). `train_shd_comparison.py --teacher FILE`
distills a teacher (logits from `cnn_int8_baseline.py --save-teacher FILE`) and `--time-shift N`
shifts the training samples by up to N ticks; neither changes the deployed model.

