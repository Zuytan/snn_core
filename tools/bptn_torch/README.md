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
