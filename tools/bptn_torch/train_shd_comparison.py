#!/usr/bin/env python3
"""
train_shd_comparison.py

Compares models of the same size on Spiking Heidelberg Digits (SHD, 20 classes):

1. BPTN: one layer of H BPTN neurons over the whole 64-tick window, trained with a straight-through
   estimator (STE) on binary kernel and polarity masks, integer thresholds at deployment.
2. BNN: one binarized layer of H neurons (weights in {-1, +1}, binary activations, STE), same input.
3. LIF: one feedforward layer of H leaky integrate-and-fire neurons (snnTorch), stepped over the
   64 ticks, same input; `rlif` adds H x H recurrent weights (snnTorch RLeaky).
4. LUT (`lut`): temporal LUT neurons. Each reads k bits at fixed positions of the input history,
   forms a k-bit index and outputs the bit stored at that index of a learned table of 2^k entries.
   One or more LUT layers, then a group-sum readout (the LUTs of the last layer are split into 20
   groups; the class score is the number of 1s in its group). Trained with the multilinear
   extension of the lookup and a straight-through estimator on the table bits. Sizes are set with
   `--lut-sizes` (for example 2048 or 4096,2000) and do not follow `--hidden`.
5. Learned branching programs (`bp`): a forest of T binary decision trees of depth D whose nodes are
   LUT neurons (k learned input bits, learned table). The node output chooses the child; the leaf
   reached adds a vector of 20 class scores (int8 when deployed). One inference reads T x D nodes
   and T leaves, not the whole model. Training routes hard in the forward pass (as deployed) and
   passes gradients through the routing probabilities (straight-through).

BPTN, BNN and LIF use the same readout: an int8 linear layer from the H hidden outputs to the 20
classes. The LUT model uses a group-sum readout instead (no weights).

The script reports, for each model: test accuracy (at the epoch with the best validation accuracy),
accuracy after quantization to the deployed format, Flash and SRAM computed from the data layout,
and an estimate of the Wio-E5 cycle cost built from the per-operation costs measured on the board
(CLAIMS.md). The cycle figures are estimates, not measurements of these models on the board.

Input encoding (same for all models): the 700 SHD channels are grouped into 64 channels, the first
`--t-max` seconds are split into 64 time bins, and a bin is 1 if at least one spike falls in it.

Key identity used for BPTN: with x, k, p in {0, 1},
    V = sum k * (x XOR p) = x . (k * (1 - 2p)) + sum(k * p)
so a BPTN neuron is a perceptron with ternary weights {-1, 0, +1} on the flattened window plus a
constant offset. The script checks this identity against the bitwise popcount form.

Usage:
    python3 tools/bptn_torch/train_shd_comparison.py --data-dir ~/.cache/snn_core/shd
    python3 tools/bptn_torch/train_shd_comparison.py --synthetic --epochs 3   # smoke test, no download

Requirements: numpy, torch, snntorch, h5py (see requirements-shd.txt).
"""

import argparse
import gzip
import json
import math
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch is required: pip install -r tools/bptn_torch/requirements-shd.txt")

N_CLASSES = 20
SHD_CHANNELS = 700
SHD_URL = "https://zenkelab.org/datasets/{name}.h5.gz"

# ----------------------------------------------------------------------------
# Per-operation costs measured on the Wio-E5 (STM32WLE5JC, 48 MHz, 2 Flash wait states).
# Source: CLAIMS.md, measurements of 2026-10-25. Used only to estimate cycle costs.
# ----------------------------------------------------------------------------
CLOCK_HZ = 48_000_000
ASSUMED_ACTIVE_POWER_W = 3.3 * 4.2e-3  # assumed run current, not measured
# BPTN layer 16 x 16 (256 neuron-channel words of 64 bits): 11,972 cycles -> per word.
CYCLES_PER_WINDOW_WORD = 11_972 / 256
# LIF neuron step (subtract reset) and CSR propagation per synapse (Fix32 and INT8).
CYCLES_LIF_STEP = 17.0
CYCLES_SYNAPSE_FIX32 = 20.9
CYCLES_SYNAPSE_INT8 = 18.5
# Shifting one input spike into a 64-bit framing buffer: NOT measured, assumed.
CYCLES_FRAMING_PUSH_ASSUMED = 4.0
# Temporal LUT neuron, k = 6, group-sum included (examples/wio_e5_benchmark/src/bin/lut_benchmark.rs):
# generated code with constant bit positions, and the same layer driven by an address table.
CYCLES_LUT6_CODEGEN = 49.72
CYCLES_LUT6_TABLE = 77.68
FLASH_LUT6_CODEGEN_BYTES = 24_874 / 256  # measured code size of the generated layer, per LUT
FLASH_LUT6_TABLE_BYTES = 6 * 2 + 8  # 6 (word, bit) byte pairs + a 64-bit table
# LUT evaluation inside a large model on the board (2,000-LUT network: 223,791 cycles / 2,000),
# used for the branching-program nodes; the cost of adding a 20-class int8 leaf is assumed.
CYCLES_LUT6_BOARD = 111.9
CYCLES_LEAF_ADD_ASSUMED = 100.0


# ============================================================================
# Data
# ============================================================================
def download_shd(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("shd_train", "shd_test"):
        h5 = data_dir / f"{name}.h5"
        if h5.exists():
            continue
        gz = data_dir / f"{name}.h5.gz"
        if not gz.exists():
            url = SHD_URL.format(name=name)
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, gz)
        with gzip.open(gz, "rb") as src, open(h5, "wb") as dst:
            shutil.copyfileobj(src, dst)


def encode_shd(h5_path: Path, channels: int, ticks: int, t_max: float) -> tuple:
    """Bins one SHD file into a binary tensor [N, channels, ticks] (tick 0 = oldest)."""
    import h5py

    group = math.ceil(SHD_CHANNELS / channels)
    with h5py.File(h5_path, "r") as f:
        times = f["spikes"]["times"]
        units = f["spikes"]["units"]
        labels = np.asarray(f["labels"], dtype=np.int64)
        x = np.zeros((len(labels), channels, ticks), dtype=np.uint8)
        for i in range(len(labels)):
            t = np.asarray(times[i])
            u = np.asarray(units[i])
            keep = t < t_max
            ch = np.minimum(u[keep] // group, channels - 1)
            tk = np.minimum((t[keep] / t_max * ticks).astype(np.int64), ticks - 1)
            x[i, ch, tk] = 1
    return x, labels


def load_shd(data_dir: Path, channels: int, ticks: int, t_max: float, download: bool):
    cache = data_dir / f"shd_encoded_c{channels}_t{ticks}_{t_max:.2f}s.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    if download:
        download_shd(data_dir)
    for name in ("shd_train", "shd_test"):
        if not (data_dir / f"{name}.h5").exists():
            sys.exit(f"{data_dir / name}.h5 not found; pass --download or place the SHD files there.")
    x_train, y_train = encode_shd(data_dir / "shd_train.h5", channels, ticks, t_max)
    x_test, y_test = encode_shd(data_dir / "shd_test.h5", channels, ticks, t_max)
    np.savez_compressed(cache, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return x_train, y_train, x_test, y_test


def synthetic_dataset(channels: int, ticks: int, seed: int = 0):
    """Small labelled dataset with class-specific spatio-temporal patterns, for smoke tests."""
    rng = np.random.default_rng(seed)
    prototypes = rng.random((N_CLASSES, channels, ticks)) < 0.08

    def make(n):
        y = rng.integers(0, N_CLASSES, n)
        noise = rng.random((n, channels, ticks)) < 0.03
        keep = rng.random((n, channels, ticks)) < 0.7
        x = (prototypes[y] & keep) | noise
        return x.astype(np.uint8), y

    x_train, y_train = make(2000)
    x_test, y_test = make(500)
    return x_train, y_train, x_test, y_test


# ============================================================================
# Shared pieces
# ============================================================================
class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside step forward, fast-sigmoid surrogate gradient backward."""

    @staticmethod
    def forward(ctx, v, slope):
        ctx.save_for_backward(v)
        ctx.slope = slope
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad):
        (v,) = ctx.saved_tensors
        return grad / (ctx.slope * v.abs() + 1.0).pow(2), None


def spike(v, slope=5.0):
    return FastSigmoidSpike.apply(v, slope)


def ste_binary(p_continuous):
    """Forward: p >= 0.5 in {0, 1}; backward: identity on the continuous value."""
    hard = (p_continuous >= 0.5).to(p_continuous.dtype)
    return hard.detach() - p_continuous.detach() + p_continuous


def quantize_int8(w: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor int8 quantization, returned dequantized (for accuracy evaluation)."""
    scale = w.abs().max().clamp(min=1e-8) / 127.0
    return torch.round(w / scale).clamp(-127, 127) * scale


class Readout(nn.Module):
    """Linear readout from H hidden outputs to the classes; int8 weights when deployed."""

    def __init__(self, hidden):
        super().__init__()
        self.fc = nn.Linear(hidden, N_CLASSES)
        self.deployed = False

    def forward(self, h):
        if self.deployed:
            return F.linear(h, quantize_int8(self.fc.weight), self.fc.bias)
        return self.fc(h)


# ============================================================================
# Models
# ============================================================================
class BptnNet(nn.Module):
    """H BPTN neurons over the full window, then the readout."""

    def __init__(self, channels, ticks, hidden):
        super().__init__()
        self.fan_in = channels * ticks
        self.raw_kernel = nn.Parameter(torch.randn(hidden, self.fan_in) * 0.5 - 1.0)
        self.raw_polarity = nn.Parameter(torch.randn(hidden, self.fan_in) * 0.5)
        self.threshold = nn.Parameter(torch.zeros(hidden))
        self.scale = math.sqrt(self.fan_in) / 4.0
        self.readout = Readout(hidden)
        self.deployed = False
        self.thresholds_initialized = False

    def masks(self):
        k = torch.sigmoid(self.raw_kernel)
        p = torch.sigmoid(self.raw_polarity)
        if self.deployed:
            return (k >= 0.5).float(), (p >= 0.5).float()
        return ste_binary(k), ste_binary(p)

    def potential(self, x_flat):
        k, p = self.masks()
        w = k * (1.0 - 2.0 * p)  # ternary {-1, 0, +1}
        offset = (k * p).sum(dim=1)
        return x_flat @ w.t() + offset

    def integer_thresholds(self):
        # V is an integer, so V >= theta <=> V >= ceil(theta).
        return torch.ceil(self.threshold.detach()).clamp(min=0)

    def hidden_spikes(self, x_flat):
        v = self.potential(x_flat)
        if not self.thresholds_initialized and self.training:
            with torch.no_grad():
                self.threshold.copy_(v.mean(dim=0))
            self.thresholds_initialized = True
        if self.deployed:
            return (v >= self.integer_thresholds()).float()
        return spike((v - self.threshold) / self.scale)

    def forward(self, x):
        h = self.hidden_spikes(x.flatten(1))
        return self.readout(h), h


class BnnNet(nn.Module):
    """H binarized neurons (weights +-1, inputs mapped to +-1), then the readout."""

    def __init__(self, channels, ticks, hidden):
        super().__init__()
        self.fan_in = channels * ticks
        self.weight = nn.Parameter(torch.empty(hidden, self.fan_in).uniform_(-1, 1))
        self.threshold = nn.Parameter(torch.zeros(hidden))
        self.scale = math.sqrt(self.fan_in) / 4.0
        self.readout = Readout(hidden)
        self.deployed = False
        self.thresholds_initialized = False

    def binary_weight(self):
        w = self.weight.clamp(-1, 1)
        hard = torch.where(w >= 0, 1.0, -1.0)
        if self.deployed:
            return hard
        return hard.detach() - w.detach() + w

    def hidden_spikes(self, x_flat):
        v = (2.0 * x_flat - 1.0) @ self.binary_weight().t()  # XNOR-popcount form
        if not self.thresholds_initialized and self.training:
            with torch.no_grad():
                self.threshold.copy_(v.mean(dim=0))
            self.thresholds_initialized = True
        if self.deployed:
            return (v >= torch.ceil(self.threshold.detach())).float()
        return spike((v - self.threshold) / self.scale)

    def forward(self, x):
        h = self.hidden_spikes(x.flatten(1))
        return self.readout(h), h


class LifNet(nn.Module):
    """C inputs -> H LIF neurons stepped over the ticks; readout on the hidden spike counts.

    With `recurrent=True` the hidden layer also has H x H recurrent weights (snnTorch RLeaky).
    """

    def __init__(self, channels, ticks, hidden, beta=0.9, init_gain=1.0, recurrent=False):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate

        self.ticks = ticks
        self.fc = nn.Linear(channels, hidden)
        # The default init gives currents far below the threshold of 1 on sparse binary inputs, so
        # almost no neuron fires at the start; a larger gain lets training start.
        with torch.no_grad():
            self.fc.weight.mul_(init_gain)
        self.recurrent = recurrent
        if recurrent:
            self.lif = snn.RLeaky(
                beta=beta,
                learn_beta=True,
                linear_features=hidden,
                spike_grad=surrogate.fast_sigmoid(slope=5),
                reset_mechanism="subtract",
            )
        else:
            self.lif = snn.Leaky(
                beta=beta,
                learn_beta=True,
                spike_grad=surrogate.fast_sigmoid(slope=5),
                reset_mechanism="subtract",
            )
        self.readout = Readout(hidden)
        self.deployed = False

    def forward(self, x):
        # x: [B, C, T]; tick 0 is the oldest.
        weight = quantize_int8(self.fc.weight) if self.deployed else self.fc.weight
        if self.recurrent:
            spk, mem = self.lif.init_rleaky()
        else:
            mem = self.lif.init_leaky()
        count = 0.0
        for t in range(self.ticks):
            cur = F.linear(x[:, :, t], weight, self.fc.bias)
            if self.recurrent:
                spk, mem = self.lif(cur, spk, mem)
            else:
                spk, mem = self.lif(cur, mem)
            count = count + spk
        # Spike counts normalized by the window length (a rate in [0, 1]).
        rate = count / self.ticks
        return self.readout(rate), count


class LutLayer(nn.Module):
    """n LUT neurons with k inputs each, reading k positions of a binary input vector.

    The lookup is trained through its multilinear extension: with soft inputs x in [0, 1], the
    output is sum_a table[a] * prod_i (x_i if a_i else 1 - x_i), which equals the lookup on binary
    inputs. Table bits are binarized in the forward pass (straight-through estimator), so the
    training forward pass is the deployed computation.

    With `learn_mapping`, each of the n * k inputs has a score for every input position (as in
    Differentiable Weightless Neural Networks): the forward pass reads the position with the
    highest score, and the backward pass goes through a softmax over the scores (straight-through).
    The scores can start with a bonus on the initial `mapping` (`map_init`; 0 by default, which
    worked better on SHD than starting from the fixed random mapping).
    """

    def __init__(self, in_features, n, k, mapping, learn_mapping=False, map_init=0.0, map_tau=1.0):
        super().__init__()
        self.k = k
        self.n = n
        self.learn_mapping = learn_mapping
        self.map_tau = map_tau
        if learn_mapping:
            scores = torch.randn(n * k, in_features) * 0.1
            scores[torch.arange(n * k), mapping.flatten()] += map_init
            self.map_scores = nn.Parameter(scores)
            self.register_buffer("initial_index", mapping.clone())
        else:
            self.register_buffer("fixed_index", mapping)  # [n, k] positions in the input
        self.logits = nn.Parameter(torch.randn(n, 2**k) * 0.1)
        self.deployed = False

    def index(self):
        """Current [n, k] input positions (the deployed connections)."""
        if self.learn_mapping:
            return self.map_scores.detach().argmax(dim=1).view(self.n, self.k)
        return self.fixed_index

    def select(self, x):
        hard = x[:, self.index()]  # [B, n, k]
        if not self.learn_mapping or self.deployed:
            return hard
        soft = (x @ torch.softmax(self.map_scores / self.map_tau, dim=1).t()).view(-1, self.n, self.k)
        return hard + soft - soft.detach()

    def table(self):
        t = torch.sigmoid(self.logits)
        hard = (t >= 0.5).float()
        if self.deployed:
            return hard
        return hard.detach() - t.detach() + t

    def forward(self, x):
        sel = self.select(x)  # [B, n, k]
        weights = torch.ones(x.shape[0], self.n, 1, device=x.device)
        for i in range(self.k):
            xi = sel[:, :, i : i + 1]
            # Address bit i has weight 2^i, as in the firmware: addresses with bit i = 1 are the
            # upper half of the new vector.
            weights = torch.cat((weights * (1 - xi), weights * xi), dim=-1)
        return (weights * self.table()).sum(-1)


def lut_address_order_ok(k):
    """Checks that the weight vector built in LutLayer.forward puts address a at index a."""
    layer = LutLayer(k, 1, k, torch.arange(k).view(1, k))
    with torch.no_grad():
        layer.logits.zero_()
        for a in range(2**k):
            bits = torch.tensor([[(a >> i) & 1 for i in range(k)]], dtype=torch.float32)
            layer.logits.fill_(-10.0)
            layer.logits[0, a] = 10.0
            layer.deployed = True
            if layer(bits).item() != 1.0:
                return False
    return True


def lut_mapping(in_channels, in_ticks, n, k, kind, generator):
    """Positions of the k inputs of each LUT in the flattened [channels, ticks] input."""
    if kind == "random":
        return torch.randint(0, in_channels * in_ticks, (n, k), generator=generator)
    # "local": all k inputs within a small spatio-temporal neighbourhood (+-4 channels, +-8 ticks).
    c0 = torch.randint(0, in_channels, (n, 1), generator=generator)
    t0 = torch.randint(0, in_ticks, (n, 1), generator=generator)
    dc = torch.randint(-4, 5, (n, k), generator=generator)
    dt = torch.randint(-8, 9, (n, k), generator=generator)
    c = (c0 + dc).clamp(0, in_channels - 1)
    t = (t0 + dt).clamp(0, in_ticks - 1)
    return c * in_ticks + t


class LutNet(nn.Module):
    """LUT layers on the binary input window, then a group-sum readout."""

    def __init__(self, channels, ticks, sizes, k, mapping, tau, seed, learn_mapping=False, map_init=0.0, map_tau=1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        layers = []
        in_features = channels * ticks
        for depth, n in enumerate(sizes):
            if depth == 0:
                idx = lut_mapping(channels, ticks, n, k, mapping, g)
            else:
                idx = torch.randint(0, in_features, (n, k), generator=g)
            layers.append(LutLayer(in_features, n, k, idx, learn_mapping, map_init, map_tau))
            in_features = n
        if sizes[-1] % N_CLASSES:
            sys.exit("the last LUT layer size must be a multiple of 20 (group-sum readout)")
        self.layers = nn.ModuleList(layers)
        self.tau = tau
        self.deployed = False

    def forward(self, x):
        h = x.flatten(1)
        for i, layer in enumerate(self.layers):
            layer.deployed = self.deployed
            h = layer(h)
            if i < len(self.layers) - 1 and not self.deployed:
                h = ste_binary(h)
        scores = h.view(h.shape[0], N_CLASSES, -1).sum(-1)
        return scores / self.tau, h


class BranchingNet(nn.Module):
    """Forest of `trees` LUT decision trees of depth `depth` (heap order: children of node i are
    2i + 1 for output 0 and 2i + 2 for output 1), with learned connections; leaves hold class scores."""

    def __init__(self, channels, ticks, trees, depth, k, seed, map_tau=1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.trees, self.depth = trees, depth
        self.nodes_per_tree = 2**depth - 1
        self.leaves_per_tree = 2**depth
        n = trees * self.nodes_per_tree
        idx = torch.randint(0, channels * ticks, (n, k), generator=g)
        self.nodes = LutLayer(channels * ticks, n, k, idx, learn_mapping=True, map_init=0.0, map_tau=map_tau)
        self.leaf = nn.Parameter(torch.randn(trees, self.leaves_per_tree, N_CLASSES) * 0.01)
        self.deployed = False

    def leaf_values(self):
        return quantize_int8(self.leaf) if self.deployed else self.leaf

    def forward(self, x):
        self.nodes.deployed = self.deployed
        b = x.shape[0]
        out = self.nodes(x.flatten(1)).view(b, self.trees, self.nodes_per_tree)  # hard in forward
        prob = torch.ones(b, self.trees, 1, device=x.device)
        for level in range(self.depth):
            o = out[:, :, 2**level - 1 : 2 ** (level + 1) - 1]
            prob = torch.stack((prob * (1 - o), prob * o), dim=-1).flatten(2)
        logits = torch.einsum("btl,tlc->bc", prob, self.leaf_values())
        return logits, prob.argmax(-1).float()


def branching_integer_reference(model: BranchingNet, x_flat: np.ndarray):
    """Deployed forest with integer operations, as the firmware runs it. Returns
    (predictions, node positions [n, k], tables [n] as u64, int8 leaves [T, L, 20])."""
    set_deployed(model, True)
    idx = model.nodes.index().cpu().numpy().astype(np.int64)
    bits = model.nodes.table().detach().cpu().numpy().astype(np.uint64)
    tables = (bits << np.arange(bits.shape[1], dtype=np.uint64)[None, :]).sum(axis=1, dtype=np.uint64)
    w = model.leaf.detach().cpu()
    scale = w.abs().max().clamp(min=1e-8) / 127.0
    leaves = torch.round(w / scale).clamp(-127, 127).numpy().astype(np.int8)
    nn_, nt = model.nodes_per_tree, model.trees
    scores = np.zeros((len(x_flat), N_CLASSES), dtype=np.int64)
    for t in range(nt):
        node = np.zeros(len(x_flat), dtype=np.int64)
        for _ in range(model.depth):
            n = t * nn_ + node
            address = np.zeros(len(x_flat), dtype=np.uint64)
            for j in range(idx.shape[1]):
                address |= x_flat[np.arange(len(x_flat)), idx[n, j]].astype(np.uint64) << np.uint64(j)
            o = ((tables[n] >> address) & np.uint64(1)).astype(np.int64)
            node = 2 * node + 1 + o
        scores += leaves[t, node - nn_].astype(np.int64)
    return scores.argmax(1), idx, tables, leaves


def export_branching(model: BranchingNet, x_test, y_test, out_dir: Path, meta: dict):
    """Writes the forest and the packed test set for `bp_shd_eval.rs`: bp_addr.bin (n x k x 2
    bytes: word, bit), bp_table.bin (n x u64), bp_leaf.bin (T x L x 20 int8), test_x.bin,
    test_y.bin, host_pred.bin (integer reference), meta.json. Nodes are stored tree by tree."""
    x_flat = x_test.reshape(len(x_test), -1).astype(np.uint8)
    pred, idx, tables, leaves = branching_integer_reference(model, x_flat)
    with torch.no_grad():
        logits, _ = model(torch.as_tensor(x_test, dtype=torch.float32, device=model.leaf.device))
    agree = float((logits.argmax(1).cpu().numpy() == pred).mean())
    addr = np.stack((idx // 32, idx % 32), axis=-1).astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bp_addr.bin").write_bytes(addr.tobytes())
    (out_dir / "bp_table.bin").write_bytes(tables.astype("<u8").tobytes())
    (out_dir / "bp_leaf.bin").write_bytes(leaves.tobytes())
    (out_dir / "test_x.bin").write_bytes(pack_bits(x_flat).astype("<u4").tobytes())
    (out_dir / "test_y.bin").write_bytes(y_test.astype(np.uint8).tobytes())
    (out_dir / "host_pred.bin").write_bytes(pred.astype(np.uint8).tobytes())
    acc = float((pred == y_test).mean())
    meta = dict(meta, trees=model.trees, depth=model.depth, k=int(idx.shape[1]), words=int(x_flat.shape[1] // 32),
                samples=int(len(x_flat)), host_integer_accuracy=acc, torch_integer_agreement=agree)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  exported to {out_dir}: integer reference accuracy {acc * 100:.2f}%, agreement with PyTorch {agree * 100:.2f}%")


# ============================================================================
# Training and evaluation
# ============================================================================
def set_deployed(model, flag):
    model.deployed = flag
    if hasattr(model, "readout"):
        model.readout.deployed = flag


def evaluate(model, x, y, device, batch=512):
    model.eval()
    correct, hidden_activity = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(x[i : i + batch], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[i : i + batch], device=device)
            logits, h = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            hidden_activity += h.sum().item()
    return correct / len(x), hidden_activity / len(x)


def time_shift(xb, max_shift, rng):
    """Shifts each sample of xb [B, C, T] by a random number of ticks in [-max_shift, max_shift],
    filling with zeros (training-time augmentation; the deployed model is unchanged)."""
    t = xb.shape[-1]
    shifts = torch.as_tensor(rng.integers(-max_shift, max_shift + 1, len(xb)), device=xb.device)
    src = torch.arange(t, device=xb.device)[None, :] - shifts[:, None]  # [B, T]
    valid = ((src >= 0) & (src < t)).to(xb.dtype)
    out = torch.gather(xb, 2, src.clamp(0, t - 1)[:, None, :].expand(-1, xb.shape[1], -1))
    return out * valid[:, None, :]


def distillation_loss(logits, yb, teacher_logits, alpha, temperature):
    """(1 - alpha) * cross-entropy + alpha * T^2 * KL(teacher || student) at temperature T (Hinton et al.)."""
    hard = F.cross_entropy(logits, yb)
    soft = F.kl_div(F.log_softmax(logits / temperature, dim=1), F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean") * temperature**2
    return (1 - alpha) * hard + alpha * soft


def train(model, x_tr, y_tr, x_val, y_val, args, device, teacher=None):
    if isinstance(model, LifNet):
        lr = args.lr_lif
    elif isinstance(model, (LutNet, BranchingNet)):
        lr = args.lr_lut
    else:
        lr = args.lr_binary
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = (-1.0, None)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        set_deployed(model, False)
        order = rng.permutation(len(x_tr))
        t0 = time.time()
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            xb = torch.as_tensor(x_tr[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_tr[idx], device=device)
            if args.time_shift:
                xb = time_shift(xb, args.time_shift, rng)
            logits, _ = model(xb)
            if teacher is not None:
                tb = torch.as_tensor(teacher[idx], device=device)
                loss = distillation_loss(logits, yb, tb, args.distill_alpha, args.distill_temp)
            else:
                loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        val_acc, _ = evaluate(model, x_val, y_val, device)
        if val_acc > best[0]:
            best = (val_acc, {k: v.detach().clone() for k, v in model.state_dict().items()})
        print(
            f"  epoch {epoch + 1:3d}/{args.epochs}  loss {loss.item():.3f}  "
            f"val {val_acc * 100:5.1f}%  ({time.time() - t0:.1f}s)"
        )
    model.load_state_dict(best[1])
    return best[0]


# ============================================================================
# Footprint and cost estimates
# ============================================================================
@dataclass
class Result:
    model: str
    hidden: int
    val_acc: float
    test_acc_train_format: float
    test_acc_deployed: float
    flash_bytes: int
    sram_bytes: int
    est_cycles_per_sample: float
    est_ms_per_sample: float
    est_uj_per_sample: float
    hidden_activity_per_sample: float
    notes: str


def readout_flash(hidden):
    return hidden * N_CLASSES + N_CLASSES * 4  # int8 weights + int32 biases


def readout_cycles(active_hidden):
    return active_hidden * N_CLASSES * CYCLES_SYNAPSE_INT8


def binary_layer_cost(channels, ticks, hidden, active_hidden, input_spikes):
    words = hidden * channels * math.ceil(ticks / 64)
    framing = channels * ticks * CYCLES_FRAMING_PUSH_ASSUMED
    return words * CYCLES_PER_WINDOW_WORD + framing + readout_cycles(active_hidden)


def lif_cost(channels, ticks, hidden, input_spikes, hidden_spikes):
    steps = ticks * hidden * CYCLES_LIF_STEP
    propagation = input_spikes * hidden * CYCLES_SYNAPSE_INT8
    # Readout once per sample on the spike counts (dense: H x classes).
    return steps + propagation + hidden * N_CLASSES * CYCLES_SYNAPSE_INT8


def to_time_energy(cycles):
    seconds = cycles / CLOCK_HZ
    return seconds * 1e3, seconds * ASSUMED_ACTIVE_POWER_W * 1e6


# ============================================================================
# Sanity check of the BPTN identity against the bitwise popcount form
# ============================================================================
def check_bptn_identity(model: BptnNet, x: np.ndarray, channels: int, ticks: int):
    set_deployed(model, True)
    k, p = model.masks()
    k = k.cpu().numpy().astype(bool).reshape(-1, channels, ticks)
    p = p.cpu().numpy().astype(bool).reshape(-1, channels, ticks)
    xs = x[:8].astype(bool)
    popcount = np.array(
        [[np.sum(k[n] & (xs[i] ^ p[n])) for n in range(k.shape[0])] for i in range(len(xs))]
    )
    with torch.no_grad():
        flat = torch.as_tensor(x[:8].reshape(len(xs), -1), dtype=torch.float32, device=model.threshold.device)
        matmul = model.potential(flat).cpu().numpy()
    if not np.allclose(popcount, matmul):
        sys.exit("BPTN identity check failed: popcount and ternary matmul disagree")
    print("  BPTN identity check: popcount form == ternary matmul form on 8 samples")


def check_lut_bitlevel(model: LutNet, x: np.ndarray, n_samples=200):
    """Recomputes the deployed LUT network with integer bit operations, as the firmware does."""
    set_deployed(model, True)
    xs = x[:n_samples].reshape(min(n_samples, len(x)), -1).astype(np.int64)
    h = xs
    for layer in model.layers:
        idx = layer.index().cpu().numpy()
        table = layer.table().detach().cpu().numpy().astype(np.int64)
        address = np.zeros((h.shape[0], idx.shape[0]), dtype=np.int64)
        for i in range(idx.shape[1]):
            address |= h[:, idx[:, i]] << i
        h = table[np.arange(idx.shape[0])[None, :], address]
    scores = h.reshape(h.shape[0], N_CLASSES, -1).sum(-1)
    with torch.no_grad():
        logits, _ = model(torch.as_tensor(x[:n_samples], dtype=torch.float32, device=model.layers[0].logits.device))
    torch_scores = (logits * model.tau).round().cpu().numpy().astype(np.int64)
    if not np.array_equal(scores, torch_scores):
        sys.exit("LUT bit-level check failed: integer lookup and PyTorch model disagree")
    print(f"  LUT bit-level check: integer lookups == PyTorch model on {h.shape[0]} samples")


def export_lut(model: LutNet, x_test: np.ndarray, y_test: np.ndarray, out_dir: Path, meta: dict):
    """Writes a one-layer LUT network and the packed test set for the Wio-E5 firmware.

    Layout (little-endian), matching `lut_shd_eval.rs`:
      lut_addr.bin   n x k x 2 bytes: (word, bit) of each LUT input; input position f = c * T + t
                     is bit f % 32 of u32 word f // 32
      lut_table.bin  n x u64: bit a of the word is the LUT output for address a (input j = bit j)
      test_x.bin     samples x (C * T / 32) u32 words, packed the same way
      test_y.bin     samples x u8 labels
      host_pred.bin  samples x u8 predictions of the integer reference below (argmax, lowest
                     class on ties), which the firmware must reproduce exactly
    LUT n belongs to class group n // (n_luts / 20).
    """
    if len(model.layers) != 1:
        sys.exit("export supports one LUT layer")
    set_deployed(model, True)
    layer = model.layers[0]
    idx = layer.index().cpu().numpy().astype(np.int64)  # [n, k]
    if idx.max() >= 256 * 32:
        sys.exit("input too large for byte word indices")
    table_bits = layer.table().detach().cpu().numpy().astype(np.uint64)  # [n, 2^k]
    if table_bits.shape[1] != 64:
        sys.exit("export supports k = 6 (64-entry tables)")
    tables = (table_bits << np.arange(64, dtype=np.uint64)[None, :]).sum(axis=1, dtype=np.uint64)
    addr = np.stack((idx // 32, idx % 32), axis=-1).astype(np.uint8)  # [n, k, 2]

    flat = x_test.reshape(len(x_test), -1).astype(np.uint32)
    n_words = flat.shape[1] // 32
    weights = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    words = (flat.reshape(len(flat), n_words, 32) * weights).sum(-1, dtype=np.uint32)

    # Integer reference, written the way the firmware computes it.
    address = np.zeros((len(flat), idx.shape[0]), dtype=np.int64)
    for j in range(idx.shape[1]):
        address |= flat[:, idx[:, j]].astype(np.int64) << j
    outputs = (tables[None, :] >> address.astype(np.uint64)) & np.uint64(1)
    counts = outputs.reshape(len(flat), N_CLASSES, -1).sum(-1)
    host_pred = counts.argmax(1).astype(np.uint8)

    with torch.no_grad():
        logits, _ = model(torch.as_tensor(x_test, dtype=torch.float32, device=layer.logits.device))
    torch_counts = (logits * model.tau).round().cpu().numpy().astype(np.int64)
    if not np.array_equal(torch_counts, counts):
        sys.exit("export check failed: integer reference and PyTorch disagree")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lut_addr.bin").write_bytes(addr.tobytes())
    (out_dir / "lut_table.bin").write_bytes(tables.astype("<u8").tobytes())
    (out_dir / "test_x.bin").write_bytes(words.astype("<u4").tobytes())
    (out_dir / "test_y.bin").write_bytes(y_test.astype(np.uint8).tobytes())
    (out_dir / "host_pred.bin").write_bytes(host_pred.tobytes())
    acc = float((host_pred == y_test).mean())
    meta = dict(meta, n_luts=int(idx.shape[0]), k=int(idx.shape[1]), words=int(n_words),
                groups=N_CLASSES, samples=int(len(flat)), host_integer_accuracy=acc)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  exported to {out_dir}: integer reference accuracy {acc * 100:.2f}% on {len(flat)} test samples")


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Packs [..., D] bits into [..., D / 32] u32 words: bit f % 32 of word f // 32 is element f."""
    words = bits.reshape(*bits.shape[:-1], -1, 32).astype(np.uint32)
    return (words << np.arange(32, dtype=np.uint32)).sum(-1, dtype=np.uint32)


def export_bnn(model: BnnNet, x_test: np.ndarray, y_test: np.ndarray, out_dir: Path, meta: dict):
    """Writes the binarized network and the packed test set for `bnn_shd_eval.rs`.

    Integer model, as the firmware computes it (D = channels x ticks inputs, H hidden neurons):
      mismatches_n = sum over words of popcount(x_word XOR w_word[n])   (w bit 1 means weight +1)
      hidden_n     = mismatches_n <= max_mismatch[n]                    (same as V_n >= ceil(theta_n),
                                                                        with V_n = D - 2 mismatches_n)
      score_c      = sum over firing n of readout_w[c][n] + readout_b[c] (int8 weights, int32 bias)
      prediction   = argmax score (lowest class on ties)
    The int32 bias is round(bias / scale) of the int8 readout, so this integer model can differ
    from the PyTorch deployed model on rare near-ties; both accuracies are recorded.
    Files (little-endian): bnn_weights.bin H x (D / 32) u32, bnn_max_mismatch.bin H x i32,
    readout_w.bin 20 x H i8, readout_b.bin 20 x i32, test_x.bin, test_y.bin, host_pred.bin.
    """
    set_deployed(model, True)
    w_bits = (model.binary_weight().detach().cpu().numpy() > 0).astype(np.uint8)  # [H, D]
    d = w_bits.shape[1]
    theta = np.ceil(model.threshold.detach().cpu().numpy()).astype(np.int64)
    max_mismatch = np.floor((d - theta) / 2).astype(np.int32)
    w = model.readout.fc.weight.detach().cpu()
    scale = w.abs().max().clamp(min=1e-8) / 127.0
    w_int = torch.round(w / scale).clamp(-127, 127).numpy().astype(np.int8)  # [20, H]
    b_int = torch.round(model.readout.fc.bias.detach().cpu() / scale).numpy().astype(np.int32)

    x_flat = x_test.reshape(len(x_test), -1).astype(np.uint8)
    mismatches = (x_flat[:, None, :] != w_bits[None, :, :]).sum(-1)  # [N, H]
    hidden = (mismatches <= max_mismatch[None, :]).astype(np.int64)
    scores = hidden @ w_int.astype(np.int64).T + b_int[None, :]
    host_pred = scores.argmax(1).astype(np.uint8)

    with torch.no_grad():
        logits, _ = model(torch.as_tensor(x_test, dtype=torch.float32, device=model.weight.device))
    torch_pred = logits.argmax(1).cpu().numpy()
    agree = float((torch_pred == host_pred).mean())

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bnn_weights.bin").write_bytes(pack_bits(w_bits).astype("<u4").tobytes())
    (out_dir / "bnn_max_mismatch.bin").write_bytes(max_mismatch.astype("<i4").tobytes())
    (out_dir / "readout_w.bin").write_bytes(w_int.tobytes())
    (out_dir / "readout_b.bin").write_bytes(b_int.astype("<i4").tobytes())
    (out_dir / "test_x.bin").write_bytes(pack_bits(x_flat).astype("<u4").tobytes())
    (out_dir / "test_y.bin").write_bytes(y_test.astype(np.uint8).tobytes())
    (out_dir / "host_pred.bin").write_bytes(host_pred.tobytes())
    acc = float((host_pred == y_test).mean())
    meta = dict(meta, hidden=int(w_bits.shape[0]), inputs=int(d), words=int(d // 32), groups=N_CLASSES,
                samples=int(len(x_flat)), host_integer_accuracy=acc, torch_integer_agreement=agree)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"  exported to {out_dir}: integer reference accuracy {acc * 100:.2f}% on {len(x_flat)} test "
        f"samples, agreement with PyTorch {agree * 100:.2f}%"
    )


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path.home() / ".cache" / "snn_core" / "shd")
    ap.add_argument("--download", action="store_true", help="download SHD (about 169 MB) if missing")
    ap.add_argument("--synthetic", action="store_true", help="use a small synthetic dataset (smoke test)")
    ap.add_argument("--data-npz", type=Path, default=None,
                    help="encoded dataset (.npz with x_train, y_train, x_test, y_test) instead of SHD")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=64)
    ap.add_argument("--t-max", type=float, default=1.0, help="seconds of each sample kept")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr-binary", type=float, default=5e-3)
    ap.add_argument("--lr-lif", type=float, default=2e-3)
    ap.add_argument("--lif-init-gain", type=float, default=4.0)
    ap.add_argument("--lif-beta", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="bptn,bnn,lif,rlif")
    ap.add_argument("--lut-sizes", default="2000", help="LUTs per layer, e.g. 2000 or 4000,2000")
    ap.add_argument("--lut-k", type=int, default=6)
    ap.add_argument("--lut-mapping", choices=("random", "local"), default="local")
    ap.add_argument("--lut-tau", type=float, default=None, help="group-sum temperature")
    ap.add_argument("--lr-lut", type=float, default=1e-2)
    ap.add_argument("--lut-learn-mapping", action="store_true", help="learn the LUT input positions")
    ap.add_argument("--lut-map-init", type=float, default=0.0, help="score bonus of the initial positions")
    ap.add_argument("--lut-map-tau", type=float, default=1.0, help="softmax temperature of the scores")
    ap.add_argument("--out", type=Path, default=None, help="JSON file for the results")
    ap.add_argument("--export-lut", type=Path, default=None, help="directory for the firmware export of the LUT model")
    ap.add_argument("--export-bnn", type=Path, default=None, help="directory for the firmware export of the BNN model")
    ap.add_argument("--bp-trees", type=int, default=64)
    ap.add_argument("--bp-depth", type=int, default=6)
    ap.add_argument("--export-bp", type=Path, default=None, help="directory for the firmware export of the branching model")
    ap.add_argument("--teacher", type=Path, default=None,
                    help="teacher logits on the whole training set (.npy from cnn_int8_baseline.py --save-teacher)")
    ap.add_argument("--distill-alpha", type=float, default=0.7, help="weight of the distillation term")
    ap.add_argument("--distill-temp", type=float, default=4.0, help="distillation temperature")
    ap.add_argument("--time-shift", type=int, default=0, help="random shift of training samples by up to N ticks")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    global N_CLASSES
    if args.synthetic:
        x_train, y_train, x_test, y_test = synthetic_dataset(args.channels, args.ticks, args.seed)
        dataset = "synthetic"
    elif args.data_npz:
        # Any dataset encoded the same way: x [N, channels, ticks] in {0, 1}, integer labels.
        d = np.load(args.data_npz.expanduser())
        x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
        dataset = args.data_npz.stem
        args.channels, args.ticks = x_train.shape[1], x_train.shape[2]
        N_CLASSES = int(max(y_train.max(), y_test.max()) + 1)
    else:
        x_train, y_train, x_test, y_test = load_shd(
            args.data_dir.expanduser(), args.channels, args.ticks, args.t_max, args.download
        )
        dataset = "SHD"

    # Validation split from the training set (SHD has no official validation set).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(x_train))
    n_val = len(x_train) // 10
    x_val, y_val = x_train[perm[:n_val]], y_train[perm[:n_val]]
    x_tr, y_tr = x_train[perm[n_val:]], y_train[perm[n_val:]]
    teacher = None
    if args.teacher:
        teacher_all = np.load(args.teacher.expanduser())
        assert len(teacher_all) == len(x_train), "teacher logits must cover the training set in file order"
        teacher = teacher_all[perm[n_val:]]
        t_acc = (teacher.argmax(1) == y_tr).mean() * 100
        print(f"distillation from {args.teacher.name} (train accuracy {t_acc:.1f}%), "
              f"alpha {args.distill_alpha}, T {args.distill_temp}")
    input_spikes = float(x_test.reshape(len(x_test), -1).sum(1).mean())
    print(
        f"{dataset}: {len(x_tr)} train / {len(x_val)} val / {len(x_test)} test, "
        f"input {args.channels} ch x {args.ticks} ticks, density {x_test.mean() * 100:.1f}%, "
        f"{input_spikes:.0f} active bins per sample, device {device}"
    )

    c, t, h = args.channels, args.ticks, args.hidden
    results = []
    for name in args.models.split(","):
        torch.manual_seed(args.seed)
        if name == "bptn":
            model = BptnNet(c, t, h)
        elif name == "bnn":
            model = BnnNet(c, t, h)
        elif name == "lut":
            if not lut_address_order_ok(args.lut_k):
                sys.exit("LUT address order check failed")
            sizes = [int(v) for v in args.lut_sizes.split(",")]
            tau = args.lut_tau or math.sqrt(sizes[-1] / N_CLASSES)
            model = LutNet(
                c, t, sizes, args.lut_k, args.lut_mapping, tau, args.seed,
                args.lut_learn_mapping, args.lut_map_init, args.lut_map_tau,
            )
        elif name == "bp":
            model = BranchingNet(c, t, args.bp_trees, args.bp_depth, args.lut_k, args.seed, args.lut_map_tau)
        elif name in ("lif", "rlif"):
            model = LifNet(
                c, t, h, beta=args.lif_beta, init_gain=args.lif_init_gain, recurrent=name == "rlif"
            )
        else:
            sys.exit(f"unknown model {name}")
        model.to(device)
        learned = " learned" if args.lut_learn_mapping else ""
        if name == "bp":
            learned = f" trees={args.bp_trees} depth={args.bp_depth}"
        label = f"sizes={args.lut_sizes} k={args.lut_k} {args.lut_mapping}{learned}" if name == "lut" else f"hidden={h}"
        print(f"\n[{name.upper()}] {label}")
        val_acc = train(model, x_tr, y_tr, x_val, y_val, args, device, teacher)

        set_deployed(model, False)
        acc_train_fmt, _ = evaluate(model, x_test, y_test, device)
        set_deployed(model, True)
        acc_deployed, activity = evaluate(model, x_test, y_test, device)

        if name == "bp":
            x_flat = x_test.reshape(len(x_test), -1).astype(np.uint8)
            ref_pred, *_ = branching_integer_reference(model, x_flat)
            ref_acc = float((ref_pred == y_test).mean())
            print(f"  integer reference: {ref_acc * 100:.2f}% (PyTorch deployed {acc_deployed * 100:.2f}%)")
            if args.export_bp:
                export_branching(model, x_test, y_test, args.export_bp.expanduser(), {
                    "dataset": dataset, "seed": args.seed, "epochs": args.epochs,
                    "test_acc_deployed_torch": acc_deployed,
                })
            nt, d = args.bp_trees, args.bp_depth
            n_nodes, n_leaves = nt * (2**d - 1), nt * 2**d
            flash = n_nodes * FLASH_LUT6_TABLE_BYTES + n_leaves * N_CLASSES
            cycles = nt * (d * CYCLES_LUT6_BOARD + CYCLES_LEAF_ADD_ASSUMED)
            ms, uj = to_time_energy(cycles)
            results.append(Result(
                model="bp", hidden=n_nodes, val_acc=val_acc, test_acc_train_format=acc_train_fmt,
                test_acc_deployed=ref_acc, flash_bytes=int(flash), sram_bytes=c * t // 8 + N_CLASSES * 4,
                est_cycles_per_sample=cycles, est_ms_per_sample=ms, est_uj_per_sample=uj,
                hidden_activity_per_sample=float(nt * d),
                notes=(f"{nt} trees of depth {d}, LUT nodes k={args.lut_k} with learned inputs, int8 leaves; "
                       f"cycles from the board LUT cost x {nt * d} nodes + assumed leaf cost"),
            ))
            print(f"  test {acc_train_fmt * 100:.2f}% (training format), {ref_acc * 100:.2f}% (integer, deployed)")
            continue
        if name == "lut":
            check_lut_bitlevel(model, x_test)
            if args.export_lut:
                export_lut(model, x_test, y_test, args.export_lut.expanduser(), {
                    "dataset": dataset, "seed": args.seed, "epochs": args.epochs,
                    "lut_sizes": args.lut_sizes, "learned_mapping": args.lut_learn_mapping,
                    "test_acc_deployed_torch": acc_deployed,
                })
            for depth, layer in enumerate(model.layers):
                if layer.learn_mapping:
                    moved = (layer.index() != layer.initial_index).float().mean().item()
                    print(f"  layer {depth}: {moved * 100:.1f}% of LUT inputs moved from their initial position")
            n_luts = sum(sizes)
            hidden_bits = sum(sizes[:-1])
            sram = c * t // 8 + math.ceil(hidden_bits / 8) + N_CLASSES * 2
            measured = args.lut_k == 6
            for variant, cyc, fl in (
                ("lut-codegen", CYCLES_LUT6_CODEGEN, FLASH_LUT6_CODEGEN_BYTES),
                ("lut-table", CYCLES_LUT6_TABLE, FLASH_LUT6_TABLE_BYTES),
            ):
                # Same framing cost as BPTN/BNN: the input history is packed into words.
                framing = c * t * CYCLES_FRAMING_PUSH_ASSUMED
                cycles = n_luts * cyc + framing if measured else float("nan")
                ms, uj = to_time_energy(cycles)
                results.append(
                    Result(
                        model=variant,
                        hidden=n_luts,
                        val_acc=val_acc,
                        test_acc_train_format=acc_train_fmt,
                        test_acc_deployed=acc_deployed,
                        flash_bytes=int(n_luts * fl) if measured else -1,
                        sram_bytes=int(sram),
                        est_cycles_per_sample=cycles,
                        est_ms_per_sample=ms,
                        est_uj_per_sample=uj,
                        hidden_activity_per_sample=activity,
                        notes=(
                            f"layers {args.lut_sizes}, k={args.lut_k}, {args.lut_mapping} initial mapping"
                            f"{', learned' if args.lut_learn_mapping else ', fixed'}, "
                            f"group-sum readout; per-LUT cost and Flash measured on the Wio-E5 for k=6 "
                            f"({'generated code' if variant == 'lut-codegen' else 'address table'})"
                        ),
                    )
                )
            print(
                f"  test {acc_train_fmt * 100:.2f}% (training format), {acc_deployed * 100:.2f}% (deployed)"
            )
            continue
        if name == "bptn":
            check_bptn_identity(model, x_test, c, t)
            k, _ = model.masks()
            density = k.mean().item()
            flash = h * c * math.ceil(t / 64) * 16 + h * 4 + readout_flash(h)
            sram = c * 8 + math.ceil(h / 8) + N_CLASSES * 4
            cycles = binary_layer_cost(c, t, h, activity, input_spikes)
            notes = (
                f"kernel and polarity masks, 2 x 64 bits per neuron-channel; kernel density "
                f"{density * 100:.1f}%; hidden = spikes per sample"
            )
        elif name == "bnn":
            if args.export_bnn:
                export_bnn(model, x_test, y_test, args.export_bnn.expanduser(), {
                    "dataset": dataset, "seed": args.seed, "epochs": args.epochs, "hidden": h,
                    "test_acc_deployed_torch": acc_deployed,
                })
            flash = h * c * math.ceil(t / 64) * 8 + h * 4 + readout_flash(h)
            sram = c * 8 + math.ceil(h / 8) + N_CLASSES * 4
            cycles = binary_layer_cost(c, t, h, activity, input_spikes)
            notes = "1 x 64 bits per neuron-channel; cost assumed equal to BPTN per word (XNOR + popcount)"
        else:
            recurrent = name == "rlif"
            # int8 input (and recurrent) weights, Fix32 biases and betas.
            flash = c * h + (h * h if recurrent else 0) + h * 4 + h * 4 + readout_flash(h)
            sram = h * 9 + h * 4 + N_CLASSES * 4  # 9 B per neuron (Fix32) + accumulators + logits
            cycles = lif_cost(c, t, h, input_spikes, activity)
            if recurrent:
                cycles += activity * h * CYCLES_SYNAPSE_INT8  # each hidden spike feeds H synapses
            notes = (
                "int8 weights (recurrent H x H included); hidden = total hidden spikes per sample"
                if recurrent
                else "int8 input weights; hidden = total hidden spikes per sample"
            )

        ms, uj = to_time_energy(cycles)
        res = Result(
            model=name,
            hidden=h,
            val_acc=val_acc,
            test_acc_train_format=acc_train_fmt,
            test_acc_deployed=acc_deployed,
            flash_bytes=int(flash),
            sram_bytes=int(sram),
            est_cycles_per_sample=cycles,
            est_ms_per_sample=ms,
            est_uj_per_sample=uj,
            hidden_activity_per_sample=activity,
            notes=notes,
        )
        results.append(res)
        print(
            f"  test {acc_train_fmt * 100:.2f}% (training format), {acc_deployed * 100:.2f}% (deployed)"
        )

    print(f"\nResults on {dataset} (seed {args.seed}, {args.epochs} epochs, hidden {h})")
    header = f"{'model':<6} | {'test acc':>9} | {'deployed':>9} | {'Flash':>9} | {'SRAM':>7} | {'est. cycles':>12} | {'est. ms':>8} | {'est. uJ':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.model:<6} | {r.test_acc_train_format * 100:8.2f}% | {r.test_acc_deployed * 100:8.2f}% | "
            f"{r.flash_bytes:>7} B | {r.sram_bytes:>5} B | {r.est_cycles_per_sample:>12,.0f} | "
            f"{r.est_ms_per_sample:>8.2f} | {r.est_uj_per_sample:>8.1f}"
        )
    print(
        "\nCycles, ms and uJ are estimates from per-operation costs measured on the Wio-E5 (48 MHz) and an\n"
        "assumed 13.86 mW; framing-buffer cost is assumed. Flash and SRAM come from the data layout."
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": dataset,
            "args": {k: str(v) for k, v in vars(args).items()},
            "torch": torch.__version__,
            "cost_constants": {
                "cycles_per_window_word": CYCLES_PER_WINDOW_WORD,
                "cycles_lif_step": CYCLES_LIF_STEP,
                "cycles_synapse_int8": CYCLES_SYNAPSE_INT8,
                "cycles_framing_push_assumed": CYCLES_FRAMING_PUSH_ASSUMED,
                "clock_hz": CLOCK_HZ,
                "assumed_active_power_w": ASSUMED_ACTIVE_POWER_W,
            },
            "results": [asdict(r) for r in results],
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
