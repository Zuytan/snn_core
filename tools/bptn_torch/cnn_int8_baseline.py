#!/usr/bin/env python3
"""
cnn_int8_baseline.py

Int8 convolutional baseline for the SHD comparison, run with CMSIS-NN on the Wio-E5
(`examples/wio_e5_benchmark/src/bin/cnn_shd_eval.rs`).

Same input and split as `train_shd_comparison.py`: 64 channels x 64 bins, binary, seed-0 split with
10 % validation. The network is a 1-D CNN over time, in the NHWC layout CMSIS-NN uses (H = 1,
W = 64 ticks, C = 64 channels):

    conv(64 -> c1, kernel k1, stride s1) + BatchNorm + ReLU
    conv(c1 -> c2, kernel k2, stride s2) + BatchNorm + ReLU
    global average pooling over time
    fully connected (c2 -> 20)

After float training, BatchNorm is folded into the convolutions and the model is quantized after
training, the TensorFlow Lite way that CMSIS-NN implements: int8 weights (per output channel for
the convolutions, per tensor for the fully connected layer), int32 biases, int8 activations
calibrated on the training split (ReLU outputs use zero point -128), and fixed-point multipliers
and shifts. The script runs an integer reference that reproduces CMSIS-NN's arithmetic
(arm_nn_requantize without single rounding, arm_avgpool_s8 rounding), reports float and integer
accuracy, the number of multiply-accumulates, and exports one configuration for the firmware.

Usage:
    python3 tools/bptn_torch/cnn_int8_baseline.py --data ~/.cache/snn_core/shd/shd_encoded_c64_t64_1.00s.npz \\
        --configs tiny,small --export-dir ~/.cache/snn_core --out tools/bptn_torch/results/cnn_int8.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 20
# name: (c1, k1, s1, c2, k2, s2)
CONFIGS = {
    "micro": (4, 16, 16, 16, 2, 1),
    "tiny": (8, 8, 8, 16, 3, 1),
    "small": (16, 8, 4, 32, 3, 2),
    "medium": (32, 8, 4, 64, 3, 2),
    "large": (64, 8, 2, 64, 3, 2),
}


def split(path: Path, seed: int):
    d = np.load(path)
    x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    perm = np.random.default_rng(seed).permutation(len(x_train))
    n_val = len(x_train) // 10
    return (x_train[perm[n_val:]], y_train[perm[n_val:]], x_train[perm[:n_val]], y_train[perm[:n_val]], x_test, y_test)


class Cnn(nn.Module):
    def __init__(self, c1, k1, s1, c2, k2, s2, channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, c1, k1, stride=s1)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, k2, stride=s2)
        self.bn2 = nn.BatchNorm1d(c2)
        self.fc = nn.Linear(c2, N_CLASSES)

    def features(self, x):
        a1 = F.relu(self.bn1(self.conv1(x)))
        a2 = F.relu(self.bn2(self.conv2(a1)))
        return a1, a2

    def forward(self, x):
        _, a2 = self.features(x)
        return self.fc(a2.mean(-1))


def train(model, data, epochs, lr, seed, device):
    x_tr, y_tr, x_val, y_val, *_ = data
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)
    best = (-1.0, None)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(x_tr))
        for i in range(0, len(order), 128):
            idx = order[i : i + 128]
            xb = torch.as_tensor(x_tr[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_tr[idx], device=device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xv).argmax(1).cpu().numpy() == y_val).mean()
        if acc > best[0]:
            best = (acc, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    model.eval()
    return float(best[0])


# ----------------------------------------------------------------------------
# Quantization (TensorFlow Lite style, as consumed by CMSIS-NN)
# ----------------------------------------------------------------------------
def fold_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    w = conv.weight.detach().cpu().double().numpy()
    b = conv.bias.detach().cpu().double().numpy()
    gamma = bn.weight.detach().cpu().double().numpy()
    beta = bn.bias.detach().cpu().double().numpy()
    mean = bn.running_mean.detach().cpu().double().numpy()
    var = bn.running_var.detach().cpu().double().numpy()
    scale = gamma / np.sqrt(var + bn.eps)
    return w * scale[:, None, None], (b - mean) * scale + beta


def quantize_multiplier(m: float):
    """Real multiplier -> (int32 multiplier, shift) with m = mult / 2^31 * 2^shift."""
    if m == 0:
        return 0, 0
    mant, exp = math.frexp(m)
    q = int(round(mant * (1 << 31)))
    if q == 1 << 31:
        q //= 2
        exp += 1
    return q, exp


def doubling_high_mult_no_sat(a, b):
    return (a.astype(np.int64) * b + (1 << 30)) >> 31


def divide_by_power_of_two(x, exponent):
    mask = (np.int64(1) << exponent) - 1
    remainder = x & mask
    result = x >> exponent
    threshold = (mask >> 1) + (result < 0)
    return result + (remainder > threshold)


def requantize(acc, mult, shift):
    """arm_nn_requantize (default build: no single rounding, no inline assembly)."""
    acc = acc.astype(np.int64)
    mult = np.asarray(mult, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    v = (acc * (np.int64(1) << left)).astype(np.int32).astype(np.int64)  # int32 wrap as in C
    return divide_by_power_of_two(doubling_high_mult_no_sat(v, mult), right)


def quantize_model(model: Cnn, calib_x: np.ndarray, device):
    """Returns a dict of integer parameters and quantization constants."""
    model.eval()
    with torch.no_grad():
        xc = torch.as_tensor(calib_x, dtype=torch.float32, device=device)
        a1, a2 = model.features(xc)
        logits = model(xc)
    s_in = 1.0 / 127.0
    s_a1 = float(a1.max()) / 255.0
    s_a2 = float(a2.max()) / 255.0
    s_out = float(logits.abs().max()) / 127.0

    q = {"s_in": s_in, "s_a1": s_a1, "s_a2": s_a2, "s_out": s_out}
    for name, conv, bn, s_x, s_y in (("conv1", model.conv1, model.bn1, s_in, s_a1), ("conv2", model.conv2, model.bn2, s_a1, s_a2)):
        w, b = fold_bn(conv, bn)
        s_w = np.maximum(np.abs(w).reshape(len(w), -1).max(1), 1e-12) / 127.0
        w_q = np.clip(np.round(w / s_w[:, None, None]), -127, 127).astype(np.int8)
        b_q = np.round(b / (s_x * s_w)).astype(np.int64)
        mults, shifts = zip(*(quantize_multiplier(s_x * sw / s_y) for sw in s_w))
        q[name] = {
            # torch [c_out, c_in, k] -> CMSIS [c_out, 1, k, c_in], stored as [c_out, k, c_in]
            "w": np.ascontiguousarray(w_q.transpose(0, 2, 1)),
            "b": b_q.astype(np.int32),
            "mult": np.array(mults, dtype=np.int32),
            "shift": np.array(shifts, dtype=np.int32),
            "stride": conv.stride[0],
        }
    w = model.fc.weight.detach().cpu().double().numpy()
    b = model.fc.bias.detach().cpu().double().numpy()
    s_wf = max(np.abs(w).max(), 1e-12) / 127.0
    m, sh = quantize_multiplier(s_a2 * s_wf / s_out)
    q["fc"] = {
        "w": np.clip(np.round(w / s_wf), -127, 127).astype(np.int8),  # [20, c2]
        "b": np.round(b / (s_a2 * s_wf)).astype(np.int32),
        "mult": np.int32(m),
        "shift": np.int32(sh),
    }
    return q


def conv_int(x, layer, input_offset, output_offset):
    """x: [N, W, C_in] int; CMSIS-NN arm_convolve_s8 with valid padding."""
    w = layer["w"].astype(np.int64)  # [c_out, k, c_in] (CMSIS [c_out, 1, k, c_in])
    c_out, k, c_in = w.shape
    s = layer["stride"]
    n, width, _ = x.shape
    w_out = (width - k) // s + 1
    xo = x.astype(np.int64) + input_offset
    acc = np.zeros((n, w_out, c_out), dtype=np.int64)
    for j in range(w_out):
        patch = xo[:, j * s : j * s + k, :].reshape(n, -1)  # [N, k * c_in]
        acc[:, j, :] = patch @ w.reshape(c_out, -1).T
    acc += layer["b"].astype(np.int64)
    out = requantize(acc, layer["mult"][None, None, :], layer["shift"][None, None, :]) + output_offset
    return np.clip(out, -128, 127)


def int_forward(q, x_bits):
    """Integer reference of the firmware. x_bits: [N, C, T] in {0, 1}. Returns int8 logits."""
    x = (x_bits.transpose(0, 2, 1).astype(np.int64) * 127)  # NHWC with H = 1: [N, W = T, C]
    a1 = conv_int(x, q["conv1"], input_offset=0, output_offset=-128)
    a2 = conv_int(a1, q["conv2"], input_offset=128, output_offset=-128)
    count = a2.shape[1]
    total = a2.sum(1)  # [N, c2]
    half = count // 2
    pooled = np.where(total > 0, (total + half) // count, -((-(total - half)) // count))  # C division truncates
    pooled = np.clip(pooled, -128, 127)
    fc = q["fc"]
    acc = (pooled + 128) @ fc["w"].astype(np.int64).T + fc["b"].astype(np.int64)
    out = requantize(acc, fc["mult"], fc["shift"])
    return np.clip(out, -128, 127)


def macs(cfg, width=64, channels=64):
    c1, k1, s1, c2, k2, s2 = cfg
    w1 = (width - k1) // s1 + 1
    w2 = (w1 - k2) // s2 + 1
    return w1 * c1 * k1 * channels + w2 * c2 * k2 * c1 + c2 * N_CLASSES, w1, w2


def export(name, cfg, q, x_test, y_test, pred, out_dir: Path, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer in ("conv1", "conv2"):
        for key in ("w", "b", "mult", "shift"):
            arr = q[layer][key]
            (out_dir / f"{layer}_{key}.bin").write_bytes(arr.astype("<i1" if key == "w" else "<i4").tobytes())
    (out_dir / "fc_w.bin").write_bytes(q["fc"]["w"].astype("<i1").tobytes())
    (out_dir / "fc_b.bin").write_bytes(q["fc"]["b"].astype("<i4").tobytes())
    flat = x_test.reshape(len(x_test), -1).astype(np.uint32)
    words = (flat.reshape(len(flat), -1, 32) << np.arange(32, dtype=np.uint32)).sum(-1, dtype=np.uint32)
    (out_dir / "test_x.bin").write_bytes(words.astype("<u4").tobytes())
    (out_dir / "test_y.bin").write_bytes(y_test.astype(np.uint8).tobytes())
    (out_dir / "host_pred.bin").write_bytes(pred.astype(np.uint8).tobytes())
    c1, k1, s1, c2, k2, s2 = cfg
    meta = dict(meta, name=name, c1=c1, k1=k1, s1=s1, c2=c2, k2=k2, s2=s2,
                fc_mult=int(q["fc"]["mult"]), fc_shift=int(q["fc"]["shift"]), samples=int(len(x_test)))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  exported to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-dir", type=Path, default=None, help="writes <dir>/cnn_export_<name> per config")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = split(args.data.expanduser(), args.seed)
    x_tr, y_tr, x_val, y_val, x_te, y_te = data
    calib = x_tr[np.random.default_rng(args.seed).permutation(len(x_tr))[:2000]]

    results = []
    for name in args.configs.split(","):
        cfg = CONFIGS[name]
        torch.manual_seed(args.seed)
        model = Cnn(*cfg).to(device)
        val = train(model, data, args.epochs, args.lr, args.seed, device)
        with torch.no_grad():
            float_pred = model(torch.as_tensor(x_te, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
        q = quantize_model(model, calib, device)
        int_pred = int_forward(q, x_te).argmax(1)
        int_val = int_forward(q, x_val).argmax(1)
        n_macs, w1, w2 = macs(cfg)
        c1, k1, s1, c2, k2, s2 = cfg
        flash = (c1 * k1 * 64 + c2 * k2 * c1 + c2 * N_CLASSES) + 12 * (c1 + c2) + 4 * N_CLASSES
        res = {
            "name": name, "config": cfg, "val_float": val, "val_int": float((int_val == y_val).mean()),
            "test_float": float((float_pred == y_te).mean()), "test_int": float((int_pred == y_te).mean()),
            "macs": n_macs, "w1": w1, "w2": w2, "weights_bytes": flash,
        }
        results.append(res)
        print(f"{name:7} {cfg}: test float {res['test_float'] * 100:.2f}%, int8 {res['test_int'] * 100:.2f}% "
              f"(val int8 {res['val_int'] * 100:.1f}%), {n_macs:,} MACs, weights {flash / 1024:.1f} KB", flush=True)
        if args.export_dir:
            export(name, cfg, q, x_te, y_te, int_pred, args.export_dir.expanduser() / f"cnn_export_{name}",
                   {"seed": args.seed, "epochs": args.epochs, "test_int": res["test_int"], "test_float": res["test_float"], "macs": n_macs})
    if args.out:
        args.out.write_text(json.dumps({"seed": args.seed, "results": results}, indent=2))
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
