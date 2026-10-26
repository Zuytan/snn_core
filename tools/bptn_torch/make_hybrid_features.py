#!/usr/bin/env python3
"""
make_hybrid_features.py

Adds a cheap temporal aggregation layer in front of the forest of LUT trees: for each input
channel, the number of set bits in 4 windows of 16 ticks and in the whole 64-tick window, each
count turned into 4 bits by thresholds (thermometer code). The thresholds are the 20, 40, 60 and
80 % quantiles of that count on the training split; bit k is 1 when the count is above threshold k.

Output: the raw 64 x 64 bits followed by the 64 x 5 x 4 = 1,280 aggregated bits, as one flat input
of 5,376 bits, shaped [N, 1, 5376] so that `train_shd_comparison.py --data-npz` and
`tree_baselines.py` read it unchanged. The thresholds are stored in the .npz.

On a Cortex-M4 the aggregation is 320 popcounts of 16- or 64-bit slices and 1,280 comparisons;
its cycle cost is not measured here.

Usage:
    python3 tools/bptn_torch/make_hybrid_features.py --data ~/.cache/snn_core/gsc/gsc_encoded_b64_f64.npz \\
        --out ~/.cache/snn_core/gsc/gsc_hybrid.npz
"""

import argparse
from pathlib import Path

import numpy as np

WINDOWS = [(0, 16), (16, 32), (32, 48), (48, 64), (0, 64)]
QUANTILES = (0.2, 0.4, 0.6, 0.8)


def counts(x):
    """x: [N, C, 64] bits -> [N, C, len(WINDOWS)] counts."""
    return np.stack([x[:, :, a:b].sum(-1) for a, b in WINDOWS], axis=-1).astype(np.int16)


def thermometer(c, thresholds):
    """c: [N, C, W]; thresholds: [C, W, 4] -> [N, C * W * 4] bits."""
    bits = c[..., None] > thresholds[None]
    return bits.reshape(len(c), -1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.data.expanduser())
    x_train, x_test = d["x_train"], d["x_test"]
    assert x_train.shape[2] == 64, "expects 64 ticks per channel"
    c_train, c_test = counts(x_train), counts(x_test)
    thresholds = np.quantile(c_train, QUANTILES, axis=0).transpose(1, 2, 0)  # [C, W, 4]
    out = {}
    for split, x, c in (("train", x_train, c_train), ("test", x_test, c_test)):
        flat = np.concatenate([x.reshape(len(x), -1), thermometer(c, thresholds)], axis=1)
        out[f"x_{split}"] = flat[:, None, :]
        out[f"y_{split}"] = d[f"y_{split}"]
    np.savez_compressed(args.out.expanduser(), **out, thresholds=thresholds)
    agg = out["x_train"][:, 0, 4096:]
    print(f"{out['x_train'].shape[2]} input bits ({agg.shape[1]} aggregated, density {agg.mean() * 100:.1f}%); "
          f"written to {args.out}")


if __name__ == "__main__":
    main()
