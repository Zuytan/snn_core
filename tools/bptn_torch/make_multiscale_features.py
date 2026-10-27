#!/usr/bin/env python3
"""
make_multiscale_features.py

Builds a bank of multi-scale count inputs for the forest of LUT trees. The learned connections of
the forest then choose which inputs each node reads, so the aggregation itself is selected by
training instead of being fixed by hand (as in `make_hybrid_features.py`).

Counts of set bits over rectangles of the 64 channels x 64 ticks input:
- channel groups of 1, 4 and 16 adjacent channels (64 + 16 + 4 = 84 groups);
- time windows of 4, 8, 16, 32 and 64 ticks, overlapping by half (31 + 15 + 7 + 3 + 1 = 57);
- so 84 x 57 = 4,788 counts, each turned into 3 bits by the 25, 50 and 75 % quantiles of that
  count on the training split (bit k = count above threshold k; thresholds stored in the .npz).

Output: the raw 4,096 bits followed by the 14,364 count bits, as one flat input of 18,460 bits shaped
[N, 1, 18460], readable by `train_shd_comparison.py --data-npz` and `tree_baselines.py`.

Cost model on a Cortex-M4 (estimate, not measured): an integral image of the 64 x 64 bits (about
4,096 additions) is built once per inference; each count read by a node is then 4 lookups and 3
additions, plus one comparison per bit.

Usage:
    python3 tools/bptn_torch/make_multiscale_features.py --data ~/.cache/snn_core/gsc/gsc_encoded_b64_f64.npz \\
        --out ~/.cache/snn_core/gsc/gsc_multiscale.npz
"""

import argparse
from pathlib import Path

import numpy as np

GROUPS = (1, 4, 16)
WINDOWS = (4, 8, 16, 32, 64)
QUANTILES = (0.25, 0.5, 0.75)


def rectangles(channels=64, ticks=64):
    """List of (c0, c1, t0, t1) half-open rectangles, in output order."""
    rects = []
    for g in GROUPS:
        for c0 in range(0, channels, g):
            for w in WINDOWS:
                for t0 in range(0, ticks - w + 1, w // 2):
                    rects.append((c0, c0 + g, t0, t0 + w))
    return rects


def counts(x, rects, batch=4096):
    """x: [N, 64, 64] bits -> [N, len(rects)] counts, through an integral image."""
    out = np.empty((len(x), len(rects)), dtype=np.int16)
    r = np.array(rects)
    for i in range(0, len(x), batch):
        xb = x[i : i + batch].astype(np.int32)
        s = np.zeros((len(xb), xb.shape[1] + 1, xb.shape[2] + 1), dtype=np.int32)
        s[:, 1:, 1:] = xb.cumsum(1).cumsum(2)
        out[i : i + batch] = (
            s[:, r[:, 1], r[:, 3]] - s[:, r[:, 0], r[:, 3]] - s[:, r[:, 1], r[:, 2]] + s[:, r[:, 0], r[:, 2]]
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.data.expanduser())
    x_train, x_test = d["x_train"], d["x_test"]
    assert x_train.shape[1:] == (64, 64), "expects 64 channels x 64 ticks"
    rects = rectangles()
    c_train, c_test = counts(x_train, rects), counts(x_test, rects)
    thresholds = np.quantile(c_train, QUANTILES, axis=0).T  # [counts, 3]
    out = {}
    for split, x, c in (("train", x_train, c_train), ("test", x_test, c_test)):
        bits = (c[:, :, None] > thresholds[None]).reshape(len(c), -1).astype(np.uint8)
        out[f"x_{split}"] = np.concatenate([x.reshape(len(x), -1), bits], axis=1)[:, None, :]
        out[f"y_{split}"] = d[f"y_{split}"]
    np.savez(args.out.expanduser(), **out, thresholds=thresholds, rectangles=np.array(rects))
    print(f"{len(rects)} counts -> {out['x_train'].shape[2]} input bits, "
          f"count-bit density {out['x_train'][:, 0, 4096:].mean() * 100:.1f}%; written to {args.out}")


if __name__ == "__main__":
    main()
