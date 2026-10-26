#!/usr/bin/env python3
"""
encode_autohack.py

Encodes the AutoHack CAN intrusion dataset (Zenodo 10.5281/zenodo.19661007, CC BY 4.0;
`Train/autohack_train_both.csv` and `Test/autohack_test_both.csv`) into the binary 64 x 64 input
used by the SHD comparison scripts.

- A window is 64 consecutive CAN frames of the official stream (no overlap), in file order.
- Each frame becomes 64 bits: the 11-bit arbitration identifier, the 4-bit DLC, 1 bit set when the
  gap to the previous frame is below `--short-gap-us` (a DoS signature), and the first 6 data bytes
  (48 bits; missing bytes are 0). Bit j of frame t is x[j, t].
- Window label: the most frequent attack type among its attack frames, or Normal when it has none.
  Classes: Normal, DoS, Fuzzing, Replay, Spoofing, UDS Spoofing.
- Attack frames are rare, so the script keeps every attack window and as many Normal windows,
  drawn at random (seeded), in both splits; accuracy is then not dominated by Normal traffic.

Output: an .npz with x_train, y_train, x_test, y_test ([N, 64 bits, 64 frames], uint8), the class
names and the window counts before balancing.

Usage:
    python3 tools/bptn_torch/encode_autohack.py --root ~/.cache/snn_core/autohack/Autohack2025_Dataset \\
        --out ~/.cache/snn_core/autohack/autohack_w64.npz
"""

import argparse
import csv
from pathlib import Path

import numpy as np

WINDOW = 64
CLASSES = ["Normal", "DoS", "Fuzzing", "Replay", "Spoofing", "UDS Spoofing"]


def label_index(label: str) -> int:
    label = label.strip()
    if label.startswith("UDS"):
        return CLASSES.index("UDS Spoofing")
    return CLASSES.index(label)


def read_stream(path: Path, short_gap_us: float):
    """Returns per-frame 64-bit codes [n, 64] (uint8 bits) and labels [n]."""
    ids, dlcs, gaps, data, labels = [], [], [], [], []
    prev = None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            t = float(row[col["Timestamp"]])
            ids.append(int(row[col["Arbitration_ID"]], 16) & 0x7FF)
            dlcs.append(int(row[col["DLC"]]) & 0xF)
            gaps.append(prev is not None and (t - prev) * 1e6 < short_gap_us)
            prev = t
            payload = bytes.fromhex(row[col["Data"]].replace(" ", ""))[:6]
            data.append(payload + bytes(6 - len(payload)))
            labels.append(label_index(row[col["Label"]]))
    n = len(ids)
    bits = np.zeros((n, 64), dtype=np.uint8)
    ids = np.array(ids, dtype=np.int64)
    dlcs = np.array(dlcs, dtype=np.int64)
    for b in range(11):
        bits[:, b] = (ids >> b) & 1
    for b in range(4):
        bits[:, 11 + b] = (dlcs >> b) & 1
    bits[:, 15] = np.array(gaps, dtype=np.uint8)
    raw = np.frombuffer(b"".join(data), dtype=np.uint8).reshape(n, 6)
    bits[:, 16:] = np.unpackbits(raw, axis=1, bitorder="little")
    return bits, np.array(labels, dtype=np.int64)


def windows(bits, labels):
    n = len(labels) // WINDOW
    x = bits[: n * WINDOW].reshape(n, WINDOW, 64).transpose(0, 2, 1)  # [N, 64 bits, 64 frames]
    lab = labels[: n * WINDOW].reshape(n, WINDOW)
    attack_counts = np.stack([(lab == c).sum(1) for c in range(1, len(CLASSES))], axis=1)  # [N, attacks]
    y = np.where(attack_counts.sum(1) > 0, attack_counts.argmax(1) + 1, 0)
    return np.ascontiguousarray(x), y


def balance(x, y, rng):
    attack = np.flatnonzero(y > 0)
    normal = np.flatnonzero(y == 0)
    keep = np.concatenate([attack, rng.choice(normal, size=min(len(normal), len(attack)), replace=False)])
    keep.sort()
    return x[keep], y[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--short-gap-us", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = args.root.expanduser()
    rng = np.random.default_rng(args.seed)

    out, counts = {}, {}
    for split, rel in (("train", "Train/autohack_train_both.csv"), ("test", "Test/autohack_test_both.csv")):
        bits, labels = read_stream(root / rel, args.short_gap_us)
        x, y = windows(bits, labels)
        counts[split] = {CLASSES[c]: int((y == c).sum()) for c in range(len(CLASSES))}
        x, y = balance(x, y, rng)
        out[f"x_{split}"], out[f"y_{split}"] = x, y
        print(f"{split}: {len(labels)} frames, windows {counts[split]}, kept {len(y)} "
              f"(density {x.mean() * 100:.1f}%)", flush=True)
    np.savez_compressed(args.out.expanduser(), **out, classes=np.array(CLASSES),
                        counts=np.array(str(counts)))
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
