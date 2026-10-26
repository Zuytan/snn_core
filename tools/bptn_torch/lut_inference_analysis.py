#!/usr/bin/env python3
"""
lut_inference_analysis.py

Host analysis of two ways to read less of an exported temporal LUT network per inference
(export of `train_shd_comparison.py --export-lut`), on the encoded SHD data:

1. Early exit of the group-sum readout. LUTs are evaluated in rounds (one LUT per class per
   round). The exact rule stops as soon as no other class can overtake the leader with the LUTs
   it has left (same prediction as the full evaluation, by construction). The margin rule stops
   when the leader is ahead by at least `delta` votes; `delta` is chosen on the validation split
   (the same 10 % split as the training script) and reported on the test set. Two orders are
   compared: the exported order, and LUTs sorted inside each group by how well their output
   separates that class from the others on the training split.

2. Incremental streaming. In a stream, the 64-tick window slides by one tick per step, so a LUT
   input changes when the bit at its position differs from the next-newer bit. The script
   counts, per tick, how many LUTs have at least one changed input and would need recomputing.

Usage:
    python3 tools/bptn_torch/lut_inference_analysis.py --model-dir ~/.cache/snn_core/lut_export_2000 \\
        --data ~/.cache/snn_core/shd/shd_encoded_c64_t64_1.00s.npz --out results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

N_CLASSES = 20


def load_model(model_dir: Path):
    meta = json.loads((model_dir / "meta.json").read_text())
    addr = np.frombuffer((model_dir / "lut_addr.bin").read_bytes(), dtype=np.uint8).reshape(-1, meta["k"], 2)
    positions = addr[..., 0].astype(np.int64) * 32 + addr[..., 1]  # flattened input positions [n, k]
    tables = np.frombuffer((model_dir / "lut_table.bin").read_bytes(), dtype="<u8")
    return meta, positions, tables


def lut_outputs(x_flat: np.ndarray, positions: np.ndarray, tables: np.ndarray) -> np.ndarray:
    address = np.zeros((len(x_flat), positions.shape[0]), dtype=np.uint64)
    for j in range(positions.shape[1]):
        address |= x_flat[:, positions[:, j]].astype(np.uint64) << np.uint64(j)
    return ((tables[None, :] >> address) & np.uint64(1)).astype(np.int16)


def argmax_low(counts):
    return counts.argmax(-1)  # numpy returns the lowest index on ties, as the firmware


def group_order(outputs_train, y_train, per_group, sort):
    """LUT indices in round order: round j holds the j-th LUT of every class group."""
    order = np.arange(N_CLASSES * per_group).reshape(N_CLASSES, per_group)
    if sort:
        for g in range(N_CLASSES):
            luts = order[g]
            o = outputs_train[:, luts].astype(np.float64)
            score = o[y_train == g].mean(0) - o[y_train != g].mean(0)
            order[g] = luts[np.argsort(-score, kind="stable")]
    return order  # [classes, per_group]


def early_exit(outputs, order, delta=None):
    """Returns (predictions, rounds used) for the exact rule (delta=None) or the margin rule."""
    n, per_group = len(outputs), order.shape[1]
    counts = np.zeros((n, N_CLASSES), dtype=np.int32)
    done = np.zeros(n, dtype=bool)
    pred = np.zeros(n, dtype=np.int64)
    rounds = np.full(n, per_group)
    classes = np.arange(N_CLASSES)
    for j in range(per_group):
        counts += outputs[:, order[:, j]]
        remaining = per_group - j - 1
        leader = argmax_low(counts)
        lead = counts[np.arange(n), leader]
        if delta is None:
            other = counts + remaining
            beats = (lead[:, None] > other) | ((lead[:, None] == other) & (leader[:, None] < classes[None, :]))
            beats[np.arange(n), leader] = True
            stop = beats.all(1)
        else:
            second = np.sort(counts, axis=1)[:, -2]
            stop = (lead - second >= delta) | (remaining == 0)
        newly = stop & ~done
        pred[newly] = leader[newly]
        rounds[newly] = j + 1
        done |= stop
        if done.all():
            break
    return pred, rounds


def streaming_stats(x: np.ndarray, positions: np.ndarray):
    """Per-tick fraction of LUTs with a changed input when a sample is streamed after silence."""
    n, c, t = x.shape
    prev = np.zeros((n, c, t), dtype=np.uint8)
    lut_frac, bit_frac = [], []
    for tau in range(t):
        window = np.zeros((n, c, t), dtype=np.uint8)
        window[:, :, t - 1 - tau :] = x[:, :, : tau + 1]
        changed = (window != prev).reshape(n, -1)
        bit_frac.append(changed.mean())
        lut_frac.append(changed[:, positions].any(-1).mean())
        prev = window
    return np.array(lut_frac), np.array(bit_frac)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="encoded SHD .npz written by the training script")
    ap.add_argument("--max-accuracy-drop", type=float, default=0.005, help="allowed validation drop for delta")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meta, positions, tables = load_model(args.model_dir.expanduser())
    d = np.load(args.data.expanduser())
    x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    rng = np.random.default_rng(meta["seed"])
    perm = rng.permutation(len(x_train))
    n_val = len(x_train) // 10
    x_val, y_val = x_train[perm[:n_val]], y_train[perm[:n_val]]
    x_tr, y_tr = x_train[perm[n_val:]], y_train[perm[n_val:]]

    n_luts = positions.shape[0]
    per_group = n_luts // N_CLASSES
    out_tr = lut_outputs(x_tr.reshape(len(x_tr), -1), positions, tables)
    out_val = lut_outputs(x_val.reshape(len(x_val), -1), positions, tables)
    out_test = lut_outputs(x_test.reshape(len(x_test), -1), positions, tables)
    full_test = argmax_low(out_test.reshape(len(out_test), N_CLASSES, per_group).sum(-1))
    full_val = argmax_low(out_val.reshape(len(out_val), N_CLASSES, per_group).sum(-1))
    acc_full = float((full_test == y_test).mean())
    acc_full_val = float((full_val == y_val).mean())
    print(f"Full evaluation: {n_luts} LUTs, test {acc_full * 100:.2f}%, validation {acc_full_val * 100:.2f}%")

    report = {"model": meta, "full_test_accuracy": acc_full, "early_exit": [], "streaming": {}}
    for sort in (False, True):
        order = group_order(out_tr, y_tr, per_group, sort)
        name = "sorted by training separation" if sort else "exported order"
        pred, rounds = early_exit(out_test, order)
        assert np.array_equal(pred, full_test), "exact early exit changed a prediction"
        frac = float(rounds.mean() / per_group)
        print(f"\n[{name}] exact rule: {frac * 100:.1f}% of LUTs evaluated on average (test), same predictions")
        report["early_exit"].append({"order": name, "rule": "exact", "delta": None, "test_accuracy": acc_full,
                                     "mean_fraction": frac, "max_fraction": float(rounds.max() / per_group)})
        best = None
        for delta in range(1, per_group + 1):
            p_val, r_val = early_exit(out_val, order, delta)
            if (p_val == y_val).mean() >= acc_full_val - args.max_accuracy_drop:
                best = delta
                break
        p_test, r_test = early_exit(out_test, order, best)
        acc = float((p_test == y_test).mean())
        frac = float(r_test.mean() / per_group)
        print(f"[{name}] margin rule, delta={best} (chosen on validation): test {acc * 100:.2f}%, "
              f"{frac * 100:.1f}% of LUTs on average")
        report["early_exit"].append({"order": name, "rule": "margin", "delta": best, "test_accuracy": acc,
                                     "mean_fraction": frac, "max_fraction": float(r_test.max() / per_group)})
        if not sort:
            # Expected board predictions for the firmware early exit (exported order, round-major).
            (args.model_dir.expanduser() / f"host_pred_ee{best}.bin").write_bytes(p_test.astype(np.uint8).tobytes())
            report["early_exit_delta_exported_order"] = best

    lut_frac, bit_frac = streaming_stats(x_test, positions)
    print(f"\nStreaming (sample after silence, 64 ticks): changed input bits {bit_frac.mean() * 100:.1f}% per tick, "
          f"LUTs with a changed input {lut_frac.mean() * 100:.1f}% per tick "
          f"(min {lut_frac.min() * 100:.1f}%, max {lut_frac.max() * 100:.1f}%)")
    report["streaming"] = {"changed_bits_mean": float(bit_frac.mean()), "luts_to_update_mean": float(lut_frac.mean()),
                           "luts_to_update_per_tick": lut_frac.tolist()}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
