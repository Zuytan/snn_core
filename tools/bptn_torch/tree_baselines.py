#!/usr/bin/env python3
"""
tree_baselines.py

Classic tree ensembles on the same SHD input as `train_shd_comparison.py` (64 channels x 64 bins,
binary, 4,096 features; same 10 % validation split), to check whether the forest of LUT trees
gains anything over ordinary trees whose nodes test one bit.

Models (scikit-learn):
- random forest: `trees` trees of maximum depth `depth`; leaves hold class probabilities, deployed
  as 20 int8 scores (round(127 p)), summed over the trees;
- gradient boosting (HistGradientBoostingClassifier): one tree per class per iteration; leaves hold
  one value, deployed as int16 (one global scale), summed per class with the int16 baseline.

For each configuration the script reports validation and test accuracy of the integer model that
the firmware would run, the mean and maximum number of internal nodes visited per inference, and
the Flash of a compact layout: 6 bytes per internal node (u16 feature, u16 left, u16 right; a child
with bit 15 set is a leaf id), the leaf payload (20 B per random-forest leaf, 2 B per boosting leaf),
and 3 bytes per tree (u16 root, u8 class). `--export DIR` writes one configuration for the firmware `tree_shd_eval.rs`.

Usage:
    python3 tools/bptn_torch/tree_baselines.py --data ~/.cache/snn_core/shd/shd_encoded_c64_t64_1.00s.npz \\
        --out tools/bptn_torch/results/tree_baselines.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

N_CLASSES = 20
LEAF = 0xFFFF


def split(data_path: Path, seed: int):
    d = np.load(data_path)
    x_train, y_train = d["x_train"].reshape(len(d["x_train"]), -1), d["y_train"]
    x_test, y_test = d["x_test"].reshape(len(d["x_test"]), -1), d["y_test"]
    perm = np.random.default_rng(seed).permutation(len(x_train))
    n_val = len(x_train) // 10
    return (x_train[perm[n_val:]], y_train[perm[n_val:]], x_train[perm[:n_val]], y_train[perm[:n_val]], x_test, y_test)


# ----------------------------------------------------------------------------
# Flattened integer model: nodes[i] = (feature or LEAF, left, right); for a leaf, left = leaf id.
# Go left when the bit is 0, as scikit-learn does with threshold 0.5 on {0, 1} features.
# ----------------------------------------------------------------------------
def flatten_rf(model: RandomForestClassifier):
    nodes, roots, leaves = [], [], []
    for est in model.estimators_:
        t = est.tree_
        base = len(nodes)
        roots.append(base)
        for i in range(t.node_count):
            if t.children_left[i] == -1:
                p = t.value[i, 0] / t.value[i, 0].sum()
                full = np.zeros(N_CLASSES)
                full[est.classes_.astype(int)] = p
                nodes.append((LEAF, len(leaves), 0))
                leaves.append(np.round(full * 127).astype(np.int8))
            else:
                nodes.append((int(t.feature[i]), base + int(t.children_left[i]), base + int(t.children_right[i])))
    return np.array(nodes, dtype=np.int64), np.array(roots), np.array(leaves, dtype=np.int8)


def flatten_hgb(model: HistGradientBoostingClassifier):
    """Returns nodes, roots, tree classes, int16 leaf values, int32 baseline and the scale."""
    raw_nodes, roots, tree_class, values = [], [], [], []
    for it in model._predictors:
        for k, pred in enumerate(it):
            base = len(raw_nodes)
            roots.append(base)
            tree_class.append(k)
            for n in pred.nodes:
                if n["is_leaf"]:
                    raw_nodes.append((LEAF, len(values), 0))
                    values.append(float(n["value"]))
                else:
                    assert n["num_threshold"] == 0.5 or (0 < n["num_threshold"] < 1), "unexpected threshold"
                    raw_nodes.append((int(n["feature_idx"]), base + int(n["left"]), base + int(n["right"])))
    baseline = model._baseline_prediction.reshape(-1)
    scale = max(np.abs(values).max(), 1e-12) / 32000.0
    q_values = np.round(np.array(values) / scale).astype(np.int16)
    # The baseline is a per-class constant that can change the argmax: kept as an int32 score.
    q_base = np.round(baseline / scale).astype(np.int64)
    return np.array(raw_nodes, dtype=np.int64), np.array(roots), np.array(tree_class), q_values, q_base, scale


def traverse(nodes, roots, x):
    """Integer traversal of every tree; returns leaf ids [N, trees] and internal nodes visited [N]."""
    n = len(x)
    leaf_ids = np.zeros((n, len(roots)), dtype=np.int64)
    visited = np.zeros(n, dtype=np.int64)
    rows = np.arange(n)
    for t, root in enumerate(roots):
        node = np.full(n, root, dtype=np.int64)
        while True:
            feat = nodes[node, 0]
            internal = feat != LEAF
            if not internal.any():
                break
            visited += internal
            bit = np.where(internal, x[rows, np.where(internal, feat, 0)], 0)
            node = np.where(internal, np.where(bit == 0, nodes[node, 1], nodes[node, 2]), node)
        leaf_ids[:, t] = nodes[node, 1]
    return leaf_ids, visited


def rf_integer(model, x):
    nodes, roots, leaves = flatten_rf(model)
    leaf_ids, visited = traverse(nodes, roots, x)
    scores = leaves[leaf_ids].astype(np.int64).sum(1)
    return scores.argmax(1), visited, (nodes, roots, leaves)


def hgb_integer(model, x):
    nodes, roots, tree_class, values, base, scale = flatten_hgb(model)
    leaf_ids, visited = traverse(nodes, roots, x)
    scores = np.tile(base, (len(x), 1))
    contrib = values[leaf_ids].astype(np.int64)
    for c in range(N_CLASSES):
        scores[:, c] += contrib[:, tree_class == c].sum(1)
    # Check against scikit-learn's own raw prediction (float).
    raw = model._raw_predict(x[:256].astype(np.float64))
    if raw.shape[0] == N_CLASSES:
        raw = raw.T
    assert (scores[:256].argmax(1) == raw.argmax(1)).mean() > 0.99, "integer boosting model disagrees with sklearn"
    return scores.argmax(1), visited, (nodes, roots, tree_class, values, base)


def flash_bytes(kind, nodes, n_trees, n_leaves):
    internal = int((nodes[:, 0] != LEAF).sum())
    leaf_payload = N_CLASSES if kind == "rf" else 2
    return internal * 6 + n_leaves * leaf_payload + n_trees * 3 + N_CLASSES * 4


def compact(nodes, roots):
    """Drops leaf nodes: internal node i -> [feature, left, right] where a child (or a root) with
    bit 15 set is the leaf id in its low 15 bits."""
    internal = np.flatnonzero(nodes[:, 0] != LEAF)
    if len(internal) >= 0x8000 or nodes[nodes[:, 0] == LEAF, 1].max() >= 0x8000:
        raise SystemExit("model too large for 15-bit indices")
    new_index = np.full(len(nodes), -1, dtype=np.int64)
    new_index[internal] = np.arange(len(internal))

    def ref(i):
        return 0x8000 | int(nodes[i, 1]) if nodes[i, 0] == LEAF else int(new_index[i])

    out = np.array([(nodes[i, 0], ref(nodes[i, 1]), ref(nodes[i, 2])) for i in internal], dtype=np.int64)
    return out, np.array([ref(r) for r in roots], dtype=np.int64)


def run(kind, cfg, data, seed):
    x_tr, y_tr, x_val, y_val, x_te, y_te = data
    t0 = time.time()
    if kind == "rf":
        trees, depth = cfg
        model = RandomForestClassifier(n_estimators=trees, max_depth=depth, max_features="sqrt", n_jobs=-1, random_state=seed)
        model.fit(x_tr, y_tr)
        integer = rf_integer
    else:
        iters, depth = cfg
        model = HistGradientBoostingClassifier(max_iter=iters, max_depth=depth, learning_rate=0.1,
                                               early_stopping=False, random_state=seed)
        model.fit(x_tr, y_tr)
        integer = hgb_integer
    pred_val, _, _ = integer(model, x_val)
    pred_te, visited, parts = integer(model, x_te)
    nodes = parts[0]
    n_trees = len(parts[1])
    n_leaves = int((nodes[:, 0] == LEAF).sum())
    res = {
        "kind": kind, "config": list(cfg), "trees": n_trees,
        "val_acc": float((pred_val == y_val).mean()), "test_acc": float((pred_te == y_te).mean()),
        "nodes_visited_mean": float(visited.mean()), "nodes_visited_max": int(visited.max()),
        "flash_bytes": flash_bytes(kind, nodes, n_trees, n_leaves), "train_s": time.time() - t0,
    }
    print(f"{kind:3} {str(cfg):12} trees {n_trees:5}  val {res['val_acc'] * 100:5.1f}  test {res['test_acc'] * 100:5.2f}  "
          f"nodes read {res['nodes_visited_mean']:7.1f} (max {res['nodes_visited_max']:5})  Flash {res['flash_bytes'] / 1024:7.1f} KB  "
          f"({res['train_s']:.0f}s)", flush=True)
    return res, model, pred_te


def export(kind, model, data, out_dir: Path, meta):
    """Writes the flattened ensemble and the packed test set for `tree_shd_eval.rs`."""
    *_, x_te, y_te = data
    if kind == "rf":
        pred, _, (nodes, roots, leaves) = rf_integer(model, x_te)
        payload = leaves.astype(np.int8).tobytes()
        tree_class = np.zeros(len(roots), dtype=np.uint8)
        base = np.zeros(N_CLASSES, dtype=np.int32)
    else:
        pred, _, (nodes, roots, tree_class, values, base) = hgb_integer(model, x_te)
        payload = values.astype("<i2").tobytes()
        base = base.astype(np.int32)
    c_nodes, c_roots = compact(nodes, roots)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tree_nodes.bin").write_bytes(c_nodes.astype("<u2").tobytes())
    (out_dir / "tree_roots.bin").write_bytes(c_roots.astype("<u2").tobytes())
    (out_dir / "tree_class.bin").write_bytes(np.asarray(tree_class).astype(np.uint8).tobytes())
    (out_dir / "tree_leaves.bin").write_bytes(payload)
    (out_dir / "tree_base.bin").write_bytes(base.astype("<i4").tobytes())
    words = x_te.reshape(len(x_te), -1, 32).astype(np.uint32)
    packed = (words << np.arange(32, dtype=np.uint32)).sum(-1, dtype=np.uint32)
    (out_dir / "test_x.bin").write_bytes(packed.astype("<u4").tobytes())
    (out_dir / "test_y.bin").write_bytes(y_te.astype(np.uint8).tobytes())
    (out_dir / "host_pred.bin").write_bytes(pred.astype(np.uint8).tobytes())
    acc = float((pred == y_te).mean())
    meta = dict(meta, kind=kind, leaf_kind="vector_i8" if kind == "rf" else "scalar_i16", nodes=int(len(nodes)),
                trees=int(len(roots)), samples=int(len(x_te)), host_integer_accuracy=acc)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"exported {kind} {meta['config']} to {out_dir}: integer accuracy {acc * 100:.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rf", default="64x8,128x8,256x8,64x12,128x12,256x12,128x16,256x16",
                    help="random-forest configs TREESxDEPTH")
    ap.add_argument("--hgb", default="10x4,25x4,50x4,100x4,25x6,50x6", help="boosting configs ITERxDEPTH")
    ap.add_argument("--export", type=Path, default=None)
    ap.add_argument("--export-config", default=None, help="config to export, e.g. rf:256x12 or hgb:50x4")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = split(args.data.expanduser(), args.seed)
    results = []
    parse = lambda spec: [tuple(int(v) for v in c.split("x")) for c in spec.split(",") if c]
    for kind, spec in (("rf", args.rf), ("hgb", args.hgb)):
        for cfg in parse(spec):
            res, model, _ = run(kind, cfg, data, args.seed)
            results.append(res)
            if args.export and args.export_config == f"{kind}:{cfg[0]}x{cfg[1]}":
                export(kind, model, data, args.export.expanduser(), {"config": list(cfg), "seed": args.seed})
    if args.out:
        args.out.write_text(json.dumps({"seed": args.seed, "results": results}, indent=2))
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
