#!/usr/bin/env python3
"""
train_lut_boosting.py

Gradient boosting of LUT decision trees, without gradient descent on the model.

The model is the forest of LUT trees of `train_shd_comparison.py --models bp` (same format, same
firmware `bp_shd_eval`): T trees of depth D, heap order, every node reads k = 6 input bits, forms a
6-bit address and reads one bit of a 64-entry table to choose its child; every leaf adds a vector of
int8 class scores. Instead of training all trees together with straight-through estimators, the
trees are grown one after the other, as in gradient boosting with a softmax loss:

- each tree fits the Newton step of the current scores: per sample, gradient g = p - y and hessian
  h = p (1 - p) for every class; one multi-output tree per iteration (the leaf holds all classes);
- at each node, the 6 input bits are chosen greedily: at step j, the bit added is the one that
  maximizes sum over cells of ||G_cell||^2 / (H_cell + lambda) when the node samples are split into
  the 2^j cells of the address built so far (a gain computed for all 4,096 candidate bits at once
  with one matrix product);
- the table (which of the 64 cells go to the right child) is then chosen to maximize the split gain
  sum over the 2 children of sum_c G_c^2 / (H_c + lambda): the best threshold split of the cells
  sorted along each class, then single-cell moves while they improve the gain;
- leaf value: learning_rate * (-G / (H + lambda)) per class.

An integer reference (int8 leaves, global scale, as in the forest export) checks the deployed model.

Usage:
    ~/.cache/snn_core/venv/bin/python -u tools/bptn_torch/train_lut_boosting.py \\
        --data-npz ~/.cache/snn_core/shd/shd_encoded_c64_t64_1.00s.npz --trees 64 --depth 4 \\
        --export ~/.cache/snn_core/lutboost_export_shd_t64_d4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import train_shd_comparison as tsc


def softmax(f):
    e = np.exp(f - f.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def split_gain(g, h, lam):
    """g, h: [..., C] sums -> sum_c g^2 / (h + lambda)."""
    return (g * g / (h + lam)).sum(-1)


def choose_bits(x, g, h, k, lam, allowed=None):
    """Greedy choice of k input bits for the samples of one node. x [n, F] float32 bits,
    g, h [n, C]; `allowed` (optional boolean [F]) restricts the candidate bits. Returns the bit
    positions and the cell index (address) of every sample."""
    n, f = x.shape
    c = g.shape[1]
    cell = np.zeros(n, dtype=np.int64)
    bits = []
    gh = np.concatenate([g, h], axis=1)  # [n, 2C]
    for j in range(k):
        cells = 2**j
        onehot = np.zeros((n, cells), dtype=np.float32)
        onehot[np.arange(n), cell] = 1.0
        weighted = (onehot[:, :, None] * gh[:, None, :]).reshape(n, cells * 2 * c)
        s1 = (x.T @ weighted).reshape(f, cells, 2, c)  # sums over samples with bit = 1, per cell
        tot = weighted.sum(0).reshape(1, cells, 2, c)
        s0 = tot - s1
        score = (split_gain(s1[:, :, 0], s1[:, :, 1], lam) + split_gain(s0[:, :, 0], s0[:, :, 1], lam)).sum(1)
        score[bits] = -np.inf
        if allowed is not None:
            score[~allowed] = -np.inf
        b = int(np.argmax(score))
        bits.append(b)
        cell = cell | (x[:, b].astype(np.int64) << j)  # address bit j has weight 2^j, as in the firmware
    return bits, cell


def choose_table(cg, ch, lam):
    """cg, ch [64, C] cell sums -> table [64] of 0/1 (1 = right child) maximizing the split gain."""
    cells = cg.shape[0]
    tg, th = cg.sum(0), ch.sum(0)

    def gain(side):
        rg, rh = cg[side == 1].sum(0), ch[side == 1].sum(0)
        return split_gain(rg, rh, lam) + split_gain(tg - rg, th - rh, lam)

    best_side, best = np.zeros(cells, dtype=np.int64), -np.inf
    for cl in range(cg.shape[1]):
        order = np.argsort(cg[:, cl] / (ch[:, cl] + lam))
        for cut in range(1, cells):
            side = np.zeros(cells, dtype=np.int64)
            side[order[cut:]] = 1
            v = gain(side)
            if v > best:
                best, best_side = v, side
    improved = True
    while improved:
        improved = False
        for i in range(cells):
            side = best_side.copy()
            side[i] ^= 1
            v = gain(side)
            if v > best + 1e-9:
                best, best_side, improved = v, side, True
    return best_side


def grow_tree(x, g, h, depth, k, lam, lr, min_samples, col_frac, rng):
    """One multi-output LUT tree. Returns (bits [nodes, k], tables [nodes, 64], leaves [2^depth, C],
    leaf index of every sample)."""
    n_nodes = 2**depth - 1
    c = g.shape[1]
    bits = np.zeros((n_nodes, k), dtype=np.int64)
    tables = np.zeros((n_nodes, 2**k), dtype=np.int64)
    node = np.zeros(len(x), dtype=np.int64)
    for level in range(depth):
        for nd in range(2**level - 1, 2 ** (level + 1) - 1):
            m = np.flatnonzero(node == nd)
            if len(m) < min_samples:
                bits[nd] = np.arange(k)  # unused node: any bits, table 0 (left child)
                continue
            allowed = rng.random(x.shape[1]) < col_frac if col_frac < 1 else None
            b, cell = choose_bits(x[m], g[m], h[m], k, lam, allowed)
            cg = np.zeros((2**k, c)); ch = np.zeros((2**k, c))
            np.add.at(cg, cell, g[m]); np.add.at(ch, cell, h[m])
            bits[nd], tables[nd] = b, choose_table(cg, ch, lam)
        out = tables[node, address(x, bits[node])]
        node = 2 * node + 1 + out
    leaf = node - n_nodes
    lg = np.zeros((2**depth, c)); lh = np.zeros((2**depth, c))
    np.add.at(lg, leaf, g); np.add.at(lh, leaf, h)
    return bits, tables, -lr * lg / (lh + lam), leaf


def address(x, node_bits):
    """x [n, F], node_bits [n, k] -> k-bit address of every sample."""
    rows = np.arange(len(x))[:, None]
    sel = x[rows, node_bits].astype(np.int64)
    return (sel << np.arange(node_bits.shape[1])).sum(1)


def tree_leaf(x, bits, tables, depth):
    node = np.zeros(len(x), dtype=np.int64)
    for _ in range(depth):
        node = 2 * node + 1 + tables[node, address(x, bits[node])]
    return node - (2**depth - 1)


def to_branching_net(trees, depth, k, classes, all_bits, all_tables, all_leaves, n_inputs):
    """Builds a BranchingNet holding the boosted trees, so that the forest export and integer
    reference of train_shd_comparison.py apply unchanged."""
    tsc.N_CLASSES = classes
    model = tsc.BranchingNet(n_inputs, 1, trees, depth, k, seed=0)
    n = trees * (2**depth - 1)
    scores = torch.zeros(n * k, n_inputs)
    scores[torch.arange(n * k), torch.as_tensor(np.concatenate(all_bits).reshape(-1))] = 1.0
    with torch.no_grad():
        model.nodes.map_scores.copy_(scores)
        model.nodes.logits.copy_(torch.as_tensor(np.concatenate(all_tables) * 2.0 - 1.0, dtype=torch.float32))
        model.leaf.copy_(torch.as_tensor(np.stack(all_leaves), dtype=torch.float32))
    tsc.set_deployed(model, True)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-npz", type=Path, required=True)
    ap.add_argument("--trees", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.1, help="shrinkage of every tree")
    ap.add_argument("--lam", type=float, default=1.0, help="L2 regularization of the leaf values")
    ap.add_argument("--min-samples", type=int, default=20, help="minimum samples to split a node")
    ap.add_argument("--max-node-samples", type=int, default=20000,
                    help="subsample of the training set used to grow each tree (all samples get updated)")
    ap.add_argument("--row-frac", type=float, default=1.0, help="fraction of the training samples drawn for each tree")
    ap.add_argument("--col-frac", type=float, default=1.0, help="fraction of the input bits that are candidates at each node")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--export", type=Path, default=None, help="directory for the firmware export (bp_shd_eval format)")
    args = ap.parse_args()

    d = np.load(args.data_npz.expanduser())
    x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    classes = int(max(y_train.max(), y_test.max()) + 1)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(x_train))  # same validation split as train_shd_comparison.py
    n_val = len(x_train) // 10
    flat = lambda a: a.reshape(len(a), -1).astype(np.float32)
    x_val, y_val = flat(x_train[perm[:n_val]]), y_train[perm[:n_val]]
    x_tr, y_tr = flat(x_train[perm[n_val:]]), y_train[perm[n_val:]]
    x_te = flat(x_test)
    print(f"{args.data_npz.stem}: {len(x_tr)} train / {len(x_val)} val / {len(x_te)} test, {x_tr.shape[1]} input bits, "
          f"{classes} classes; {args.trees} trees of depth {args.depth}, k {args.k}, lr {args.lr}, lambda {args.lam}")

    y1 = np.eye(classes)[y_tr]
    f_tr, f_val, f_te = (np.zeros((len(a), classes)) for a in (x_tr, x_val, x_te))
    all_bits, all_tables, all_leaves, curve = [], [], [], []
    t0 = time.time()
    for t in range(args.trees):
        p = softmax(f_tr)
        g, h = (p - y1), np.maximum(p * (1 - p), 1e-6)
        sub = rng.choice(len(x_tr), min(args.max_node_samples, int(len(x_tr) * args.row_frac)), replace=False)
        bits, tables, leaves, _ = grow_tree(x_tr[sub], g[sub].astype(np.float32), h[sub].astype(np.float32),
                                            args.depth, args.k, args.lam, args.lr, args.min_samples, args.col_frac, rng)
        all_bits.append(bits); all_tables.append(tables); all_leaves.append(leaves)
        for xs, fs in ((x_tr, f_tr), (x_val, f_val), (x_te, f_te)):
            fs += leaves[tree_leaf(xs, bits, tables, args.depth)]
        val_acc = float((f_val.argmax(1) == y_val).mean())
        curve.append(val_acc)
        if (t + 1) % 8 == 0 or t + 1 == args.trees:
            print(f"  tree {t + 1:4d}/{args.trees}  train {(f_tr.argmax(1) == y_tr).mean() * 100:5.1f}%  "
                  f"val {val_acc * 100:5.1f}%  ({time.time() - t0:.0f}s)", flush=True)

    test_float = float((f_te.argmax(1) == y_test).mean())
    model = to_branching_net(args.trees, args.depth, args.k, classes, all_bits, all_tables, all_leaves, x_tr.shape[1])
    x_te_bits = x_test.reshape(len(x_test), -1).astype(np.uint8)
    ref_pred, *_ = tsc.branching_integer_reference(model, x_te_bits)
    test_int = float((ref_pred == y_test).mean())
    float_pred = f_te.argmax(1)
    agree_float = float((ref_pred == float_pred).mean())
    nt, dd = args.trees, args.depth
    cycles = nt * (dd * tsc.CYCLES_LUT6_BOARD + tsc.CYCLES_LEAF_ADD_ASSUMED * classes / 20)
    flash = nt * (2**dd - 1) * tsc.FLASH_LUT6_TABLE_BYTES + nt * 2**dd * classes
    print(f"test float {test_float * 100:.2f}%, int8 reference {test_int * 100:.2f}% (agreement with float {agree_float * 100:.2f}%), "
          f"val {curve[-1] * 100:.2f}% (best {max(curve) * 100:.2f}% at {int(np.argmax(curve)) + 1} trees)")
    print(f"estimated {cycles:,.0f} cycles ({nt * dd} nodes read), Flash {flash / 1024:.1f} KB (estimates from SHD-measured per-node costs)")
    if args.export:
        tsc.export_branching(model, x_test.reshape(len(x_test), 1, -1), y_test, args.export.expanduser(), {
            "dataset": args.data_npz.stem, "seed": args.seed, "training": "lut_boosting", "lr": args.lr,
            "lam": args.lam, "test_acc_float": test_float})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(
            dataset=args.data_npz.stem, trees=nt, depth=dd, k=args.k, lr=args.lr, lam=args.lam,
            min_samples=args.min_samples, max_node_samples=args.max_node_samples, row_frac=args.row_frac,
            col_frac=args.col_frac, seed=args.seed,
            val_curve=curve, val_acc=curve[-1], test_acc_float=test_float, test_acc_int8=test_int,
            int8_float_agreement=agree_float, est_cycles=cycles, flash_bytes=flash), indent=2))


if __name__ == "__main__":
    main()
