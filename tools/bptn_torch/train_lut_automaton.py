#!/usr/bin/env python3
"""
train_lut_automaton.py

LUT automaton: a neuron whose state is a node number instead of a membrane potential.

M independent automata with S states each run over the input one tick (one frame) at a time. In
state s, automaton m reads k bits of the current frame at learned positions, forms a k-bit address
and looks up the next state in a learned table of 2^k entries. Every state holds int8 class
scores; the scores of the states visited are added over the window, and the class is the argmax.
Per tick, an automaton reads one table: the cost per frame is constant and the decision can be
taken after any frame.

Training (same principle as the forest of LUT trees in `train_shd_comparison.py`): the forward pass
is the deployed computation (hard input positions, hard transitions, one-hot states); the backward
pass goes through a softmax over the input-position scores (as in Differentiable Weightless Neural
Networks), the multilinear extension of the lookup and a softmax over the next-state logits
(straight-through estimators). An integer reference checks that the deployed model gives the same
predictions as PyTorch.

Input: an .npz with x [N, bits, ticks] in {0, 1} (bit j of frame t is x[j, t], as written by
`encode_autohack.py`; for Speech Commands and SHD, a frame is one time bin of all channels).
Validation split: 10 % of the training set, as in `train_shd_comparison.py`.

The cycle figures are estimates from per-operation costs measured on the Wio-E5 (48 MHz): one LUT
node inside a large model (111.9 cycles, measured) and an assumed int8 score addition; the automaton
itself is not measured on the board.

Usage:
    ~/.cache/snn_core/venv/bin/python -u tools/bptn_torch/train_lut_automaton.py \\
        --data-npz ~/.cache/snn_core/autohack/autohack_w64.npz --automata 16 --states 16
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_shd_comparison import ASSUMED_ACTIVE_POWER_W, CLOCK_HZ, CYCLES_LEAF_ADD_ASSUMED, CYCLES_LUT6_BOARD

# Assumed cost of adding one int8 class score to an int32 accumulator (the forest's assumed cost of
# a 20-class leaf, per class).
CYCLES_SCORE_ADD_PER_CLASS_ASSUMED = CYCLES_LEAF_ADD_ASSUMED / 20


class LutAutomaton(nn.Module):
    def __init__(self, bits, classes, automata, states, k, seed, map_tau=1.0, trans_tau=1.0, stay_bias=0.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.m, self.s, self.k, self.classes = automata, states, k, classes
        self.map_tau, self.trans_tau = map_tau, trans_tau
        n = automata * states
        self.map_scores = nn.Parameter(torch.randn(n * k, bits, generator=g) * 0.1)
        logits = torch.randn(automata, states, 2**k, states, generator=g) * 0.1
        logits[:, torch.arange(states), :, torch.arange(states)] += stay_bias  # optional bias to stay put
        self.trans_logits = nn.Parameter(logits)
        self.leaf = nn.Parameter(torch.randn(automata, states, classes, generator=g) * 0.01)
        self.deployed = False

    def index(self):
        """Deployed input positions [M, S, k] within the frame."""
        return self.map_scores.detach().argmax(dim=1).view(self.m, self.s, self.k)

    def next_state(self):
        """Deployed transition table [M, S, 2^k]."""
        return self.trans_logits.detach().argmax(dim=-1)

    def leaf_values(self):
        if not self.deployed:
            return self.leaf
        scale = self.leaf.abs().max().clamp(min=1e-8) / 127.0
        return torch.round(self.leaf / scale).clamp(-127, 127) * scale

    def transitions(self):
        hard = F.one_hot(self.trans_logits.argmax(-1), self.s).float()
        if self.deployed:
            return hard
        soft = torch.softmax(self.trans_logits / self.trans_tau, dim=-1)
        return hard + soft - soft.detach()

    def address_weights(self, frame, index, soft_map):
        """frame [B, bits] -> [B, M, S, 2^k] weights of each address (one-hot in the forward pass)."""
        b = frame.shape[0]
        sel = frame[:, index]  # [B, M, S, k]
        if soft_map is not None:
            soft = (frame @ soft_map).view(b, self.m, self.s, self.k)
            sel = sel + soft - soft.detach()
        w = torch.ones(b, self.m, self.s, 1, device=frame.device)
        for i in range(self.k):
            xi = sel[..., i : i + 1]
            w = torch.cat((w * (1 - xi), w * xi), dim=-1)  # address bit i has weight 2^i
        return w

    def forward(self, x):
        b, _, ticks = x.shape
        index = self.index()
        soft_map = None if self.deployed else torch.softmax(self.map_scores / self.map_tau, dim=1).t()
        trans = self.transitions()
        p = torch.zeros(b, self.m, self.s, device=x.device)
        p[:, :, 0] = 1.0
        occupancy = torch.zeros_like(p)
        for t in range(ticks):
            w = self.address_weights(x[:, :, t], index, soft_map)
            p = torch.einsum("bmsa,msaz->bmz", w * p.unsqueeze(-1), trans)
            occupancy = occupancy + p
        logits = torch.einsum("bms,msc->bc", occupancy, self.leaf_values()) / ticks
        return logits


def integer_reference(model: LutAutomaton, x: np.ndarray):
    """Deployed automata with integer operations, as a firmware would run them. Returns the
    predictions after the whole window, the int8 scores, and the predictions after each frame
    [ticks, N] (decision taken early from the scores accumulated so far)."""
    idx = model.index().cpu().numpy()
    nxt = model.next_state().cpu().numpy()
    w = model.leaf.detach().cpu()
    scale = w.abs().max().clamp(min=1e-8) / 127.0
    leaves = torch.round(w / scale).clamp(-127, 127).numpy().astype(np.int64)
    n, ar = len(x), np.arange(model.m)
    state = np.zeros((n, model.m), dtype=np.int64)
    scores = np.zeros((n, model.classes), dtype=np.int64)
    rows = np.arange(n)[:, None]
    early = np.zeros((x.shape[2], n), dtype=np.int64)
    for t in range(x.shape[2]):
        frame = x[:, :, t]
        address = np.zeros_like(state)
        for j in range(model.k):
            address |= frame[rows, idx[ar[None, :], state, j]].astype(np.int64) << j
        state = nxt[ar[None, :], state, address]
        scores += leaves[ar[None, :], state].sum(1)
        early[t] = scores.argmax(1)
    return scores.argmax(1), leaves, early


def evaluate(model, x, y, device, batch=512):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(x[i : i + batch], dtype=torch.float32, device=device)
            correct += (model(xb).argmax(1).cpu().numpy() == y[i : i + batch]).sum()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-npz", type=Path, required=True)
    ap.add_argument("--automata", type=int, default=16)
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--map-tau", type=float, default=1.0)
    ap.add_argument("--trans-tau", type=float, default=1.0)
    ap.add_argument("--stay-bias", type=float, default=0.0, help="initial logit bonus for staying in the same state")
    ap.add_argument("--max-train", type=int, default=0, help="cap on training samples (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-model", type=Path, default=None, help="file for the trained state dict")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = np.load(args.data_npz.expanduser())
    x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    classes = int(max(y_train.max(), y_test.max()) + 1)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(x_train))
    n_val = len(x_train) // 10
    x_val, y_val = x_train[perm[:n_val]], y_train[perm[:n_val]]
    x_tr, y_tr = x_train[perm[n_val:]], y_train[perm[n_val:]]
    if args.max_train:
        x_tr, y_tr = x_tr[: args.max_train], y_tr[: args.max_train]
    bits, ticks = x_train.shape[1], x_train.shape[2]
    print(f"{args.data_npz.stem}: {len(x_tr)} train / {len(x_val)} val / {len(x_test)} test, "
          f"{bits} bits x {ticks} frames, {classes} classes, device {device}")

    model = LutAutomaton(bits, classes, args.automata, args.states, args.k, args.seed,
                         args.map_tau, args.trans_tau, args.stay_bias).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = (-1.0, None)
    for epoch in range(args.epochs):
        model.train()
        model.deployed = False
        order = rng.permutation(len(x_tr))
        t0 = time.time()
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            xb = torch.as_tensor(x_tr[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_tr[idx], device=device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.deployed = True
        val_acc = evaluate(model, x_val, y_val, device)
        if val_acc > best[0]:
            best = (val_acc, {k: v.detach().clone() for k, v in model.state_dict().items()})
        print(f"  epoch {epoch + 1:3d}/{args.epochs}  loss {loss.item():.3f}  val {val_acc * 100:5.1f}%  "
              f"({time.time() - t0:.1f}s)")
    model.load_state_dict(best[1])
    model.deployed = True

    test_acc = evaluate(model, x_test, y_test, device)
    pred, _, early = integer_reference(model, x_test.astype(np.uint8))
    frames_read = [f for f in (1, 2, 4, 8, 16, 24, 32, 48, 64) if f <= ticks]
    acc_by_frames = {f: float((early[f - 1] == y_test).mean()) for f in frames_read}
    int_acc = float((pred == y_test).mean())
    with torch.no_grad():
        torch_pred = np.concatenate([
            model(torch.as_tensor(x_test[i : i + 512], dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
            for i in range(0, len(x_test), 512)])
    agree = float((torch_pred == pred).mean())
    states_used = int(np.mean([len(np.unique(model.next_state()[m].cpu().numpy())) for m in range(args.automata)]))

    cycles_tick = args.automata * (CYCLES_LUT6_BOARD + classes * CYCLES_SCORE_ADD_PER_CLASS_ASSUMED)
    state_bits = max(1, int(np.ceil(np.log2(args.states))))
    flash = args.automata * args.states * (2**args.k * state_bits / 8 + args.k + classes)
    res = dict(dataset=args.data_npz.stem, automata=args.automata, states=args.states, k=args.k,
               epochs=args.epochs, seed=args.seed, stay_bias=args.stay_bias, val_acc=best[0],
               test_acc=float(test_acc), integer_test_acc=int_acc, integer_agreement=agree,
               mean_target_states=states_used, est_cycles_per_frame=cycles_tick,
               est_cycles_per_window=cycles_tick * ticks, est_us_per_frame=cycles_tick / CLOCK_HZ * 1e6,
               est_uj_per_frame=cycles_tick / CLOCK_HZ * ASSUMED_ACTIVE_POWER_W * 1e6, flash_bytes=flash,
               sram_bytes=args.automata + 4 * classes, test_acc_by_frames_read=acc_by_frames)
    print(f"test {test_acc * 100:.2f}%  integer reference {int_acc * 100:.2f}% (agreement {agree * 100:.2f}%)  "
          f"states reached per automaton {states_used}/{args.states}")
    print(f"estimated {cycles_tick:,.0f} cycles per frame ({res['est_us_per_frame']:.1f} us), "
          f"{cycles_tick * ticks:,.0f} per {ticks}-frame window; Flash {flash / 1024:.1f} KB, "
          f"SRAM {res['sram_bytes']} B (estimates, not measured)")
    print("accuracy after f frames: " + ", ".join(f"{f}: {a * 100:.1f}%" for f, a in acc_by_frames.items()))
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
