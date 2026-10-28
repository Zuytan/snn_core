#!/usr/bin/env python3
"""
encode_mitbih.py

Encodes the MIT-BIH Arrhythmia Database (PhysioNet, ODC-By 1.0; `mitdb` 1.0.0) into the binary
64 x 64 input used by the SHD comparison scripts.

- Beats: the annotated beats of the MLII lead (360 Hz), mapped to the AAMI classes
  N (normal / bundle branch / escape), S (supraventricular ectopic), V (ventricular ectopic) and
  F (fusion). Q (paced / unclassifiable) beats are dropped, as are the four paced records
  102, 104, 107 and 217.
- Split: inter-patient, de Chazal et al. 2004 (DS1 for training, DS2 for testing). No record
  appears in both splits.
- A window is 192 samples centred on the annotated R peak (0.53 s), decimated by 3 to 64 ticks,
  after subtracting the window median (baseline removal). The RR intervals are NOT given to the
  model, so S beats are hard by design: this tests the waveform alone.
- Each tick becomes 64 bits: a 32-level thermometer code of the amplitude and a 32-level
  thermometer code of the first difference. Thresholds are quantiles of the training signal, so
  the bit density is about 50 % and the test split is encoded with the training thresholds.
  Bit j of tick t is x[j, t].
- With `--rr`, 64 more bits per tick (constant over the ticks) give the model the rhythm context:
  the previous and next RR intervals and each divided by the mean of the 8 previous RR intervals,
  4 features x 16-level thermometer codes at training quantiles. Without it the model sees the
  waveform only.
- The N class dominates (about 90 %). Each class is capped at the size of the largest non-N class
  in the same split (seeded random draw), so accuracy is not dominated by Normal beats.

Output: an .npz with x_train, y_train, x_test, y_test ([N, 64 bits, 64 ticks], uint8), the class
names, the thresholds and the beat counts before capping.

Requires `pip install wfdb numpy`; the records are fetched with `wfdb.dl_database("mitdb", dir)`.

Usage:
    python3 tools/bptn_torch/encode_mitbih.py --root ~/.cache/snn_core/mitdb \\
        --out ~/.cache/snn_core/mitdb/mitbih_w64.npz
"""

import argparse
from pathlib import Path

import numpy as np

CLASSES = ["N", "S", "V", "F"]
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
       219, 221, 222, 228, 231, 232, 233, 234]
AAMI = {
    **dict.fromkeys("NLRej", 0),
    **dict.fromkeys("AaJS", 1),
    **dict.fromkeys("VE", 2),
    "F": 3,
}
HALF = 96
DECIMATE = 3
LEVELS = 32
RR_LEVELS = 16
RR_LOCAL = 8
# Every annotation that marks a beat (used to measure the RR intervals, whatever the class).
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ")


def rr_features(beat_samples, k):
    """[previous RR, next RR, previous RR / local mean, next RR / local mean] of the k-th beat."""
    rr = np.diff(beat_samples).astype(np.float64)
    pre = rr[k - 1] if k > 0 else rr[k]
    post = rr[k] if k < len(rr) else rr[k - 1]
    local = rr[max(0, k - RR_LOCAL): k].mean() if k > 0 else post
    return [pre, post, pre / local, post / local]


def beats(root: Path, record: int):
    """Returns the raw windows [n, 2 * HALF], the AAMI labels [n] and the RR features [n, 4]."""
    import wfdb

    rec = wfdb.rdrecord(str(root / str(record)))
    ann = wfdb.rdann(str(root / str(record)), "atr")
    lead = next((i for i, name in enumerate(rec.sig_name) if name == "MLII"), None)
    if lead is None:
        raise ValueError(f"record {record} has no MLII lead: {rec.sig_name}")
    sig = rec.p_signal[:, lead]
    beat_samples = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS])
    windows, labels, rr = [], [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol not in AAMI or sample < HALF or sample + HALF > len(sig):
            continue
        w = sig[sample - HALF: sample + HALF]
        windows.append(w - np.median(w))
        labels.append(AAMI[symbol])
        rr.append(rr_features(beat_samples, int(np.searchsorted(beat_samples, sample))))
    return np.array(windows), np.array(labels, dtype=np.int64), np.array(rr)


def load(root: Path, records):
    xs, ys, rrs = [], [], []
    for r in records:
        x, y, rr = beats(root, r)
        xs.append(x)
        ys.append(y)
        rrs.append(rr)
    x = np.concatenate(xs)
    x = x.reshape(len(x), -1, DECIMATE).mean(axis=2)  # [N, 64 ticks]
    return x, np.concatenate(ys), np.concatenate(rrs)


def thresholds(values):
    return np.quantile(values, (np.arange(LEVELS) + 0.5) / LEVELS)


def encode(x, amp_thr, diff_thr):
    diff = np.diff(x, axis=1, prepend=x[:, :1])
    amp = (x[:, None, :] > amp_thr[None, :, None]).astype(np.uint8)
    dif = (diff[:, None, :] > diff_thr[None, :, None]).astype(np.uint8)
    return np.ascontiguousarray(np.concatenate([amp, dif], axis=1))  # [N, 64 bits, 64 ticks]


def encode_rr(rr, rr_thr, ticks):
    """Thermometer codes of the RR features, held constant over the ticks: [N, features * RR_LEVELS, ticks]."""
    bits = (rr[:, :, None] > rr_thr[None, :, :]).astype(np.uint8)  # [N, features, RR_LEVELS]
    bits = bits.reshape(len(rr), -1)
    return np.ascontiguousarray(np.repeat(bits[:, :, None], ticks, axis=2))


def cap(x, y, rng):
    size = max(int((y == c).sum()) for c in range(1, len(CLASSES)))
    keep = np.concatenate([
        rng.choice(np.flatnonzero(y == c), size=min(size, int((y == c).sum())), replace=False)
        for c in range(len(CLASSES))
    ])
    keep.sort()
    return x[keep], y[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rr", action="store_true",
                    help="append 64 bits (4 RR features x 16 levels, constant over the ticks): 128 bits per tick")
    ap.add_argument("--causal", action="store_true",
                    help="with --rr, keep only the previous RR and its ratio (32 bits, no look-ahead)")
    args = ap.parse_args()
    root = args.root.expanduser()
    rng = np.random.default_rng(args.seed)

    raw = {split: load(root, recs) for split, recs in (("train", DS1), ("test", DS2))}
    amp_thr = thresholds(raw["train"][0].ravel())
    diff_thr = thresholds(np.diff(raw["train"][0], axis=1).ravel())
    rr_thr = np.quantile(raw["train"][2], (np.arange(RR_LEVELS) + 0.5) / RR_LEVELS, axis=0).T  # [4, RR_LEVELS]

    out, counts = {}, {}
    for split, (x, y, rr) in raw.items():
        counts[split] = {CLASSES[c]: int((y == c).sum()) for c in range(len(CLASSES))}
        enc = encode(x, amp_thr, diff_thr)
        if args.rr:
            keep = [0, 2] if args.causal else [0, 1, 2, 3]  # previous RR and ratio; next RR and ratio look ahead
            enc = np.concatenate([enc, encode_rr(rr[:, keep], rr_thr[keep], enc.shape[2])], axis=1)
        x, y = cap(enc, y, rng)
        out[f"x_{split}"], out[f"y_{split}"] = x, y
        print(f"{split}: beats {counts[split]}, kept {len(y)} (density {x.mean() * 100:.1f}%)", flush=True)
    np.savez_compressed(args.out.expanduser(), **out, classes=np.array(CLASSES),
                        amp_thresholds=amp_thr, diff_thresholds=diff_thr, rr_thresholds=rr_thr,
                        counts=np.array(str(counts)))
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
