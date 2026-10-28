#!/usr/bin/env python3
"""
mitbih_recall.py

Per-class recall of the 50 x 4 gradient boosting model on an encoded MIT-BIH .npz (see
encode_mitbih.py): plain accuracy hides the weak S and F classes, the macro recall does not.

Usage:
    python3 tools/bptn_torch/mitbih_recall.py --data ~/.cache/snn_core/mitdb/mitbih_w64_rr.npz --out recall.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.data.expanduser())
    classes = [str(c) for c in d["classes"]]
    xtr, xte = (d[k].reshape(len(d[k]), -1) for k in ("x_train", "x_test"))
    model = HistGradientBoostingClassifier(max_iter=50, max_depth=4, learning_rate=0.1,
                                           early_stopping=False, random_state=args.seed)
    model.fit(xtr, d["y_train"])
    pred = model.predict(xte)
    cm = confusion_matrix(d["y_test"], pred, labels=range(len(classes)))
    recall = cm.diagonal() / cm.sum(axis=1)
    result = {
        "data": str(args.data),
        "accuracy": float(cm.diagonal().sum() / cm.sum()),
        "recall": {c: float(r) for c, r in zip(classes, recall)},
        "macro_recall": float(recall.mean()),
        "confusion": cm.tolist(),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
