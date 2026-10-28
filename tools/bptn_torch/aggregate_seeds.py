#!/usr/bin/env python3
"""
aggregate_seeds.py

Mean and standard deviation over seeds of the test accuracy of every model of a task, and paired
comparisons against a reference model (WP1 of docs/plan_consolidation.md).

Input: JSON files named <model>_seed<s>.json (results/seeds/<task>/). Each script's layout is read
by an adapter:
- train_shd_comparison.py: results[].test_acc_deployed, est_cycles_per_sample (estimated);
- train_lut_boosting.py: test_acc_int8, est_cycles (estimated);
- train_lut_automaton.py: integer_test_acc, est_cycles_per_window (estimated);
- tree_baselines.py: results[].test_acc, one model per configuration (<model>:hgb50x4), nodes read;
- cnn_int8_baseline.py: results[].test_int, one model per configuration (<model>:tiny); cycles
  are the lower bound MACs / 2 (estimated).
Files without a known layout are listed and skipped.

Paired rule: for models A and B on their common seeds, d = A - B; "A > B" when mean(d) > 2 std(d)
and d > 0 on every seed, "A < B" symmetrically, otherwise "tie". With fewer than 2 common seeds
the verdict is "n/a".

Usage:
    python3 tools/bptn_torch/aggregate_seeds.py --dir tools/bptn_torch/results/seeds/autohack \\
        --ref cnn:tiny --md --out tools/bptn_torch/results/summary/autohack.json
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

SEED_RE = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)$")


def entries(d, stem):
    """Yields (model name, accuracy, cycles, cycles status, flash bytes) for one JSON file."""
    if "test_acc_int8" in d:
        yield stem, d["test_acc_int8"], d.get("est_cycles"), "estimated", d.get("flash_bytes")
    elif "integer_test_acc" in d:
        yield stem, d["integer_test_acc"], d.get("est_cycles_per_window"), "estimated", d.get("flash_bytes")
    for r in d.get("results", []) if isinstance(d.get("results"), list) else []:
        if "test_acc_deployed" in r:
            yield stem, r["test_acc_deployed"], r.get("est_cycles_per_sample"), "estimated", r.get("flash_bytes")
        elif "kind" in r and "test_acc" in r:
            cfg = "x".join(str(v) for v in r["config"])
            yield f"{stem}:{r['kind']}{cfg}", r["test_acc"], None, f"{r.get('nodes_visited_mean', 0):.0f} nodes", r.get("flash_bytes")
        elif "name" in r and "test_int" in r:
            yield f"{stem}:{r['name']}", r["test_int"], r["macs"] / 2, "lower bound", r.get("weights_bytes")


def verdict(a, b):
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    d = [a[s] - b[s] for s in common]
    m, sd = statistics.mean(d), statistics.stdev(d)
    if m > 2 * sd and all(x > 0 for x in d):
        v = "better"
    elif -m > 2 * sd and all(x < 0 for x in d):
        v = "worse"
    else:
        v = "tie"
    return dict(seeds=common, mean_diff=m, std_diff=sd, verdict=v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="directory of <model>_seed<s>.json files")
    ap.add_argument("--select", default=None, help="regular expression on model names to keep")
    ap.add_argument("--ref", default=None, help="reference model for the paired comparisons")
    ap.add_argument("--md", action="store_true", help="print a Markdown table")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    acc = defaultdict(dict)  # model -> seed -> accuracy (%)
    cost = {}
    skipped = []
    for f in sorted(args.dir.glob("*.json")):
        m = SEED_RE.match(f.stem)
        if not m:
            skipped.append(f.name)
            continue
        found = False
        for name, a, cycles, status, flash in entries(json.loads(f.read_text()), m["model"]):
            if args.select and not re.search(args.select, name):
                continue
            acc[name][int(m["seed"])] = 100.0 * a
            cost[name] = dict(cycles=cycles, cycles_status=status, flash_bytes=flash)
            found = True
        if not found:
            skipped.append(f.name)

    rows = []
    for name in sorted(acc):
        v = acc[name]
        vals = [v[s] for s in sorted(v)]
        row = dict(model=name, n=len(vals), seeds=sorted(v), values=vals, mean=statistics.mean(vals),
                   std=statistics.stdev(vals) if len(vals) > 1 else None, **cost[name])
        if args.ref and name != args.ref and args.ref in acc:
            row["vs_ref"] = verdict(v, acc[args.ref])
        rows.append(row)

    if args.md:
        ref = f" vs {args.ref}" if args.ref else ""
        print(f"| Model | n | Test accuracy | Per seed | Cycles | Flash |{' Paired' + ref + ' |' if ref else ''}")
        print(f"| :--- | ---: | ---: | :--- | :--- | ---: |{' :--- |' if ref else ''}")
        for r in rows:
            sd = f" ± {r['std']:.2f}" if r["std"] is not None else " (1 seed)"
            cyc = f"{r['cycles']:,.0f} ({r['cycles_status']})" if r["cycles"] else r["cycles_status"]
            fl = f"{r['flash_bytes'] / 1024:.1f} KB" if r["flash_bytes"] else ""
            line = f"| {r['model']} | {r['n']} | {r['mean']:.2f}{sd} | {' / '.join(f'{x:.2f}' for x in r['values'])} | {cyc} | {fl} |"
            if ref:
                c = r.get("vs_ref")
                line += f" {c['verdict']} ({c['mean_diff']:+.2f} ± {c['std_diff']:.2f}, n={len(c['seeds'])}) |" if c else " — |"
            print(line)
    if skipped:
        print(f"skipped (no seed in the name or unknown layout): {', '.join(skipped)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(dir=str(args.dir), ref=args.ref, models=rows), indent=2))


if __name__ == "__main__":
    main()
