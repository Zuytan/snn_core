#!/usr/bin/env python3
"""
run_lut_shd_on_board.py

Runs the whole SHD test set through a trained network on the Wio-E5, chunk by chunk: the temporal
LUT network (`--kind lut`, firmware `lut_shd_eval`) or the binarized baseline (`--kind bnn`,
firmware `bnn_shd_eval`), or the forest of LUT decision trees (`--kind bp`, firmware
`bp_shd_eval`).

The test set does not fit in Flash with the model, so for each chunk the script rebuilds
`examples/wio_e5_benchmark` with `LUT_MODEL_DIR` or `BNN_MODEL_DIR`, `SHD_CHUNK_START` and
`SHD_CHUNK_LEN`, flashes and runs the firmware with `probe-rs run` (it stops on a breakpoint), and parses the
RTT output. It checks every board prediction against the host integer reference and writes the
accuracy and the measured cycles per inference to a JSON file.

Usage:
    python3 tools/run_lut_shd_on_board.py --model-dir ~/.cache/snn_core/lut_export_2000 \\
        --out tools/bptn_torch/results/lut_board_2000.json
    python3 tools/run_lut_shd_on_board.py --kind bnn --model-dir ~/.cache/snn_core/bnn_export_64 \\
        --out tools/bptn_torch/results/bnn_board_64.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIRMWARE = ROOT / "examples" / "wio_e5_benchmark"
RELEASE = FIRMWARE / "target" / "thumbv7em-none-eabi" / "release"
KINDS = {
    "lut": ("lut_shd_eval", "LUT_MODEL_DIR"),
    "lut-ee": ("lut_shd_early_exit", "LUT_MODEL_DIR"),
    "lut-stream": ("lut_stream_eval", "LUT_MODEL_DIR"),
    "bnn": ("bnn_shd_eval", "BNN_MODEL_DIR"),
    "bp": ("bp_shd_eval", "BP_MODEL_DIR"),
    "tree": ("tree_shd_eval", "TREE_MODEL_DIR"),
}
# Optional fields of the CHUNK line (early exit and streaming binaries).
EXTRA_RE = re.compile(r"\b(rounds_sum|ticks|updated_sum|counts_ok)=(\d+)")
CHUNK_RE = re.compile(
    r"CHUNK start=(\d+) samples=(\d+) correct=(\d+) host_agree=(\d+) cycles_min=(\d+) "
    r"cycles_mean=([\d.]+) cycles_max=(\d+) us_mean=([\d.]+) clock_hz=(\d+)"
)


def run_chunk(kind: str, model_dir: Path, start: int, length: int, chip: str, timeout: int, extra_env: dict) -> dict:
    binary, env_var = KINDS[kind]
    env = dict(os.environ, SHD_CHUNK_START=str(start), SHD_CHUNK_LEN=str(length), **extra_env)
    env[env_var] = str(model_dir)
    subprocess.run(
        ["cargo", "build", "--release", "--bin", binary], cwd=FIRMWARE, env=env, check=True,
        stdout=subprocess.DEVNULL,
    )
    proc = subprocess.run(
        ["probe-rs", "run", "--chip", chip, str(RELEASE / binary)], cwd=FIRMWARE, capture_output=True, text=True,
        timeout=timeout,
    )
    text = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
    m = CHUNK_RE.search(text)
    pred = re.search(r"PRED (\d+) ([0-9a-f]*)", text)
    if not m or not pred:
        sys.exit(f"chunk {start}: no result in the probe-rs output:\n{text[-2000:]}")
    keys = ["start", "samples", "correct", "host_agree", "cycles_min", "cycles_mean", "cycles_max", "us_mean", "clock_hz"]
    res = {k: (float(v) if "." in v else int(v)) for k, v in zip(keys, m.groups())}
    chunk_line = text[m.start() : text.find("\n", m.start())]
    for key, value in EXTRA_RE.findall(chunk_line):
        res[key] = int(value)
    res["predictions"] = [int(pred.group(2)[i : i + 2], 16) for i in range(0, len(pred.group(2)), 2)]
    if len(res["predictions"]) != res["samples"]:
        sys.exit(f"chunk {start}: {len(res['predictions'])} predictions for {res['samples']} samples")
    print(
        f"chunk {start:5d}: {res['samples']} samples, {res['correct']} correct, host agreement "
        f"{res['host_agree']}/{res['samples']}, cycles {res['cycles_min']}..{res['cycles_max']} (mean {res['cycles_mean']:.0f})"
    )
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=sorted(KINDS), default="lut")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=320, help="test samples per firmware build")
    ap.add_argument("--chip", default="STM32WLE5JC")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env", action="append", default=[], help="extra build variable KEY=VALUE (e.g. LUT_EE_DELTA=7)")
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    meta = json.loads((model_dir / "meta.json").read_text())
    samples = meta["samples"]
    chunks = [
        run_chunk(args.kind, model_dir, s, args.chunk, args.chip, args.timeout, dict(e.split("=", 1) for e in args.env))
        for s in range(0, samples, args.chunk)
    ]

    total = sum(c["samples"] for c in chunks)
    correct = sum(c["correct"] for c in chunks)
    agree = sum(c["host_agree"] for c in chunks)
    mean_cycles = sum(c["cycles_mean"] * c["samples"] for c in chunks) / total
    clock = chunks[0]["clock_hz"]
    summary = {
        "kind": args.kind,
        "model": meta,
        "board": "Wio-E5 (STM32WLE5JC)",
        "clock_hz": clock,
        "samples": total,
        "board_accuracy": correct / total,
        "host_integer_accuracy": meta["host_integer_accuracy"],
        "host_agreement": agree / total,
        "cycles_min": min(c["cycles_min"] for c in chunks),
        "cycles_mean": mean_cycles,
        "cycles_max": max(c["cycles_max"] for c in chunks),
        "us_mean": mean_cycles / clock * 1e6,
        "build_env": args.env,
        "chunks": [{k: v for k, v in c.items() if k != "predictions"} for c in chunks],
    }
    if all("rounds_sum" in c for c in chunks):
        summary["rounds_mean"] = sum(c["rounds_sum"] for c in chunks) / total
    if all("ticks" in c for c in chunks):
        # Streaming: cycles are per tick; also report the LUTs recomputed per tick and the checks.
        ticks = sum(c["ticks"] for c in chunks)
        summary["cycles_unit"] = "tick"
        summary["cycles_mean"] = sum(c["cycles_mean"] * c["ticks"] for c in chunks) / ticks
        summary["us_mean"] = summary["cycles_mean"] / clock * 1e6
        summary["luts_updated_per_tick"] = sum(c["updated_sum"] for c in chunks) / ticks
        summary["final_state_checks_ok"] = sum(c["counts_ok"] for c in chunks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    if "luts_updated_per_tick" in summary:
        print(
            f"Streaming: {summary['cycles_mean']:.0f} cycles per tick ({summary['us_mean']:.1f} us), "
            f"{summary['luts_updated_per_tick']:.0f} LUTs recomputed per tick, final state check "
            f"{summary['final_state_checks_ok']}/{total}"
        )
    print(
        f"\nBoard: {correct}/{total} correct ({summary['board_accuracy'] * 100:.2f}%), host reference "
        f"{meta['host_integer_accuracy'] * 100:.2f}%, agreement {agree}/{total}; cycles per inference "
        f"{summary['cycles_min']}..{summary['cycles_max']} (mean {mean_cycles:.0f}, {summary['us_mean']:.1f} us). "
        f"Written to {args.out}"
    )


if __name__ == "__main__":
    main()
