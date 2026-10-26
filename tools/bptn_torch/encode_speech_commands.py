#!/usr/bin/env python3
"""
encode_speech_commands.py

Encodes Google Speech Commands v2 (35 words) into the binary 64 x 64 input used by the SHD
comparison scripts, so that every model (forest of LUT trees, int8 CNN, gradient boosting) sees the
same bits:

- each clip is padded or cut to 1 s at 16 kHz;
- 64 frames of 30 ms (Hann window, 512-point FFT, hop 246 samples) and 64 mel bands from 20 Hz to
  7.6 kHz, natural log of the band energy;
- binarization per utterance: a cell is 1 if its log energy is in the top 25 % of that utterance
  (density 25 %, close to the 26.5 % of the SHD encoding).

Split: the official `testing_list.txt` is the test set; the official training and validation clips
together form the training set (the comparison scripts take their own 10 % validation from it, as
for SHD). Output: an .npz with x_train, y_train, x_test, y_test (x is [N, bands, frames], uint8) and
the class names.

Usage:
    python3 tools/bptn_torch/encode_speech_commands.py --root ~/.cache/snn_core/gsc \\
        --out ~/.cache/snn_core/gsc/gsc_encoded_b64_f64.npz
"""

import argparse
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RATE = 16_000
BANDS = 64
FRAMES = 64
WIN = 480
NFFT = 512
HOP = (RATE - WIN) // (FRAMES - 1)  # 246
TOP_FRACTION = 0.25


def mel_filterbank():
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(20.0), hz_to_mel(7600.0), BANDS + 2)
    bins = np.floor((NFFT + 1) * mel_to_hz(mels) / RATE).astype(int)
    fb = np.zeros((BANDS, NFFT // 2 + 1))
    for b in range(BANDS):
        lo, mid, hi = bins[b], bins[b + 1], bins[b + 2]
        for k in range(lo, mid):
            fb[b, k] = (k - lo) / max(mid - lo, 1)
        for k in range(mid, hi):
            fb[b, k] = (hi - k) / max(hi - mid, 1)
    return fb


FILTERBANK = mel_filterbank()
WINDOW = np.hanning(WIN)


def encode(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == RATE and w.getsampwidth() == 2 and w.getnchannels() == 1, path
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    audio = np.pad(audio, (0, max(0, RATE - len(audio))))[:RATE]
    frames = np.stack([audio[i * HOP : i * HOP + WIN] * WINDOW for i in range(FRAMES)])
    power = np.abs(np.fft.rfft(frames, NFFT)) ** 2  # [frames, bins]
    logmel = np.log(power @ FILTERBANK.T + 1e-10).T  # [bands, frames]
    threshold = np.quantile(logmel, 1.0 - TOP_FRACTION)
    return (logmel > threshold).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="extracted speech_commands_v0.02 directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = args.root.expanduser()
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    test_list = set((root / "testing_list.txt").read_text().split())
    files = sorted(str(p.relative_to(root)) for c in classes for p in (root / c).glob("*.wav"))
    print(f"{len(classes)} classes, {len(files)} clips ({len(test_list)} in the official test list)")

    with ProcessPoolExecutor(args.workers) as pool:
        x = np.stack(list(pool.map(encode, (root / f for f in files), chunksize=256)))
    y = np.array([classes.index(f.split("/")[0]) for f in files], dtype=np.int64)
    is_test = np.array([f in test_list for f in files])
    np.savez_compressed(
        args.out.expanduser(), x_train=x[~is_test], y_train=y[~is_test], x_test=x[is_test], y_test=y[is_test],
        classes=np.array(classes),
    )
    print(f"train {int((~is_test).sum())}, test {int(is_test.sum())}, density {x.mean() * 100:.1f}%; written to {args.out}")


if __name__ == "__main__":
    main()
