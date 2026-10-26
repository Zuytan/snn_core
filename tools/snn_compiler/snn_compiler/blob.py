"""
snn_compiler.blob
Writer for the versioned model container (`.snnb`, format 1.0). Specification:
`docs/binary_format_spec.md`; reader: `snn_core::domain::blob::ModelBlob` (Rust).
The Rust and Python encoders must produce identical bytes (checked against tests/data/model_v1.snnb).
"""
import struct
import zlib
from dataclasses import dataclass
from typing import List, Optional, Sequence

MAGIC = b"SNNB"
VERSION_MAJOR = 1
VERSION_MINOR = 0
HEADER_LEN = 32
RECORD_LEN = 64
KIND_FIX32 = 0
KIND_INT8 = 1


@dataclass
class BlobLayer:
    in_features: int
    out_features: int
    row_ptrs: Sequence[int]
    targets: Sequence[int]
    weights: Sequence[int]            # raw Q16.16 (kind fix32) or int8 values (kind int8)
    threshold: int                    # raw Q16.16
    decay_factor: int
    reset_potential: int
    refractory_period: int = 1
    tonic_bias: int = 0
    reset_subtract: bool = True
    int8_scale: Optional[int] = None  # raw Q16.16 shared scale; None => Fix32 weights
    bias: Optional[Sequence[int]] = None  # raw Q16.16 [out_features]


def _pad4(buf: bytearray) -> None:
    while len(buf) % 4:
        buf.append(0)


def encode_model(layers: List[BlobLayer]) -> bytes:
    out = bytearray()
    out += MAGIC
    out += struct.pack("<HH", VERSION_MAJOR, VERSION_MINOR)
    out += struct.pack("<IIIII", 0, 0, len(layers), HEADER_LEN, RECORD_LEN)
    out += struct.pack("<I", 0)
    assert len(out) == HEADER_LEN
    records_at = len(out)
    out += bytes(len(layers) * RECORD_LEN)
    for i, l in enumerate(layers):
        _pad4(out)
        rp_off = len(out)
        out += struct.pack(f"<{len(l.row_ptrs)}H", *l.row_ptrs)
        _pad4(out)
        tg_off = len(out)
        out += struct.pack(f"<{len(l.targets)}H", *l.targets)
        _pad4(out)
        w_off = len(out)
        if l.int8_scale is None:
            kind, scale = KIND_FIX32, 0
            out += struct.pack(f"<{len(l.weights)}i", *l.weights)
        else:
            kind, scale = KIND_INT8, l.int8_scale
            out += struct.pack(f"<{len(l.weights)}b", *l.weights)
        _pad4(out)
        bias_off = 0
        if l.bias is not None:
            bias_off = len(out)
            out += struct.pack(f"<{len(l.bias)}i", *l.bias)
        rec = struct.pack(
            "<IIIIiiiiIIiIIIII",
            kind, l.in_features, l.out_features, len(l.targets),
            l.threshold, l.decay_factor, l.reset_potential, l.tonic_bias,
            l.refractory_period, 1 if l.reset_subtract else 0, scale,
            bias_off, rp_off, tg_off, w_off, 0,
        )
        out[records_at + i * RECORD_LEN: records_at + (i + 1) * RECORD_LEN] = rec
    _pad4(out)
    out[12:16] = struct.pack("<I", len(out) + 4)
    out += struct.pack("<I", zlib.crc32(bytes(out)) & 0xFFFFFFFF)
    return bytes(out)


def sample_model() -> bytes:
    """The 2-layer model of tests/data/model_v1.snnb (must match `sample_blob()` in the Rust tests)."""
    common = dict(threshold=65536, decay_factor=58982, reset_potential=0, refractory_period=1, reset_subtract=True)
    return encode_model([
        BlobLayer(2, 2, [0, 1, 2], [0, 1], [98304, 65536], **common),
        BlobLayer(2, 2, [0, 1, 2], [1, 0], [100, -50], int8_scale=1024, bias=[0, 0], **common),
    ])


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/data/model_v1.snnb"
    if "--check" in sys.argv:
        with open(sys.argv[1], "rb") as f:
            assert f.read() == sample_model(), "golden file differs from the Python encoder"
        print("golden file matches")
    else:
        with open(path, "wb") as f:
            f.write(sample_model())
        print(f"wrote {path}")
