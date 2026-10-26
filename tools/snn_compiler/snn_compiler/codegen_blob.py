"""
snn_compiler.codegen_blob
Emits a versioned model container (`.snnb`, see docs/binary_format_spec.md) from a feed-forward
graph of Linear/Affine -> LIF/CubaLIF/IF pairs. Convolution and pooling layers are not part of
format 1.0 and are refused with an explicit error.
"""
from typing import Any, Dict, List, Tuple

from .blob import BlobLayer, encode_model
from .quantizer import float_to_fix32_bits, pade_exp_neg


def generate_blob(
    nodes: Dict[str, Any],
    edges: List[Tuple[str, str]],
    output_path: str,
    quantize_i8: bool = False,
    dt: float = 0.001,
    reset_mode: str = "subtract",
) -> int:
    syn_nodes = {k: v for k, v in nodes.items() if v.get("type") in ("Linear", "Affine", "Conv1d", "Conv2d")}
    lif_nodes = {k: v for k, v in nodes.items() if v.get("type") in ("LIF", "CubaLIF", "IF")}
    if any(v.get("type") in ("Conv1d", "Conv2d") for v in syn_nodes.values()) or any(
        v.get("type") == "MaxPool2d" for v in nodes.values()
    ):
        raise ValueError("blob format 1.0 supports only Linear/Affine layers; use --target rust or c for convolutions")

    pairs = [(u, v) for u, v in edges if u in syn_nodes and v in lif_nodes]
    if not pairs:
        raise ValueError("no Linear -> LIF layer found: nothing to put in the blob")

    layers: List[BlobLayer] = []
    for s_name, l_name in pairs:
        s, l = syn_nodes[s_name], lif_nodes[l_name]
        tau = l.get("tau", 0.01)
        alpha = pade_exp_neg(dt / tau) if tau > 0 else 0.0
        common = dict(
            threshold=float_to_fix32_bits(l.get("v_threshold", 1.0)),
            decay_factor=float_to_fix32_bits(alpha),
            reset_potential=float_to_fix32_bits(l.get("v_reset", 0.0)),
            refractory_period=1,
            tonic_bias=0,
            reset_subtract=(reset_mode == "subtract"),
        )
        bias = [float_to_fix32_bits(b) for b in s["bias"]] if s.get("bias") else None
        weights = s.get("weights", [])
        if quantize_i8 and ("weights_i8" in s or weights):
            if "weights_i8" in s:
                w_i8, scale = list(s["weights_i8"]), s["scale"]
            else:
                max_w = max(abs(w) for w in weights) if weights else 1.0
                q = 127.0 / max_w if max_w > 0 else 1.0
                scale = 1.0 / q
                w_i8 = [max(-128, min(127, int(round(w * q)))) for w in weights]
            layers.append(BlobLayer(
                s["in_features"], s["out_features"], s["row_ptrs"], s["targets"], w_i8,
                int8_scale=float_to_fix32_bits(scale), bias=bias, **common))
        else:
            layers.append(BlobLayer(
                s["in_features"], s["out_features"], s["row_ptrs"], s["targets"],
                [float_to_fix32_bits(w) for w in weights], bias=bias, **common))

    for a, b in zip(layers, layers[1:]):
        if a.out_features != b.in_features:
            raise ValueError(f"layer chain broken: {a.out_features} outputs feed {b.in_features} inputs")
    data = encode_model(layers)
    with open(output_path, "wb") as f:
        f.write(data)
    return len(data)
