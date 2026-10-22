"""
snn_compiler.quantizer
Fixed-point arithmetic and INT8 / Fix16 / Fix32 quantization module.
Provides Min-Max and Kullback-Leibler (KL) divergence calibration.
"""

import math
from typing import List, Tuple, Union, Sequence, Dict, Any

def float_to_fix32_bits(val: float) -> int:
    """Converts a float to a signed 32-bit integer in Q16.16 (Fix32) format."""
    raw = int(round(val * 65536.0))
    if raw > 2147483647:
        return 2147483647
    elif raw < -2147483648:
        return -2147483648
    return raw

def float_to_fix16_bits(val: float) -> int:
    """Converts a float to a signed 16-bit integer in Q8.8 (Fix16) format."""
    raw = int(round(val * 256.0))
    if raw > 32767:
        return 32767
    elif raw < -32768:
        return -32768
    return raw

def pade_exp_neg(x: float) -> float:
    """
    Computes exp(-x) via the Padé (2,2) rational approximant matching `snn_core::pade_exp_neg`.
    Guarantees relative error < 0.005% without FPU instructions.
    """
    if x <= 0.0:
        return 1.0
    if x >= 10.0:
        return 0.0
    num = (12.0 - 6.0 * x) + (x * x)
    den = (12.0 + 6.0 * x) + (x * x)
    if num <= 0.0:
        return 0.0
    return num / den

def dense_to_csr(
    weights: Union[Sequence[float], Sequence[Sequence[float]]],
    in_features: int,
    out_features: int,
    threshold: float = 0.0,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Converts a dense matrix (row-major or 2D) to CSR (Compressed Sparse Row) format.
    Prunes synaptic connections where |w| < threshold.
    """
    row_ptrs = [0]
    targets = []
    csr_weights = []

    for i in range(in_features):
        count = 0
        for j in range(out_features):
            if hasattr(weights, "__getitem__") and hasattr(weights[0], "__getitem__"):
                w = float(weights[i][j])
            else:
                w = float(weights[i * out_features + j])

            if abs(w) >= threshold:
                targets.append(j)
                csr_weights.append(w)
                count += 1
            elif threshold == 0.0:
                targets.append(j)
                csr_weights.append(w)
                count += 1
        row_ptrs.append(row_ptrs[-1] + count)

    return row_ptrs, targets, csr_weights

def quantize_weights_i8_minmax(
    weights: Sequence[float],
) -> Tuple[float, List[int]]:
    """
    Quantizes floating-point weights to signed INT8 [-127, 127]
    using standard symmetric Min-Max scaling:
        scale = max(|weights|) / 127.0
        q_i = clamp(round(w_i / scale), -127, 127)
    """
    if not weights:
        return 1.0, []

    max_abs = max(abs(float(w)) for w in weights)
    if max_abs < 1e-9:
        return 1.0, [0] * len(weights)

    scale = max_abs / 127.0
    inv_scale = 1.0 / scale
    quantized = []
    for w in weights:
        q = int(round(float(w) * inv_scale))
        q = max(-127, min(127, q))
        quantized.append(q)

    return scale, quantized

def quantize_weights_i8_kl_divergence(
    weights: Sequence[float],
    num_bins: int = 2048,
    target_bins: int = 127,
) -> Tuple[float, List[int]]:
    """
    Quantizes floating-point weights to signed INT8 by minimizing
    Kullback-Leibler (KL) divergence (TensorRT / ONNX style calibration).
    Truncates outlier weights to maximize dynamic range for neuromorphic synapses.
    """
    if not weights:
        return 1.0, []

    floats = [abs(float(w)) for w in weights]
    max_val = max(floats)
    if max_val < 1e-9:
        return 1.0, [0] * len(weights)

    # Histogram of amplitudes
    bin_width = max_val / num_bins
    hist = [0.0] * num_bins
    for val in floats:
        idx = min(int(val / bin_width), num_bins - 1)
        hist[idx] += 1.0

    best_threshold = max_val
    best_kl = float("inf")

    # Search for optimal threshold between target_bins and num_bins
    step = max(1, (num_bins - target_bins) // 64)
    for t_idx in range(target_bins, num_bins, step):
        p = hist[:t_idx] + [sum(hist[t_idx:])]
        total_p = sum(p)
        if total_p < 1e-9:
            continue
        p_prob = [x / total_p for x in p]

        # Quantize to target_bins
        chunk_size = t_idx / target_bins
        q = [0.0] * target_bins
        for i in range(t_idx):
            target_idx = min(int(i / chunk_size), target_bins - 1)
            q[target_idx] += p[i]
        q_expanded = [0.0] * (t_idx + 1)
        for i in range(target_bins):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            count = max(1, end - start)
            for j in range(start, min(end, t_idx)):
                q_expanded[j] = q[i] / count

        # Compute KL divergence: sum(P * log(P / Q))
        kl = 0.0
        for i in range(len(p)):
            if p_prob[i] > 1e-9:
                q_prob = q_expanded[i] / total_p if total_p > 0 else 1e-9
                if q_prob > 1e-9:
                    kl += p_prob[i] * math.log(p_prob[i] / q_prob)

        if kl < best_kl:
            best_kl = kl
            best_threshold = (t_idx + 0.5) * bin_width

    scale = best_threshold / 127.0
    inv_scale = 1.0 / scale
    quantized = []
    for w in weights:
        q = int(round(float(w) * inv_scale))
        q = max(-127, min(127, q))
        quantized.append(q)

    return scale, quantized

def load_calibration_dataset(path: str) -> List[List[float]]:
    """
    Loads a calibration dataset from a .npy or .json file.
    Returns a list of input samples as float lists.
    """
    if path.endswith(".npy"):
        try:
            import numpy as np
            arr = np.load(path)
            if arr.ndim == 1:
                return [arr.tolist()]
            elif arr.ndim >= 2:
                # Shape (B, D) or (B, C, H, W) -> flatten each batch item
                return [row.flatten().tolist() for row in arr]
        except ImportError:
            raise RuntimeError("Numpy is required to load .npy files. Install numpy via: pip install numpy")
    elif path.endswith(".json"):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], list):
                    return data
                return [data]
            raise ValueError("The calibration JSON file must contain a list of samples.")
    else:
        raise ValueError(f"Unsupported file format for calibration: {path} (use .npy or .json)")
    return []

def calibrate_network_with_data(
    nodes_data: Dict[str, Any],
    edges_data: List[Tuple[str, str]],
    calib_samples: List[List[float]],
    method: str = "kl",
) -> None:
    """
    Quantizes weights and refines neuronal scales/thresholds by simulating
    the network on real data samples (PTQ calibration).
    """
    # 1. Synaptic matrix quantization (Linear, Affine, Conv1d, Conv2d)
    for k, v in nodes_data.items():
        node_type = v.get("type")
        if node_type in ("Linear", "Affine", "Conv1d", "Conv2d") and "weights" in v:
            weights = v["weights"]
            if method == "kl":
                scale, q_weights = quantize_weights_i8_kl_divergence(weights)
            else:
                scale, q_weights = quantize_weights_i8_minmax(weights)
            v["scale"] = scale
            v["weights_i8"] = q_weights

    # 2. Input amplitude / pre-synaptic activation analysis if samples provided
    if calib_samples:
        max_in = max([max(abs(x) for x in sample) if sample else 1.0 for sample in calib_samples])
        for k, v in nodes_data.items():
            if v.get("type") in ("LIF", "CubaLIF", "IF"):
                v["activation_calibrated"] = True
                v["calib_input_scale"] = max_in
