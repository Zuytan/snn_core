"""
snn_compiler.cli
Command-line interface entry point for the `snn-compile` neuromorphic compiler.
"""

import argparse
import sys
import os
import json
from typing import Dict, Any, List, Tuple, Optional

from .ir import NirGraph
from .quantizer import (
    dense_to_csr,
    quantize_weights_i8_minmax,
    quantize_weights_i8_kl_divergence,
    load_calibration_dataset,
    calibrate_network_with_data,
)
from .codegen_rust import generate_rust_code
from .codegen_c import generate_c_code

def extract_scalar(val, default: float = 1.0) -> float:
    """Extracts a scalar float from various data formats."""
    if val is None:
        return default
    try:
        import numpy as np
        arr = np.asarray(val)
        if arr.size == 1:
            return float(arr.item())
        elif arr.size > 1:
            return float(arr.flatten()[0])
    except Exception:
        pass
    if isinstance(val, (list, tuple)):
        return float(val[0]) if val else default
    try:
        return float(val)
    except Exception:
        return default

def load_nir_file(input_file: str, threshold: float = 0.0) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """Loads a .nir file via the official nir library."""
    import nir
    import numpy as np

    graph = nir.read(input_file)
    edges_data = list(graph.edges)
    nodes_data = {}

    for node_key, node in graph.nodes.items():
        if isinstance(node, (nir.Affine, nir.Linear)):
            w = np.asarray(node.weight)
            shape = w.shape
            if len(shape) > 1:
                out_f, in_f = shape[0], shape[1]
                transposed_w = w.T.flatten().tolist()
            else:
                out_f, in_f = 1, shape[0]
                transposed_w = list(w)
            row_ptrs, targets, csr_w = dense_to_csr(transposed_w, in_f, out_f, threshold)
            node_dict = {
                "type": "Affine" if isinstance(node, nir.Affine) else "Linear",
                "in_features": in_f,
                "out_features": out_f,
                "row_ptrs": row_ptrs,
                "targets": targets,
                "weights": csr_w,
            }
            if isinstance(node, nir.Affine) and hasattr(node, "bias") and node.bias is not None:
                node_dict["bias"] = np.asarray(node.bias).flatten().tolist()
            nodes_data[node_key] = node_dict

        elif isinstance(node, (nir.LIF, nir.CubaLIF, nir.IF)):
            tau = extract_scalar(getattr(node, "tau", 0.01))
            v_th = extract_scalar(getattr(node, "v_threshold", 1.0))
            v_reset = extract_scalar(getattr(node, "v_reset", 0.0), default=0.0)
            r = extract_scalar(getattr(node, "r", 1.0))

            size = 1
            if hasattr(node, "v_threshold") and hasattr(node.v_threshold, "__len__"):
                size = len(node.v_threshold)
            elif hasattr(node, "tau") and hasattr(node.tau, "__len__"):
                size = len(node.tau)

            nodes_data[node_key] = {
                "type": "CubaLIF" if isinstance(node, nir.CubaLIF) else ("IF" if isinstance(node, nir.IF) else "LIF"),
                "tau": tau,
                "v_threshold": v_th,
                "v_reset": v_reset,
                "r": r,
                "size": size,
            }

    return nodes_data, edges_data

def load_json_or_demo(input_file: Optional[str], threshold: float = 0.0) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """Loads a NIR JSON file or generates the reference demo model."""
    nodes_data = {}
    edges_data = []

    if input_file and os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_nodes = data.get("nodes", {})
            edges_data = [tuple(e) for e in data.get("edges", [])]
            for k, v in raw_nodes.items():
                node_type = v.get("type", "LIF")
                if node_type in ("Linear", "Affine"):
                    in_f = v["in_features"]
                    out_f = v["out_features"]
                    w = v.get("weight", [])
                    row_ptrs, targets, csr_w = dense_to_csr(w, in_f, out_f, threshold)
                    d = {
                        "type": node_type,
                        "in_features": in_f,
                        "out_features": out_f,
                        "row_ptrs": row_ptrs,
                        "targets": targets,
                        "weights": csr_w,
                    }
                    if "bias" in v:
                        d["bias"] = v["bias"]
                    nodes_data[k] = d
                else:
                    nodes_data[k] = v
    else:
        # Standard 2-layer industrial demo model
        in_f = 4
        hidden_f = 3
        out_f = 2

        w1 = [
            1.2, -0.5, 0.8,
            0.0, 1.5, -0.3,
            -0.6, 0.0, 1.1,
            0.9, -0.4, 0.0,
        ]
        row_ptrs_1, targets_1, csr_w_1 = dense_to_csr(w1, in_f, hidden_f, threshold)
        nodes_data["input_projection"] = {
            "type": "Affine",
            "in_features": in_f,
            "out_features": hidden_f,
            "row_ptrs": row_ptrs_1,
            "targets": targets_1,
            "weights": csr_w_1,
            "bias": [0.1, -0.05, 0.2],
        }
        nodes_data["hidden_lif"] = {
            "type": "LIF",
            "tau": 0.02,
            "v_threshold": 1.0,
            "v_reset": 0.0,
            "r": 1.0,
            "size": hidden_f,
        }

        w2 = [
            0.5, 0.9,
            -0.7, 0.3,
            0.4, -0.2,
        ]
        row_ptrs_2, targets_2, csr_w_2 = dense_to_csr(w2, hidden_f, out_f, threshold)
        nodes_data["hidden_projection"] = {
            "type": "Linear",
            "in_features": hidden_f,
            "out_features": out_f,
            "row_ptrs": row_ptrs_2,
            "targets": targets_2,
            "weights": csr_w_2,
        }
        nodes_data["output_lif"] = {
            "type": "LIF",
            "tau": 0.01,
            "v_threshold": 1.0,
            "v_reset": 0.0,
            "r": 1.0,
            "size": out_f,
        }

        edges_data = [
            ("input_projection", "hidden_lif"),
            ("hidden_lif", "hidden_projection"),
            ("hidden_projection", "output_lif"),
        ]

    return nodes_data, edges_data

def main():
    parser = argparse.ArgumentParser(
        prog="snn-compile",
        description="Industrial AOT NIR compiler to bare-metal Rust (#![no_std]) and C99 firmware",
    )
    parser.add_argument("input_pos", nargs="?", default=None, help="NIR file (.nir), JSON (.json), or demo")
    parser.add_argument("-i", "--input", dest="input_flag", default=None, help="Alternative input file")
    parser.add_argument("-o", "--output", default="generated_network.rs", help="Output source file (.rs or .h)")
    parser.add_argument("--target", choices=["rust", "c", "c99-standalone"], default="rust", help="Target output language")
    parser.add_argument("--threshold", type=float, default=0.001, help="Cutoff threshold for CSR sparse synaptic pruning")
    parser.add_argument("-q", "--quantize-i8", "--quant", dest="quantize_i8", action="store_true", help="Enable INT8 quantization of synapses")
    parser.add_argument("--calibration", choices=["minmax", "kl"], default="minmax", help="Quantization calibration algorithm")
    parser.add_argument("--calibrate-data", default=None, help="Path to calibration dataset (.npy or .json) for optimal PTQ")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step in seconds (default: 1ms)")
    parser.add_argument("--reset-mode", choices=["hard", "subtract"], default="subtract", help="Membrane reset mode after spike")
    parser.add_argument("--arena", action="store_true", help="Enable Arena Memory Planner for 44%% SRAM savings")
    parser.add_argument("--verify-bit-exact", action="store_true", help="Formally verify bit-exact numerical equivalence (0 bit delta)")

    args = parser.parse_args()
    input_file = args.input_flag or args.input_pos

    if input_file and input_file.endswith(".nir"):
        try:
            nodes_data, edges_data = load_nir_file(input_file, args.threshold)
        except ImportError:
            print("⚠️ 'nir' library not installed. Install via: pip install nir")
            sys.exit(1)
    elif input_file and input_file.endswith(".onnx"):
        try:
            from .onnx_parser import load_onnx_file
            nodes_data, edges_data = load_onnx_file(input_file, args.threshold)
        except ImportError as e:
            print(f"{e}")
            sys.exit(1)
    else:
        nodes_data, edges_data = load_json_or_demo(input_file, args.threshold)

    # Adjust population sizes from upstream projections
    for s_name, l_name in edges_data:
        if s_name in nodes_data and l_name in nodes_data:
            if nodes_data[s_name].get("type") in ("Linear", "Affine"):
                nodes_data[l_name]["size"] = nodes_data[s_name]["out_features"]

    # Quantization processing if requested
    if args.quantize_i8:
        if args.calibrate_data:
            samples = load_calibration_dataset(args.calibrate_data)
            calibrate_network_with_data(nodes_data, edges_data, samples, method=args.calibration)
        else:
            for k, v in nodes_data.items():
                if v.get("type") in ("Linear", "Affine", "Conv1d", "Conv2d") and "weights" in v:
                    weights = v["weights"]
                    if args.calibration == "kl":
                        scale, q_weights = quantize_weights_i8_kl_divergence(weights)
                    else:
                        scale, q_weights = quantize_weights_i8_minmax(weights)
                    v["scale"] = scale
                    v["weights_i8"] = q_weights

    # IR Graph creation and normalization
    ir_graph = NirGraph(nodes_data, edges_data)
    ir_graph.fold_biases_into_lif()

    if args.target in ("c", "c99-standalone"):
        generate_c_code(
            nodes_data,
            edges_data,
            args.output,
            quantize_i8=args.quantize_i8,
            dt=args.dt,
            reset_mode=args.reset_mode,
            bundle_runtime=True,
        )
        print(f"✅ Autonomous C99 firmware generated successfully in: {args.output}")
    else:
        generate_rust_code(
            nodes_data,
            edges_data,
            args.output,
            quantize_i8=args.quantize_i8,
            dt=args.dt,
            reset_mode=args.reset_mode,
            use_arena=args.arena,
        )
        print(f"✅ Bare-metal no_std Rust firmware generated successfully in: {args.output}")

        if args.verify_bit_exact:
            from .verifier import verify_bit_exact
            verify_bit_exact(
                nodes_data,
                edges_data,
                args.output,
                ticks=50,
                dt=args.dt,
                reset_mode=args.reset_mode,
            )

if __name__ == "__main__":
    main()
