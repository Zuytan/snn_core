"""
snn_compiler.onnx_parser
ONNX model parser targeting snn_core Intermediate Representation (IR).
Enables direct ingestion of standard PyTorch/ONNX models without passing through NIR.
Supports Gemm, MatMul, Conv (1D/2D), MaxPool, BatchNormalization (fused), and LIF activations.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
from .quantizer import dense_to_csr

def load_onnx_file(input_file: str, threshold: float = 0.0) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """
    Loads computational graph from a standard .onnx file and extracts
    synaptic projections (Gemm / MatMul / Conv), pooling operations (MaxPool),
    BatchNorm fusions, and spiking neuron layers.
    """
    try:
        import onnx
        from onnx import numpy_helper
        import numpy as np
    except ImportError as e:
        raise ImportError("⚠️ 'onnx' and 'numpy' libraries are required to load ONNX models. Install via: pip install onnx numpy") from e

    model = onnx.load(input_file)
    graph = model.graph

    # Extraction of initializer tensors (weights and biases)
    initializers: Dict[str, np.ndarray] = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    nodes_data: Dict[str, Any] = {}
    edges_data: List[Tuple[str, str]] = []

    last_projection_node = None
    node_idx = 0

    for node in graph.node:
        op = node.op_type
        name = node.name or f"node_{node_idx}_{op}"
        node_idx += 1

        if op in ("Gemm", "MatMul"):
            w_name = None
            b_name = None
            for inp in node.input:
                if inp in initializers:
                    if w_name is None and initializers[inp].ndim >= 2:
                        w_name = inp
                    elif b_name is None and initializers[inp].ndim == 1:
                        b_name = inp

            if w_name is not None:
                w = initializers[w_name]
                if w.ndim == 2:
                    out_f, in_f = w.shape[0], w.shape[1]
                    transposed_w = w.T.flatten().tolist()
                else:
                    out_f, in_f = 1, w.size
                    transposed_w = list(w.flatten())

                row_ptrs, targets, csr_w = dense_to_csr(transposed_w, in_f, out_f, threshold)

                node_dict = {
                    "type": "Affine" if b_name is not None else "Linear",
                    "in_features": in_f,
                    "out_features": out_f,
                    "row_ptrs": row_ptrs,
                    "targets": targets,
                    "weights": csr_w,
                }
                if b_name is not None:
                    node_dict["bias"] = initializers[b_name].flatten().tolist()

                nodes_data[name] = node_dict

                if last_projection_node is not None:
                    edges_data.append((last_projection_node, name))
                last_projection_node = name

        elif op == "Conv":
            w_name = None
            b_name = None
            for inp in node.input:
                if inp in initializers:
                    if w_name is None and initializers[inp].ndim in (3, 4):
                        w_name = inp
                    elif b_name is None and initializers[inp].ndim == 1:
                        b_name = inp

            if w_name is not None:
                w = initializers[w_name]
                out_ch = w.shape[0]
                in_ch = w.shape[1]
                strides = [1, 1]
                pads = [0, 0, 0, 0]
                kernel_shape = list(w.shape[2:])

                for attr in node.attribute:
                    if attr.name == "strides":
                        strides = list(attr.ints)
                    elif attr.name == "pads":
                        pads = list(attr.ints)
                    elif attr.name == "kernel_shape":
                        kernel_shape = list(attr.ints)

                if len(kernel_shape) == 2:
                    kh, kw = kernel_shape[0], kernel_shape[1]
                    sh = strides[0] if len(strides) > 0 else 1
                    sw = strides[1] if len(strides) > 1 else sh
                    pad_h = pads[0] if len(pads) > 0 else 0
                    pad_w = pads[1] if len(pads) > 1 else pad_h

                    node_dict = {
                        "type": "Conv2d",
                        "in_channels": in_ch,
                        "out_channels": out_ch,
                        "kernel_h": kh,
                        "kernel_w": kw,
                        "stride_h": sh,
                        "stride_w": sw,
                        "padding_h": pad_h,
                        "padding_w": pad_w,
                        "weights": list(w.flatten()),
                    }
                    if b_name is not None:
                        node_dict["bias"] = initializers[b_name].flatten().tolist()

                    nodes_data[name] = node_dict

                    if last_projection_node is not None:
                        edges_data.append((last_projection_node, name))
                    last_projection_node = name

        elif op == "MaxPool":
            kernel_shape = [2, 2]
            strides = [2, 2]
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_shape = list(attr.ints)
                elif attr.name == "strides":
                    strides = list(attr.ints)

            kh = kernel_shape[0] if len(kernel_shape) > 0 else 2
            kw = kernel_shape[1] if len(kernel_shape) > 1 else kh
            sh = strides[0] if len(strides) > 0 else kh
            sw = strides[1] if len(strides) > 1 else sh

            node_dict = {
                "type": "MaxPool2d",
                "kernel_h": kh,
                "kernel_w": kw,
                "stride_h": sh,
                "stride_w": sw,
            }
            nodes_data[name] = node_dict

            if last_projection_node is not None:
                edges_data.append((last_projection_node, name))
            last_projection_node = name

        elif op == "BatchNormalization":
            gamma_name = node.input[1] if len(node.input) > 1 else None
            beta_name = node.input[2] if len(node.input) > 2 else None
            mean_name = node.input[3] if len(node.input) > 3 else None
            var_name = node.input[4] if len(node.input) > 4 else None
            eps = 1e-5
            for attr in node.attribute:
                if attr.name == "epsilon":
                    eps = float(attr.f)

            if last_projection_node and last_projection_node in nodes_data:
                prev = nodes_data[last_projection_node]
                if (gamma_name in initializers and beta_name in initializers
                        and mean_name in initializers and var_name in initializers):
                    gamma = initializers[gamma_name]
                    beta = initializers[beta_name]
                    mean = initializers[mean_name]
                    var = initializers[var_name]
                    scale = gamma / np.sqrt(var + eps)

                    if prev.get("type") == "Conv2d" and "weights" in prev:
                        w_arr = np.array(prev["weights"]).reshape(prev["out_channels"], -1)
                        w_arr = w_arr * scale[:, np.newaxis]
                        prev["weights"] = list(w_arr.flatten())
                        b_arr = np.array(prev.get("bias", np.zeros(prev["out_channels"])))
                        b_arr = (b_arr - mean) * scale + beta
                        prev["bias"] = list(b_arr.flatten())
                    elif prev.get("type") in ("Linear", "Affine") and "bias" in prev:
                        b_arr = np.array(prev["bias"])
                        b_arr = (b_arr - mean) * scale + beta
                        prev["bias"] = list(b_arr.flatten())

        elif op in ("Flatten", "Reshape"):
            pass

        elif op in ("Relu", "LIF", "LeakyRelu", "Spike"):
            lif_name = f"lif_{name}"
            tau = 0.010
            v_th = 1.0
            v_reset = 0.0

            for attr in node.attribute:
                if attr.name in ("tau", "decay"):
                    tau = float(attr.f)
                elif attr.name in ("v_threshold", "threshold"):
                    v_th = float(attr.f)
                elif attr.name in ("v_reset", "reset"):
                    v_reset = float(attr.f)

            size = nodes_data[last_projection_node].get("out_features", 1) if last_projection_node else 1

            nodes_data[lif_name] = {
                "type": "LIF",
                "tau": tau,
                "v_threshold": v_th,
                "v_reset": v_reset,
                "r": 1.0,
                "size": size,
            }

            if last_projection_node is not None:
                edges_data.append((last_projection_node, lif_name))
                last_projection_node = lif_name

    return nodes_data, edges_data
