"""
🧠 snn_compiler
Industrial AOT compiler for NIR neuromorphic models to bare-metal Rust (#![no_std]) and C99.
"""

from .ir import NirGraph
from .quantizer import (
    float_to_fix32_bits,
    float_to_fix16_bits,
    pade_exp_neg,
    dense_to_csr,
    quantize_weights_i8_minmax,
    quantize_weights_i8_kl_divergence,
)
from .codegen_rust import generate_rust_code
from .codegen_c import generate_c_code

__version__ = "0.4.0"

def compile_nir_to_rust(
    input_file: str,
    output_path: str = "generated_network.rs",
    quantize_i8: bool = False,
    calibration: str = "minmax",
    threshold: float = 0.001,
    dt: float = 0.001,
    reset_mode: str = "subtract",
    use_arena: bool = False,
):
    """Compiles a NIR model (.nir or .json) into static bare-metal Rust code."""
    from .cli import load_nir_file, load_json_or_demo

    if input_file.endswith(".nir"):
        nodes_data, edges_data = load_nir_file(input_file, threshold)
    else:
        nodes_data, edges_data = load_json_or_demo(input_file, threshold)

    if quantize_i8:
        for v in nodes_data.values():
            if v.get("type") in ("Linear", "Affine") and "weights" in v:
                if calibration == "kl":
                    scale, q_weights = quantize_weights_i8_kl_divergence(v["weights"])
                else:
                    scale, q_weights = quantize_weights_i8_minmax(v["weights"])
                v["scale"] = scale
                v["weights_i8"] = q_weights

    ir_graph = NirGraph(nodes_data, edges_data)
    ir_graph.fold_biases_into_lif()

    generate_rust_code(
        nodes_data,
        edges_data,
        output_path,
        quantize_i8=quantize_i8,
        dt=dt,
        reset_mode=reset_mode,
        use_arena=use_arena,
    )

def compile_nir_to_c(
    input_file: str,
    output_path: str = "generated_network.h",
    threshold: float = 0.001,
    dt: float = 0.001,
    reset_mode: str = "subtract",
):
    """Compiles a NIR model (.nir or .json) into a C99 header with Flash ROM (.rodata) tables."""
    from .cli import load_nir_file, load_json_or_demo

    if input_file.endswith(".nir"):
        nodes_data, edges_data = load_nir_file(input_file, threshold)
    else:
        nodes_data, edges_data = load_json_or_demo(input_file, threshold)

    ir_graph = NirGraph(nodes_data, edges_data)
    ir_graph.fold_biases_into_lif()

    generate_c_code(
        nodes_data,
        edges_data,
        output_path,
        dt=dt,
        reset_mode=reset_mode,
    )

from .verifier import verify_bit_exact

__all__ = [
    "__version__",
    "NirGraph",
    "float_to_fix32_bits",
    "float_to_fix16_bits",
    "pade_exp_neg",
    "dense_to_csr",
    "quantize_weights_i8_minmax",
    "quantize_weights_i8_kl_divergence",
    "generate_rust_code",
    "generate_c_code",
    "compile_nir_to_rust",
    "compile_nir_to_c",
    "verify_bit_exact",
]
