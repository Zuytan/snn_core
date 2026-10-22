"""
bptn_torch: Differentiable Training and Crystallization for Bit-Parallel Temporal Neurons (BPTN).

Implements the continuous multilinear relaxation and surrogate gradients defined in
Section 7 of `docs/formal_specification_bptn.md`.
"""

from .crystallize import (
    crystallize_to_rust_struct,
    crystallize_to_rust_constants,
    export_rust_header,
)

__all__ = [
    "crystallize_to_rust_struct",
    "crystallize_to_rust_constants",
    "export_rust_header",
]
