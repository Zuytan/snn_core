//! 🌐 NIR (Neuromorphic Intermediate Representation) Interoperability Module in pure `no_std`.
//!
//! Defines standardized NIR graph descriptors enabling direct deployment in `snn_core`
//! of SNN models trained in `snnTorch`, `SpikingJelly`, `Norse`, or `Sinabs`.

use crate::domain::models::cuba_lif::CubaLifConfig;
use crate::domain::models::lif::LifConfig;
use crate::domain::numeric::Fix32;
use crate::domain::static_csr::StaticCsrSynapses;

/// Computes exp(-x) in pure fixed-point `Fix32` without FPU via Padé approximant (2,2):
/// $$R_{2,2}(x) = \frac{12 - 6x + x^2}{12 + 6x + x^2}$$
///
/// Reduces discretization error from >5% (Euler approximation 1 - dt/tau) to <0.02%
/// compared to PyTorch neuromorphic simulators (snnTorch, Norse).
#[inline]
pub fn pade_exp_neg(x: Fix32) -> Fix32 {
    if x <= Fix32::ZERO {
        return Fix32::ONE;
    }
    if x >= Fix32::from_num(10.0) {
        return Fix32::ZERO;
    }

    let twelve = Fix32::from_num(12.0);
    let six = Fix32::from_num(6.0);
    let six_x = six * x;
    let x_sq = x * x;

    let num = (twelve - six_x) + x_sq;
    let den = (twelve + six_x) + x_sq;

    if num <= Fix32::ZERO {
        Fix32::ZERO
    } else {
        num.checked_div(den).unwrap_or(Fix32::ZERO)
    }
}

/// LIF neuron node descriptor conforming to the NIR standard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NirLifNode {
    pub tau_mem: Fix32,
    pub v_threshold: Fix32,
    pub v_reset: Fix32,
    pub r: Fix32,
}

impl NirLifNode {
    /// Converts the NIR node into `LifConfig` for `snn_core`.
    /// `dt` is the discretization time step (e.g. 1 ms).
    ///
    /// Takes into account membrane resistance `r` by adapting the effective threshold
    /// $V_{\text{th}} / R$ and reset potential $V_{\text{reset}} / R$
    /// according to the official NIR continuous equation: $\tau \dot{v} = -v + R I_{\text{in}}$.
    pub fn to_lif_config(&self, dt: Fix32) -> LifConfig {
        let decay = if self.tau_mem > Fix32::ZERO {
            pade_exp_neg(dt / self.tau_mem)
        } else {
            Fix32::ZERO
        };

        let effective_threshold = if self.r > Fix32::ZERO && self.r != Fix32::ONE {
            self.v_threshold / self.r
        } else {
            self.v_threshold
        };

        let effective_reset = if self.r > Fix32::ZERO && self.r != Fix32::ONE {
            self.v_reset / self.r
        } else {
            self.v_reset
        };

        LifConfig::new(effective_threshold, decay, effective_reset, 1)
    }
}

/// CUBA-LIF neuron node descriptor conforming to the NIR standard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NirCubaLifNode {
    pub tau_mem: Fix32,
    pub tau_syn: Fix32,
    pub v_threshold: Fix32,
    pub v_reset: Fix32,
    pub r: Fix32,
}

impl NirCubaLifNode {
    /// Converts the NIR node into `CubaLifConfig` for `snn_core`.
    pub fn to_cuba_lif_config(&self, dt: Fix32) -> CubaLifConfig {
        let decay_mem = if self.tau_mem > Fix32::ZERO {
            pade_exp_neg(dt / self.tau_mem)
        } else {
            Fix32::ZERO
        };

        let decay_syn = if self.tau_syn > Fix32::ZERO {
            pade_exp_neg(dt / self.tau_syn)
        } else {
            Fix32::ZERO
        };

        let effective_threshold = if self.r > Fix32::ZERO && self.r != Fix32::ONE {
            self.v_threshold / self.r
        } else {
            self.v_threshold
        };

        let effective_reset = if self.r > Fix32::ZERO && self.r != Fix32::ONE {
            self.v_reset / self.r
        } else {
            self.v_reset
        };

        CubaLifConfig::new(
            effective_threshold,
            decay_mem,
            decay_syn,
            effective_reset,
            1,
        )
    }
}

/// Dense or sparse linear projection descriptor conforming to the NIR standard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NirLinearNode<'a> {
    /// Weights in row-major format (length in_features * out_features).
    pub weights: &'a [Fix32],
    pub in_features: usize,
    pub out_features: usize,
}

impl<'a> NirLinearNode<'a> {
    pub const fn new(weights: &'a [Fix32], in_features: usize, out_features: usize) -> Self {
        Self {
            weights,
            in_features,
            out_features,
        }
    }

    /// Converts the linear NIR matrix to a static sparse CSR synaptic matrix `StaticCsrSynapses`.
    /// Only weights with |w| >= min_weight_threshold are preserved to maximize sparsity.
    #[allow(clippy::needless_range_loop)]
    pub fn to_static_csr<const N_INPUTS_PLUS_1: usize, const MAX_SYNAPSES: usize>(
        &self,
        min_weight_threshold: Fix32,
    ) -> StaticCsrSynapses<N_INPUTS_PLUS_1, MAX_SYNAPSES> {
        let mut row_ptrs = [0u32; N_INPUTS_PLUS_1];
        let mut targets = [0u32; MAX_SYNAPSES];
        let mut weights = [Fix32::ZERO; MAX_SYNAPSES];

        let mut syn_count = 0;
        let num_inputs = self.in_features.min(N_INPUTS_PLUS_1.saturating_sub(1));

        for i in 0..num_inputs {
            row_ptrs[i] = syn_count as u32;
            let row_offset = i * self.out_features;
            for j in 0..self.out_features {
                if row_offset + j < self.weights.len() {
                    let w = self.weights[row_offset + j];
                    if w.abs() >= min_weight_threshold && syn_count < MAX_SYNAPSES {
                        targets[syn_count] = j as u32;
                        weights[syn_count] = w;
                        syn_count += 1;
                    }
                }
            }
        }
        for i in num_inputs..N_INPUTS_PLUS_1 {
            row_ptrs[i] = syn_count as u32;
        }

        StaticCsrSynapses {
            row_ptrs,
            targets,
            weights,
            total_synapses: syn_count,
        }
    }

    /// Quantizes weights to signed `i8` integers with a scaling factor `scale`
    /// using nearest-integer rounding to minimize quantization noise.
    #[allow(clippy::needless_range_loop)]
    pub fn quantize_i8(&self, scale: Fix32, out_weights: &mut [i8]) -> usize {
        let n = self.weights.len().min(out_weights.len());
        for i in 0..n {
            let scaled = self.weights[i] * scale;
            let val = ((scaled.0 + (1 << 15)) >> 16).clamp(i8::MIN as i32, i8::MAX as i32) as i8;
            out_weights[i] = val;
        }
        n
    }
}

/// Affine layer descriptor conforming to NIR (Linear projection + Bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NirAffineNode<'a> {
    pub linear: NirLinearNode<'a>,
    pub bias: &'a [Fix32],
}

impl<'a> NirAffineNode<'a> {
    pub const fn new(linear: NirLinearNode<'a>, bias: &'a [Fix32]) -> Self {
        Self { linear, bias }
    }

    /// Converts the underlying linear matrix to a static sparse CSR synaptic matrix.
    pub fn to_static_csr<const N_INPUTS_PLUS_1: usize, const MAX_SYNAPSES: usize>(
        &self,
        min_weight_threshold: Fix32,
    ) -> StaticCsrSynapses<N_INPUTS_PLUS_1, MAX_SYNAPSES> {
        self.linear.to_static_csr(min_weight_threshold)
    }

    /// Returns a reference to the static bias vector.
    pub const fn bias(&self) -> &'a [Fix32] {
        self.bias
    }
}

/// 1D convolution layer descriptor conforming to NIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NirConv1dNode<'a> {
    pub kernel: &'a [Fix32],
    pub stride: usize,
    pub padding: usize,
}

impl<'a> NirConv1dNode<'a> {
    pub const fn new(kernel: &'a [Fix32], stride: usize, padding: usize) -> Self {
        Self {
            kernel,
            stride,
            padding,
        }
    }

    /// Converts the NIR descriptor to a 1D convolutional synaptic matrix `Conv1dSynapses`.
    pub fn to_conv1d_synapses(
        &self,
        in_channels: usize,
        out_channels: usize,
        in_len: usize,
    ) -> crate::domain::layers::Conv1dSynapses<'a> {
        let kernel_size = self
            .kernel
            .len()
            .checked_div(in_channels * out_channels)
            .unwrap_or(0);
        crate::domain::layers::Conv1dSynapses::with_padding(
            self.kernel,
            in_channels,
            out_channels,
            kernel_size,
            self.stride,
            self.padding,
            in_len,
        )
    }
}

/// 2D convolution layer descriptor conforming to NIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NirConv2dNode<'a> {
    pub kernel: &'a [Fix32],
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
}

impl<'a> NirConv2dNode<'a> {
    pub const fn new(
        kernel: &'a [Fix32],
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Self {
        Self {
            kernel,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        }
    }

    /// Converts the NIR descriptor to a 2D convolutional synaptic matrix `Conv2dSynapses`.
    pub fn to_conv2d_synapses(
        &self,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        in_h: usize,
        in_w: usize,
    ) -> crate::domain::layers::Conv2dSynapses<'a> {
        crate::domain::layers::Conv2dSynapses::with_padding(
            self.kernel,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            in_h,
            in_w,
        )
    }
}

/// AOT (Ahead-of-Time) Rust source code generator for static CSR tables.
/// Enables a build script (`build.rs`) or host tooling to produce
/// `.rodata` directly for bare-metal (`no_std`) microcontrollers.
#[cfg(feature = "alloc")]
pub fn emit_rust_static_csr(
    prefix: &str,
    row_ptrs: &[u32],
    targets: &[u32],
    weights: &[Fix32],
) -> alloc::string::String {
    use core::fmt::Write;
    let mut out = alloc::string::String::new();
    let _ = write!(
        out,
        "pub static {prefix}_ROW_PTRS: [u32; {}] = [",
        row_ptrs.len()
    );
    for (i, p) in row_ptrs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{p}");
    }
    out.push_str("];\n");

    let _ = write!(
        out,
        "pub static {prefix}_TARGETS: [u32; {}] = [",
        targets.len()
    );
    for (i, t) in targets.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{t}");
    }
    out.push_str("];\n");

    let _ = write!(
        out,
        "pub static {prefix}_WEIGHTS: [Fix32; {}] = [",
        weights.len()
    );
    for (i, w) in weights.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "Fix32::from_bits({})", w.0);
    }
    out.push_str("];\n");

    out
}

/// AOT Rust source code generator for INT8 quantized static CSR tables with scale factor.
#[cfg(feature = "alloc")]
pub fn emit_rust_static_quantized_i8(
    prefix: &str,
    row_ptrs: &[u32],
    targets: &[u32],
    weights: &[i8],
    scale_factor: Fix32,
) -> alloc::string::String {
    use core::fmt::Write;
    let mut out = alloc::string::String::new();
    let _ = write!(
        out,
        "pub static {prefix}_ROW_PTRS: [u32; {}] = [",
        row_ptrs.len()
    );
    for (i, p) in row_ptrs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{p}");
    }
    out.push_str("];\n");

    let _ = write!(
        out,
        "pub static {prefix}_TARGETS: [u32; {}] = [",
        targets.len()
    );
    for (i, t) in targets.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{t}");
    }
    out.push_str("];\n");

    let _ = write!(
        out,
        "pub static {prefix}_WEIGHTS_I8: [i8; {}] = [",
        weights.len()
    );
    for (i, w) in weights.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{w}");
    }
    out.push_str("];\n");

    let _ = writeln!(
        out,
        "pub const {prefix}_SCALE: Fix32 = Fix32::from_bits({});",
        scale_factor.0
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pade_exp_neg_accuracy() {
        // For x = 0.1, exp(-0.1) = 0.904837418...
        let x = Fix32::from_num(0.1);
        let exp_approx = pade_exp_neg(x);
        let expected = Fix32::from_num(0.904837);
        let err = (exp_approx - expected).abs();
        assert!(
            err < Fix32::from_num(0.0001),
            "Padé error should be < 0.0001, got {:?}",
            err
        );

        // x = 0 -> 1.0
        assert_eq!(pade_exp_neg(Fix32::ZERO), Fix32::ONE);
        // x >= 10 -> 0.0
        assert_eq!(pade_exp_neg(Fix32::from_num(10.0)), Fix32::ZERO);
    }

    #[test]
    fn test_nir_lif_node_conversion_with_resistance() {
        let nir_lif = NirLifNode {
            tau_mem: Fix32::from_num(10.0), // 10 ms
            v_threshold: Fix32::from_num(2.0),
            v_reset: Fix32::ZERO,
            r: Fix32::from_num(2.0), // Resistance of 2.0 -> effective threshold normalized to 1.0!
        };

        let dt = Fix32::from_num(1.0); // 1 ms
        let lif_cfg = nir_lif.to_lif_config(dt);

        assert_eq!(lif_cfg.threshold, Fix32::ONE);
        assert_eq!(lif_cfg.reset_potential, Fix32::ZERO);
        // exp(-0.1) ~ 0.9048
        let expected_decay = Fix32::from_num(0.9048);
        assert!((lif_cfg.decay_factor - expected_decay).abs() < Fix32::from_num(0.001));
    }

    #[test]
    fn test_nir_linear_to_static_csr_and_quantize() {
        let weights = [
            Fix32::from_num(0.8),
            Fix32::from_num(0.02), // Input neuron 0
            Fix32::from_num(-0.5),
            Fix32::from_num(0.9), // Input neuron 1
        ];
        let nir_lin = NirLinearNode::new(&weights, 2, 2);

        // CSR conversion with threshold 0.05 (eliminates weight 0.02)
        let csr: StaticCsrSynapses<3, 4> = nir_lin.to_static_csr(Fix32::from_num(0.05));
        assert_eq!(csr.row_ptrs, [0, 1, 3]);
        assert_eq!(csr.targets[0], 0);
        assert_eq!(csr.weights[0], Fix32::from_num(0.8));
        assert_eq!(csr.targets[1], 0);
        assert_eq!(csr.weights[1], Fix32::from_num(-0.5));
        assert_eq!(csr.targets[2], 1);
        assert_eq!(csr.weights[2], Fix32::from_num(0.9));

        // i8 quantization
        let mut q_weights = [0i8; 4];
        nir_lin.quantize_i8(Fix32::from_num(100.0), &mut q_weights);
        assert_eq!(q_weights[0], 80);
        assert_eq!(q_weights[1], 2);
        assert_eq!(q_weights[2], -50);
        assert_eq!(q_weights[3], 90);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_nir_emit_rust_static_code() {
        let row_ptrs = [0u32, 2, 3];
        let targets = [0u32, 1, 0];
        let weights = [Fix32::ONE, Fix32::from_num(-0.5), Fix32::from_num(1.5)];

        let code = emit_rust_static_csr("LAYER1", &row_ptrs, &targets, &weights);
        assert!(code.contains("pub static LAYER1_ROW_PTRS: [u32; 3] = [0, 2, 3];"));
        assert!(code.contains("pub static LAYER1_TARGETS: [u32; 3] = [0, 1, 0];"));
        assert!(code.contains("pub static LAYER1_WEIGHTS: [Fix32; 3] = ["));

        let q_weights = [127i8, -64, 100];
        let code_i8 = emit_rust_static_quantized_i8(
            "LAYER1",
            &row_ptrs,
            &targets,
            &q_weights,
            Fix32::from_num(0.01),
        );
        assert!(code_i8.contains("pub static LAYER1_WEIGHTS_I8: [i8; 3] = [127, -64, 100];"));
        assert!(code_i8.contains("pub const LAYER1_SCALE: Fix32 ="));
    }
}
