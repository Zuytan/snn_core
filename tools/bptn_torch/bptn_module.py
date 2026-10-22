"""
bptn_module.py

PyTorch module for differentiable training of Bit-Parallel Temporal Neurons (BPTN).
Implements the continuous mathematical formulation from Section 7 of `docs/formal_specification_bptn.md`.
"""

from typing import Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class FastSigmoidSurrogate(torch.autograd.Function):
        """
        Fast-Sigmoid surrogate derivative for the Heaviside step operator:
        f(z) = 1_{z >= 0}
        f'(z) = gamma / (2 * (1 + gamma * |z|)^2)
        """
        @staticmethod
        def forward(ctx, input_tensor, gamma=10.0):
            ctx.save_for_backward(input_tensor)
            ctx.gamma = gamma
            return (input_tensor >= 0.0).to(input_tensor.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (input_tensor,) = ctx.saved_tensors
            gamma = ctx.gamma
            grad_input = grad_output * (gamma / (2.0 * (1.0 + gamma * torch.abs(input_tensor)).pow(2)))
            return grad_input, None

    class BPTNLayer(nn.Module):
        """
        Differentiable BPTN Feedforward Layer over a temporal window of W=64 time steps.
        
        Input: Tensor of shape [Batch, in_channels, 64] with values in [0, 1].
        Output: Tensor of shape [Batch, out_neurons] with values in {0, 1} (or [0, 1]).
        """
        def __init__(
            self,
            in_channels: int,
            out_neurons: int,
            surrogate_gamma: float = 10.0,
            use_ste: bool = True,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_neurons = out_neurons
            self.surrogate_gamma = surrogate_gamma
            self.use_ste = use_ste

            # Unconstrained continuous latent parameters
            # Shape: [out_neurons, in_channels, 64]
            self.raw_kernel = nn.Parameter(torch.randn(out_neurons, in_channels, 64) * 0.1)
            self.raw_polarity = nn.Parameter(torch.randn(out_neurons, in_channels, 64) * 0.1)
            
            # Latent continuous threshold
            self.raw_threshold = nn.Parameter(torch.ones(out_neurons) * 2.0)

        @property
        def kernel(self) -> torch.Tensor:
            """Continuous synaptic support mask k in [0, 1]."""
            k = torch.sigmoid(self.raw_kernel)
            if self.use_ste and not self.training:
                return (k >= 0.5).to(k.dtype)
            elif self.use_ste and self.training:
                # Straight-Through Estimator: binarized in forward, continuous gradient in backward
                k_bin = (k >= 0.5).to(k.dtype)
                return k_bin.detach() - k.detach() + k
            return k

        @property
        def polarity(self) -> torch.Tensor:
            """Continuous logic polarity mask p in [0, 1]."""
            p = torch.sigmoid(self.raw_polarity)
            if self.use_ste and not self.training:
                return (p >= 0.5).to(p.dtype)
            elif self.use_ste and self.training:
                p_bin = (p >= 0.5).to(p.dtype)
                return p_bin.detach() - p.detach() + p
            return p

        @property
        def threshold(self) -> torch.Tensor:
            """Continuous positive threshold."""
            return F.softplus(self.raw_threshold)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Continuous multilinear forward pass (Section 7, Definition 7.1).
            
            x: [Batch, in_channels, 64]
            Returns: (spikes [Batch, out_neurons], potential [Batch, out_neurons])
            """
            batch_size = x.shape[0]
            k = self.kernel          # [out_neurons, in_channels, 64]
            p = self.polarity        # [out_neurons, in_channels, 64]
            th = self.threshold      # [out_neurons]

            # 1. Continuous XOR polarity evaluation: u = x(1 - p) + (1 - x)p
            # x_exp: [Batch, 1, in_channels, 64]
            # p_exp: [1, out_neurons, in_channels, 64]
            x_exp = x.unsqueeze(1)
            p_exp = p.unsqueeze(0)
            u = x_exp * (1.0 - p_exp) + (1.0 - x_exp) * p_exp

            # 2. Continuous Hamming correlation: V = sum_{j, tau} u_{j, tau} * k_{j, tau}
            # k_exp: [1, out_neurons, in_channels, 64]
            k_exp = k.unsqueeze(0)
            active_contributions = u * k_exp  # [Batch, out_neurons, in_channels, 64]
            
            # Sum across all synaptic channels and all temporal steps
            v_potential = active_contributions.sum(dim=(2, 3))  # [Batch, out_neurons]

            # 3. Thresholded spike emission via surrogate gradient
            v_diff = v_potential - th.unsqueeze(0)  # [Batch, out_neurons]
            spikes = FastSigmoidSurrogate.apply(v_diff, self.surrogate_gamma)

            return spikes, v_potential

        def binarization_loss(self) -> torch.Tensor:
            """
            Concave entropic binarization penalty R_bin(k, p) (Theorem 7.2):
            Vanishes strictly if and only if k and p belong to {0, 1}.
            """
            k = torch.sigmoid(self.raw_kernel)
            p = torch.sigmoid(self.raw_polarity)
            loss_k = (k * (1.0 - k)).sum()
            loss_p = (p * (1.0 - p)).sum()
            return loss_k + loss_p

        def crystallize_to_rust(self, name_prefix: str) -> str:
            """
            Exports the layer directly to static Rust u64 code.
            """
            from .crystallize import crystallize_to_rust_struct
            k_np = (torch.sigmoid(self.raw_kernel) >= 0.5).cpu().detach().numpy().astype(np.uint8)
            p_np = (torch.sigmoid(self.raw_polarity) >= 0.5).cpu().detach().numpy().astype(np.uint8)
            th_np = self.threshold.cpu().detach().numpy()
            return crystallize_to_rust_struct(name_prefix, k_np, p_np, th_np)

else:
    # Stubs if PyTorch is not installed in the environment
    class BPTNLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed in this environment. Use `tools/bptn_torch/bptn_numpy.py` or install torch via `pip install torch`.")
