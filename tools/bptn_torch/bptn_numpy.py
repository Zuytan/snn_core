"""
bptn_numpy.py

Pure and self-contained NumPy implementation of Axis 2 (Section 7).
Provides a full differentiable training engine via gradient descent (Adam)
and a compiler to Rust without requiring PyTorch installation.
"""

from typing import Tuple
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def fast_sigmoid_derivative(z: np.ndarray, gamma: float = 5.0) -> np.ndarray:
    """Fast-Sigmoid surrogate derivative: psi_gamma(z)."""
    denom = 1.0 + gamma * np.abs(z)
    return gamma / (2.0 * denom * denom)


class BptnNumpyLayer:
    """
    Differentiable BPTN layer in pure NumPy vectorized arithmetic.
    
    in_channels: number of input temporal channels (M)
    out_neurons: number of output neurons (N)
    """
    def __init__(self, in_channels: int, out_neurons: int, seed: int = 42):
        self.in_channels = in_channels
        self.out_neurons = out_neurons
        rng = np.random.RandomState(seed)

        # Unconstrained latent weights (sparse init to avoid saturation)
        self.raw_kernel = rng.normal(-1.0, 0.5, (out_neurons, in_channels, 64))
        self.raw_polarity = rng.normal(0.0, 0.5, (out_neurons, in_channels, 64))
        self.threshold = np.ones(out_neurons, dtype=np.float64) * 2.0

        # Optimizer Adam states
        self.m_k = np.zeros_like(self.raw_kernel)
        self.v_k = np.zeros_like(self.raw_kernel)
        self.m_p = np.zeros_like(self.raw_polarity)
        self.v_p = np.zeros_like(self.raw_polarity)
        self.m_th = np.zeros_like(self.threshold)
        self.v_th = np.zeros_like(self.threshold)
        self.t_adam = 0

    @property
    def kernel(self) -> np.ndarray:
        return sigmoid(self.raw_kernel)

    @property
    def polarity(self) -> np.ndarray:
        return sigmoid(self.raw_polarity)

    def forward(self, x: np.ndarray, gamma: float = 5.0) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Continuous forward pass (Section 7, Definition 7.1).
        x: [Batch, in_channels, 64] with x in [0, 1]
        """
        batch_size = x.shape[0]
        k = self.kernel      # [out_neurons, in_channels, 64]
        p = self.polarity    # [out_neurons, in_channels, 64]
        th = self.threshold  # [out_neurons]

        # 1. Continuous polarity: u = x(1 - p) + (1 - x)p
        # x_exp: [Batch, 1, in_channels, 64]
        x_exp = np.expand_dims(x, 1)
        p_exp = np.expand_dims(p, 0)
        u = x_exp * (1.0 - p_exp) + (1.0 - x_exp) * p_exp

        # 2. Continuous Hamming correlation: V = sum_{j, tau} u * k
        k_exp = np.expand_dims(k, 0)
        active = u * k_exp  # [Batch, out_neurons, in_channels, 64]
        v_potential = np.sum(active, axis=(2, 3))  # [Batch, out_neurons]

        # 3. Thresholded spike emission via smooth surrogate
        v_diff = v_potential - np.expand_dims(th, 0)  # [Batch, out_neurons]
        y_prob = sigmoid(gamma * v_diff)

        cache = {
            "x": x,
            "x_exp": x_exp,
            "u": u,
            "k": k,
            "p": p,
            "th": th,
            "v_potential": v_potential,
            "v_diff": v_diff,
            "y_prob": y_prob,
            "gamma": gamma,
        }
        return y_prob, v_potential, cache

    def backward(self, grad_y: np.ndarray, cache: dict, reg_bin_lambda: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Exact analytical gradient backpropagation.
        """
        batch_size = grad_y.shape[0]
        x_exp = cache["x_exp"]
        u = cache["u"]
        k = cache["k"]
        p = cache["p"]
        v_diff = cache["v_diff"]
        y_prob = cache["y_prob"]
        gamma = cache["gamma"]

        # 1. Gradient through threshold operator (Surrogate Gradient)
        # dy/dv_diff = gamma * y_prob * (1 - y_prob)
        d_v = grad_y * (gamma * y_prob * (1.0 - y_prob))  # [Batch, out_neurons]

        # 2. Gradient w.r.t threshold th
        grad_th = -np.sum(d_v, axis=0)  # [out_neurons]

        # 3. Gradient w.r.t potential: d_active = d_v [Batch, out_neurons, 1, 1]
        d_v_exp = np.expand_dims(np.expand_dims(d_v, 2), 3)  # [Batch, out_neurons, 1, 1]

        # d_k = sum_batch (d_v * u)
        grad_k = np.sum(d_v_exp * u, axis=0)  # [out_neurons, in_channels, 64]

        # d_u = d_v * k_exp
        d_u = d_v_exp * np.expand_dims(k, 0)  # [Batch, out_neurons, in_channels, 64]

        # d_p = sum_batch (d_u * du/dp) where du/dp = 1 - 2*x_exp
        du_dp = 1.0 - 2.0 * x_exp
        grad_p = np.sum(d_u * du_dp, axis=0)  # [out_neurons, in_channels, 64]

        # 4. Concave binarization penalty R_bin = k*(1-k) + p*(1-p)
        # d/dk R_bin = 1 - 2k (pushes toward {0, 1})
        grad_k += reg_bin_lambda * (1.0 - 2.0 * k)
        grad_p += reg_bin_lambda * (1.0 - 2.0 * p)

        # 5. Backpropagation through latent sigmoid parameters
        # dk/draw_kernel = k * (1 - k)
        grad_raw_k = grad_k * (k * (1.0 - k))
        grad_raw_p = grad_p * (p * (1.0 - p))

        return grad_raw_k, grad_raw_p, grad_th

    def step_adam(self, grad_raw_k: np.ndarray, grad_raw_p: np.ndarray, grad_th: np.ndarray, lr: float = 0.05):
        """Adam optimizer step."""
        self.t_adam += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for param, grad, m, v in [
            (self.raw_kernel, grad_raw_k, self.m_k, self.v_k),
            (self.raw_polarity, grad_raw_p, self.m_p, self.v_p),
            (self.threshold, grad_th, self.m_th, self.v_th),
        ]:
            m[:] = beta1 * m + (1.0 - beta1) * grad
            v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - beta1 ** self.t_adam)
            v_hat = v / (1.0 - beta2 ** self.t_adam)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def crystallize(self, name_prefix: str) -> str:
        """Exports the model directly to static Rust u64 code."""
        from .crystallize import crystallize_to_rust_struct
        bin_k = (self.kernel >= 0.5).astype(np.uint8)
        bin_p = (self.polarity >= 0.5).astype(np.uint8)
        return crystallize_to_rust_struct(name_prefix, bin_k, bin_p, self.threshold)
