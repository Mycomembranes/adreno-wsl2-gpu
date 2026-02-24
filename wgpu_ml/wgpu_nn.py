"""
Neural network module library for wgpu GPU backend.

Provides building blocks (Linear, LayerNorm, Embedding, MultiHeadAttention, etc.)
for transformer models, with automatic differentiation support.
"""

import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any

from wgpu_ml.wgpu_tensor import WgpuTensor, _get_device
from wgpu_ml.wgpu_autograd import (
    GradNode, WgpuParameter, backward,
    matmul as autograd_matmul, add as autograd_add, gelu as autograd_gelu,
    softmax as autograd_softmax, layer_norm as autograd_layer_norm,
    embedding_lookup as autograd_embedding_lookup, reshape as autograd_reshape,
    transpose as autograd_transpose, cat as autograd_cat, split as autograd_split,
)


# ============================================================================
# Base Module Class
# ============================================================================

class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._parameters: Dict[str, WgpuParameter] = {}
        self._modules: Dict[str, "Module"] = {}
        self.training = True

    def __setattr__(self, name: str, value: Any) -> None:
        """Register parameters and modules."""
        if isinstance(value, WgpuParameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                super().__setattr__("_modules", {})
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def parameters(self):
        """Yield all parameters recursively."""
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix: str = ""):
        """Yield (name, param) tuples recursively."""
        for name, p in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, p
        for name, m in self._modules.items():
            subprefix = f"{prefix}.{name}" if prefix else name
            yield from m.named_parameters(prefix=subprefix)

    def train(self, mode: bool = True) -> "Module":
        """Set training mode."""
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        """Set eval mode."""
        return self.train(False)

    def __call__(self, *args, **kwargs):
        """Forward pass via __call__."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Override this in subclasses."""
        raise NotImplementedError


# ============================================================================
# Linear Layer
# ============================================================================

class Linear(Module):
    """Linear transformation: y = x @ W.T + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier uniform initialization
        limit = math.sqrt(6.0 / (in_features + out_features))
        weight_data = np.random.uniform(
            -limit, limit, size=(out_features, in_features)
        ).astype(np.float32)
        self.weight = WgpuParameter(WgpuTensor.from_numpy(weight_data))

        if bias:
            bias_data = np.zeros(out_features, dtype=np.float32)
            self.bias = WgpuParameter(WgpuTensor.from_numpy(bias_data))
        else:
            self.bias = None

    def forward(self, x: WgpuTensor) -> WgpuTensor:
        """
        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)
        """
        # x @ weight.T
        weight_T = autograd_transpose(self.weight, axes=(-2, -1))
        out = autograd_matmul(x, weight_T)

        if self.bias is not None:
            out = autograd_add(out, self.bias)

        return out


# ============================================================================
# Layer Normalization
# ============================================================================

class LayerNorm(Module):
    """Layer normalization with learnable scale and bias."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # gamma (scale) initialized to 1
        gamma_data = np.ones(normalized_shape, dtype=np.float32)
        self.gamma = WgpuParameter(WgpuTensor.from_numpy(gamma_data))

        # beta (shift) initialized to 0
        beta_data = np.zeros(normalized_shape, dtype=np.float32)
        self.beta = WgpuParameter(WgpuTensor.from_numpy(beta_data))

    def forward(self, x: WgpuTensor) -> WgpuTensor:
        """
        Normalize over the last dimension.
        Args:
            x: (..., normalized_shape)
        Returns:
            (..., normalized_shape)
        """
        return autograd_layer_norm(
            x, self.gamma, self.beta, eps=self.eps, axis=-1
        )


# ============================================================================
# Embedding Layer
# ============================================================================

class Embedding(Module):
    """Token embedding lookup table."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize from normal distribution
        weight_data = np.random.normal(
            0, 1, size=(num_embeddings, embedding_dim)
        ).astype(np.float32)
        self.weight = WgpuParameter(WgpuTensor.from_numpy(weight_data))

    def forward(self, indices: WgpuTensor) -> WgpuTensor:
        """
        Look up embeddings.
        Args:
            indices: (...,) with values in [0, num_embeddings)
        Returns:
            (..., embedding_dim)
        """
        return autograd_embedding_lookup(indices, self.weight)


# ============================================================================
# Multi-Head Attention
# ============================================================================

class MultiHeadAttention(Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections for Q, K, V, output
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

        # dropout is not used during wgpu training (simplification)
        self.dropout = dropout

    def forward(
        self,
        query: WgpuTensor,
        key: Optional[WgpuTensor] = None,
        value: Optional[WgpuTensor] = None,
        mask: Optional[WgpuTensor] = None,
        return_attention: bool = False,
    ) -> Tuple[WgpuTensor, Optional[WgpuTensor]]:
        """
        Args:
            query: (B, L_q, d_model)
            key: (B, L_k, d_model) or None (use query)
            value: (B, L_v, d_model) or None (use key)
            mask: (L_q, L_k) or None
            return_attention: whether to return attention weights

        Returns:
            output: (B, L_q, d_model)
            attn_weights: (B, n_heads, L_q, L_k) or None
        """
        if key is None:
            key = query
        if value is None:
            value = key

        B = query.shape[0]
        L_q = query.shape[1]
        L_k = key.shape[1]

        # Project Q, K, V
        Q = self.W_q(query)  # (B, L_q, d_model)
        K = self.W_k(key)    # (B, L_k, d_model)
        V = self.W_v(value)  # (B, L_v, d_model)

        # Reshape for multi-head: (B, L, d_model) -> (B, L, n_heads, d_head) -> (B, n_heads, L, d_head)
        Q = autograd_reshape(Q, (B, L_q, self.n_heads, self.d_head))
        Q = autograd_transpose(Q, axes=(0, 2, 1, 3))  # (B, n_heads, L_q, d_head)

        K = autograd_reshape(K, (B, L_k, self.n_heads, self.d_head))
        K = autograd_transpose(K, axes=(0, 2, 1, 3))  # (B, n_heads, L_k, d_head)

        V = autograd_reshape(V, (B, L_k, self.n_heads, self.d_head))
        V = autograd_transpose(V, axes=(0, 2, 1, 3))  # (B, n_heads, L_v, d_head)

        # Compute attention scores: (B, n_heads, L_q, d_head) @ (B, n_heads, d_head, L_k)
        K_T = autograd_transpose(K, axes=(-2, -1))  # (B, n_heads, d_head, L_k)
        scores = autograd_matmul(Q, K_T)  # (B, n_heads, L_q, L_k)

        # Scale by sqrt(d_head)
        scale_factor = 1.0 / math.sqrt(self.d_head)
        scores = scores * scale_factor

        # Apply mask if provided (causal or padding)
        if mask is not None:
            # mask shape: (L_q, L_k) or (1, L_q, L_k)
            # Add large negative value where mask is False/0
            neg_inf = -1e9
            # Broadcast and add: scores += mask * neg_inf
            # For simplicity, assume mask is (L_q, L_k) with 1s for valid, 0s for invalid
            mask_expanded = WgpuTensor.from_numpy(
                (1 - mask.to_numpy()) * neg_inf
            )
            scores = autograd_add(scores, mask_expanded)

        # Softmax over last dimension (L_k)
        attn = autograd_softmax(scores, axis=-1)  # (B, n_heads, L_q, L_k)

        # Apply attention to values: (B, n_heads, L_q, L_k) @ (B, n_heads, L_k, d_head)
        out = autograd_matmul(attn, V)  # (B, n_heads, L_q, d_head)

        # Reshape back: (B, n_heads, L_q, d_head) -> (B, L_q, n_heads, d_head) -> (B, L_q, d_model)
        out = autograd_transpose(out, axes=(0, 2, 1, 3))  # (B, L_q, n_heads, d_head)
        out = autograd_reshape(out, (B, L_q, self.d_model))  # (B, L_q, d_model)

        # Final linear projection
        out = self.W_o(out)

        attn_weights = attn if return_attention else None
        return out, attn_weights


# ============================================================================
# Transformer Layer
# ============================================================================

class TransformerLayer(Module):
    """Single transformer layer with pre-norm residuals."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        # Layer norms
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Attention and feedforward
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ff_linear1 = Linear(d_model, d_ff)
        self.ff_linear2 = Linear(d_ff, d_model)

    def forward(
        self,
        x: WgpuTensor,
        mask: Optional[WgpuTensor] = None,
        return_attention: bool = False,
    ) -> Tuple[WgpuTensor, Optional[WgpuTensor]]:
        """
        Args:
            x: (B, L, d_model)
            mask: (L, L) causal mask or None
            return_attention: whether to return attention weights

        Returns:
            out: (B, L, d_model)
            attn_weights: (B, n_heads, L, L) or None
        """
        # Pre-norm attention
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, mask=mask, return_attention=return_attention)
        x = autograd_add(x, attn_out)

        # Pre-norm feedforward
        h = self.norm2(x)
        h = autograd_gelu(self.ff_linear1(h))
        h = self.ff_linear2(h)
        x = autograd_add(x, h)

        return x, attn_weights


# ============================================================================
# Sequential Container
# ============================================================================

class Sequential(Module):
    """Sequential container of modules."""

    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def forward(self, x: WgpuTensor) -> WgpuTensor:
        """Pass input through all modules in sequence."""
        for m in self._modules.values():
            x = m(x)
        return x


# ============================================================================
# ModuleList Container
# ============================================================================

class ModuleList(Module):
    """List container for modules."""

    def __init__(self, modules: Optional[List[Module]] = None):
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                self._modules[str(i)] = m

    def __getitem__(self, idx: int) -> Module:
        """Get module by index."""
        return self._modules[str(idx)]

    def __len__(self) -> int:
        """Number of modules."""
        return len(self._modules)

    def __iter__(self):
        """Iterate over modules."""
        for i in range(len(self)):
            yield self._modules[str(i)]


# ============================================================================
# Adam Optimizer
# ============================================================================

class AdamW:
    """AdamW optimizer with weight decay."""

    def __init__(
        self,
        parameters,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.param_groups = []
        if not isinstance(parameters, list):
            parameters = list(parameters)

        self.param_groups = [{"params": parameters}]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # State for each parameter (m and v)
        self.state = {}
        for param in parameters:
            self.state[id(param)] = {
                "step": 0,
                "m": None,  # First moment (mean)
                "v": None,  # Second moment (variance)
            }

    def zero_grad(self):
        """Zero out all parameter gradients."""
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad = None

    def step(self):
        """Perform a single optimization step."""
        device = _get_device()
        beta1, beta2 = self.betas

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                param_id = id(param)
                state = self.state[param_id]

                # Initialize moments on first step
                if state["m"] is None:
                    m_data = np.zeros_like(param.data.to_numpy())
                    v_data = np.zeros_like(param.data.to_numpy())
                    state["m"] = WgpuTensor.from_numpy(m_data)
                    state["v"] = WgpuTensor.from_numpy(v_data)

                state["step"] += 1
                step = state["step"]

                # Get gradient
                grad = param.grad.data

                # Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
                grad_np = grad.to_numpy()
                m_np = state["m"].to_numpy()
                m_np = beta1 * m_np + (1 - beta1) * grad_np
                state["m"] = WgpuTensor.from_numpy(m_np)

                # Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
                v_np = state["v"].to_numpy()
                v_np = beta2 * v_np + (1 - beta2) * (grad_np ** 2)
                state["v"] = WgpuTensor.from_numpy(v_np)

                # Bias correction
                m_hat_np = m_np / (1 - beta1 ** step)
                v_hat_np = v_np / (1 - beta2 ** step)

                # Update parameters: param = param - lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
                param_np = param.data.to_numpy()
                update = (
                    self.lr
                    * (m_hat_np / (np.sqrt(v_hat_np) + self.eps) + self.weight_decay * param_np)
                )
                param_np = param_np - update
                param.data = WgpuTensor.from_numpy(param_np)


# ============================================================================
# Weight Loading from Checkpoint
# ============================================================================

def load_from_numpy(module: Module, state_dict: Dict[str, np.ndarray]) -> None:
    """
    Load parameters from a numpy state dict (e.g., from MLX checkpoint).

    Args:
        module: The neural network module
        state_dict: Dict mapping parameter names (dot-separated) to numpy arrays
    """
    named_params = dict(module.named_parameters())

    for name, array in state_dict.items():
        if name in named_params:
            tensor = WgpuTensor.from_numpy(array.astype(np.float32))
            named_params[name].data = tensor
        else:
            print(f"Warning: {name} not found in module")


# ============================================================================
# Positional Encoding
# ============================================================================

def sinusoidal_position_encoding(max_len: int, d_model: int) -> WgpuTensor:
    """
    Generate sinusoidal positional encoding.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        (max_len, d_model) positional encoding tensor
    """
    # Compute on CPU (small, one-time)
    position = np.arange(max_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
    )

    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    # Upload to GPU
    return WgpuTensor.from_numpy(pe)


# ============================================================================
# Utility: Create Causal Mask
# ============================================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (lower triangular) mask for decoder self-attention.

    Args:
        seq_len: Sequence length

    Returns:
        (seq_len, seq_len) mask with 1s in valid positions (lower triangular)
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return mask


# ============================================================================
# Test and Example Usage
# ============================================================================

if __name__ == "__main__":
    print("wgpu_nn module loaded successfully.")

    # Example: Create a simple transformer layer
    d_model = 128
    n_heads = 4
    d_ff = 512

    layer = TransformerLayer(d_model, n_heads, d_ff)
    print(f"TransformerLayer created with {d_model} dims, {n_heads} heads, {d_ff} ff dims")

    # Count parameters
    total_params = sum(p.data.size for p in layer.parameters())
    print(f"Total parameters: {total_params}")

    # Example: Positional encoding
    pe = sinusoidal_position_encoding(max_len=512, d_model=d_model)
    print(f"Positional encoding shape: {pe.shape}")
