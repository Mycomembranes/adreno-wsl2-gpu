"""GPU-accelerated transformer with hybrid wgpu + PyTorch backward pass.

This module demonstrates the hybrid training pattern:
- Forward pass: wgpu compute shaders on GPU (matmul, softmax, layer_norm, GELU)
- Backward pass: PyTorch autograd on CPU (fused kernels, multi-threaded BLAS)

The backward pass is the bottleneck in GPU training on drivers with high
per-dispatch overhead (e.g., ~25-60ms on Adreno X1-85 via Mesa Dozen).
PyTorch autograd eliminates this by running the backward entirely on CPU
with optimized fused kernels and multi-threaded BLAS.

Architecture:
    Token Embedding (vocab_size -> d_model) + Sinusoidal Position Encoding
    -> N x TransformerLayer:
        Pre-LayerNorm -> MultiHeadAttention -> Residual
        Pre-LayerNorm -> FeedForward(GELU) -> Residual
    -> Final LayerNorm
    -> MLM Head: Linear -> GELU -> LayerNorm -> Linear(vocab_size)

Usage:
    model = TransformerWGPU(d_model=128, n_layers=6, vocab_size=256)
    model.init_torch_backward()

    # Training loop
    logits, cache = model.forward_mlm(token_ids)
    loss = model.pytorch_backward(token_ids, mask_pos, labels, cache)
    optimizer.step(model.params, model.grads)
"""

import re
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Import wgpu tensor ops — falls back gracefully if wgpu not available
try:
    from wgpu_ml.wgpu_tensor import (
        WgpuTensor, matmul, softmax, layer_norm, gelu,
        embedding_lookup, add, scalar_mul,
    )
    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False
    logger.warning("wgpu_ml not available; GPU forward pass disabled")


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax on CPU."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class TransformerWGPU:
    """GPU-accelerated transformer with hybrid backward pass.

    Weights are stored as numpy arrays (self.params). The forward pass
    optionally runs on GPU via wgpu compute shaders. The backward pass
    uses PyTorch autograd for fused, multi-threaded gradient computation.

    Args:
        d_model: Model dimension (default 128)
        n_layers: Number of transformer layers (default 6)
        n_heads: Number of attention heads (default 4)
        d_ff: Feedforward hidden dimension (default 512)
        vocab_size: Vocabulary size (default 256)
        max_len: Maximum sequence length (default 1024)
        eps: LayerNorm epsilon
    """

    def __init__(self, d_model=128, n_layers=6, n_heads=4, d_ff=512,
                 vocab_size=256, max_len=1024, eps=1e-5):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.eps = eps
        self.d_head = d_model // n_heads

        self.params = {}
        self.grads = {}
        self.gpu_params = {}

        self._pos_encoding = None
        self._init_params()

    def _init_params(self):
        """Initialize all parameters with Xavier uniform."""
        # Embedding
        self.params["token_embedding.weight"] = self._xavier_uniform(
            self.vocab_size, self.d_model
        )

        # Transformer layers
        for i in range(self.n_layers):
            # LayerNorm 1
            self.params[f"layers.{i}.norm1.gamma"] = np.ones(self.d_model, dtype=np.float32)
            self.params[f"layers.{i}.norm1.beta"] = np.zeros(self.d_model, dtype=np.float32)

            # Multi-head attention
            for proj in ["W_q", "W_k", "W_v", "W_o"]:
                self.params[f"layers.{i}.attn.{proj}.weight"] = self._xavier_uniform(
                    self.d_model, self.d_model
                )
                self.params[f"layers.{i}.attn.{proj}.bias"] = np.zeros(
                    self.d_model, dtype=np.float32
                )

            # LayerNorm 2
            self.params[f"layers.{i}.norm2.gamma"] = np.ones(self.d_model, dtype=np.float32)
            self.params[f"layers.{i}.norm2.beta"] = np.zeros(self.d_model, dtype=np.float32)

            # FeedForward
            self.params[f"layers.{i}.ff.linear1.weight"] = self._xavier_uniform(
                self.d_ff, self.d_model
            )
            self.params[f"layers.{i}.ff.linear1.bias"] = np.zeros(self.d_ff, dtype=np.float32)
            self.params[f"layers.{i}.ff.linear2.weight"] = self._xavier_uniform(
                self.d_model, self.d_ff
            )
            self.params[f"layers.{i}.ff.linear2.bias"] = np.zeros(self.d_model, dtype=np.float32)

        # Final LayerNorm
        self.params["final_norm.gamma"] = np.ones(self.d_model, dtype=np.float32)
        self.params["final_norm.beta"] = np.zeros(self.d_model, dtype=np.float32)

        # MLM Head
        self.params["mlm_head.dense.weight"] = self._xavier_uniform(
            self.d_model, self.d_model
        )
        self.params["mlm_head.dense.bias"] = np.zeros(self.d_model, dtype=np.float32)
        self.params["mlm_head.norm.gamma"] = np.ones(self.d_model, dtype=np.float32)
        self.params["mlm_head.norm.beta"] = np.zeros(self.d_model, dtype=np.float32)
        self.params["mlm_head.proj.weight"] = self._xavier_uniform(
            self.vocab_size, self.d_model
        )
        self.params["mlm_head.proj.bias"] = np.zeros(self.vocab_size, dtype=np.float32)

        # Initialize gradients
        for name in self.params:
            self.grads[name] = np.zeros_like(self.params[name])

    def _xavier_uniform(self, rows: int, cols: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (rows + cols))
        return np.random.uniform(-limit, limit, (rows, cols)).astype(np.float32)

    def _sinusoidal_position_encoding(self, seq_len: int) -> np.ndarray:
        """Compute sinusoidal position encoding."""
        if self._pos_encoding is not None and self._pos_encoding.shape[0] >= seq_len:
            return self._pos_encoding[:seq_len]

        import math
        pe = np.zeros((self.max_len, self.d_model), dtype=np.float32)
        position = np.arange(self.max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self._pos_encoding = pe
        return pe[:seq_len]

    # ---- Forward Pass (CPU reference — replace with wgpu GPU ops) ----

    def forward_mlm(self, token_ids: np.ndarray) -> tuple:
        """Forward pass returning MLM logits and a cache for backward.

        Args:
            token_ids: (B, L) int32 token IDs

        Returns:
            logits: (B, L, vocab_size) MLM logits
            cache: dict with intermediate activations
        """
        B, L = token_ids.shape
        cache = {}

        # Embedding
        tok_emb = self.params["token_embedding.weight"][token_ids]  # (B, L, D)
        pos_emb = self._sinusoidal_position_encoding(L)  # (L, D)
        x = tok_emb + pos_emb
        cache["embedding_output"] = x.copy()

        # Transformer layers
        for i in range(self.n_layers):
            # Pre-norm attention
            x_norm = self._layernorm_np(x, f"layers.{i}.norm1")
            attn_out = self._attention_forward_np(x_norm, i)
            x = x + attn_out

            # Pre-norm FFN
            x_norm2 = self._layernorm_np(x, f"layers.{i}.norm2")
            ff_out = self._ffn_forward_np(x_norm2, i)
            x = x + ff_out

        # Final norm
        cache["pre_final_norm"] = x.copy()
        x = self._layernorm_np(x, "final_norm")
        cache["final_output"] = x.copy()

        # MLM head
        dense_out = x @ self.params["mlm_head.dense.weight"].T + self.params["mlm_head.dense.bias"]
        gelu_out = self._gelu_np(dense_out)
        normed = self._layernorm_np(gelu_out, "mlm_head.norm")
        logits = normed @ self.params["mlm_head.proj.weight"].T + self.params["mlm_head.proj.bias"]

        return logits, cache

    def _layernorm_np(self, x: np.ndarray, prefix: str) -> np.ndarray:
        gamma = self.params[f"{prefix}.gamma"]
        beta = self.params[f"{prefix}.beta"]
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return x_norm * gamma + beta

    def _attention_forward_np(self, x_norm: np.ndarray, layer_idx: int) -> np.ndarray:
        B, L, D = x_norm.shape
        xn = x_norm.reshape(-1, D)

        Q = (xn @ self.params[f"layers.{layer_idx}.attn.W_q.weight"].T
             + self.params[f"layers.{layer_idx}.attn.W_q.bias"])
        K = (xn @ self.params[f"layers.{layer_idx}.attn.W_k.weight"].T
             + self.params[f"layers.{layer_idx}.attn.W_k.bias"])
        V = (xn @ self.params[f"layers.{layer_idx}.attn.W_v.weight"].T
             + self.params[f"layers.{layer_idx}.attn.W_v.bias"])

        Q = Q.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        scores = np.einsum("bhqd,bhkd->bhqk", Q, K) / np.sqrt(self.d_head)
        probs = softmax_numpy(scores)
        attn_out = np.einsum("bhqk,bhkd->bhqd", probs, V)
        attn_concat = attn_out.transpose(0, 2, 1, 3).reshape(B, L, D)

        output = attn_concat.reshape(-1, D) @ self.params[f"layers.{layer_idx}.attn.W_o.weight"].T
        output += self.params[f"layers.{layer_idx}.attn.W_o.bias"]
        return output.reshape(B, L, D)

    def _ffn_forward_np(self, x_norm: np.ndarray, layer_idx: int) -> np.ndarray:
        B, L, D = x_norm.shape
        xn = x_norm.reshape(-1, D)
        hidden = xn @ self.params[f"layers.{layer_idx}.ff.linear1.weight"].T
        hidden += self.params[f"layers.{layer_idx}.ff.linear1.bias"]
        hidden = self._gelu_np(hidden)
        output = hidden @ self.params[f"layers.{layer_idx}.ff.linear2.weight"].T
        output += self.params[f"layers.{layer_idx}.ff.linear2.bias"]
        return output.reshape(B, L, D)

    def _gelu_np(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    # ---- Hybrid PyTorch Backward Pass ----

    # Key mapping: wgpu param names -> PyTorch state_dict names
    _WGPU_TO_TORCH_PATTERNS = [
        # LayerNorm gamma/beta -> weight/bias
        (re.compile(r'^(.+)\.gamma$'), r'\1.weight'),
        (re.compile(r'^(.+)\.beta$'), r'\1.bias'),
        # FFN Sequential indexing: linear1 -> ff.0, linear2 -> ff.3
        (re.compile(r'^(layers\.\d+)\.ff\.linear1\.(.+)$'), r'\1.ff.0.\2'),
        (re.compile(r'^(layers\.\d+)\.ff\.linear2\.(.+)$'), r'\1.ff.3.\2'),
    ]

    def _wgpu_key_to_torch(self, wgpu_key: str) -> str:
        """Convert a wgpu param key to PyTorch state_dict key."""
        result = wgpu_key
        for pattern, replacement in self._WGPU_TO_TORCH_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def _torch_key_to_wgpu(self, torch_key: str) -> str:
        """Convert a PyTorch state_dict key to wgpu param key."""
        result = torch_key
        result = re.sub(
            r'^((?:layers\.\d+\.norm[12]|final_norm|mlm_head\.norm))\.weight$',
            r'\1.gamma', result)
        result = re.sub(
            r'^((?:layers\.\d+\.norm[12]|final_norm|mlm_head\.norm))\.bias$',
            r'\1.beta', result)
        result = re.sub(r'^(layers\.\d+)\.ff\.0\.(.+)$', r'\1.ff.linear1.\2', result)
        result = re.sub(r'^(layers\.\d+)\.ff\.3\.(.+)$', r'\1.ff.linear2.\2', result)
        return result

    def init_torch_backward(self):
        """Initialize PyTorch model mirror for autograd backward pass.

        Creates a Transformer (from torch_transformer.py) with identical
        architecture, builds key mappings, and syncs weights.
        """
        import torch
        from .torch_transformer import Transformer

        self.torch_model = Transformer(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            max_len=self.max_len,
            dropout=0.0,
            vocab_size=self.vocab_size,
        )
        self.torch_model.train()

        # Build key mapping
        torch_keys = set(self.torch_model.state_dict().keys())
        torch_keys.discard('_pos_enc')

        self._torch_key_map = {}
        mapped_torch = set()
        for wk in self.params:
            tk = self._wgpu_key_to_torch(wk)
            if tk in torch_keys:
                self._torch_key_map[wk] = tk
                mapped_torch.add(tk)

        unmapped_torch = torch_keys - mapped_torch
        if unmapped_torch:
            logger.debug(f"Torch params without wgpu mapping: {unmapped_torch}")

        self._sync_numpy_to_torch()
        self._torch_params_dirty = False

        logger.info(f"PyTorch backward initialized: {len(self._torch_key_map)} params mapped")

    def _sync_numpy_to_torch(self):
        """Sync current numpy params into PyTorch model."""
        import torch
        state_dict = {}
        for wk, tk in self._torch_key_map.items():
            state_dict[tk] = torch.from_numpy(self.params[wk].copy())
        self.torch_model.load_state_dict(state_dict, strict=False)

    def _extract_torch_grads(self):
        """Extract gradients from PyTorch model into self.grads."""
        named_params = dict(self.torch_model.named_parameters())
        for wk, tk in self._torch_key_map.items():
            param = named_params.get(tk)
            if param is not None and param.grad is not None:
                self.grads[wk] += param.grad.detach().numpy()

    def pytorch_backward(self, token_ids, mask_pos, labels, cache):
        """Use PyTorch autograd for the backward pass.

        Args:
            token_ids: (B, L) int32 token IDs
            mask_pos: (B, L) bool/int mask positions (1 = masked)
            labels: (B, L) int32 target labels at masked positions
            cache: Forward pass cache (unused, for API compatibility)

        Returns:
            float: loss value
        """
        import torch
        import torch.nn.functional as F

        if self._torch_params_dirty:
            self._sync_numpy_to_torch()
            self._torch_params_dirty = False

        self.torch_model.zero_grad()

        t_tokens = torch.from_numpy(token_ids.astype(np.int64))
        t_labels = torch.from_numpy(labels.astype(np.int64))
        t_mask = torch.from_numpy(mask_pos.astype(bool))

        logits = self.torch_model.forward_mlm(t_tokens)

        masked_logits = logits[t_mask]
        masked_labels = t_labels[t_mask]

        if masked_logits.shape[0] > 0:
            loss = F.cross_entropy(masked_logits, masked_labels, ignore_index=-1)
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        loss.backward()
        self._extract_torch_grads()

        return loss.item()

    # ---- Utilities ----

    def load_weights(self, state_dict: Dict[str, np.ndarray]):
        """Load weights from a numpy dictionary."""
        for name, value in state_dict.items():
            if name in self.params:
                if self.params[name].shape == value.shape:
                    self.params[name] = value.astype(np.float32)
                else:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected {self.params[name].shape}, "
                        f"got {value.shape}"
                    )
        if hasattr(self, '_torch_params_dirty'):
            self._torch_params_dirty = True

    def save_weights(self, path: str):
        """Save weights to .npz file."""
        np.savez(path, **self.params)

    def count_parameters(self) -> int:
        total = 0
        for arr in self.params.values():
            total += np.prod(arr.shape)
        return total
