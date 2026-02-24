"""
wgpu_ml: GPU deep learning library for Adreno GPUs via wgpu-py.

Provides a PyTorch-like API for tensor operations, automatic differentiation,
and neural network modules running on Qualcomm Adreno GPUs through
Vulkan/D3D12 compute shaders.

Modules:
    wgpu_tensor  - GPU tensor class with WGSL compute shaders
    wgpu_autograd - Reverse-mode automatic differentiation
    wgpu_nn      - Neural network modules (Linear, LayerNorm, Attention, etc.)
"""

from wgpu_ml.wgpu_tensor import (
    WgpuTensor,
    add, mul, sub, matmul,
    gelu, relu, sigmoid, tanh_act,
    softmax, layer_norm,
    sum_reduce, max_reduce, mean_reduce,
    cross_entropy, focal_bce,
    scalar_mul, neg, transpose_2d,
    embedding_lookup,
)

from wgpu_ml.wgpu_autograd import (
    GradNode, WgpuParameter,
    backward, zero_grad, get_parameters,
)

from wgpu_ml.wgpu_nn import (
    Module, Linear, LayerNorm, Embedding,
    MultiHeadAttention, TransformerLayer,
    Sequential, ModuleList, AdamW,
)

__all__ = [
    # Tensor
    "WgpuTensor",
    "add", "mul", "sub", "matmul",
    "gelu", "relu", "sigmoid", "tanh_act",
    "softmax", "layer_norm",
    "sum_reduce", "max_reduce", "mean_reduce",
    "cross_entropy", "focal_bce",
    "scalar_mul", "neg", "transpose_2d",
    "embedding_lookup",
    # Autograd
    "GradNode", "WgpuParameter",
    "backward", "zero_grad", "get_parameters",
    # NN Modules
    "Module", "Linear", "LayerNorm", "Embedding",
    "MultiHeadAttention", "TransformerLayer",
    "Sequential", "ModuleList", "AdamW",
]
