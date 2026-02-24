"""Hybrid wgpu forward + PyTorch backward pass for transformer training.

Demonstrates a pattern for training transformer models on GPUs with high
per-dispatch overhead (e.g., Adreno X1-85 via Mesa Dozen on WSL2):

- Forward pass: wgpu compute shaders on GPU (few large dispatches)
- Backward pass: PyTorch autograd on CPU (fused kernels, multi-threaded BLAS)

This avoids the ~25-60ms per-dispatch overhead that makes GPU backward passes
(dozens of small dispatches) slower than optimized CPU backward passes.

Key files:
    - wgpu_transformer.py: GPU-accelerated transformer with hybrid backward
    - torch_transformer.py: PyTorch mirror model for autograd
    - (uses wgpu_ml.wgpu_tensor for GPU tensor operations)

Usage:
    from wgpu_ml.hybrid_backward.wgpu_transformer import TransformerWGPU

    model = TransformerWGPU(d_model=128, n_layers=6, vocab_size=256)
    model.init_torch_backward()

    # Forward on GPU, backward via PyTorch autograd on CPU
    loss = model.pytorch_backward(token_ids, mask_pos, labels, cache)

See docs/HYBRID_BACKWARD_PASS.md for architecture details and benchmarks.
"""
