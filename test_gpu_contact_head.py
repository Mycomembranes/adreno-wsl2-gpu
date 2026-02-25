"""Test GPU-accelerated contact head matches numpy reference within float32 tolerance."""

import numpy as np
import sys
import os

# Ensure we can import from this directory
sys.path.insert(0, os.path.dirname(__file__))

from wgpu_tensor import WgpuTensor, matmul, add, relu, relu_backward, sigmoid, transpose_2d


def test_forward_mlp_gpu_vs_cpu():
    """Verify GPU contact MLP forward matches numpy."""
    np.random.seed(42)
    l_a, l_b = 15, 20
    N = l_a * l_b

    # Random features and weights
    feat_flat = np.random.randn(N, 57).astype(np.float32)
    W0 = np.random.randn(128, 57).astype(np.float32) * 0.1
    b0 = np.random.randn(128).astype(np.float32) * 0.01
    W2 = np.random.randn(1, 128).astype(np.float32) * 0.1
    b2 = np.random.randn(1).astype(np.float32) * 0.01

    # --- CPU reference ---
    hidden_cpu = feat_flat @ W0.T + b0
    pre_relu_cpu = hidden_cpu.copy()
    hidden_cpu = np.maximum(0, hidden_cpu)
    logits_cpu = hidden_cpu @ W2.T + b2

    # --- GPU path ---
    feat_gpu = WgpuTensor.from_numpy(feat_flat)
    W0_gpu = WgpuTensor.from_numpy(W0.T.copy())
    b0_gpu = WgpuTensor.from_numpy(np.broadcast_to(b0, (N, 128)).copy())
    pre_relu_gpu = add(matmul(feat_gpu, W0_gpu), b0_gpu)
    pre_relu_gpu_np = pre_relu_gpu.numpy()
    hidden_gpu = relu(pre_relu_gpu)
    hidden_gpu_np = hidden_gpu.numpy()

    W2_gpu = WgpuTensor.from_numpy(W2.T.copy())
    b2_gpu = WgpuTensor.from_numpy(np.broadcast_to(b2, (N, 1)).copy())
    logits_gpu_val = add(matmul(hidden_gpu, W2_gpu), b2_gpu).numpy()

    # --- Compare ---
    assert np.allclose(pre_relu_gpu_np, pre_relu_cpu, atol=1e-4), \
        f"Pre-relu mismatch: max diff = {np.max(np.abs(pre_relu_gpu_np - pre_relu_cpu))}"
    assert np.allclose(hidden_gpu_np, hidden_cpu, atol=1e-4), \
        f"Hidden mismatch: max diff = {np.max(np.abs(hidden_gpu_np - hidden_cpu))}"
    assert np.allclose(logits_gpu_val, logits_cpu, atol=1e-4), \
        f"Logits mismatch: max diff = {np.max(np.abs(logits_gpu_val - logits_cpu))}"

    print(f"PASS forward: max logits diff = {np.max(np.abs(logits_gpu_val - logits_cpu)):.2e}")


def test_backward_mlp_gpu_vs_cpu():
    """Verify GPU contact MLP backward matches numpy."""
    np.random.seed(42)
    l_a, l_b = 15, 20
    N = l_a * l_b

    # Random data
    features = np.random.randn(N, 57).astype(np.float32)
    W0 = np.random.randn(128, 57).astype(np.float32) * 0.1
    b0 = np.random.randn(128).astype(np.float32) * 0.01
    W2 = np.random.randn(1, 128).astype(np.float32) * 0.1

    # Forward (CPU) to get caches
    pre_relu = features @ W0.T + b0
    hidden = np.maximum(0, pre_relu)
    logits = hidden @ W2.T

    # Random grad_logits (simulating BCE gradient)
    grad_logits = np.random.randn(N).astype(np.float32) * 0.01

    # --- CPU backward ---
    dW2_cpu = grad_logits[:, np.newaxis].T @ hidden
    grad_hidden_cpu = grad_logits[:, np.newaxis] @ W2
    grad_hidden_cpu *= (pre_relu > 0).astype(np.float32)
    dW0_cpu = grad_hidden_cpu.T @ features
    grad_features_cpu = grad_hidden_cpu @ W0

    # --- GPU backward ---
    grad_logits_col = grad_logits[:, np.newaxis]
    grad_logits_gpu = WgpuTensor.from_numpy(grad_logits_col)
    hidden_gpu = WgpuTensor.from_numpy(hidden)

    grad_logits_T_gpu = transpose_2d(grad_logits_gpu)
    dW2_gpu_val = matmul(grad_logits_T_gpu, hidden_gpu).numpy()

    W2_gpu = WgpuTensor.from_numpy(W2)
    grad_hidden_gpu = matmul(grad_logits_gpu, W2_gpu)

    pre_relu_gpu = WgpuTensor.from_numpy(pre_relu)
    grad_hidden_gpu = relu_backward(grad_hidden_gpu, pre_relu_gpu)
    grad_hidden_np = grad_hidden_gpu.numpy()

    features_gpu = WgpuTensor.from_numpy(features)
    grad_hidden_T_gpu = transpose_2d(grad_hidden_gpu)
    dW0_gpu_val = matmul(grad_hidden_T_gpu, features_gpu).numpy()

    W0_gpu = WgpuTensor.from_numpy(W0)
    grad_features_gpu_val = matmul(grad_hidden_gpu, W0_gpu).numpy()

    # --- Compare ---
    assert np.allclose(dW2_gpu_val, dW2_cpu, atol=1e-4), \
        f"dW2 mismatch: max diff = {np.max(np.abs(dW2_gpu_val - dW2_cpu))}"
    assert np.allclose(grad_hidden_np, grad_hidden_cpu, atol=1e-4), \
        f"grad_hidden mismatch: max diff = {np.max(np.abs(grad_hidden_np - grad_hidden_cpu))}"
    assert np.allclose(dW0_gpu_val, dW0_cpu, atol=1e-4), \
        f"dW0 mismatch: max diff = {np.max(np.abs(dW0_gpu_val - dW0_cpu))}"
    assert np.allclose(grad_features_gpu_val, grad_features_cpu, atol=1e-4), \
        f"grad_features mismatch: max diff = {np.max(np.abs(grad_features_gpu_val - grad_features_cpu))}"

    print(f"PASS backward: max grad_features diff = {np.max(np.abs(grad_features_gpu_val - grad_features_cpu)):.2e}")


def test_sigmoid_gpu_vs_cpu():
    """Verify GPU sigmoid matches numpy."""
    np.random.seed(42)
    logits = np.random.randn(300, 1).astype(np.float32) * 5.0

    cpu_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    gpu_sigmoid = sigmoid(WgpuTensor.from_numpy(logits)).numpy()

    assert np.allclose(gpu_sigmoid, cpu_sigmoid, atol=1e-5), \
        f"Sigmoid mismatch: max diff = {np.max(np.abs(gpu_sigmoid - cpu_sigmoid))}"
    print(f"PASS sigmoid: max diff = {np.max(np.abs(gpu_sigmoid - cpu_sigmoid)):.2e}")


if __name__ == "__main__":
    test_forward_mlp_gpu_vs_cpu()
    test_backward_mlp_gpu_vs_cpu()
    test_sigmoid_gpu_vs_cpu()
    print("\nAll GPU contact head tests passed!")
