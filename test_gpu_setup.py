#!/usr/bin/env python3
"""Test that GPU is properly configured and compute shaders work.

Checks:
1. Vulkan adapter is real GPU (not llvmpipe CPU)
2. wgpu device creation succeeds
3. Compute shader dispatch works (matmul, relu, sigmoid)
4. GPU results match numpy reference (float32 tolerance)
5. Benchmark GPU vs CPU for contact head workloads

Usage:
    # First set env vars (see GPU_SETUP.md):
    export VK_ICD_FILENAMES=/home/mukshud/mesa-dozen-install/share/vulkan/icd.d/dzn_icd.aarch64.json
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/mukshud/mesa-dozen-install/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

    python3 test_gpu_setup.py
"""

import sys
import os
import time
import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))


def check_env_vars():
    """Check that required environment variables are set."""
    print("=" * 60)
    print("STEP 1: Environment Variables")
    print("=" * 60)

    vk_icd = os.environ.get("VK_ICD_FILENAMES", "")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    if "dzn_icd" in vk_icd:
        print(f"  VK_ICD_FILENAMES: {vk_icd}")
        print("  [OK] Dozen ICD configured")
    else:
        print("  [FAIL] VK_ICD_FILENAMES not set to dozen ICD")
        print("  Run: export VK_ICD_FILENAMES=/home/mukshud/mesa-dozen-install/share/vulkan/icd.d/dzn_icd.aarch64.json")
        return False

    if "/usr/lib/wsl/lib" in ld_path:
        print(f"  LD_LIBRARY_PATH includes WSL lib: OK")
    else:
        print("  [WARN] LD_LIBRARY_PATH missing /usr/lib/wsl/lib")

    return True


def check_gpu_adapter():
    """Check that wgpu finds a real GPU (not llvmpipe)."""
    print("\n" + "=" * 60)
    print("STEP 2: GPU Adapter Detection")
    print("=" * 60)

    import wgpu
    adapters = wgpu.gpu.enumerate_adapters_sync()

    gpu_found = False
    for i, adapter in enumerate(adapters):
        summary = adapter.summary
        is_gpu = "IntegratedGPU" in summary or "DiscreteGPU" in summary
        marker = " <<<" if is_gpu else ""
        print(f"  Adapter {i}: {summary}{marker}")
        if is_gpu:
            gpu_found = True

    if gpu_found:
        print("  [OK] Real GPU adapter found")
    else:
        print("  [FAIL] No real GPU found — only CPU/llvmpipe")
        print("  Check GPU_SETUP.md for environment variable setup")

    return gpu_found


def check_compute_shaders():
    """Test that compute shaders execute correctly on GPU."""
    print("\n" + "=" * 60)
    print("STEP 3: Compute Shader Tests")
    print("=" * 60)

    from wgpu_tensor import WgpuTensor, matmul, add, relu, relu_backward, sigmoid, transpose_2d, mul

    np.random.seed(42)
    all_pass = True

    # Test matmul
    a = np.random.randn(64, 32).astype(np.float32)
    b = np.random.randn(32, 16).astype(np.float32)
    expected = a @ b
    result = matmul(WgpuTensor.from_numpy(a), WgpuTensor.from_numpy(b)).numpy()
    ok = np.allclose(result, expected, atol=1e-4)
    print(f"  matmul(64x32, 32x16): {'PASS' if ok else 'FAIL'} (max diff: {np.max(np.abs(result - expected)):.2e})")
    all_pass &= ok

    # Test add
    c = np.random.randn(64, 16).astype(np.float32)
    expected = result + c
    gpu_result = add(WgpuTensor.from_numpy(result), WgpuTensor.from_numpy(c)).numpy()
    ok = np.allclose(gpu_result, expected, atol=1e-6)
    print(f"  add(64x16):           {'PASS' if ok else 'FAIL'} (max diff: {np.max(np.abs(gpu_result - expected)):.2e})")
    all_pass &= ok

    # Test relu
    x = np.random.randn(100, 50).astype(np.float32)
    expected = np.maximum(0, x)
    result = relu(WgpuTensor.from_numpy(x)).numpy()
    ok = np.allclose(result, expected, atol=1e-6)
    print(f"  relu(100x50):         {'PASS' if ok else 'FAIL'} (max diff: {np.max(np.abs(result - expected)):.2e})")
    all_pass &= ok

    # Test relu_backward
    grad = np.random.randn(100, 50).astype(np.float32)
    expected = grad * (x > 0).astype(np.float32)
    result = relu_backward(WgpuTensor.from_numpy(grad), WgpuTensor.from_numpy(x)).numpy()
    ok = np.allclose(result, expected, atol=1e-6)
    print(f"  relu_backward:        {'PASS' if ok else 'FAIL'} (max diff: {np.max(np.abs(result - expected)):.2e})")
    all_pass &= ok

    # Test sigmoid
    logits = np.random.randn(200, 1).astype(np.float32) * 5
    expected = 1.0 / (1.0 + np.exp(-logits))
    result = sigmoid(WgpuTensor.from_numpy(logits)).numpy()
    ok = np.allclose(result, expected, atol=1e-5)
    print(f"  sigmoid(200x1):       {'PASS' if ok else 'FAIL'} (max diff: {np.max(np.abs(result - expected)):.2e})")
    all_pass &= ok

    # Test transpose_2d
    m = np.random.randn(30, 50).astype(np.float32)
    expected = m.T
    result = transpose_2d(WgpuTensor.from_numpy(m)).numpy()
    ok = np.allclose(result, expected, atol=1e-6)
    print(f"  transpose_2d(30x50):  {'PASS' if ok else 'FAIL'} (max diff: {np.max(np.abs(result - expected)):.2e})")
    all_pass &= ok

    # Test mul
    p = np.random.randn(64, 16).astype(np.float32)
    q = np.random.randn(64, 16).astype(np.float32)
    expected = p * q
    result = mul(WgpuTensor.from_numpy(p), WgpuTensor.from_numpy(q)).numpy()
    ok = np.allclose(result, expected, atol=1e-6)
    print(f"  mul(64x16):           {'PASS' if ok else 'FAIL'} (max diff: {np.max(np.abs(result - expected)):.2e})")
    all_pass &= ok

    if all_pass:
        print("  [OK] All compute shaders pass")
    else:
        print("  [FAIL] Some shaders failed")

    return all_pass


def benchmark_contact_head():
    """Benchmark GPU vs CPU for contact head MLP workloads."""
    print("\n" + "=" * 60)
    print("STEP 4: Contact Head Benchmark (GPU vs numpy)")
    print("=" * 60)

    from wgpu_tensor import WgpuTensor, matmul, add, relu

    np.random.seed(42)
    N = 300  # typical l_a * l_b
    iters = 10

    sizes = [
        ("Layer0: (N,58)@(58,256)", N, 58, 256),
        ("Layer2: (N,256)@(256,128)", N, 256, 128),
        ("Layer4: (N,128)@(128,1)", N, 128, 1),
    ]

    total_np = 0
    total_gpu = 0

    for label, M, K, O in sizes:
        a = np.random.randn(M, K).astype(np.float32)
        W = np.random.randn(K, O).astype(np.float32)
        b = np.random.randn(M, O).astype(np.float32) * 0.01

        # Numpy baseline
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = np.maximum(0, a @ W + b)
        t_np = (time.perf_counter() - t0) / iters

        # GPU
        a_gpu = WgpuTensor.from_numpy(a)
        W_gpu = WgpuTensor.from_numpy(W)
        b_gpu = WgpuTensor.from_numpy(b)
        _ = relu(add(matmul(a_gpu, W_gpu), b_gpu)).numpy()  # warmup

        t0 = time.perf_counter()
        for _ in range(iters):
            r = relu(add(matmul(a_gpu, W_gpu), b_gpu))
            _ = r.numpy()
        t_gpu = (time.perf_counter() - t0) / iters

        speedup = t_np / t_gpu
        winner = "GPU" if speedup > 1 else "numpy"
        print(f"  {label}")
        print(f"    numpy: {t_np*1000:.2f}ms  GPU: {t_gpu*1000:.2f}ms  speedup: {speedup:.2f}x ({winner})")

        total_np += t_np
        total_gpu += t_gpu

    overall = total_np / total_gpu
    print(f"\n  Overall MLP speedup: {overall:.2f}x")
    return overall


def main():
    print("OperonFold GPU Setup Test")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    ok = check_env_vars()
    if not ok:
        print("\n[RESULT] FAIL — environment variables not configured")
        sys.exit(1)

    gpu_ok = check_gpu_adapter()
    shader_ok = check_compute_shaders()

    if gpu_ok and shader_ok:
        speedup = benchmark_contact_head()
        print("\n" + "=" * 60)
        if speedup > 1.0:
            print(f"[RESULT] PASS — GPU active, {speedup:.1f}x faster than numpy")
        else:
            print(f"[RESULT] PASS — GPU active, compute shaders work")
            print(f"  Note: GPU {speedup:.1f}x vs numpy — overhead dominates at this size")
        print("=" * 60)
    elif shader_ok:
        print("\n[RESULT] PARTIAL — shaders work but on CPU (llvmpipe)")
        print("  Set env vars per GPU_SETUP.md for real GPU acceleration")
    else:
        print("\n[RESULT] FAIL — compute shaders not working")
        sys.exit(1)


if __name__ == "__main__":
    main()
