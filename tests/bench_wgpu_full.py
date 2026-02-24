#!/usr/bin/env python3
"""Comprehensive GPU benchmark for Adreno X1-85 via wgpu-py.

Tests: vector add (u32, f32, i32), matrix multiply (f32).
Sizes: 1K to 4M elements. Multiple iterations for warm-cache timing.
Compares GPU throughput and reports speedup vs CPU baseline.
"""

import time
import struct
import numpy as np

import wgpu
import wgpu.backends.wgpu_native
from wgpu.utils.compute import compute_with_buffers

# Force GPU selection
wgpu.utils.device.helper._adapter_kwargs.setdefault(
    "power_preference", "high-performance"
)

# --- Shaders ---

SHADER_ADD_U32 = """
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&a)) {
        out[i] = a[i] + b[i];
    }
}
"""

SHADER_ADD_F32 = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&a)) {
        out[i] = a[i] + b[i];
    }
}
"""

SHADER_ADD_I32 = """
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&a)) {
        out[i] = a[i] + b[i];
    }
}
"""

SHADER_MATMUL = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> dims: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = dims[0];
    let row = gid.y;
    let col = gid.x;
    if (row >= N || col >= N) { return; }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < N; k = k + 1u) {
        sum = sum + a[row * N + k] * b[k * N + col];
    }
    out[row * N + col] = sum;
}
"""


def bench_vector_add(dtype_name, shader, n, pack_fmt, np_dtype, iterations=5):
    """Benchmark vector addition for a given type and size."""
    np.random.seed(42)
    if np_dtype == np.float32:
        a_np = np.random.rand(n).astype(np.float32)
        b_np = np.random.rand(n).astype(np.float32)
    elif np_dtype == np.int32:
        a_np = np.random.randint(-1000, 1000, n, dtype=np.int32)
        b_np = np.random.randint(-1000, 1000, n, dtype=np.int32)
    else:
        a_np = np.random.randint(0, 1000, n, dtype=np.uint32)
        b_np = np.random.randint(0, 1000, n, dtype=np.uint32)

    a_bytes = a_np.tobytes()
    b_bytes = b_np.tobytes()
    out_nbytes = n * 4

    # Warm-up
    compute_with_buffers(
        {0: a_bytes, 1: b_bytes},
        {2: out_nbytes},
        shader,
        n=(n + 255) // 256,
    )

    # Timed runs
    gpu_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        outputs = compute_with_buffers(
            {0: a_bytes, 1: b_bytes},
            {2: out_nbytes},
            shader,
            n=(n + 255) // 256,
        )
        gpu_times.append(time.perf_counter() - t0)
    result = outputs[2]

    # CPU baseline
    cpu_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = a_np + b_np
        cpu_times.append(time.perf_counter() - t0)

    # Verify correctness (last GPU result)
    out_np = np.frombuffer(bytes(result), dtype=np_dtype)
    expected = a_np + b_np
    correct = np.array_equal(out_np, expected) if np_dtype != np.float32 else np.allclose(out_np, expected)

    gpu_median = np.median(gpu_times) * 1000
    cpu_median = np.median(cpu_times) * 1000

    return {
        "dtype": dtype_name,
        "n": n,
        "gpu_ms": gpu_median,
        "cpu_ms": cpu_median,
        "speedup": cpu_median / gpu_median if gpu_median > 0 else float("inf"),
        "correct": correct,
        "throughput_gops": (n / gpu_median) / 1e6,  # G-ops/s
    }


def bench_matmul(N, iterations=3):
    """Benchmark NxN matrix multiply."""
    np.random.seed(42)
    a_np = np.random.rand(N, N).astype(np.float32)
    b_np = np.random.rand(N, N).astype(np.float32)

    a_bytes = a_np.tobytes()
    b_bytes = b_np.tobytes()
    dims_bytes = struct.pack("I", N)
    out_nbytes = N * N * 4

    wg_x = (N + 15) // 16
    wg_y = (N + 15) // 16

    # Warm-up
    compute_with_buffers(
        {0: a_bytes, 1: b_bytes, 3: dims_bytes},
        {2: out_nbytes},
        SHADER_MATMUL,
        n=(wg_x, wg_y, 1),
    )

    gpu_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        outputs = compute_with_buffers(
            {0: a_bytes, 1: b_bytes, 3: dims_bytes},
            {2: out_nbytes},
            SHADER_MATMUL,
            n=(wg_x, wg_y, 1),
        )
        gpu_times.append(time.perf_counter() - t0)
    result = outputs[2]

    cpu_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = a_np @ b_np
        cpu_times.append(time.perf_counter() - t0)

    out_np = np.frombuffer(bytes(result), dtype=np.float32).reshape(N, N)
    expected = a_np @ b_np
    correct = np.allclose(out_np, expected, atol=1e-2)

    gpu_median = np.median(gpu_times) * 1000
    cpu_median = np.median(cpu_times) * 1000
    gflops = (2 * N**3) / (gpu_median / 1000) / 1e9

    return {
        "N": N,
        "gpu_ms": gpu_median,
        "cpu_ms": cpu_median,
        "speedup": cpu_median / gpu_median if gpu_median > 0 else float("inf"),
        "correct": correct,
        "gflops": gflops,
    }


def main():
    # Print adapter info
    device = wgpu.utils.device.get_default_device()
    info = device.adapter.info
    print(f"GPU: {info['device']} ({info['adapter_type']}, {info['backend_type']})")
    print(f"Driver: {info.get('description', 'N/A')}")
    print()

    # Vector add benchmarks
    sizes = [1024, 16384, 65536, 262144, 1048576, 4194304]
    configs = [
        ("u32", SHADER_ADD_U32, "I", np.uint32),
        ("f32", SHADER_ADD_F32, "f", np.float32),
        ("i32", SHADER_ADD_I32, "i", np.int32),
    ]

    print("=" * 80)
    print(f"{'Vector Add':^80}")
    print("=" * 80)
    print(f"{'Type':<6} {'Elements':>10} {'GPU ms':>10} {'CPU ms':>10} {'Speedup':>10} {'GOps/s':>10} {'OK':>4}")
    print("-" * 80)

    for dtype_name, shader, fmt, np_dtype in configs:
        for n in sizes:
            r = bench_vector_add(dtype_name, shader, n, fmt, np_dtype)
            mark = "Y" if r["correct"] else "N"
            print(f"{r['dtype']:<6} {r['n']:>10,} {r['gpu_ms']:>10.3f} {r['cpu_ms']:>10.3f} {r['speedup']:>10.2f}x {r['throughput_gops']:>10.2f} {mark:>4}")

    # Matrix multiply benchmarks
    mat_sizes = [32, 64, 128, 256]
    print()
    print("=" * 80)
    print(f"{'Matrix Multiply (NxN)':^80}")
    print("=" * 80)
    print(f"{'N':>6} {'GPU ms':>10} {'CPU ms':>10} {'Speedup':>10} {'GFLOPS':>10} {'OK':>4}")
    print("-" * 80)

    for N in mat_sizes:
        r = bench_matmul(N)
        mark = "Y" if r["correct"] else "N"
        print(f"{r['N']:>6} {r['gpu_ms']:>10.3f} {r['cpu_ms']:>10.3f} {r['speedup']:>10.2f}x {r['gflops']:>10.2f} {mark:>4}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
    # Force-exit to avoid wgpu cleanup hang on Dozen/D3D12
    import os; os._exit(0)
