"""Test wgpu-py GPU compute on Adreno X1-85 via Dozen/D3D12."""
import numpy as np
import time

# Try to import wgpu
import wgpu
import wgpu.utils
from wgpu.utils.compute import compute_with_buffers

# Enumerate adapters
print("=== wgpu GPU Compute Test ===")
print(f"wgpu version: {wgpu.__version__}")

# Request adapter
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
print(f"Adapter: {adapter.info}")

device = adapter.request_device_sync()
print(f"Device: {device}")

# WGSL compute shader: multiply each element by 2
shader_code = """
@group(0) @binding(0) var<storage,read> input: array<f32>;
@group(0) @binding(1) var<storage,read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&input)) {
        output[i] = input[i] * 2.0;
    }
}
"""

# Test data
N = 1024
input_data = np.arange(1, N + 1, dtype=np.float32)
print(f"\nInput (first 10): {input_data[:10]}")

# Run compute — returns dict {binding: memoryview}
t0 = time.perf_counter()
outputs = compute_with_buffers(
    input_arrays={0: input_data},
    output_arrays={1: (N, "f")},
    shader=shader_code,
    n=(N // 64, 1, 1),
)
t1 = time.perf_counter()

result = np.frombuffer(outputs[1], dtype=np.float32)
expected = input_data * 2.0

print(f"Output (first 10): {result[:10]}")
print(f"Expected (first 10): {expected[:10]}")

errors = np.sum(result != expected)
if errors == 0:
    print(f"\n===== SUCCESS =====")
    print(f"All {N} elements verified correct")
    print(f"Compute time: {(t1-t0)*1000:.3f} ms")
else:
    print(f"\n===== FAILURE =====")
    print(f"{errors}/{N} mismatches")
