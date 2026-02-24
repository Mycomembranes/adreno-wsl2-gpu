# Benchmark Results

## Hardware

- **Device**: Microsoft Surface Pro (11th Edition)
- **SoC**: Qualcomm Snapdragon X Elite
- **GPU**: Qualcomm Adreno X1-85
- **OS**: Windows 11 + WSL2 (Ubuntu)
- **Driver**: Mesa Dozen (Vulkan-over-D3D12), patched

## Compute Benchmarks (wgpu-py)

Run with: `python3 tests/bench_wgpu_full.py`

### Element-wise Operations (1M elements)

| Operation | Type | Time (ms) | Throughput |
|-----------|------|-----------|------------|
| Multiply by 2 | u32 | ~28 | ~143 Melem/s |
| Multiply by 2 | f32 | ~27 | ~148 Melem/s |
| Multiply by 2 | i32 | ~28 | ~143 Melem/s |
| Add constant | u32 | ~27 | ~148 Melem/s |
| Add constant | f32 | ~26 | ~154 Melem/s |
| Add constant | i32 | ~27 | ~148 Melem/s |
| Bitwise XOR | u32 | ~26 | ~154 Melem/s |
| Square | f32 | ~27 | ~148 Melem/s |
| Negate | i32 | ~27 | ~148 Melem/s |

### Matrix Multiplication

| Size | Time (ms) | GFLOPS |
|------|-----------|--------|
| 256x256 | ~45 | ~0.75 |
| 512x512 | ~180 | ~1.49 |

### Notes

- Times include WSL2 D3D12 interop overhead (~25-60ms fixed per dispatch)
- Throughput improves significantly with larger workloads
- First dispatch is slower due to shader compilation; subsequent dispatches are faster
- All results verified correct against CPU reference implementations

## Raw Vulkan Compute

Run with: `./tests/vulkan_compute_test`

| Test | Elements | Result |
|------|----------|--------|
| Buffer multiply | 1024 | All correct |

## Graphics

| Test | Result |
|------|--------|
| Offscreen triangle (wgpu) | Renders correctly, RGB interpolation visible |
| vkcube (5s) | Runs without crash |

## Vulkan Feature Coverage

```
fullDrawIndexUint32 = true   (patched)
logicOp = true               (patched)
geometryShader = true
tessellationShader = false    (upstream limitation)
multiDrawIndirect = true
depthClamp = true
```
