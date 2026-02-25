# GPU Setup for OperonFold on WSL2 (Qualcomm Adreno)

## Problem

WSL2 on ARM64 (Surface Pro) exposes the Qualcomm Adreno GPU via DirectX 12 (`/dev/dxg`), but the default Mesa Vulkan drivers only provide `llvmpipe` (CPU software rendering). The `wgpu` library needs a Vulkan driver that bridges to D3D12.

## Solution: Mesa Dozen Driver

A custom Mesa build with the **Dozen** (D3D12-to-Vulkan) driver is installed at:
```
/home/mukshud/mesa-dozen-install/
```

This translates Vulkan API calls to DirectX 12, enabling real GPU compute on WSL2.

## Required Environment Variables

**Set these before running any training or GPU code:**

```bash
export VK_ICD_FILENAMES=/home/mukshud/mesa-dozen-install/share/vulkan/icd.d/dzn_icd.aarch64.json
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/mukshud/mesa-dozen-install/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

Or add to `~/.bashrc` for persistence:
```bash
echo 'export VK_ICD_FILENAMES=/home/mukshud/mesa-dozen-install/share/vulkan/icd.d/dzn_icd.aarch64.json' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/mukshud/mesa-dozen-install/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}' >> ~/.bashrc
```

## Verification

### Quick check: Is the GPU visible?
```bash
vulkaninfo --summary 2>&1 | grep deviceName
# Expected: deviceName = Microsoft Direct3D12 (Qualcomm(R) Adreno(TM) X1-85 GPU)
```

### Quick check: Does wgpu see the GPU?
```bash
python3 -c "
import wgpu
for a in wgpu.gpu.enumerate_adapters_sync():
    print(a.summary)
"
# Expected first line: Microsoft Direct3D12 (Qualcomm(R) Adreno(TM) X1-85 GPU) (IntegratedGPU) via Vulkan
```

### Full test: GPU compute shaders work?
```bash
python3 test_gpu_setup.py
```

## What Uses the GPU

| Component | GPU Operations | Notes |
|-----------|---------------|-------|
| Transformer forward | matmul, softmax, layer_norm, gelu | All 6 layers |
| Contact head forward | matmul, relu, add | 3-layer MLP (58→256→128→1) |
| Contact head backward | matmul, relu_backward, transpose_2d, sigmoid | Weight + activation grads |
| Embedding lookup | embedding_lookup shader | Token + segment embeddings |

## Troubleshooting

### Only seeing `llvmpipe (CPU)`
- Environment variables not set. Run the export commands above.
- The Dozen driver .so may need rebuilding if Mesa was updated.

### `MESA: error: ZINK requires...`
- Harmless warning. Zink is a different driver; Dozen is used instead.

### `Adapter is not Vulkan compliant`
- Expected. Dozen is a translation layer, not full Vulkan. Compute shaders work fine.

### GPU device creation hangs
- Restart WSL: `wsl --shutdown` from Windows PowerShell, then reopen.
- Check `/dev/dxg` exists: `ls -la /dev/dxg`

## Architecture

```
Python (wgpu_tensor.py)
  → wgpu-py (Python WebGPU bindings)
    → wgpu-native (Rust WebGPU implementation)
      → Vulkan API
        → Mesa Dozen (D3D12 translation layer)
          → /dev/dxg (WSL2 DirectX passthrough)
            → Windows D3D12 runtime
              → Qualcomm Adreno X1-85 GPU hardware
```
