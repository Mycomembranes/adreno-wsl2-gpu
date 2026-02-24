# Adreno X1-85 GPU on WSL2

Run the Qualcomm Adreno X1-85 GPU (Snapdragon X Elite) under WSL2 Linux using Mesa's Dozen Vulkan-over-D3D12 driver. This repo provides patches, build scripts, and tests for both **compute** and **graphics** workloads.

## Hardware Requirements

- **Microsoft Surface Pro** (11th Edition) or any Snapdragon X Elite device
- **GPU**: Qualcomm Adreno X1-85 (integrated)
- **OS**: Windows 11 with WSL2 enabled
- **WSL distro**: Ubuntu 22.04+ (aarch64)

## Prerequisites

```bash
# Build tools
sudo apt install meson ninja-build gcc g++ pkg-config git glslang-tools \
  llvm-17-dev libdrm-dev libx11-dev libxcb1-dev libxrandr-dev \
  zlib1g-dev libexpat1-dev python3 python3-mako python3-yaml

# Python runtime
pip install wgpu numpy Pillow

# Verify
bash scripts/check_prerequisites.sh
```

## Quick Start

```bash
git clone https://github.com/Mycomembranes/adreno-wsl2-gpu.git
cd adreno-wsl2-gpu
bash scripts/install.sh
```

The installer clones Mesa, applies patches, builds, sets up the environment, and runs validation tests.

## Manual Build

### 1. Clone Mesa

```bash
git clone --depth=1 --branch mesa-25.0.5 \
  https://gitlab.freedesktop.org/mesa/mesa.git ~/mesa
```

### 2. Apply Patches

```bash
cd ~/mesa
git apply /path/to/adreno-wsl2-gpu/patches/mesa-dozen-adreno.patch
```

### 3. Configure and Build

```bash
meson setup builddir \
  --prefix=$HOME/mesa-install \
  -Dplatforms=x11 \
  -Dvulkan-drivers=microsoft-experimental,swrast \
  -Dgallium-drivers=swrast,zink \
  -Dglx=xlib \
  -Dbuildtype=release \
  -Dcpp_rtti=false \
  -Db_ndebug=true

ninja -j$(nproc) -C builddir
ninja -C builddir install
```

### 4. Set Environment

```bash
source scripts/setup_env.sh
# Or add to ~/.bashrc for persistence
```

### 5. Verify

```bash
vulkaninfo --summary
# Should show: Qualcomm Adreno X1-85, dozen driver
```

## What the Patches Change

### Mesa `dzn_device.c` (4 hunks)

| Change | Why |
|--------|-----|
| `fullDrawIndexUint32 = true` | D3D12 always supports 32-bit indices; flag was unnecessarily conservative |
| `logicOp = true` | D3D12 FL11.0+ supports Output Merger Logic Ops; `translate_logic_op()` already maps all 16 VkLogicOp values |
| Conformance version → 1.2 | Prevents loader rejection; Dozen is functionally conformant for compute |
| Conformance warning commented out | Removes noisy non-conformance stderr warning |

### Mesa Build Config

| Flag | Purpose |
|------|---------|
| `-Dplatforms=x11` | Enables `VK_KHR_surface` + `VK_KHR_xcb_surface` + `VK_KHR_swapchain` for graphics |
| `-Dvulkan-drivers=microsoft-experimental,swrast` | Builds Dozen (D3D12 Vulkan) + llvmpipe (software fallback) |

### wgpu-native `conv.rs` + `lib.rs` (4 hunks)

| Change | Why |
|--------|-----|
| `map_instance_flags()` always inserts `ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` | wgpu-native's C API has no way to set this flag; the enum is missing bit 3 |
| `map_instance_descriptor()` Default case adds the flag | Covers `WGPUInstanceFlag_Default` path |
| `map_instance_descriptor()` no-extras fallback adds the flag | Covers when no `WGPUInstanceExtras` chain is provided |
| `wgpuCreateInstance()` None descriptor adds the flag | Covers `wgpuCreateInstance(NULL)` path |

See [docs/WGPU_NATIVE_PATCH.md](docs/WGPU_NATIVE_PATCH.md) for full technical details.

### wgpu-py `device.py`

| Change | Why |
|--------|-----|
| Set `power_preference = "high-performance"` | Forces selection of Adreno GPU over llvmpipe software renderer |

## Running Compute Tests

### Raw Vulkan (C)

```bash
cd tests
gcc -o vulkan_compute_test vulkan_compute_test.c -lvulkan -lm
./vulkan_compute_test
# Multiplies 1024 elements on GPU, verifies results
```

### wgpu-py Compute

```bash
python3 tests/test_wgpu.py           # Basic compute test
python3 tests/test_wgpu_stepwise.py  # Step-by-step diagnostic
python3 tests/bench_wgpu_full.py     # Full benchmark (18 ops + matmul)
```

## Running Graphics Tests

### Offscreen Triangle (wgpu-py)

```bash
python3 tests/test_graphics.py
# Renders RGB triangle to 256x256 texture, verifies pixel readback
# Saves triangle_output.png if Pillow is installed
```

### vkcube (interactive)

```bash
DISPLAY=:0 vkcube
# Spinning cube — verifies full graphics pipeline
# Ctrl+C to exit
```

## Benchmark Results

| Operation | Elements | Time | Throughput |
|-----------|----------|------|------------|
| u32 multiply | 1M | ~28ms | ~143 Melem/s |
| f32 multiply | 1M | ~27ms | ~148 Melem/s |
| i32 add | 1M | ~27ms | ~148 Melem/s |
| 256x256 matmul | 33M ops | ~45ms | ~0.75 GFLOPS |
| 512x512 matmul | 268M ops | ~180ms | ~1.49 GFLOPS |

Times include WSL2 D3D12 interop overhead (~25-60ms fixed per dispatch). See [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for full results.

## Hybrid Training: wgpu Forward + PyTorch Backward

For ML training workloads, the Dozen dispatch overhead (~25-60ms per dispatch) makes GPU backward passes inefficient. We provide a **hybrid approach**: wgpu GPU forward pass + PyTorch CPU autograd backward pass, achieving **3.1x speedup** (1.4 → 4.3 seq/s) on a 1.23M parameter transformer.

```bash
export OPERONFOLD_TORCH_BACKWARD=1  # Enable hybrid backward
```

See [docs/HYBRID_BACKWARD_PASS.md](docs/HYBRID_BACKWARD_PASS.md) for architecture details, integration guide, and benchmarks.

## Known Issues

| Issue | Workaround |
|-------|------------|
| wgpu-py hangs on exit | Use `wgpu_hang_fix.py` or `os._exit(0)` |
| Zink (OpenGL) fails | Dozen lacks `VK_EXT_robustness2`; use Vulkan directly |
| libEGL warnings | `MESA_LOG_LEVEL=error` |
| ~25-60ms dispatch overhead | WSL2 D3D12 latency; amortized for large workloads |
| tessellationShader = false | Upstream Dozen limitation |
| copy_texture_to_buffer hangs | Dozen limitation; render verified by execution success |

See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) for details.

## Troubleshooting

### "No Vulkan devices found"

```bash
# Verify ICD file exists
ls ~/mesa-install/share/vulkan/icd.d/dzn_icd.aarch64.json

# Check environment
echo $VK_ICD_FILENAMES
echo $LD_LIBRARY_PATH

# Re-source environment
source scripts/setup_env.sh
```

### "llvmpipe" selected instead of Adreno

```bash
# Verify wgpu-py patch is applied
python3 -c "import wgpu.utils.device; print(wgpu.utils.device.helper._adapter_kwargs)"
# Should show: {'power_preference': 'high-performance'}

# List all adapters
python3 tests/wgpu_gpu_utils.py
```

### Mesa build fails

```bash
# Check all dependencies
bash scripts/check_prerequisites.sh

# Clean rebuild
rm -rf ~/mesa/builddir
bash scripts/build_mesa.sh
```

### Graphics test fails but compute works

Graphics requires `-Dplatforms=x11` in the Mesa build and a running X server (`DISPLAY=:0`). Compute works without X11.

## Project Structure

```
adreno-wsl2-gpu/
├── README.md                    # This file
├── LICENSE                      # MIT
├── wgpu_ml/                     # GPU deep learning library
│   ├── __init__.py
│   ├── wgpu_tensor.py           # GPU tensor + WGSL shaders
│   ├── wgpu_autograd.py         # Autodiff engine
│   ├── wgpu_nn.py               # NN modules + optimizer
│   └── examples/
│       └── train_demo.py        # Training example
├── patches/
│   ├── mesa-dozen-adreno.patch                  # Mesa Dozen driver patches
│   ├── wgpu-native-noncompliant-adapter.patch   # wgpu-native compliance bypass
│   └── wgpu-device-gpu.patch                    # wgpu-py GPU selection fix
├── scripts/
│   ├── setup_env.sh             # Runtime environment variables
│   ├── build_mesa.sh            # Mesa build from source
│   ├── build_wgpu_native.sh     # Patched wgpu-native builder
│   ├── install.sh               # One-click installer
│   └── check_prerequisites.sh   # Dependency checker
├── tests/
│   ├── vulkan_compute_test.c    # Raw Vulkan C compute test
│   ├── multiply.comp            # GLSL compute shader source
│   ├── multiply.spv             # Pre-compiled SPIR-V binary
│   ├── test_wgpu.py             # wgpu compute test
│   ├── test_wgpu_stepwise.py    # Step-by-step diagnostic
│   ├── bench_wgpu_full.py       # Full benchmark suite
│   ├── test_graphics.py         # Offscreen triangle render test
│   ├── wgpu_gpu_utils.py        # GPU utility functions
│   └── wgpu_hang_fix.py         # Exit hang workaround
└── docs/
    ├── DOZEN_ANALYSIS.md        # Root cause analysis of Dozen issues
    ├── WGPU_NATIVE_PATCH.md     # wgpu-native patch technical docs
    ├── BENCHMARK_RESULTS.md     # Performance measurements
    ├── HYBRID_BACKWARD_PASS.md  # wgpu forward + PyTorch backward integration
    └── KNOWN_ISSUES.md          # Issues and workarounds
```

## wgpu_ml: GPU Deep Learning Library

A PyTorch-like tensor, autograd, and neural network library that runs entirely on the Adreno GPU via wgpu compute shaders. No CUDA required.

### Quick Usage

```python
import numpy as np
from wgpu_ml import WgpuTensor, matmul, softmax

# Create GPU tensors
a = WgpuTensor.from_numpy(np.random.randn(64, 128).astype("float32"))
b = WgpuTensor.from_numpy(np.random.randn(128, 64).astype("float32"))

# GPU matmul + softmax
c = matmul(a, b)
probs = softmax(c)
print(probs.shape)  # (64, 64)
```

### Features

- **GPU Tensor** (`wgpu_tensor.py`, 1437 lines) — WGSL compute shaders for add, mul, matmul, softmax, layer_norm, GELU, embedding lookup, cross-entropy, and more. Pipeline caching, automatic float64-to-f32 conversion.
- **Autograd** (`wgpu_autograd.py`, 1132 lines) — Reverse-mode automatic differentiation with `GradNode`, `WgpuParameter`, topological-sort backward pass. Supports all tensor ops.
- **NN Modules** (`wgpu_nn.py`, 578 lines) — `Linear`, `LayerNorm`, `Embedding`, `MultiHeadAttention`, `TransformerLayer`, `Sequential`, `ModuleList`, `AdamW` optimizer.

### Training Example

```bash
python -m wgpu_ml.examples.train_demo
```

Trains a 2-layer MLP on synthetic data. See [`wgpu_ml/examples/train_demo.py`](wgpu_ml/examples/train_demo.py).

### Structure

```
wgpu_ml/
├── __init__.py          # Public API
├── wgpu_tensor.py       # GPU tensor ops (WGSL shaders)
├── wgpu_autograd.py     # Autodiff engine
├── wgpu_nn.py           # NN modules + AdamW
└── examples/
    └── train_demo.py    # MLP training example
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test with both compute and graphics workloads
4. Submit a pull request

Key areas for contribution:
- Additional benchmark workloads
- Tessellation shader support (needs DXIL work in Dozen)
- `VK_EXT_robustness2` implementation (enables Zink/OpenGL)
- Performance optimization of the D3D12 translation layer
