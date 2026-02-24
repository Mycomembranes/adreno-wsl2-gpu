# Hybrid wgpu Forward + PyTorch Backward Pass

## Problem

Training deep learning models on the Adreno X1-85 via wgpu compute shaders faces a backward pass bottleneck. While the forward pass benefits from GPU parallelism, the backward pass requires dozens of small GPU dispatches (one per operation: LayerNorm backward, GELU backward, matmul for each gradient, etc.). Each dispatch incurs ~25-60ms of WSL2 D3D12 interop overhead, making GPU backward passes slower than CPU.

Manual numpy analytical gradients avoid the dispatch overhead but miss out on:
- Fused kernel optimizations (e.g., GELU + bias in one pass)
- Multi-threaded BLAS for matrix operations
- In-place gradient accumulation (numpy creates temporary arrays)

## Solution: Hybrid Architecture

Keep the wgpu forward pass on GPU, replace the backward pass with PyTorch autograd on CPU.

```
tokens --> [wgpu GPU forward] --> logits, cache
                                      |
                       [PyTorch autograd backward] --> grads --> [numpy AdamW] --> params
                       (fused kernels, multi-threaded)
```

### How It Works

1. **Forward pass**: Runs on Adreno X1-85 GPU via wgpu compute shaders (matmul, softmax, layer_norm, GELU, embedding lookup)
2. **Weight sync**: Current numpy params are loaded into a PyTorch model mirror (`_sync_numpy_to_torch()`)
3. **PyTorch forward+backward**: Same input batch runs through the PyTorch model with autograd tracking, then `loss.backward()` computes all gradients
4. **Gradient extraction**: PyTorch gradients are copied back as numpy arrays (`_extract_torch_grads()`)
5. **Optimizer step**: numpy AdamW updates params, GPU params are re-synced

The PyTorch model is an exact architectural mirror of the wgpu model. Weight names are mapped between the two conventions:

| wgpu key | PyTorch key | Notes |
|----------|-------------|-------|
| `*.gamma` | `*.weight` | LayerNorm naming |
| `*.beta` | `*.bias` | LayerNorm naming |
| `layers.N.ff.linear1.*` | `layers.N.ff.0.*` | Sequential indexing |
| `layers.N.ff.linear2.*` | `layers.N.ff.3.*` | Sequential(Linear,GELU,Dropout,Linear,Dropout) |
| All other keys | Same | Identical |

### Why Not Full PyTorch?

The wgpu forward pass is still faster than CPU-only PyTorch forward for the specific ops used (batched matmul, softmax over long sequences). The GPU excels at the forward pass because it consists of fewer, larger dispatches that amortize the D3D12 overhead.

## Usage

### Quick Start

```python
from wgpu_ml.hybrid_backward.wgpu_transformer import TransformerWGPU

# Create model
model = TransformerWGPU(d_model=128, n_layers=6, vocab_size=256)
model.init_torch_backward()  # Enable PyTorch backward

# Training loop
logits, cache = model.forward_mlm(token_ids)
loss = model.pytorch_backward(token_ids, mask_pos, labels, cache)
# ... optimizer.step(model.params, model.grads) ...
```

### Key Methods

**`wgpu_transformer.py`** (GPU model):
- `init_torch_backward()` — creates PyTorch model mirror, builds key mapping
- `pytorch_backward(token_ids, mask_pos, labels, cache)` — PyTorch forward+backward, returns loss
- `_sync_numpy_to_torch()` — copies numpy params to PyTorch state_dict
- `_extract_torch_grads()` — copies PyTorch `.grad` arrays back to `model.grads`

**`torch_transformer.py`** (PyTorch mirror):
- Identical architecture to the wgpu model
- Used only for autograd computation, not for inference

### Threading Configuration

PyTorch uses 10 threads (matching the BLAS config), leaving 2 cores for wgpu/data loading:

```bash
export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
```

```python
import torch
torch.set_num_threads(10)
```

## Performance

Tested with a 1.23M parameter transformer (d_model=128, 6 layers, 4 heads), B=8, L=256:

| Configuration | Throughput | Improvement |
|--------------|-----------|-------------|
| wgpu forward + numpy backward | 1.4 seq/s | baseline |
| wgpu forward + PyTorch backward | 4.3 seq/s | **3.1x** |

**3.1x speedup** with no accuracy loss. Loss continues from checkpoint without discontinuity.

### Memory Overhead

| Component | Size |
|-----------|------|
| wgpu model params | ~5MB |
| PyTorch model mirror | ~5MB |
| PyTorch autograd graph | ~50-100MB |
| **Total overhead** | **~200MB** |

## Adapting to Your Model

To use this pattern with your own transformer architecture:

1. Create a wgpu model class with numpy `self.params` dict and a `forward()` method
2. Create a matching PyTorch `nn.Module` with identical layer structure
3. Define the key mapping between your param naming conventions
4. Copy the `init_torch_backward()`, `_sync_numpy_to_torch()`, `_extract_torch_grads()`, and `pytorch_backward()` methods

The key insight: the param key mapping only needs to handle naming differences between your numpy param dict and PyTorch's `state_dict()` convention (typically LayerNorm gamma/beta vs weight/bias, and Sequential indexing).

## Future: Pure wgpu WGSL Backward

Once the hybrid approach is validated, the backward pass can be ported to pure WGSL compute shaders:
- Custom shaders: `softmax_backward`, `layernorm_backward`, batched attention gradient
- Single command buffer with all dispatches (amortize D3D12 overhead)
- Eliminates CPU-GPU copy bottleneck entirely
