# Known Issues & Workarounds

## Active Issues

| Issue | Severity | Workaround |
|-------|----------|------------|
| wgpu-py hangs on process exit | Medium | Use `wgpu_hang_fix.py` or call `os._exit(0)` after GPU work |
| Zink (OpenGL-over-Vulkan) fails | Low | Dozen lacks `VK_EXT_robustness2`; use Vulkan/wgpu directly |
| libEGL warnings on stderr | Cosmetic | Set `MESA_LOG_LEVEL=error` |
| ~25-60ms dispatch overhead per compute call | Expected | WSL2 D3D12 interop latency; amortized for large workloads |
| `tessellationShader = false` | Upstream | Needs DXIL tessellation support in Dozen |
| `dualSrcBlend = false` | Upstream | Needs D3D12 dual-source blending support |
| `VK_EXT_memory_budget` not implemented | Informational | Warning only; no functional impact |
| `SURFACE_VIEW_FORMATS` not supported | Informational | Graphics-only; no compute impact |
| `nullDescriptor` / `VK_EXT_robustness2` missing | Upstream | Would need sentinel descriptors per type; complex |
| `copy_texture_to_buffer` hangs | Driver | Dozen texture-to-buffer copy deadlocks; render verified by execution success |

## wgpu-py Exit Hang Details

**Symptom**: Python process hangs indefinitely after `main()` returns.

**Root Cause**: wgpu-native's Rust backend holds GPU resources via Drop traits.
During Python interpreter shutdown, the destructor order is unpredictable,
causing deadlocks between the Python GIL and wgpu-native's internal locks.

**Fix**: Use the `wgpu_hang_fix.py` module which registers an `atexit` handler
that explicitly destroys devices then calls `os._exit(0)`:

```python
from wgpu_hang_fix import track_device
device = track_device(get_default_device())
# ... use device ...
# atexit handler will clean up and force-exit
```

## Zink (OpenGL) Failure

**Symptom**: `glxgears` or OpenGL apps crash with robustness2 errors.

**Root Cause**: Zink requires `VK_EXT_robustness2` (specifically `nullDescriptor`)
which Dozen doesn't support. Implementing it would require creating sentinel
descriptors for each descriptor type (buffer, image, sampler).

**Workaround**: Use Vulkan directly via wgpu-py or raw Vulkan API.
Most compute and rendering workloads work fine through Vulkan.

## Dispatch Latency

**Symptom**: First compute dispatch takes ~25-60ms.

**Root Cause**: WSL2's Vulkan-over-D3D12 path involves:
1. Vulkan API call → Mesa Dozen driver
2. Dozen translates to D3D12 API calls
3. D3D12 calls cross the WSL2 VM boundary to Windows host
4. Windows Qualcomm GPU driver executes on hardware
5. Results return via the same path

This overhead is fixed per dispatch and amortized over large workloads.
For 1M+ element operations, throughput approaches native speeds.
