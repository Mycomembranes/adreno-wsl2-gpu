# wgpu-native Noncompliant Adapter Patch

## Problem

wgpu-native hides Vulkan adapters that are not fully WebGPU-compliant. Specifically, in `wgpu-hal/src/vulkan/adapter.rs`, the `expose_adapter()` function checks `conformance_version.major` and returns `None` (hiding the adapter) when it equals 0 — unless the driver is MoltenVK or the `ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` instance flag is set.

The Mesa **dozen** driver (a Vulkan-on-D3D12 translation layer for WSL2) reports `conformance_version.major = 0` because it is not a fully conformant Vulkan implementation. This causes wgpu to hide the **Qualcomm Adreno X1-85 GPU** on Surface Pro devices running WSL2, even though the GPU hardware is fully functional for compute workloads.

The relevant code path in `wgpu-hal` v27.0.2:

```rust
// wgpu-hal/src/vulkan/adapter.rs, lines 1710-1727
if let Some(driver) = phd_capabilities.driver {
    if driver.conformance_version.major == 0 {
        if driver.driver_id == vk::DriverId::MOLTENVK {
            log::debug!("...but is MoltenVK, continuing");
        } else if self.shared.flags
            .contains(wgt::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER)
        {
            log::warn!("Adapter is not Vulkan compliant: {}", info.name);
        } else {
            log::warn!("Adapter is not Vulkan compliant, hiding adapter: {}", info.name);
            return None;  // <-- ADAPTER HIDDEN HERE
        }
    }
}
```

## Root Cause

wgpu-types defines the `ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` flag (`InstanceFlags` bit 3, `1 << 3`) and even has an environment variable `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` to control it. However, wgpu-native's C API **never maps this flag**:

1. `WGPUInstanceFlag` enum in `wgpu.h` only defines: `Default(0)`, `Debug(1)`, `Validation(2)`, `DiscardHalLabels(4)` — bit 3 (value 8) is missing
2. `map_instance_flags()` in `conv.rs` only maps Debug, Validation, and DiscardHalLabels
3. The env var `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` is read by `InstanceFlags::with_env()` in wgpu-types, but wgpu-native never calls `with_env()` — it constructs flags from the C API enum directly

This means there is **no way** to set this flag through the wgpu-native C API or environment variables.

## Solution

Patch wgpu-native to always include `ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` in the instance flags. This changes the behavior from "hide adapter" to "warn and continue", which is the intended behavior for development/compute workloads.

### Patch Sites (4 locations in 2 files)

**`src/conv.rs`** — 3 patches:

1. **`map_instance_flags()`** — When explicit flags are provided via the C API:
   ```rust
   // After the DiscardHalLabels check, add:
   result.insert(wgt::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER);
   ```

2. **`map_instance_descriptor()` Default flag case** — When `WGPUInstanceFlag_Default` is used:
   ```rust
   // Change from:
   native::WGPUInstanceFlag_Default => wgt::InstanceFlags::default(),
   // To:
   native::WGPUInstanceFlag_Default => wgt::InstanceFlags::default()
       | wgt::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
   ```

3. **`map_instance_descriptor()` no-extras fallback** — When no `WGPUInstanceExtras` is provided:
   ```rust
   // Change from:
   wgt::InstanceDescriptor::default()
   // To:
   let mut desc = wgt::InstanceDescriptor::default();
   desc.flags.insert(wgt::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER);
   desc
   ```

**`src/lib.rs`** — 1 patch:

4. **`wgpuCreateInstance()` None descriptor case** — When `wgpuCreateInstance(NULL)` is called:
   ```rust
   // Change from:
   None => wgt::InstanceDescriptor::default(),
   // To:
   None => {
       let mut desc = wgt::InstanceDescriptor::default();
       desc.flags.insert(wgt::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER);
       desc
   },
   ```

## Applying the Patch

### Automated (recommended)

```bash
./scripts/build_wgpu_native.sh
```

This clones wgpu-native, checks out v27.0.2.0, applies the patch, and builds.

### Manual

```bash
# Clone and checkout
git clone https://github.com/gfx-rs/wgpu-native.git
cd wgpu-native
git checkout v27.0.2.0
git submodule update --init --recursive

# Apply patch
git apply patches/wgpu-native-noncompliant-adapter.patch

# Build (set version for wgpu-py compatibility)
WGPU_NATIVE_VERSION=27.0.2.0 cargo build --release

# Install (backup first!)
WGPU_SO=$(python3 -c "import wgpu, pathlib; print(pathlib.Path(wgpu.__file__).parent / 'resources' / 'libwgpu_native-release.so')")
cp "$WGPU_SO" "${WGPU_SO}.bak"
cp target/release/libwgpu_native.so "$WGPU_SO"
```

## Verification

```python
import wgpu

adapters = wgpu.gpu.enumerate_adapters_sync()
for a in adapters:
    info = a.info
    print(f'{info["device"]} ({info["adapter_type"]}, {info["backend_type"]})')
```

Before patch:
```
llvmpipe (LLVM 20.1.2, 128 bits) (CPU, Vulkan)
llvmpipe (LLVM 20.1.2, 128 bits) (CPU, OpenGL)
```

After patch (with `VK_ICD_FILENAMES` including dozen):
```
Microsoft Direct3D12 (Qualcomm(R) Adreno(TM) X1-85 GPU) (IntegratedGPU, Vulkan)
llvmpipe (LLVM 20.1.2, 128 bits) (CPU, Vulkan)
llvmpipe (LLVM 20.1.2, 128 bits) (CPU, OpenGL)
```

## Compatibility

| Component | Version |
|-----------|---------|
| wgpu-native | v27.0.2.0 |
| wgpu-py | 0.30.0 |
| wgpu-hal | 27.0.2 |
| wgpu-types | 27.0.1 |
| Rust toolchain | 1.75+ (tested with 1.93.1) |
| Mesa dozen | 25.2.x |
| Platform | WSL2 on Windows 11 (aarch64) |

## Notes

- The patch is safe: it only changes the default from "hide" to "warn" for non-compliant adapters
- Adapter capabilities are still correctly reported via `DownlevelCapabilities`
- Compute shaders work correctly on the Adreno X1-85 via dozen (tested with matmul, softmax, etc.)
- The dozen driver supports: `COMPUTE_SHADERS`, `SM5`, `FRAGMENT_WRITABLE_STORAGE`, `ANISOTROPIC_FILTERING`
- Missing minor flags (`FULL_DRAW_INDEX_UINT32`, `SURFACE_VIEW_FORMATS`) are irrelevant for compute
- To revert: `cp libwgpu_native-release.so.bak libwgpu_native-release.so`
