"""GPU compute utilities configured for Adreno X1-85 via Mesa Dozen.

Usage:
    from wgpu_gpu_utils import get_gpu_device, compute_with_buffers_gpu

    # Direct device access
    device = get_gpu_device()

    # High-level compute (same API as wgpu.utils.compute_with_buffers)
    out, = compute_with_buffers_gpu({0: data}, {1: nbytes}, shader, n=1024)
"""

import wgpu
import wgpu.backends.wgpu_native
from wgpu.utils.compute import compute_with_buffers

# Ensure high-performance adapter is selected (Adreno over llvmpipe)
wgpu.utils.device.helper._adapter_kwargs.setdefault(
    "power_preference", "high-performance"
)


def get_gpu_device():
    """Get the default wgpu device, configured for discrete/integrated GPU."""
    return wgpu.utils.device.get_default_device()


def compute_with_buffers_gpu(input_arrays, output_arrays, shader, n=1):
    """Run a compute shader via wgpu, forcing GPU adapter selection.

    Same signature as wgpu.utils.compute.compute_with_buffers.
    """
    return compute_with_buffers(input_arrays, output_arrays, shader, n=n)


def list_adapters():
    """List all available GPU adapters."""
    adapters = wgpu.gpu.enumerate_adapters_sync()
    for i, adapter in enumerate(adapters):
        info = adapter.info
        print(f"  [{i}] {info['device']} ({info['adapter_type']}, {info['backend_type']})")
    return adapters


if __name__ == "__main__":
    print("Available adapters:")
    list_adapters()
    print()
    device = get_gpu_device()
    info = device.adapter.info
    print(f"Selected: {info['device']} ({info['adapter_type']}, {info['backend_type']})")
    print(f"Limits: maxComputeWorkgroupSizeX={device.limits['max-compute-workgroup-size-x']}")
