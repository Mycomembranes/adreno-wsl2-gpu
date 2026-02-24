"""Centralized atexit handler for wgpu-py cleanup hang workaround.

wgpu-py (via wgpu-native) can hang on process exit when GPU resources
aren't explicitly released. This module registers an atexit handler
that destroys tracked devices and calls os._exit(0) to force clean exit.

Usage:
    from wgpu_hang_fix import track_device
    device = track_device(wgpu.utils.device.get_default_device())
    # ... use device ...
    # Process will exit cleanly via atexit handler
"""

import atexit
import os

_devices = []


def track_device(device):
    """Track a wgpu device for cleanup at exit. Returns the device."""
    _devices.append(device)
    return device


def _cleanup():
    """Destroy tracked devices and force-exit to avoid hang."""
    for d in _devices:
        try:
            d.destroy()
        except Exception:
            pass
    os._exit(0)


atexit.register(_cleanup)
