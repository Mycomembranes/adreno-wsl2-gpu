# Dozen/D3D12 Buffer Mapping Hang Analysis

## Root Cause

The hang occurs in wgpu-py's `queue.read_buffer()` method, specifically at:

```
wgpu/backends/wgpu_native/_api.py:3962
    tmp_buffer.map_async("READ_NOSYNC").sync_wait()
```

### Call Chain

```
compute_with_buffers()
  -> device.queue.read_buffer(buffer)
    -> creates tmp_buffer with MAP_READ | COPY_DST
    -> encoder.copy_buffer_to_buffer(src, tmp_buffer)
    -> queue.submit([cmd])
    -> tmp_buffer.map_async("READ_NOSYNC")
      -> wgpuBufferMapAsync(buffer, READ, ...)
    -> .sync_wait()
      -> GPUPromise._thread_event.wait()  <--- BLOCKS FOREVER
        (waiting for PollThread to resolve)
```

The `PollThread` (in `_poller.py`) runs:
```python
while token_ids:
    self._poll_func(True)  # = wgpuDevicePoll(device, True, NULL)
```

This calls `wgpuDevicePoll(device, block=True)` which translates to Vulkan's
`vkWaitForFences` or similar. On Dozen (Vulkan-over-D3D12), this fence wait
never completes because the buffer map callback never fires.

## Why Raw Vulkan Works But wgpu-py Doesn't

The raw Vulkan test uses a different readback path:
1. `vkMapMemory` on a host-visible buffer (synchronous, no callback)
2. Direct memory copy after `vkQueueWaitIdle`

wgpu-py uses the WebGPU buffer mapping model:
1. `wgpuBufferMapAsync` (asynchronous, callback-based)
2. Requires device polling to drive the callback
3. Dozen may not properly signal the Vulkan fence that wgpu-native waits on

## Missing Downlevel Flags Impact

- **FULL_DRAW_INDEX_UINT32**: Graphics-only, not relevant to compute
- **SURFACE_VIEW_FORMATS**: Presentation-only, not relevant to compute
- **VK_KHR_swapchain**: Presentation-only, not relevant to compute

None of these should block compute dispatch. The issue is specifically in the
buffer mapping/readback path.

## Possible Dozen Bugs

1. **vkMapMemory on readback heaps**: Dozen translates Vulkan memory mapping to
   D3D12 readback heap access. The async mapping callback may not fire because
   Dozen doesn't implement `VK_KHR_map_memory_placed` or the D3D12 fence
   signaling for copy operations may not propagate through the Vulkan layer.

2. **Command buffer completion signaling**: The copy command
   (`copy_buffer_to_buffer`) submits work via `vkQueueSubmit`. Dozen translates
   this to `ID3D12CommandQueue::ExecuteCommandLists`. The subsequent fence that
   wgpu-native checks via `vkWaitForFences` may not be properly connected to
   the D3D12 fence.

3. **Polling model mismatch**: wgpu-native's polling model expects
   `vkGetFenceStatus` to eventually return `VK_SUCCESS`. If Dozen doesn't
   update fence status correctly after GPU work completes, the poll loop spins
   forever.

## Workarounds

### 1. compushady (Recommended)
Talks to D3D12 directly, no Vulkan translation. See `test_compushady.py`.

### 2. Raw Vulkan (Proven)
Already works. Use `vkQueueWaitIdle` + `vkMapMemory` instead of async mapping.

### 3. WGPU_BACKENDS=dx12
If wgpu-native has a D3D12 backend on Linux (unlikely but worth checking):
```bash
WGPU_BACKENDS=dx12 python test_wgpu.py
```

### 4. Mesa/Dozen Fix
File a bug against Mesa's Dozen driver for buffer map callback not firing
after compute dispatch + copy_buffer_to_buffer. The relevant code is in
`src/microsoft/vulkan/dzn_device.c` and `dzn_cmd_buffer.c`.
