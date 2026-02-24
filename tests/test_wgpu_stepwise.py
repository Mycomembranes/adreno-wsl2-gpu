"""Step-by-step wgpu compute test to find exactly where it hangs."""
import sys
import time
import numpy as np

def step(msg):
    print(f"[{time.perf_counter():.3f}] {msg}", flush=True)

step("Importing wgpu...")
import wgpu
import wgpu.backends.wgpu_native

step(f"wgpu {wgpu.__version__}")

step("Requesting adapter (high-performance)...")
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
step(f"Got: {adapter.info['device']}")

step("Requesting device...")
device = adapter.request_device_sync()
step(f"Device ready")

# Shader
shader_code = """
@group(0) @binding(0) var<storage,read_write> data: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    data[i] = data[i] * 2u;
}
"""

step("Creating shader module...")
shader_module = device.create_shader_module(code=shader_code)
step("Shader compiled")

# Data
N = 1024
input_data = np.arange(1, N + 1, dtype=np.uint32)
step(f"Input data: {input_data[:5]}...")

step("Creating GPU buffer with data...")
buf = device.create_buffer_with_data(
    data=input_data,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
)
step(f"Buffer created, size={buf.size}")

step("Creating bind group layout...")
bgl = device.create_bind_group_layout(entries=[{
    "binding": 0,
    "visibility": wgpu.ShaderStage.COMPUTE,
    "buffer": {"type": wgpu.BufferBindingType.storage},
}])
step("Bind group layout ready")

step("Creating pipeline layout...")
pl = device.create_pipeline_layout(bind_group_layouts=[bgl])
step("Pipeline layout ready")

step("Creating bind group...")
bg = device.create_bind_group(layout=bgl, entries=[{
    "binding": 0,
    "resource": {"buffer": buf, "offset": 0, "size": buf.size},
}])
step("Bind group ready")

step("Creating compute pipeline...")
pipeline = device.create_compute_pipeline(
    layout=pl,
    compute={"module": shader_module, "entry_point": "main"},
)
step("Pipeline ready")

step("Creating command encoder...")
encoder = device.create_command_encoder()
step("Encoder ready")

step("Beginning compute pass...")
cpass = encoder.begin_compute_pass()
step("Compute pass begun")

step("Setting pipeline...")
cpass.set_pipeline(pipeline)
step("Pipeline set")

step("Setting bind group...")
cpass.set_bind_group(0, bg)
step("Bind group set")

step("Dispatching workgroups (4, 1, 1)...")
cpass.dispatch_workgroups(N // 256, 1, 1)
step("Dispatch recorded")

step("Ending compute pass...")
cpass.end()
step("Compute pass ended")

step("Finishing command encoder...")
command_buffer = encoder.finish()
step("Command buffer ready")

step("Submitting to queue...")
device.queue.submit([command_buffer])
step("Submitted! Now reading back...")

step("Calling queue.read_buffer()...")
result_mem = device.queue.read_buffer(buf)
step(f"Read complete! Got {len(result_mem)} bytes")

result = np.frombuffer(result_mem, dtype=np.uint32)
expected = input_data * 2

errors = np.sum(result != expected)
if errors == 0:
    step(f"SUCCESS: All {N} elements correct")
    print(f"First 10: {result[:10]}")
else:
    step(f"FAILURE: {errors}/{N} mismatches")
    for i in range(min(10, N)):
        if result[i] != expected[i]:
            print(f"  [{i}] got={result[i]} expected={expected[i]}")
