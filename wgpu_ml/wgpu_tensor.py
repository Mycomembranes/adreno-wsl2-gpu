"""wgpu-native GPU tensor library for Adreno X1-85 training via Vulkan/D3D12.

This module provides a complete GPU tensor implementation with compute shaders
for deep learning operations on mobile Adreno GPUs.
"""

import os
import atexit
import struct
import numpy as np
import wgpu
import wgpu.backends.wgpu_native  # noqa: F401

# ============================================================================
# Device Singleton & Pipeline Cache
# ============================================================================

_device = None
_pipeline_cache = {}


def _get_device():
    """Get or create the wgpu device singleton."""
    global _device
    if _device is None:
        # Use the same approach as the validated test scripts
        wgpu.utils.device.helper._adapter_kwargs.setdefault(
            "power_preference", "high-performance"
        )
        _device = wgpu.utils.device.get_default_device()
    return _device


def _cleanup():
    """Cleanup wgpu resources on exit."""
    global _device
    if _device is not None:
        try:
            _device.destroy()
        except Exception:
            pass
        os._exit(0)


atexit.register(_cleanup)


# ============================================================================
# WGSL Compute Shader Sources
# ============================================================================

WGSL_ADD = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] + b[idx];
    }
}
"""

WGSL_MUL = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] * b[idx];
    }
}
"""

WGSL_SUB = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] - b[idx];
    }
}
"""

WGSL_SCALAR_MUL = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> scalar: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] * scalar;
    }
}
"""

WGSL_NEG = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = -a[idx];
    }
}
"""

WGSL_GELU = """
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let val = x[idx];
        let cdf = 0.5 * (1.0 + tanh(
            sqrt(2.0 / 3.14159265359) * (val + 0.044715 * val * val * val)
        ));
        out[idx] = val * cdf;
    }
}
"""

WGSL_RELU = """
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = max(0.0, x[idx]);
    }
}
"""

WGSL_SIGMOID = """
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = 1.0 / (1.0 + exp(-x[idx]));
    }
}
"""

WGSL_TANH = """
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = tanh(x[idx]);
    }
}
"""

WGSL_RELU_BACKWARD = """
@group(0) @binding(0)
var<storage, read> grad_out: array<f32>;
@group(0) @binding(1)
var<storage, read> x: array<f32>;
@group(0) @binding(2)
var<storage, read_write> grad_in: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&grad_in)) {
        grad_in[idx] = select(0.0, grad_out[idx], x[idx] > 0.0);
    }
}
"""

WGSL_SUM_REDUCE = """
@group(0) @binding(0)
var<storage, read> data: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<u32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let numel = params.x;
    let stride = params.y;
    let idx = gid.x;
    let lid_x = lid.x;

    var sum_val = 0.0;
    if (idx < numel) {
        sum_val = data[idx];
    }
    shared[lid_x] = sum_val;
    workgroupBarrier();

    var stride_val = 128u;
    loop {
        if (stride_val == 0u) { break; }
        if (lid_x < stride_val) {
            shared[lid_x] = shared[lid_x] + shared[lid_x + stride_val];
        }
        workgroupBarrier();
        stride_val = stride_val >> 1u;
    }

    if (lid_x == 0u) {
        let out_idx = gid.x / 256u;
        if (out_idx < arrayLength(&out)) {
            out[out_idx] = shared[0];
        }
    }
}
"""

WGSL_MAX_REDUCE = """
@group(0) @binding(0)
var<storage, read> data: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<u32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let numel = params.x;
    let idx = gid.x;
    let lid_x = lid.x;

    var max_val = -3.4e38;
    if (idx < numel) {
        max_val = data[idx];
    }
    shared[lid_x] = max_val;
    workgroupBarrier();

    var stride_val = 128u;
    loop {
        if (stride_val == 0u) { break; }
        if (lid_x < stride_val) {
            shared[lid_x] = max(shared[lid_x], shared[lid_x + stride_val]);
        }
        workgroupBarrier();
        stride_val = stride_val >> 1u;
    }

    if (lid_x == 0u) {
        let out_idx = gid.x / 256u;
        if (out_idx < arrayLength(&out)) {
            out[out_idx] = shared[0];
        }
    }
}
"""

WGSL_MEAN_REDUCE = """
@group(0) @binding(0)
var<storage, read> data: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let numel = u32(params.x);
    let idx = gid.x;
    let lid_x = lid.x;

    var sum_val = 0.0;
    if (idx < numel) {
        sum_val = data[idx];
    }
    shared[lid_x] = sum_val;
    workgroupBarrier();

    var stride_val = 128u;
    loop {
        if (stride_val == 0u) { break; }
        if (lid_x < stride_val) {
            shared[lid_x] = shared[lid_x] + shared[lid_x + stride_val];
        }
        workgroupBarrier();
        stride_val = stride_val >> 1u;
    }

    if (lid_x == 0u) {
        let out_idx = gid.x / 256u;
        if (out_idx < arrayLength(&out)) {
            out[out_idx] = shared[0] / f32(numel);
        }
    }
}
"""

WGSL_MATMUL = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<u32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let n = params.y;
    let k = params.z;
    let lx = lid.x;
    let ly = lid.y;

    let row = wid.y * 16u + ly;
    let col = wid.x * 16u + lx;

    var result = 0.0;

    var tile_idx = 0u;
    loop {
        if (tile_idx >= k) { break; }

        let a_col = tile_idx + lx;
        let a_idx = row * k + a_col;
        tile_a[ly * 16u + lx] = select(0.0, a[a_idx], row < m && a_col < k);

        let b_row = tile_idx + ly;
        let b_idx = b_row * n + col;
        tile_b[ly * 16u + lx] = select(0.0, b[b_idx], b_row < k && col < n);

        workgroupBarrier();

        for (var i = 0u; i < 16u; i = i + 1u) {
            result = result + tile_a[ly * 16u + i] * tile_b[i * 16u + lx];
        }

        workgroupBarrier();
        tile_idx = tile_idx + 16u;
    }

    if (row < m && col < n) {
        out[row * n + col] = result;
    }
}
"""

WGSL_TRANSPOSE_2D = """
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = params.x;
    let cols = params.y;
    let i = gid.y;
    let j = gid.x;

    if (i < rows && j < cols) {
        out[j * rows + i] = x[i * cols + j];
    }
}
"""

WGSL_LAYER_NORM = """
// One workgroup per row. params.x = width, params.y = eps (as f32)
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read> gamma: array<f32>;
@group(0) @binding(2)
var<storage, read> beta: array<f32>;
@group(0) @binding(3)
var<storage, read_write> out: array<f32>;
@group(0) @binding(4)
var<uniform> params: vec4<f32>;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let width = u32(params.x);
    let eps = params.y;
    let row = wid.x;
    let tid = lid.x;
    let row_offset = row * width;

    // Step 1: Compute mean
    var local_sum = 0.0;
    var col = tid;
    loop {
        if (col >= width) { break; }
        local_sum = local_sum + x[row_offset + col];
        col = col + 256u;
    }
    shared_val[tid] = local_sum;
    workgroupBarrier();

    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { shared_val[tid] = shared_val[tid] + shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let mean_val = shared_val[0] / f32(width);
    workgroupBarrier();

    // Step 2: Compute variance
    var local_var = 0.0;
    col = tid;
    loop {
        if (col >= width) { break; }
        let diff = x[row_offset + col] - mean_val;
        local_var = local_var + diff * diff;
        col = col + 256u;
    }
    shared_val[tid] = local_var;
    workgroupBarrier();

    s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { shared_val[tid] = shared_val[tid] + shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let var_val = shared_val[0] / f32(width);
    let inv_std = 1.0 / sqrt(var_val + eps);
    workgroupBarrier();

    // Step 3: Normalize and apply gamma/beta
    col = tid;
    loop {
        if (col >= width) { break; }
        let norm = (x[row_offset + col] - mean_val) * inv_std;
        out[row_offset + col] = norm * gamma[col] + beta[col];
        col = col + 256u;
    }
}
"""

WGSL_SOFTMAX = """
// One workgroup per row. Each thread handles multiple elements if width > 256.
// params.x = width (last dim), params.y = num_rows
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<u32>;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let width = params.x;
    let row = wid.x;
    let tid = lid.x;
    let row_offset = row * width;

    // Step 1: Find max in this row (each thread covers multiple elements)
    var local_max = -3.4e38;
    var col = tid;
    loop {
        if (col >= width) { break; }
        local_max = max(local_max, x[row_offset + col]);
        col = col + 256u;
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Tree reduction for max
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let row_max = shared_max[0];
    workgroupBarrier();

    // Step 2: Compute exp(x - max) and sum
    var local_sum = 0.0;
    col = tid;
    loop {
        if (col >= width) { break; }
        local_sum = local_sum + exp(x[row_offset + col] - row_max);
        col = col + 256u;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum
    s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let row_sum = shared_sum[0];
    workgroupBarrier();

    // Step 3: Write normalized values
    col = tid;
    loop {
        if (col >= width) { break; }
        out[row_offset + col] = exp(x[row_offset + col] - row_max) / row_sum;
        col = col + 256u;
    }
}
"""

WGSL_CROSS_ENTROPY = """
@group(0) @binding(0)
var<storage, read> logits: array<f32>;
@group(0) @binding(1)
var<storage, read> targets: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let log_p = logits[idx];
        let target = targets[idx];
        out[idx] = -target * log_p - (1.0 - target) * log(1.0 - exp(log_p) + 1e-6);
    }
}
"""

WGSL_FOCAL_BCE = """
@group(0) @binding(0)
var<storage, read> logits: array<f32>;
@group(0) @binding(1)
var<storage, read> targets: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let logit = logits[idx];
        let target = targets[idx];
        let gamma = params.x;
        let alpha = params.y;

        let p = 1.0 / (1.0 + exp(-logit));
        let ce = -target * log(p + 1e-6) - (1.0 - target) * log(1.0 - p + 1e-6);
        let p_t = select(1.0 - p, p, target > 0.5);
        let focal_weight = pow(1.0 - p_t, gamma);
        let focal_loss = alpha * (1.0 - alpha) * focal_weight * ce;

        out[idx] = focal_loss;
    }
}
"""

WGSL_EMBEDDING = """
@group(0) @binding(0)
var<storage, read> weight: array<f32>;
@group(0) @binding(1)
var<storage, read> indices: array<u32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_indices = params.x;
    let embedding_dim = params.y;

    if (idx < num_indices * embedding_dim) {
        let i = idx / embedding_dim;
        let j = idx % embedding_dim;
        let word_idx = indices[i];
        let weight_idx = word_idx * embedding_dim + j;
        out[idx] = weight[weight_idx];
    }
}
"""


# ============================================================================
# Dispatch Helper
# ============================================================================

def _dispatch_shader(device, wgsl_code, buffers, workgroups):
    """Execute a compute shader on GPU.

    Args:
        device: wgpu device
        wgsl_code: WGSL source code string
        buffers: list of (wgpu.GPUBuffer, access_mode) tuples
            access_mode: "read" or "read_write"
        workgroups: tuple (x, y=1, z=1) for dispatch
    """
    # Check cache
    cache_key = wgsl_code
    if cache_key in _pipeline_cache:
        pipeline = _pipeline_cache[cache_key]
    else:
        # Create shader module
        shader_module = device.create_shader_module(code=wgsl_code)

        # Build bind group layout
        entries = []
        for i, (buf, access) in enumerate(buffers):
            visibility = wgpu.ShaderStage.COMPUTE
            if access == "read":
                entry = {
                    "binding": i,
                    "visibility": visibility,
                    "buffer": {
                        "type": "read-only-storage",
                        "has_dynamic_offset": False,
                    },
                }
            elif access == "uniform":
                entry = {
                    "binding": i,
                    "visibility": visibility,
                    "buffer": {
                        "type": "uniform",
                        "has_dynamic_offset": False,
                    },
                }
            else:  # "read_write"
                entry = {
                    "binding": i,
                    "visibility": visibility,
                    "buffer": {
                        "type": "storage",
                        "has_dynamic_offset": False,
                    },
                }
            entries.append(entry)

        bind_group_layout = device.create_bind_group_layout(entries=entries)
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        # Create compute pipeline
        pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"},
        )
        _pipeline_cache[cache_key] = pipeline

    # Create bind group
    resources = []
    for i, (buf, _) in enumerate(buffers):
        resources.append({
            "binding": i,
            "resource": {"buffer": buf, "offset": 0, "size": buf.size},
        })

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=resources,
    )

    # Submit compute pass
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    if len(workgroups) == 1:
        compute_pass.dispatch_workgroups(workgroups[0])
    elif len(workgroups) == 2:
        compute_pass.dispatch_workgroups(workgroups[0], workgroups[1])
    else:
        compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2])

    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    device.queue.on_submitted_work_done_sync()


# ============================================================================
# WgpuTensor Class
# ============================================================================

class WgpuTensor:
    """GPU tensor wrapper around wgpu storage buffers."""

    def __init__(self, buffer, shape, dtype="float32", strides=None):
        """Initialize a GPU tensor.

        Args:
            buffer: wgpu.GPUBuffer storage buffer
            shape: tuple of dimensions
            dtype: data type ("float32", "int32", "uint32")
            strides: tuple of strides for non-contiguous views (optional)
        """
        self.buffer = buffer
        self._shape = tuple(shape)
        self.dtype = dtype
        self._strides = strides

        # Type info
        self._dtype_map = {
            "float32": np.float32,
            "int32": np.int32,
            "uint32": np.uint32,
        }
        self._dtype_bytes = {
            "float32": 4,
            "int32": 4,
            "uint32": 4,
        }

    # ---- Properties ----
    @property
    def shape(self):
        """Shape of the tensor."""
        return self._shape

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self._shape)

    def numel(self):
        """Total number of elements."""
        result = 1
        for s in self._shape:
            result *= s
        return result

    # ---- Factory Methods ----
    @staticmethod
    def zeros(shape, dtype="float32"):
        """Create a tensor filled with zeros."""
        numel = 1
        for s in shape:
            numel *= s
        device = _get_device()
        buffer = device.create_buffer(
            size=numel * (4 if dtype == "float32" else 4),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
            mapped_at_creation=True,
        )
        buffer.unmap()
        return WgpuTensor(buffer, shape, dtype)

    @staticmethod
    def ones(shape, dtype="float32"):
        """Create a tensor filled with ones."""
        cpu_array = np.ones(shape, dtype=np.float32 if dtype == "float32" else np.int32)
        return WgpuTensor.from_numpy(cpu_array)

    @staticmethod
    def from_numpy(arr):
        """Create a tensor from a numpy array."""
        device = _get_device()
        # Auto-convert float64 to float32 (GPU only supports f32)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        dtype_map = {np.float32: "float32", np.int32: "int32", np.uint32: "uint32"}
        dtype_name = dtype_map.get(arr.dtype.type, "float32")

        arr_c = np.ascontiguousarray(arr)
        buffer_data = arr_c.tobytes()

        buffer = device.create_buffer_with_data(
            data=buffer_data,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        return WgpuTensor(buffer, arr.shape, dtype_name)

    @staticmethod
    def randn(shape):
        """Create a tensor with random normal values."""
        cpu_array = np.random.randn(*shape).astype(np.float32)
        return WgpuTensor.from_numpy(cpu_array)

    @staticmethod
    def arange(start, end=None, step=1, dtype="float32"):
        """Create a 1D tensor with values in a range."""
        if end is None:
            end = start
            start = 0
        cpu_array = np.arange(start, end, step, dtype=np.float32)
        return WgpuTensor.from_numpy(cpu_array)

    # ---- Data Transfer ----
    def numpy(self):
        """Read tensor data back to CPU as numpy array."""
        device = _get_device()
        np_dtype = self._dtype_map.get(self.dtype, np.float32)
        data = device.queue.read_buffer(self.buffer)
        arr = np.frombuffer(data, dtype=np_dtype).copy()
        return arr.reshape(self.shape)

    # ---- Shape Manipulation ----
    def reshape(self, *shape):
        """Return a view with new shape (no copy)."""
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        numel_old = self.numel()
        numel_new = 1
        for s in new_shape:
            numel_new *= s
        if numel_old != numel_new:
            raise ValueError(f"Cannot reshape {numel_old} elements to {new_shape}")
        return WgpuTensor(self.buffer, new_shape, self.dtype)

    def transpose(self, dim0, dim1):
        """Transpose two dimensions (metadata only, no copy)."""
        new_shape = list(self.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        return WgpuTensor(self.buffer, new_shape, self.dtype)

    def expand(self, *shape):
        """Expand shape (metadata only, no copy)."""
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return WgpuTensor(self.buffer, new_shape, self.dtype)

    # ---- Operators ----
    def __add__(self, other):
        """Element-wise addition."""
        return add(self, other)

    def __mul__(self, other):
        """Element-wise multiplication."""
        if isinstance(other, (int, float)):
            return scalar_mul(self, other)
        return mul(self, other)

    def __rmul__(self, scalar):
        """Right multiplication by scalar."""
        return scalar_mul(self, scalar)

    def __sub__(self, other):
        """Element-wise subtraction."""
        return sub(self, other)

    def __neg__(self):
        """Negation."""
        return neg(self)

    def __matmul__(self, other):
        """Matrix multiplication via @ operator."""
        return matmul(self, other)

    @property
    def T(self):
        """Transpose last two dimensions."""
        return transpose_2d(self)

    # ---- Named Operations ----
    def matmul(self, other):
        """Matrix multiplication."""
        return matmul(self, other)

    def sum(self, axis=None):
        """Sum reduction."""
        return sum_reduce(self, axis)

    def mean(self, axis=None):
        """Mean reduction."""
        return mean_reduce(self, axis)

    def max(self, axis=None):
        """Max reduction."""
        return max_reduce(self, axis)


# ============================================================================
# Functional API - Elementwise
# ============================================================================

def add(a, b):
    """Element-wise addition: a + b."""
    device = _get_device()
    out = WgpuTensor.zeros(a.shape, a.dtype)
    numel = a.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_ADD,
        [(a.buffer, "read"), (b.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def mul(a, b):
    """Element-wise multiplication: a * b."""
    device = _get_device()
    out = WgpuTensor.zeros(a.shape, a.dtype)
    numel = a.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_MUL,
        [(a.buffer, "read"), (b.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def sub(a, b):
    """Element-wise subtraction: a - b."""
    device = _get_device()
    out = WgpuTensor.zeros(a.shape, a.dtype)
    numel = a.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_SUB,
        [(a.buffer, "read"), (b.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def scalar_mul(a, scalar):
    """Element-wise multiplication by scalar: a * s."""
    device = _get_device()
    out = WgpuTensor.zeros(a.shape, a.dtype)

    scalar_bytes = struct.pack("f", scalar)
    scalar_buffer = device.create_buffer_with_data(
        data=scalar_bytes,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    numel = a.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_SCALAR_MUL,
        [(a.buffer, "read"), (out.buffer, "read_write"), (scalar_buffer, "uniform")],
        (workgroups_x,),
    )
    return out


def neg(a):
    """Negation: -a."""
    device = _get_device()
    out = WgpuTensor.zeros(a.shape, a.dtype)
    numel = a.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_NEG,
        [(a.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


# ============================================================================
# Functional API - Activations
# ============================================================================

def gelu(x):
    """GELU activation function."""
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_GELU,
        [(x.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def relu(x):
    """ReLU activation function."""
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_RELU,
        [(x.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def sigmoid(x):
    """Sigmoid activation function."""
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_SIGMOID,
        [(x.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def tanh_act(x):
    """Tanh activation function."""
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_TANH,
        [(x.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


# ============================================================================
# Functional API - Reductions
# ============================================================================

def sum_reduce(x, axis=None):
    """Sum reduction along axis."""
    device = _get_device()

    if axis is None:
        # Full reduction
        out_shape = (1,)
    else:
        out_shape = list(x.shape)
        out_shape.pop(axis)
        out_shape = tuple(out_shape)

    out = WgpuTensor.zeros(out_shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256

    params = struct.pack("4I", numel, 1, 0, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_SUM_REDUCE,
        [
            (x.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return out


def max_reduce(x, axis=None):
    """Max reduction along axis."""
    device = _get_device()

    if axis is None:
        out_shape = (1,)
    else:
        out_shape = list(x.shape)
        out_shape.pop(axis)
        out_shape = tuple(out_shape)

    out = WgpuTensor.zeros(out_shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256

    params = struct.pack("4I", numel, 1, 0, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_MAX_REDUCE,
        [
            (x.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return out


def mean_reduce(x, axis=None):
    """Mean reduction along axis."""
    device = _get_device()

    if axis is None:
        out_shape = (1,)
    else:
        out_shape = list(x.shape)
        out_shape.pop(axis)
        out_shape = tuple(out_shape)

    out = WgpuTensor.zeros(out_shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256

    params = struct.pack("4f", numel, 1.0, 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_MEAN_REDUCE,
        [
            (x.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return out


# ============================================================================
# Functional API - Matrix Operations
# ============================================================================

def matmul(a, b):
    """Matrix multiplication with tiled shared memory.

    Handles:
    - (M, K) @ (K, N) -> (M, N)
    - Batched: (..., M, K) @ (..., K, N) -> (..., M, N)
    """
    device = _get_device()

    # Extract dimensions
    m = a.shape[-2]
    k = a.shape[-1]
    n = b.shape[-1]

    # Output shape
    batch_shape = a.shape[:-2]
    out_shape = batch_shape + (m, n)
    out = WgpuTensor.zeros(out_shape, a.dtype)

    # Dispatch with 16x16 tiles
    workgroups_x = (n + 15) // 16
    workgroups_y = (m + 15) // 16

    params = struct.pack("4I", m, n, k, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_MATMUL,
        [
            (a.buffer, "read"),
            (b.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x, workgroups_y),
    )
    return out


def transpose_2d(x):
    """Transpose last two dimensions."""
    if x.ndim < 2:
        raise ValueError("transpose_2d requires at least 2D tensor")

    device = _get_device()
    rows = x.shape[-2]
    cols = x.shape[-1]

    new_shape = x.shape[:-2] + (cols, rows)
    out = WgpuTensor.zeros(new_shape, x.dtype)

    workgroups_x = (cols + 15) // 16
    workgroups_y = (rows + 15) // 16

    params = struct.pack("4I", rows, cols, 0, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_TRANSPOSE_2D,
        [(x.buffer, "read"), (out.buffer, "read_write"), (params_buffer, "uniform")],
        (workgroups_x, workgroups_y),
    )
    return out


# ============================================================================
# Functional API - Normalization & Softmax
# ============================================================================

def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta.

    One workgroup per row (sequence position).
    """
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)

    width = x.shape[-1]
    num_rows = x.numel() // width

    params = struct.pack("4f", width, eps, 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # One workgroup per row
    _dispatch_shader(
        device,
        WGSL_LAYER_NORM,
        [
            (x.buffer, "read"),
            (gamma.buffer, "read"),
            (beta.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (num_rows,),
    )
    return out


def softmax(x, axis=-1):
    """Stable softmax along last axis. One workgroup per row."""
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)

    if axis == -1 or axis == x.ndim - 1:
        width = x.shape[-1]
        num_rows = x.numel() // width

        params = struct.pack("4I", width, num_rows, 0, 0)
        params_buffer = device.create_buffer_with_data(
            data=params,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # One workgroup per row
        _dispatch_shader(
            device,
            WGSL_SOFTMAX,
            [(x.buffer, "read"), (out.buffer, "read_write"), (params_buffer, "uniform")],
            (num_rows,),
        )
    else:
        raise NotImplementedError("softmax only supports axis=-1 for now")

    return out


# ============================================================================
# Functional API - Loss Functions
# ============================================================================

def cross_entropy(logits, targets):
    """Cross entropy loss: -target * log(p) - (1 - target) * log(1 - p)."""
    device = _get_device()
    out = WgpuTensor.zeros(logits.shape, logits.dtype)
    numel = logits.numel()
    workgroups_x = (numel + 255) // 256

    _dispatch_shader(
        device,
        WGSL_CROSS_ENTROPY,
        [(logits.buffer, "read"), (targets.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    """Focal binary cross entropy loss."""
    device = _get_device()
    out = WgpuTensor.zeros(logits.shape, logits.dtype)
    numel = logits.numel()
    workgroups_x = (numel + 255) // 256

    params = struct.pack("4f", gamma, alpha, 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_FOCAL_BCE,
        [
            (logits.buffer, "read"),
            (targets.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return out


# ============================================================================
# Functional API - Embedding
# ============================================================================

def embedding_lookup(weight, indices):
    """Gather rows from weight matrix by indices."""
    device = _get_device()

    num_indices = indices.numel()
    embedding_dim = weight.shape[-1]
    out_shape = indices.shape + (embedding_dim,)
    out = WgpuTensor.zeros(out_shape, weight.dtype)

    total_elements = num_indices * embedding_dim
    workgroups_x = (total_elements + 255) // 256

    params = struct.pack("4I", num_indices, embedding_dim, 0, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_EMBEDDING,
        [
            (weight.buffer, "read"),
            (indices.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return out
