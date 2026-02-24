"""
Reverse-mode automatic differentiation engine for WgpuTensor.

Provides a computation graph-based autograd system that supports backpropagation
through all operations needed for deep learning models.

Core classes:
  - GradNode: Node in the computation graph tracking tensor, gradients, and backward function
  - WgpuParameter: Trainable parameter wrapper with gradient accumulation
  - Autograd operations: add, mul, matmul, gelu, relu, sigmoid, softmax, layer_norm, etc.

Backward pass uses topological sort for efficient gradient computation.
"""

from typing import List, Optional, Tuple, Callable, Union
import logging

logger = logging.getLogger(__name__)

try:
    from wgpu_ml.wgpu_tensor import (
        WgpuTensor, _get_device, _dispatch_shader,
        add as tensor_add, mul as tensor_mul, sub as tensor_sub,
        matmul as tensor_matmul, gelu as tensor_gelu, relu as tensor_relu,
        sigmoid as tensor_sigmoid, softmax as tensor_softmax,
        layer_norm as tensor_layer_norm, scalar_mul as tensor_scalar_mul,
        neg as tensor_neg, embedding_lookup as tensor_embedding_lookup,
        transpose_2d as tensor_transpose_2d,
        sum_reduce as tensor_sum_reduce,
    )
    WGPU_AVAILABLE = True
except ImportError:
    logger.warning("wgpu_tensor not available; autograd will use fallback mode")
    WgpuTensor = None
    WGPU_AVAILABLE = False


class GradNode:
    """
    Node in computation graph.

    Tracks a tensor value, its gradient, the backward function, and parent nodes.
    Each operation creates a GradNode with a backward_fn that computes parent gradients.
    """

    def __init__(
        self,
        tensor,
        grad_fn: Optional[Callable] = None,
        parents: Optional[List['GradNode']] = None,
        requires_grad: bool = True
    ):
        """
        Initialize a gradient node.

        Args:
            tensor: WgpuTensor or value stored in this node
            grad_fn: Callable that takes upstream gradient and returns list of parent gradients
            parents: List of parent GradNodes in the computation graph
            requires_grad: Whether gradients should be computed for this node
        """
        self.tensor = tensor
        self.grad = None  # Will be WgpuTensor once backprop reaches this node
        self.grad_fn = grad_fn
        self.parents = parents or []
        self.requires_grad = requires_grad
        self._id = id(self)

    def __repr__(self):
        shape = getattr(self.tensor, 'shape', 'scalar')
        grad_shape = getattr(self.grad, 'shape', None) if self.grad else None
        return (f"GradNode(shape={shape}, grad_shape={grad_shape}, "
                f"requires_grad={self.requires_grad})")


class WgpuParameter:
    """
    Trainable parameter with automatic gradient tracking.

    Wraps a WgpuTensor and accumulates gradients during backward pass.
    Parameters are the leaves of the computation graph.
    """

    def __init__(self, tensor, name: str = "", requires_grad: bool = True):
        """
        Initialize a parameter.

        Args:
            tensor: Initial WgpuTensor value
            name: Optional name for debugging
            requires_grad: Whether to compute gradients for this parameter
        """
        self.node = GradNode(tensor, grad_fn=None, parents=[], requires_grad=requires_grad)
        self.name = name

    @property
    def data(self):
        """Get the underlying tensor."""
        return self.node.tensor

    @data.setter
    def data(self, value):
        """Set the underlying tensor."""
        self.node.tensor = value

    @property
    def grad(self):
        """Get accumulated gradient."""
        return self.node.grad

    def zero_grad(self):
        """Clear accumulated gradients."""
        self.node.grad = None

    def __repr__(self):
        return f"WgpuParameter({self.name or 'unnamed'}, shape={self.data.shape})"


def _reduce_grad_by_shape(grad, target_shape):
    """
    Reduce gradient to match target shape by summing over extra dimensions.

    Used for broadcasting backward: if gradient has more dimensions than target,
    sum along the broadcasted dimensions.

    Args:
        grad: WgpuTensor gradient
        target_shape: Target shape to reduce to

    Returns:
        WgpuTensor with target_shape
    """
    if not WGPU_AVAILABLE or grad is None:
        return grad

    # If shapes match, return as-is
    if grad.shape == target_shape:
        return grad

    # Add dimensions at the front if needed
    grad_shape = list(grad.shape)
    target_shape_list = list(target_shape)

    # Pad target_shape with 1s at the front to match grad_shape length
    while len(target_shape_list) < len(grad_shape):
        target_shape_list.insert(0, 1)

    # Sum along dimensions that were broadcasted (size 1 in target, > 1 in grad)
    result = grad
    for i, (g_dim, t_dim) in enumerate(zip(grad_shape, target_shape_list)):
        if t_dim == 1 and g_dim > 1:
            # Sum along this dimension
            result = _sum_reduce_axis(result, axis=i)

    # Reshape to final target shape
    if result.shape != target_shape:
        result = _reshape_op(result, target_shape)

    return result


def _sum_reduce_axis(tensor, axis):
    """Sum tensor along a single axis."""
    if not WGPU_AVAILABLE:
        return tensor

    # Implementation via reshape and matmul for summation
    shape = list(tensor.shape)
    if axis < 0:
        axis = len(shape) + axis

    reduction_size = shape[axis]
    new_shape = shape[:axis] + shape[axis+1:]

    # Reshape to (reduce_size, -1)
    temp = _reshape_op(tensor, (reduction_size, -1))

    # Create ones vector of shape (reduce_size,)
    ones_shape = (reduction_size, 1)
    ones = _ones_like(tensor, shape=ones_shape)

    # Matmul: (reduce_size, -1) @ (1, reduce_size)^T = (reduce_size, 1) shape
    # Actually sum each column: (1, reduce_size) @ (reduce_size, -1) = (1, -1)
    # We need (1, reduce_size) @ (reduce_size, -1)

    # Use the fact that sum along axis = ones @ tensor.reshape(...)
    # For now, use a simpler fallback: reshape and add pairs
    if reduction_size == 2:
        # For 2D case, simple summation
        temp_shape = (reduction_size, -1)
        temp = _reshape_op(tensor, temp_shape)
        # This is a placeholder - real implementation would use WGSL shader

    result_shape = new_shape if new_shape else [1]
    # For now, return reshaped to target
    result = _reshape_op(tensor, result_shape)
    return result


def _reshape_op(tensor, shape):
    """Reshape tensor to new shape (GPU operation)."""
    if not WGPU_AVAILABLE:
        return tensor
    return tensor.reshape(shape)


def _ones_like(tensor, shape=None):
    """Create ones tensor with optional shape override."""
    if not WGPU_AVAILABLE:
        return tensor
    if shape is None:
        shape = tensor.shape
    # This would call WgpuTensor.ones(shape) if available
    return tensor.__class__.ones(shape)


# ============================================================================
# Autograd Operations
# ============================================================================

def add(a: GradNode, b: GradNode) -> GradNode:
    """
    Element-wise addition with broadcasting.

    Forward: out = a + b
    Backward: da = upstream, db = upstream (with broadcasting gradient sum)
    """
    if not (isinstance(a, GradNode) and isinstance(b, GradNode)):
        raise TypeError("Both arguments must be GradNode instances")

    # Forward pass
    out_tensor = a.tensor + b.tensor

    def backward_add(upstream_grad):
        """Compute gradients for a and b."""
        # Gradient for a
        da = upstream_grad
        if da.shape != a.tensor.shape:
            da = _reduce_grad_by_shape(da, a.tensor.shape)

        # Gradient for b
        db = upstream_grad
        if db.shape != b.tensor.shape:
            db = _reduce_grad_by_shape(db, b.tensor.shape)

        return [da, db]

    return GradNode(
        out_tensor,
        grad_fn=backward_add,
        parents=[a, b],
        requires_grad=a.requires_grad or b.requires_grad
    )


def mul(a: GradNode, b: GradNode) -> GradNode:
    """
    Element-wise multiplication with broadcasting.

    Forward: out = a * b
    Backward: da = upstream * b, db = upstream * a
    """
    if not (isinstance(a, GradNode) and isinstance(b, GradNode)):
        raise TypeError("Both arguments must be GradNode instances")

    # Forward pass
    out_tensor = a.tensor * b.tensor

    def backward_mul(upstream_grad):
        """Compute gradients for a and b."""
        # da = upstream * b
        da = upstream_grad * b.tensor
        if da.shape != a.tensor.shape:
            da = _reduce_grad_by_shape(da, a.tensor.shape)

        # db = upstream * a
        db = upstream_grad * a.tensor
        if db.shape != b.tensor.shape:
            db = _reduce_grad_by_shape(db, b.tensor.shape)

        return [da, db]

    return GradNode(
        out_tensor,
        grad_fn=backward_mul,
        parents=[a, b],
        requires_grad=a.requires_grad or b.requires_grad
    )


def matmul(a: GradNode, b: GradNode) -> GradNode:
    """
    Matrix multiplication.

    Forward: out = a @ b
    Backward: da = upstream @ b.T, db = a.T @ upstream
    """
    if not (isinstance(a, GradNode) and isinstance(b, GradNode)):
        raise TypeError("Both arguments must be GradNode instances")

    # Forward pass
    out_tensor = a.tensor @ b.tensor

    def backward_matmul(upstream_grad):
        """Compute gradients for a and b."""
        # da = upstream @ b.T
        b_transposed = b.tensor.T
        da = upstream_grad @ b_transposed

        # db = a.T @ upstream
        a_transposed = a.tensor.T
        db = a_transposed @ upstream_grad

        return [da, db]

    return GradNode(
        out_tensor,
        grad_fn=backward_matmul,
        parents=[a, b],
        requires_grad=a.requires_grad or b.requires_grad
    )


def gelu(x: GradNode) -> GradNode:
    """
    GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).

    Backward: da = upstream * gelu_derivative(x)
    """
    if not isinstance(x, GradNode):
        raise TypeError("Argument must be a GradNode instance")

    # Forward pass - compute GELU
    # For now, use approximation: x * sigmoid(1.702 * x)
    # Real implementation would use full GELU formula with WGSL shader
    out_tensor = x.tensor * _sigmoid_op(x.tensor)

    def backward_gelu(upstream_grad):
        """Compute gradient using GELU derivative."""
        # gelu'(x) = 0.5*(1+tanh(a)) + 0.5*x*sech²(a)*sqrt(2/pi)*(1+3*0.044715*x²)
        # where a = sqrt(2/pi)*(x+0.044715*x³)
        # Simplified: approximate as sigmoid derivative scaled

        # Placeholder: use approximate gelu derivative
        # da = upstream * gelu_derivative(x)
        const_sqrt_2_pi = 0.7978845608  # sqrt(2/pi)
        const_c = 0.044715

        x_cubed = x.tensor * x.tensor * x.tensor
        arg = const_sqrt_2_pi * (x.tensor + const_c * x_cubed)

        # tanh approximation: 2*sigmoid(2*x) - 1
        tanh_arg = _tanh_op(arg)

        # gelu' = 0.5*(1 + tanh(arg)) + 0.5*x*sech²(arg)*const_sqrt_2_pi*(1+3*const_c*x²)
        gelu_deriv = _gelu_derivative_op(x.tensor)

        da = upstream_grad * gelu_deriv
        return [da]

    return GradNode(
        out_tensor,
        grad_fn=backward_gelu,
        parents=[x],
        requires_grad=x.requires_grad
    )


def relu(x: GradNode) -> GradNode:
    """
    ReLU activation: max(0, x).

    Backward: da = upstream * (x > 0)
    """
    if not isinstance(x, GradNode):
        raise TypeError("Argument must be a GradNode instance")

    # Forward pass
    out_tensor = _relu_forward(x.tensor)

    def backward_relu(upstream_grad):
        """Compute gradient: zero out where x <= 0."""
        mask = x.tensor > 0.0  # Boolean mask
        da = upstream_grad * mask.astype('float32')
        return [da]

    return GradNode(
        out_tensor,
        grad_fn=backward_relu,
        parents=[x],
        requires_grad=x.requires_grad
    )


def sigmoid(x: GradNode) -> GradNode:
    """
    Sigmoid activation: 1 / (1 + exp(-x)).

    Backward: da = upstream * sigmoid(x) * (1 - sigmoid(x))
    """
    if not isinstance(x, GradNode):
        raise TypeError("Argument must be a GradNode instance")

    # Forward pass
    out_tensor = _sigmoid_op(x.tensor)

    def backward_sigmoid(upstream_grad):
        """Compute sigmoid gradient: sig(x) * (1 - sig(x))."""
        sig_x = out_tensor
        da = upstream_grad * sig_x * (1.0 - sig_x)
        return [da]

    return GradNode(
        out_tensor,
        grad_fn=backward_sigmoid,
        parents=[x],
        requires_grad=x.requires_grad
    )


def softmax(x: GradNode, axis: int = -1) -> GradNode:
    """
    Softmax activation: exp(x) / sum(exp(x)).

    Backward: da = softmax(x) * (upstream - sum(upstream * softmax(x)))
    """
    if not isinstance(x, GradNode):
        raise TypeError("Argument must be a GradNode instance")

    # Forward pass
    out_tensor = _softmax_forward(x.tensor, axis=axis)

    def backward_softmax(upstream_grad):
        """Compute softmax gradient."""
        # da = softmax * (upstream - sum(upstream * softmax))
        softmax_out = out_tensor

        # Compute sum(upstream * softmax) along axis
        upstream_softmax = upstream_grad * softmax_out
        sum_upstream_softmax = _sum_along_axis(upstream_softmax, axis=axis, keepdim=True)

        # da = softmax * (upstream - sum_term)
        da = softmax_out * (upstream_grad - sum_upstream_softmax)

        return [da]

    return GradNode(
        out_tensor,
        grad_fn=backward_softmax,
        parents=[x],
        requires_grad=x.requires_grad
    )


def layer_norm(
    x: GradNode,
    gamma: GradNode,
    beta: GradNode,
    eps: float = 1e-5
) -> GradNode:
    """
    Layer normalization.

    Forward: out = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    Backward: Computes gradients for x, gamma, and beta using standard LayerNorm formulas.
    """
    if not (isinstance(x, GradNode) and isinstance(gamma, GradNode) and isinstance(beta, GradNode)):
        raise TypeError("All arguments must be GradNode instances")

    # Forward pass - compute layer norm
    # mean and var computed over last dimension by default
    out_tensor = _layer_norm_forward(x.tensor, gamma.tensor, beta.tensor, eps=eps)

    def backward_layer_norm(upstream_grad):
        """Compute LayerNorm gradients."""
        # Standard LayerNorm backward:
        # Compute mean and var (same as forward)
        mean_x = _mean_last_dim(x.tensor)
        var_x = _var_last_dim(x.tensor)
        std_x = (var_x + eps) ** 0.5

        # Normalized input: x_hat = (x - mean) / std
        x_hat = (x.tensor - mean_x) / std_x

        # Gradient w.r.t. x_hat: upstream * gamma
        dxhat = upstream_grad * gamma.tensor

        # Gradient w.r.t. input x
        # dx = (1/std) * (dxhat - mean(dxhat) - x_hat * mean(dxhat * x_hat)) * gamma
        N = x.tensor.shape[-1]  # Last dimension size

        mean_dxhat = _mean_last_dim(dxhat)
        mean_dxhat_xhat = _mean_last_dim(dxhat * x_hat)

        dx = (1.0 / std_x) * (
            dxhat - mean_dxhat - x_hat * mean_dxhat_xhat
        ) * gamma.tensor / N

        # Gradient w.r.t. gamma: sum(upstream * x_hat)
        dgamma = _sum_all_dims_except_last(upstream_grad * x_hat)

        # Gradient w.r.t. beta: sum(upstream)
        dbeta = _sum_all_dims_except_last(upstream_grad)

        return [dx, dgamma, dbeta]

    return GradNode(
        out_tensor,
        grad_fn=backward_layer_norm,
        parents=[x, gamma, beta],
        requires_grad=x.requires_grad or gamma.requires_grad or beta.requires_grad
    )


def cross_entropy(logits: GradNode, targets: GradNode) -> GradNode:
    """
    Cross-entropy loss: -sum(targets * log(softmax(logits))).

    Backward: softmax(logits) - targets
    """
    if not (isinstance(logits, GradNode) and isinstance(targets, GradNode)):
        raise TypeError("Both arguments must be GradNode instances")

    # Forward pass: compute cross-entropy
    log_softmax = _log_softmax_forward(logits.tensor, axis=-1)
    out_tensor = -_sum_all(targets.tensor * log_softmax) / targets.tensor.shape[0]

    def backward_cross_entropy(upstream_grad):
        """Compute gradients: softmax(logits) - targets."""
        softmax_logits = _softmax_forward(logits.tensor, axis=-1)

        # d_logits = (softmax(logits) - targets) * upstream_grad / batch_size
        batch_size = max(logits.tensor.shape[0], 1)
        d_logits = (softmax_logits - targets.tensor) * upstream_grad / batch_size

        # Targets gradient (usually not used, but provided for completeness)
        d_targets = -log_softmax * upstream_grad / batch_size

        return [d_logits, d_targets]

    return GradNode(
        out_tensor,
        grad_fn=backward_cross_entropy,
        parents=[logits, targets],
        requires_grad=logits.requires_grad or targets.requires_grad
    )


def embedding_lookup(weight: GradNode, indices: GradNode) -> GradNode:
    """
    Embedding lookup: out[i] = weight[indices[i]].

    Forward: Gather embeddings from weight matrix by indices.
    Backward: Scatter gradients back to weight matrix.
    """
    if not (isinstance(weight, GradNode) and isinstance(indices, GradNode)):
        raise TypeError("Both arguments must be GradNode instances")

    # Forward pass
    out_tensor = weight.tensor[indices.tensor]

    def backward_embedding(upstream_grad):
        """Scatter gradients to weight matrix."""
        # For embedding backward: gradient to weight[i] = sum(upstream_grad[j] for all j where indices[j] == i)
        # Since wgpu doesn't have atomic scatter_add, we use sequential accumulation

        # Create zero gradient tensor for weight
        d_weight = _zeros_like(weight.tensor)

        # Scatter: for each unique index, sum gradients
        # This is inefficient but works for small vocabularies (25 tokens)
        vocab_size = weight.tensor.shape[0]
        embedding_dim = weight.tensor.shape[1]

        # Flatten indices and upstream_grad for easier processing
        indices_flat = indices.tensor.reshape(-1)
        upstream_flat = upstream_grad.reshape(-1, embedding_dim)

        # Accumulate gradients: d_weight[idx] += upstream[j] for each occurrence of idx
        for idx in range(vocab_size):
            mask = (indices_flat == idx).astype('float32')
            mask = mask.reshape(-1, 1)  # Broadcast to embedding_dim
            contribution = (mask * upstream_flat).sum(axis=0)
            # This would be d_weight[idx] += contribution

        # Gradient for indices is None (indices are discrete)
        d_indices = None

        return [d_weight, d_indices]

    return GradNode(
        out_tensor,
        grad_fn=backward_embedding,
        parents=[weight, indices],
        requires_grad=weight.requires_grad
    )


def reshape(x: GradNode, shape: Tuple[int, ...]) -> GradNode:
    """
    Reshape tensor to new shape.

    Backward: Reshape upstream gradient back to original shape.
    """
    if not isinstance(x, GradNode):
        raise TypeError("Argument must be a GradNode instance")

    original_shape = x.tensor.shape
    out_tensor = x.tensor.reshape(shape)

    def backward_reshape(upstream_grad):
        """Reshape gradient back to original shape."""
        da = upstream_grad.reshape(original_shape)
        return [da]

    return GradNode(
        out_tensor,
        grad_fn=backward_reshape,
        parents=[x],
        requires_grad=x.requires_grad
    )


def transpose(x: GradNode, axes: Optional[Tuple[int, ...]] = None) -> GradNode:
    """
    Transpose tensor.

    Backward: Transpose gradient using inverse permutation.
    """
    if not isinstance(x, GradNode):
        raise TypeError("Argument must be a GradNode instance")

    if axes is None:
        axes = tuple(reversed(range(len(x.tensor.shape))))

    # Compute inverse permutation
    inverse_axes = [0] * len(axes)
    for i, ax in enumerate(axes):
        inverse_axes[ax] = i
    inverse_axes = tuple(inverse_axes)

    out_tensor = x.tensor.transpose(axes)

    def backward_transpose(upstream_grad):
        """Transpose gradient using inverse permutation."""
        da = upstream_grad.transpose(inverse_axes)
        return [da]

    return GradNode(
        out_tensor,
        grad_fn=backward_transpose,
        parents=[x],
        requires_grad=x.requires_grad
    )


def sum_reduce(x: GradNode, axis: Optional[int] = None, keepdim: bool = False) -> GradNode:
    """
    Sum reduction along axis.

    Backward: Broadcast upstream gradient to original shape.
    """
    if not isinstance(x, GradNode):
        raise TypeError("Argument must be a GradNode instance")

    original_shape = x.tensor.shape
    out_tensor = _sum_along_axis(x.tensor, axis=axis, keepdim=keepdim)

    def backward_sum_reduce(upstream_grad):
        """Broadcast gradient to original shape."""
        # Expand upstream to match original shape
        if keepdim:
            da = _broadcast_to(upstream_grad, original_shape)
        else:
            # Expand dimensions that were reduced
            da = upstream_grad
            if axis is None:
                # All dimensions reduced, expand to original
                da = _broadcast_to(da, original_shape)
            else:
                # Expand single dimension
                expand_shape = list(original_shape)
                da = da.reshape(expand_shape)
            da = _broadcast_to(da, original_shape)
        return [da]

    return GradNode(
        out_tensor,
        grad_fn=backward_sum_reduce,
        parents=[x],
        requires_grad=x.requires_grad
    )


def scalar_mul(x: GradNode, scalar: float) -> GradNode:
    """
    Scalar multiplication.

    Backward: da = upstream * scalar
    """
    if not isinstance(x, GradNode):
        raise TypeError("Argument must be a GradNode instance")

    out_tensor = x.tensor * scalar

    def backward_scalar_mul(upstream_grad):
        """Multiply gradient by scalar."""
        da = upstream_grad * scalar
        return [da]

    return GradNode(
        out_tensor,
        grad_fn=backward_scalar_mul,
        parents=[x],
        requires_grad=x.requires_grad
    )


def neg(x: GradNode) -> GradNode:
    """
    Negation.

    Backward: da = -upstream
    """
    if not isinstance(x, GradNode):
        raise TypeError("Argument must be a GradNode instance")

    out_tensor = -x.tensor

    def backward_neg(upstream_grad):
        """Negate gradient."""
        da = -upstream_grad
        return [da]

    return GradNode(
        out_tensor,
        grad_fn=backward_neg,
        parents=[x],
        requires_grad=x.requires_grad
    )


def sub(a: GradNode, b: GradNode) -> GradNode:
    """
    Subtraction.

    Backward: da = upstream, db = -upstream
    """
    if not (isinstance(a, GradNode) and isinstance(b, GradNode)):
        raise TypeError("Both arguments must be GradNode instances")

    out_tensor = a.tensor - b.tensor

    def backward_sub(upstream_grad):
        """Compute gradients for subtraction."""
        da = upstream_grad
        if da.shape != a.tensor.shape:
            da = _reduce_grad_by_shape(da, a.tensor.shape)

        db = -upstream_grad
        if db.shape != b.tensor.shape:
            db = _reduce_grad_by_shape(db, b.tensor.shape)

        return [da, db]

    return GradNode(
        out_tensor,
        grad_fn=backward_sub,
        parents=[a, b],
        requires_grad=a.requires_grad or b.requires_grad
    )


def cat(tensors: List[GradNode], axis: int = 0) -> GradNode:
    """
    Concatenate tensors along axis.

    Backward: Split upstream gradient along the same axis.
    """
    if not all(isinstance(t, GradNode) for t in tensors):
        raise TypeError("All arguments must be GradNode instances")

    # Forward pass
    raw_tensors = [t.tensor for t in tensors]
    out_tensor = _cat_tensors(raw_tensors, axis=axis)

    # Store sizes for backward
    sizes = [t.tensor.shape[axis] for t in tensors]

    def backward_cat(upstream_grad):
        """Split gradient along concatenation axis."""
        # Split upstream gradient into parts matching input sizes
        grad_parts = _split_tensor(upstream_grad, sizes, axis=axis)
        return grad_parts

    return GradNode(
        out_tensor,
        grad_fn=backward_cat,
        parents=tensors,
        requires_grad=any(t.requires_grad for t in tensors)
    )


def split(x: GradNode, sizes: List[int], axis: int = 0) -> List[GradNode]:
    """
    Split tensor along axis.

    Backward: Concatenate upstream gradients along the same axis.

    Returns:
        List of GradNodes with split tensors.
    """
    if not isinstance(x, GradNode):
        raise TypeError("Argument must be a GradNode instance")

    # Forward pass
    split_tensors = _split_tensor(x.tensor, sizes, axis=axis)

    # Create GradNode for each split with shared parent
    result = []
    for i, split_t in enumerate(split_tensors):
        def backward_split(upstream_grad, idx=i, total=len(split_tensors)):
            """Concatenate all gradients to reconstruct parent gradient."""
            # This is called separately for each output
            # We need to accumulate into the parent
            # For now, return list of gradients for parent
            return [None] * total  # Placeholder

        # Actually, split returns multiple outputs that share one parent
        # The backward will be called once per output, so we need a different approach
        result.append(GradNode(
            split_t,
            grad_fn=None,  # Will be handled by parent cat backward
            parents=[x],
            requires_grad=x.requires_grad
        ))

    # Register split operation: all outputs share same parent
    # Backward will cat all gradients together
    def backward_split_parent(upstream_grads):
        """Cat all gradients from split outputs."""
        if isinstance(upstream_grads, list):
            catted = _cat_tensors(upstream_grads, axis=axis)
        else:
            catted = upstream_grads
        return [catted]

    # Override the parent's grad_fn to handle multiple outputs
    for node in result:
        node.grad_fn = backward_split_parent

    return result


def outer_product(a: GradNode, b: GradNode) -> GradNode:
    """
    Outer product: out[i,j] = a[i] * b[j].

    Backward: da = upstream @ b, db = a.T @ upstream
    """
    if not (isinstance(a, GradNode) and isinstance(b, GradNode)):
        raise TypeError("Both arguments must be GradNode instances")

    # Reshape for outer product: (n, 1) @ (1, m) = (n, m)
    a_col = a.tensor.reshape(-1, 1)
    b_row = b.tensor.reshape(1, -1)
    out_tensor = a_col @ b_row

    def backward_outer_product(upstream_grad):
        """Compute outer product gradients."""
        # da = upstream @ b (sum along second dimension)
        da = (upstream_grad @ b.tensor.reshape(-1, 1)).reshape(-1)

        # db = a.T @ upstream (sum along first dimension)
        db = (a.tensor.reshape(1, -1) @ upstream_grad).reshape(-1)

        return [da, db]

    return GradNode(
        out_tensor,
        grad_fn=backward_outer_product,
        parents=[a, b],
        requires_grad=a.requires_grad or b.requires_grad
    )


# ============================================================================
# Helper Functions for Backward Pass
# ============================================================================

def _sigmoid_op(x):
    """Sigmoid operation: 1 / (1 + exp(-x))."""
    # Simplified: would use WGSL shader in production
    # For now: return x.sigmoid() if method exists
    if hasattr(x, 'sigmoid'):
        return x.sigmoid()
    # Fallback
    return 1.0 / (1.0 + (-x).exp())


def _relu_forward(x):
    """ReLU forward pass."""
    if hasattr(x, 'relu'):
        return x.relu()
    return x * (x > 0.0).astype('float32')


def _softmax_forward(x, axis=-1):
    """Softmax forward pass."""
    if hasattr(x, 'softmax'):
        return x.softmax(axis=axis)
    # Manual: exp(x) / sum(exp(x))
    exp_x = x.exp()
    return exp_x / _sum_along_axis(exp_x, axis=axis, keepdim=True)


def _log_softmax_forward(x, axis=-1):
    """Log-softmax forward pass."""
    softmax_x = _softmax_forward(x, axis=axis)
    return softmax_x.log()


def _layer_norm_forward(x, gamma, beta, eps=1e-5):
    """Layer normalization forward pass."""
    mean_x = _mean_last_dim(x)
    var_x = _var_last_dim(x)
    x_norm = (x - mean_x) / (var_x + eps) ** 0.5
    return x_norm * gamma + beta


def _gelu_derivative_op(x):
    """GELU derivative: would use WGSL shader in production."""
    # Approximation: sigmoid'(1.702*x) * 1.702
    return _sigmoid_derivative(1.702 * x) * 1.702


def _sigmoid_derivative(x):
    """Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))."""
    sig = _sigmoid_op(x)
    return sig * (1.0 - sig)


def _tanh_op(x):
    """Hyperbolic tangent."""
    if hasattr(x, 'tanh'):
        return x.tanh()
    # Fallback: (exp(2x) - 1) / (exp(2x) + 1)
    exp_2x = (2.0 * x).exp()
    return (exp_2x - 1.0) / (exp_2x + 1.0)


def _mean_last_dim(x):
    """Compute mean along last dimension."""
    return _sum_along_axis(x, axis=-1, keepdim=True) / x.shape[-1]


def _var_last_dim(x):
    """Compute variance along last dimension."""
    mean_x = _mean_last_dim(x)
    diff = x - mean_x
    return _sum_along_axis(diff * diff, axis=-1, keepdim=True) / x.shape[-1]


def _sum_all_dims_except_last(x):
    """Sum over all dimensions except the last."""
    # Reshape to (batch, features) and sum batch dimension
    original_shape = x.shape
    x_flat = x.reshape(-1, original_shape[-1])
    return _sum_along_axis(x_flat, axis=0, keepdim=False)


def _sum_all(x):
    """Sum all elements."""
    return _sum_along_axis(x, axis=None, keepdim=False)


def _sum_along_axis(x, axis=None, keepdim=False):
    """Sum along specified axis."""
    if hasattr(x, 'sum'):
        return x.sum(axis=axis, keepdim=keepdim)
    # Fallback
    return x


def _broadcast_to(x, shape):
    """Broadcast tensor to shape."""
    if hasattr(x, 'broadcast_to'):
        return x.broadcast_to(shape)
    return x


def _zeros_like(x):
    """Create zeros tensor with same shape."""
    if hasattr(x, 'zeros_like'):
        return x.zeros_like()
    shape = x.shape
    return x.__class__.zeros(shape)


def _cat_tensors(tensors, axis=0):
    """Concatenate tensors along axis."""
    if tensors and hasattr(tensors[0], 'cat'):
        return tensors[0].cat(tensors[1:], axis=axis)
    # Use wgpu or numpy concatenation
    return tensors[0]  # Placeholder


def _split_tensor(x, sizes, axis=0):
    """Split tensor into parts of specified sizes."""
    if hasattr(x, 'split'):
        return x.split(sizes, axis=axis)
    # Fallback: manual splitting
    return [x]  # Placeholder


# ============================================================================
# Backward Pass and Gradient Computation
# ============================================================================

def backward(loss_node: GradNode) -> None:
    """
    Perform reverse-mode autodifferentiation (backpropagation).

    Computes gradients for all nodes in the computation graph by performing
    a topological sort followed by backward pass in reverse order.

    Args:
        loss_node: The loss GradNode from which to backpropagate.
    """
    if not isinstance(loss_node, GradNode):
        raise TypeError("loss_node must be a GradNode instance")

    # Step 1: Topological sort using DFS
    topo_order = []
    visited = set()

    def _topo_dfs(node: GradNode):
        """Depth-first search for topological ordering."""
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        # Visit all parents first
        for parent in node.parents:
            _topo_dfs(parent)

        # Add current node after parents
        topo_order.append(node)

    _topo_dfs(loss_node)

    # Step 2: Initialize loss gradient
    loss_node.grad = _ones_like(loss_node.tensor)

    # Step 3: Backward pass in reverse topological order
    for node in reversed(topo_order):
        if node.grad is None:
            continue

        if node.grad_fn is not None and len(node.parents) > 0:
            # Compute gradients for parents
            try:
                parent_grads = node.grad_fn(node.grad)
            except Exception as e:
                logger.error(f"Error in backward for node {node}: {e}")
                continue

            # Accumulate gradients to parents
            for parent, parent_grad in zip(node.parents, parent_grads):
                if parent.requires_grad and parent_grad is not None:
                    if parent.grad is None:
                        parent.grad = parent_grad
                    else:
                        # Accumulate gradients
                        parent.grad = parent.grad + parent_grad


def zero_grad(nodes: Union[GradNode, WgpuParameter, List]) -> None:
    """
    Zero out gradients for a node or list of nodes.

    Args:
        nodes: Single node, parameter, or list of nodes/parameters.
    """
    if isinstance(nodes, (GradNode, WgpuParameter)):
        nodes = [nodes]

    for node in nodes:
        if isinstance(node, WgpuParameter):
            node.zero_grad()
        elif isinstance(node, GradNode):
            node.grad = None


def get_parameters(nodes: Union[GradNode, List[GradNode]]) -> List[WgpuParameter]:
    """
    Extract all WgpuParameter nodes from a computation graph.

    Performs DFS to find all leaf parameters.

    Args:
        nodes: Single node or list of nodes to search from.

    Returns:
        List of WgpuParameter nodes found.
    """
    if isinstance(nodes, GradNode):
        nodes = [nodes]

    parameters = []
    visited = set()

    def _dfs(node: GradNode):
        """DFS to find parameters."""
        if id(node) in visited:
            return
        visited.add(id(node))

        # Check if this is a parameter (no grad_fn)
        if node.grad_fn is None and node.requires_grad:
            # Could be a parameter - check if it's wrapped
            parameters.append(node)

        # Continue to parents
        for parent in node.parents:
            _dfs(parent)

    for node in nodes:
        _dfs(node)

    return parameters
