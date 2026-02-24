#!/usr/bin/env python3
"""
Minimal training demo for wgpu_ml.

Trains a 2-layer MLP on synthetic classification data using the Adreno GPU
via wgpu compute shaders. Demonstrates: tensor creation, forward pass,
loss computation, backward pass, and AdamW optimization.

Usage:
    python -m wgpu_ml.examples.train_demo
    # or
    python wgpu_ml/examples/train_demo.py
"""

import sys
import os
import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wgpu_ml.wgpu_tensor import WgpuTensor
from wgpu_ml.wgpu_autograd import (
    GradNode, WgpuParameter, backward, zero_grad,
    add as ag_add, matmul as ag_matmul, gelu as ag_gelu,
    cross_entropy as ag_cross_entropy,
)
from wgpu_ml.wgpu_nn import Linear, Sequential, AdamW


class SimpleMLP:
    """2-layer MLP for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, x: GradNode) -> GradNode:
        h = self.fc1(x)
        h = ag_gelu(h)
        out = self.fc2(h)
        return out

    def parameters(self):
        yield from self.fc1.parameters()
        yield from self.fc2.parameters()


def generate_data(n_samples: int, input_dim: int, n_classes: int, seed: int = 42):
    """Generate synthetic linearly-separable classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim).astype(np.float32)
    # Simple linear boundary
    w_true = rng.randn(input_dim).astype(np.float32)
    scores = X @ w_true
    if n_classes == 2:
        y = (scores > 0).astype(np.int32)
    else:
        thresholds = np.quantile(scores, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.digitize(scores, thresholds).astype(np.int32)
    return X, y


def main():
    print("wgpu_ml Training Demo")
    print("=" * 50)

    # Hyperparameters
    input_dim = 16
    hidden_dim = 32
    n_classes = 4
    n_samples = 256
    batch_size = 64
    n_epochs = 20
    lr = 1e-3

    # Generate data
    X_all, y_all = generate_data(n_samples, input_dim, n_classes)
    print(f"Data: {n_samples} samples, {input_dim} features, {n_classes} classes")

    # Build model
    model = SimpleMLP(input_dim, hidden_dim, n_classes)
    params = list(model.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=0.01)
    print(f"Model: {sum(p.data.to_numpy().size for p in params)} parameters")
    print()

    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_correct = 0
        n_total = 0

        for i in range(0, n_samples, batch_size):
            x_batch = X_all[i:i + batch_size]
            y_batch = y_all[i:i + batch_size]
            bs = x_batch.shape[0]

            # Create GPU tensors
            x_tensor = WgpuTensor.from_numpy(x_batch)
            x_node = GradNode(x_tensor, requires_grad=False)

            # One-hot encode targets for cross entropy
            y_onehot = np.zeros((bs, n_classes), dtype=np.float32)
            y_onehot[np.arange(bs), y_batch] = 1.0
            y_tensor = WgpuTensor.from_numpy(y_onehot)
            y_node = GradNode(y_tensor, requires_grad=False)

            # Forward
            logits = model.forward(x_node)

            # Loss
            loss = ag_cross_entropy(logits, y_node)

            # Backward
            optimizer.zero_grad()
            backward(loss)

            # Update
            optimizer.step()

            # Metrics
            loss_val = loss.tensor.to_numpy().item()
            epoch_loss += loss_val * bs

            # Accuracy
            pred = logits.tensor.to_numpy().argmax(axis=-1)
            n_correct += (pred == y_batch).sum()
            n_total += bs

        avg_loss = epoch_loss / n_total
        accuracy = n_correct / n_total * 100
        print(f"Epoch {epoch + 1:3d}/{n_epochs}  loss={avg_loss:.4f}  acc={accuracy:.1f}%")

    print()
    print("Training complete!")
    print(f"Final accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
