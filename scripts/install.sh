#!/bin/bash
# One-click installer: Adreno X1-85 GPU support on WSL2
#
# This script:
# 1. Checks prerequisites
# 2. Clones and patches Mesa
# 3. Builds Mesa with Dozen driver
# 4. Sets up environment
# 5. Installs Python dependencies
# 6. Runs validation tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  Adreno X1-85 WSL2 GPU — Full Installer"
echo "============================================"
echo

# Step 1: Prerequisites
echo ">>> Step 1/6: Checking prerequisites..."
if ! bash "$SCRIPT_DIR/check_prerequisites.sh"; then
    echo "Please install missing dependencies and re-run."
    exit 1
fi
echo

# Step 2: Build Mesa
echo ">>> Step 2/6: Building Mesa with Dozen driver..."
bash "$SCRIPT_DIR/build_mesa.sh"
echo

# Step 3: Environment
echo ">>> Step 3/6: Setting up environment..."
source "$SCRIPT_DIR/setup_env.sh"

# Add to .bashrc if not already there
MARKER="# Adreno WSL2 GPU"
if ! grep -q "$MARKER" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "$MARKER" >> ~/.bashrc
    echo "source $SCRIPT_DIR/setup_env.sh" >> ~/.bashrc
    echo "Added to ~/.bashrc"
fi
echo

# Step 4: Python deps
echo ">>> Step 4/6: Installing Python dependencies..."
pip install --quiet wgpu numpy Pillow 2>/dev/null || \
    pip install --quiet --user wgpu numpy Pillow
echo

# Step 5: Vulkan verification
echo ">>> Step 5/6: Verifying Vulkan..."
if command -v vulkaninfo &>/dev/null; then
    echo "Vulkan devices:"
    vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|apiVersion" || true
else
    echo "vulkaninfo not found — install vulkan-tools for detailed info"
fi
echo

# Step 6: Run tests
echo ">>> Step 6/6: Running validation tests..."
cd "$REPO_DIR/tests"

echo "--- Compute test (wgpu) ---"
python3 test_wgpu.py && echo "PASS" || echo "FAIL"

echo
echo "--- Graphics test ---"
python3 test_graphics.py && echo "PASS" || echo "FAIL"

echo
echo "============================================"
echo "  Installation complete!"
echo "  Run 'source scripts/setup_env.sh' in new shells"
echo "  or restart your terminal."
echo "============================================"
