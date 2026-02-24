#!/bin/bash
set -e

# Build patched wgpu-native to allow non-compliant Vulkan adapters
#
# Problem: wgpu-native hides Vulkan adapters where conformance_version.major == 0.
# The Mesa dozen driver (Vulkan-on-D3D12 translation for WSL2) reports
# conformance_version.major=0, causing the Adreno X1-85 GPU to be hidden.
#
# Solution: Patch wgpu-native to always set the ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER
# flag in InstanceFlags, which makes wgpu emit a warning instead of hiding the adapter.
#
# Applies to: wgpu-native v27.0.2.0 (compatible with wgpu-py 0.30.0)
#
# Prerequisites:
#   - Rust toolchain (rustup): https://rustup.rs
#   - Git with submodule support
#   - ~2GB disk space for build artifacts

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/../patches"
PATCH_FILE="$PATCH_DIR/wgpu-native-noncompliant-adapter.patch"

WGPU_VERSION="v27.0.2.0"
BUILD_DIR="${WGPU_BUILD_DIR:-/tmp/wgpu-native-build}"
INSTALL_TARGET="${WGPU_INSTALL_TARGET:-}"

echo "=== Building patched wgpu-native $WGPU_VERSION ==="
echo "  Build dir: $BUILD_DIR"

# Check prerequisites
if ! command -v cargo &>/dev/null && ! command -v "$HOME/.cargo/bin/cargo" &>/dev/null; then
    echo "ERROR: Rust toolchain not found. Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
CARGO="${CARGO:-$(command -v cargo 2>/dev/null || echo "$HOME/.cargo/bin/cargo")}"
echo "  Cargo: $CARGO ($($CARGO --version))"

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Patch file not found: $PATCH_FILE"
    exit 1
fi

# Clone or reuse
if [ -d "$BUILD_DIR/.git" ]; then
    echo "  Reusing existing clone at $BUILD_DIR"
    cd "$BUILD_DIR"
    git checkout -- . 2>/dev/null || true
else
    echo "  Cloning wgpu-native..."
    git clone https://github.com/gfx-rs/wgpu-native.git "$BUILD_DIR"
    cd "$BUILD_DIR"
fi

# Checkout correct version
echo "  Checking out $WGPU_VERSION..."
git fetch --tags 2>/dev/null || true
git checkout "$WGPU_VERSION" 2>/dev/null
git submodule update --init --recursive

# Apply patch
echo "  Applying noncompliant adapter patch..."
git apply "$PATCH_FILE"

# Build
echo "  Building (release mode)..."
WGPU_NATIVE_VERSION="${WGPU_VERSION#v}" $CARGO build --release

# Verify version
OUTPUT="$BUILD_DIR/target/release/libwgpu_native.so"
if [ ! -f "$OUTPUT" ]; then
    echo "ERROR: Build output not found: $OUTPUT"
    exit 1
fi

echo "  Built: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"

# Install if target specified
if [ -n "$INSTALL_TARGET" ]; then
    if [ -f "$INSTALL_TARGET" ]; then
        BACKUP="${INSTALL_TARGET}.bak"
        if [ ! -f "$BACKUP" ]; then
            echo "  Backing up original to $BACKUP"
            cp "$INSTALL_TARGET" "$BACKUP"
        fi
    fi
    echo "  Installing to $INSTALL_TARGET"
    cp "$OUTPUT" "$INSTALL_TARGET"
fi

# Auto-detect wgpu-py install location
if [ -z "$INSTALL_TARGET" ]; then
    WGPU_SO=$(python3 -c "import wgpu; import pathlib; print(pathlib.Path(wgpu.__file__).parent / 'resources' / 'libwgpu_native-release.so')" 2>/dev/null || true)
    if [ -n "$WGPU_SO" ] && [ -f "$WGPU_SO" ]; then
        echo ""
        echo "  Detected wgpu-py library at: $WGPU_SO"
        echo "  To install, run:"
        echo "    cp $WGPU_SO ${WGPU_SO}.bak  # backup"
        echo "    cp $OUTPUT $WGPU_SO          # install"
    fi
fi

echo ""
echo "=== Build complete ==="
echo "  Library: $OUTPUT"
echo "  Version: ${WGPU_VERSION#v}"
echo "  Patch: ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER always enabled"
