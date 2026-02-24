#!/bin/bash
# Build Mesa from source with Dozen (Vulkan-over-D3D12) driver for Adreno on WSL2
#
# Prerequisites: meson, ninja, llvm-17+, libdrm-dev, python3, glslang-tools
# See check_prerequisites.sh for full dependency list.

set -euo pipefail

MESA_SRC="${MESA_SRC:-$HOME/mesa}"
MESA_INSTALL="${MESA_INSTALL:-$HOME/mesa-install}"
PATCH_DIR="$(cd "$(dirname "$0")/../patches" && pwd)"
JOBS="${JOBS:-$(nproc)}"

echo "=== Mesa Dozen Build for Adreno WSL2 ==="
echo "Source:  $MESA_SRC"
echo "Install: $MESA_INSTALL"
echo "Jobs:    $JOBS"
echo

# Clone if needed
if [ ! -d "$MESA_SRC" ]; then
    echo ">>> Cloning Mesa..."
    git clone --depth=1 --branch mesa-25.0.5 https://gitlab.freedesktop.org/mesa/mesa.git "$MESA_SRC"
fi

# Apply patch
if [ -f "$PATCH_DIR/mesa-dozen-adreno.patch" ]; then
    echo ">>> Applying Adreno patches..."
    cd "$MESA_SRC"
    git apply --check "$PATCH_DIR/mesa-dozen-adreno.patch" 2>/dev/null && \
        git apply "$PATCH_DIR/mesa-dozen-adreno.patch" || \
        echo "Patch already applied or conflicts — skipping"
fi

# Configure with meson
echo ">>> Configuring Mesa with meson..."
cd "$MESA_SRC"
if [ ! -d builddir ]; then
    meson setup builddir \
        --prefix="$MESA_INSTALL" \
        -Dplatforms=x11 \
        -Dvulkan-drivers=microsoft-experimental,swrast \
        -Dgallium-drivers=swrast,zink \
        -Dglx=xlib \
        -Dbuildtype=release \
        -Dcpp_rtti=false \
        -Db_ndebug=true
else
    echo "builddir exists, reconfiguring..."
    meson configure builddir \
        --prefix="$MESA_INSTALL" \
        -Dplatforms=x11 \
        -Dvulkan-drivers=microsoft-experimental,swrast \
        -Dgallium-drivers=swrast,zink \
        -Dglx=xlib \
        -Dbuildtype=release
fi

# Build
echo ">>> Building Mesa (${JOBS} jobs)..."
ninja -j"$JOBS" -C "$MESA_SRC/builddir"

# Install
echo ">>> Installing to $MESA_INSTALL..."
ninja -C "$MESA_SRC/builddir" install

echo
echo "=== Build complete ==="
echo "Set environment with: source scripts/setup_env.sh"
echo "Verify with: vulkaninfo --summary"
