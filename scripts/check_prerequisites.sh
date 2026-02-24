#!/bin/bash
# Check prerequisites for building Mesa Dozen on WSL2

set -u

PASS=0
FAIL=0
WARN=0

check() {
    local name="$1" cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo "  [OK]   $name"
        ((PASS++))
    else
        echo "  [FAIL] $name"
        ((FAIL++))
    fi
}

warn() {
    local name="$1" cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo "  [OK]   $name"
        ((PASS++))
    else
        echo "  [WARN] $name (optional)"
        ((WARN++))
    fi
}

echo "=== Adreno WSL2 GPU Prerequisites Check ==="
echo

echo "Build tools:"
check "meson"           "command -v meson"
check "ninja"           "command -v ninja"
check "gcc/g++"         "command -v gcc && command -v g++"
check "pkg-config"      "command -v pkg-config"
check "git"             "command -v git"
check "glslangValidator" "command -v glslangValidator"

echo
echo "LLVM:"
check "llvm-config"     "command -v llvm-config || command -v llvm-config-17 || command -v llvm-config-18"

echo
echo "Libraries:"
check "libdrm-dev"      "pkg-config --exists libdrm"
check "libx11-dev"      "pkg-config --exists x11"
check "libxcb-dev"      "pkg-config --exists xcb"
check "libxrandr-dev"   "pkg-config --exists xrandr"
check "zlib"            "pkg-config --exists zlib"
check "libexpat"        "pkg-config --exists expat"

echo
echo "Python:"
check "python3"         "command -v python3"
check "python3-mako"    "python3 -c 'import mako'"
check "python3-yaml"    "python3 -c 'import yaml'"

echo
echo "WSL2 specific:"
check "/usr/lib/wsl/lib exists" "test -d /usr/lib/wsl/lib"
check "libd3d12.so"     "test -f /usr/lib/wsl/lib/libd3d12.so"
check "libdxcore.so"    "test -f /usr/lib/wsl/lib/libdxcore.so"

echo
echo "Runtime (optional):"
warn "wgpu-py"          "python3 -c 'import wgpu'"
warn "numpy"            "python3 -c 'import numpy'"
warn "PIL/Pillow"       "python3 -c 'from PIL import Image'"
warn "vulkaninfo"       "command -v vulkaninfo"
warn "vkcube"           "command -v vkcube"

echo
echo "=== Results: $PASS passed, $FAIL failed, $WARN warnings ==="
if [ "$FAIL" -gt 0 ]; then
    echo
    echo "Install missing dependencies:"
    echo "  sudo apt install meson ninja-build gcc g++ pkg-config git glslang-tools \\"
    echo "    llvm-17-dev libdrm-dev libx11-dev libxcb1-dev libxrandr-dev \\"
    echo "    zlib1g-dev libexpat1-dev python3 python3-mako python3-yaml"
    echo "  pip install wgpu numpy Pillow"
    exit 1
fi
