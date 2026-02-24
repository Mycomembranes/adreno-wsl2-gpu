#!/bin/bash
# Environment setup for Adreno X1-85 GPU on WSL2 via Mesa Dozen
#
# Source this file or add to ~/.bashrc:
#   source /path/to/setup_env.sh

export VK_ICD_FILENAMES="$HOME/mesa-install/share/vulkan/icd.d/dzn_icd.aarch64.json"
export LD_LIBRARY_PATH="$HOME/mesa-install/lib/aarch64-linux-gnu:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DISPLAY="${DISPLAY:-:0}"
export MESA_LOG_LEVEL=error          # Suppress libEGL/Zink warnings
export LIBGL_ALWAYS_SOFTWARE=0

echo "Adreno WSL2 GPU environment configured."
echo "  VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
