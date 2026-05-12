#!/usr/bin/env bash
# sanity-check.sh
# Verifies the container is correctly configured.
# Run INSIDE the container after entry-setup.sh.
#
# Usage:
#   bash /workspace/sanity-check.sh
#
# Each check is independent; the script reports all results before exiting.

set -uo pipefail

# Don't use set -e; we want to run all checks and report
ISAAC_SIM_PYTHON="/isaac-sim/python.sh"

PASS=0
FAIL=0
declare -a RESULTS=()

check() {
    local name="$1"
    local result="$2"
    if [ "$result" -eq 0 ]; then
        RESULTS+=("PASS: $name")
        PASS=$((PASS + 1))
    else
        RESULTS+=("FAIL: $name")
        FAIL=$((FAIL + 1))
    fi
}

echo "======================================================================"
echo "  rwmu-cogar sanity check"
echo "======================================================================"
echo

# ---------------------------------------------------------------------------
# 1. nvidia-smi works
# ---------------------------------------------------------------------------
echo "[1/8] nvidia-smi"
nvidia-smi > /tmp/nvidia-smi.out 2>&1
check "nvidia-smi works inside container" $?
echo "    First line of output:"
head -1 /tmp/nvidia-smi.out | sed 's/^/    /'
echo

# ---------------------------------------------------------------------------
# 2. GPU visible to torch
# ---------------------------------------------------------------------------
echo "[2/8] torch sees GPU"
$ISAAC_SIM_PYTHON -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'    torch: {torch.__version__}, cuda_version: {torch.version.cuda}')
print(f'    devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    device {i}: {torch.cuda.get_device_name(i)}')
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / (1024**3)
    print(f'              VRAM: {vram_gb:.1f} GB')
"
check "torch sees GPU" $?
echo

# ---------------------------------------------------------------------------
# 3. Isaac Sim modules importable
# ---------------------------------------------------------------------------
echo "[3/8] isaacsim import"
$ISAAC_SIM_PYTHON -c "
import isaacsim
print(f'    isaacsim: {isaacsim.__version__ if hasattr(isaacsim, \"__version__\") else \"(no __version__ attr)\"}')"
check "isaacsim imports" $?
echo

# ---------------------------------------------------------------------------
# 4. Isaac Lab importable
# ---------------------------------------------------------------------------
echo "[4/8] isaaclab import"
$ISAAC_SIM_PYTHON -c "
import isaaclab
print(f'    isaaclab: {isaaclab.__file__}')"
check "isaaclab imports" $?
echo

# ---------------------------------------------------------------------------
# 5. Project mbrl extension importable
# ---------------------------------------------------------------------------
echo "[5/8] mbrl import (project extension)"
$ISAAC_SIM_PYTHON -c "
import mbrl
print(f'    mbrl: {mbrl.__file__}')"
check "mbrl imports" $?
echo

# ---------------------------------------------------------------------------
# 6. Custom rsl_rl_lib importable
# ---------------------------------------------------------------------------
echo "[6/8] rsl_rl import (project fork)"
$ISAAC_SIM_PYTHON -c "
import rsl_rl
print(f'    rsl_rl: {rsl_rl.__file__}')"
check "rsl_rl imports" $?
echo

# ---------------------------------------------------------------------------
# 7. Data science extras importable
# ---------------------------------------------------------------------------
echo "[7/8] data science extras"
$ISAAC_SIM_PYTHON -c "
import pandas, scipy, matplotlib, wandb, tensorboard, h5py, safetensors
print(f'    pandas {pandas.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__}')
print(f'    wandb {wandb.__version__}, h5py {h5py.__version__}, safetensors {safetensors.__version__}')"
check "data science extras importable" $?
echo

# ---------------------------------------------------------------------------
# 8. Pretrained LFS checkpoint present (needed for Finetune-v0)
# ---------------------------------------------------------------------------
echo "[8/8] pretrained checkpoint (LFS)"
CKPT="/workspace/Uncertainty-Aware-Robotic-World-Model-Go2/upstream/robotic_world_model/assets/models/pretrain_rnn_ens.pt"
if [ -f "$CKPT" ]; then
    SIZE_MB=$(($(stat -c%s "$CKPT") / 1024 / 1024))
    if [ "$SIZE_MB" -gt 10 ]; then
        echo "    Found: $CKPT (${SIZE_MB} MB)"
        check "pretrained checkpoint present and non-trivial" 0
    else
        echo "    Found but suspiciously small: $CKPT (${SIZE_MB} MB) — LFS pull may have failed"
        check "pretrained checkpoint non-trivial size" 1
    fi
else
    echo "    NOT FOUND: $CKPT"
    check "pretrained checkpoint exists" 1
fi
echo

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  Results"
echo "======================================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo "----------------------------------------------------------------------"
echo "  ${PASS} passed, ${FAIL} failed"
echo "======================================================================"

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
