#!/usr/bin/env bash
# sanity-check.sh
# Verifies the lab container is correctly configured for this project.
# Run INSIDE the container after entry-setup.sh:
#
#   cd /workspace/Uncertainty-Aware-Robotic-World-Model-Go2
#   bash lab/sanity-check.sh

set -uo pipefail

ISAAC_SIM_PYTHON="/isaac-sim/python.sh"
PROJECT_ROOT="/workspace/Uncertainty-Aware-Robotic-World-Model-Go2"
RWM="${PROJECT_ROOT}/upstream/robotic_world_model"
RSL="${PROJECT_ROOT}/upstream/rsl_rl_rwm"

export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:${RWM}/source:${RSL}:${PYTHONPATH:-}"

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
echo "  rwmu-cogar lab sanity check"
echo "======================================================================"
echo

echo "Project root: ${PROJECT_ROOT}"
echo "RWM:          ${RWM}"
echo "RSL:          ${RSL}"
echo "Python:       ${ISAAC_SIM_PYTHON}"
echo

# ---------------------------------------------------------------------------
# 1. nvidia-smi works
# ---------------------------------------------------------------------------
echo "[1/9] nvidia-smi"
nvidia-smi > /tmp/nvidia-smi.out 2>&1
check "nvidia-smi works inside container" $?
head -20 /tmp/nvidia-smi.out | sed 's/^/    /'
echo

# ---------------------------------------------------------------------------
# 2. GPU visible to torch
# ---------------------------------------------------------------------------
echo "[2/9] torch sees GPU"
$ISAAC_SIM_PYTHON - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f"    torch: {torch.__version__}, cuda_version: {torch.version.cuda}")
print(f"    torch file: {torch.__file__}")
print(f"    devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"    device {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"              VRAM: {props.total_memory / (1024**3):.1f} GB")
PY
check "torch sees GPU" $?
echo

# ---------------------------------------------------------------------------
# 3. Isaac Sim importable
# ---------------------------------------------------------------------------
echo "[3/9] isaacsim import"
$ISAAC_SIM_PYTHON - <<'PY'
import isaacsim
print("    isaacsim:", getattr(isaacsim, "__file__", ""))
PY
check "isaacsim imports" $?
echo

# ---------------------------------------------------------------------------
# 4. Isaac Lab base importable
# ---------------------------------------------------------------------------
echo "[4/9] isaaclab import"
$ISAAC_SIM_PYTHON - <<'PY'
import isaaclab
print("    isaaclab:", isaaclab.__file__)
PY
check "isaaclab imports" $?
echo

# ---------------------------------------------------------------------------
# 5. omni.client path available
# ---------------------------------------------------------------------------
echo "[5/9] omni.client import"
$ISAAC_SIM_PYTHON - <<'PY'
import omni.client
print("    omni.client:", omni.client.__file__)
PY
check "omni.client imports" $?
echo

# ---------------------------------------------------------------------------
# 6. Project packages importable without forcing full Isaac asset import
# ---------------------------------------------------------------------------
echo "[6/9] project package imports"
$ISAAC_SIM_PYTHON - <<'PY'
import rsl_rl
print("    rsl_rl:", rsl_rl.__file__)

# mbrl imports Isaac Lab modules, so this is the real project extension smoke test.
import mbrl
print("    mbrl:", mbrl.__file__)
PY
check "mbrl and rsl_rl import" $?
echo

# ---------------------------------------------------------------------------
# 7. Data science extras importable
# ---------------------------------------------------------------------------
echo "[7/9] data science extras"
$ISAAC_SIM_PYTHON - <<'PY'
import pandas, scipy, matplotlib, wandb, tensorboard, h5py, safetensors
print(f"    pandas {pandas.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__}")
print(f"    wandb {wandb.__version__}, h5py {h5py.__version__}, safetensors {safetensors.__version__}")
PY
check "data science extras importable" $?
echo

# ---------------------------------------------------------------------------
# 8. Pretrained LFS checkpoint present
# ---------------------------------------------------------------------------
echo "[8/9] pretrained checkpoint and dataset"
CKPT="${RWM}/assets/models/pretrain_rnn_ens.pt"
DATA="${RWM}/assets/data/state_action_data_0.csv"

if [ -f "$CKPT" ]; then
    SIZE_BYTES=$(stat -c%s "$CKPT")
    echo "    checkpoint: $CKPT (${SIZE_BYTES} bytes)"
    if [ "$SIZE_BYTES" -gt 10000000 ]; then
        CKPT_OK=0
    else
        CKPT_OK=1
    fi
else
    echo "    missing checkpoint: $CKPT"
    CKPT_OK=1
fi

if [ -f "$DATA" ]; then
    ROWS=$(wc -l < "$DATA")
    echo "    dataset: $DATA (${ROWS} rows)"
    DATA_OK=0
else
    echo "    missing dataset: $DATA"
    DATA_OK=1
fi

if [ "$CKPT_OK" -eq 0 ] && [ "$DATA_OK" -eq 0 ]; then
    check "checkpoint and dataset present" 0
else
    check "checkpoint and dataset present" 1
fi
echo

# ---------------------------------------------------------------------------
# 9. RWM train entrypoint parses arguments
# ---------------------------------------------------------------------------
echo "[9/9] RWM train.py --help"
cd "$RWM"
$ISAAC_SIM_PYTHON scripts/reinforcement_learning/rsl_rl/train.py --help > /tmp/rwm-train-help.out 2>&1
HELP_STATUS=$?
head -40 /tmp/rwm-train-help.out | sed 's/^/    /'
check "RWM train.py --help works" "$HELP_STATUS"
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
