# Baseline Smoke Command

## Purpose
This is the canonical bounded smoke test for the untouched upstream ANYmal-D initialization baseline.

## Environment
- Conda env: `env_isaaclab_src`
- Launcher: `upstream/IsaacLab/isaaclab.sh`
- Task: `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0`

## Canonical command
```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_isaaclab_src
export OMNI_KIT_ACCEPT_EULA=YES

PROJECT_ROOT="$HOME/Desktop/Robotics_Masters_UniGe/Sem2/CoGaR/Uncertainty-Aware-Robotic-World-Model-Go2"
ISAACLAB_ROOT="$PROJECT_ROOT/upstream/IsaacLab"
UPSTREAM_RWM="$PROJECT_ROOT/upstream/robotic_world_model"

pkill -f isaacsim || true
pkill -f "kit_" || true
sleep 2

cd "$UPSTREAM_RWM"
timeout --signal=INT --kill-after=20s 120s \
  "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Template-Isaac-Velocity-Flat-Anymal-D-Init-v0 \
  --headless
