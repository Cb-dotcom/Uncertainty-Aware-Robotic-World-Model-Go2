# Baseline Status

## Current status
The untouched upstream ANYmal-D initialization baseline from `robotic_world_model` runs locally in the source-based Isaac Lab environment `env_isaaclab_src`.

## What has been validated
- Isaac Lab launcher starts correctly
- Isaac Sim headless execution works
- `mbrl` is installed in the active environment
- model-based `rsl_rl_rwm` integration is active
- the task `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0` runs headlessly
- PPO training advances for many iterations with stable metrics

## Canonical smoke command
```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_isaaclab_src
export OMNI_KIT_ACCEPT_EULA=YES

PROJECT_ROOT="$HOME/Desktop/Robotics_Masters_UniGe/Sem2/CoGaR/Uncertainty-Aware-Robotic-World-Model-Go2"
ISAACLAB_ROOT="$PROJECT_ROOT/upstream/IsaacLab"
UPSTREAM_RWM="$PROJECT_ROOT/upstream/robotic_world_model"

cd "$UPSTREAM_RWM"
timeout 120s "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/reinforcement_learning/rsl_rl/train.py --task=Template-Isaac-Velocity-Flat-Anymal-D-Init-v0 --headless
```

## Known caveats
- Warp CUDA startup warnings appear, but training proceeds.
- Timeout-driven shutdown may end with a carb.tasking Mutex assertion.

