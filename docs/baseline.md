# Baseline

This document covers the validated upstream baseline run, the canonical smoke command, and known caveats. It is the single reference for verifying that the local stack is in a working state.

## Status

The untouched upstream ANYmal-D initialization baseline from `robotic_world_model` runs locally inside the source-based Isaac Lab environment `env_isaaclab_src`.

Validated:

- The Isaac Lab launcher starts correctly.
- Isaac Sim runs headless.
- The `mbrl` package is installed in the active environment.
- The model-based `rsl_rl_rwm` integration is active.
- The task `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0` runs headlessly.
- PPO training reaches learning iterations and proceeds with stable metrics.

A bounded smoke run is considered successful when environment setup completes and the log line `Learning iteration 0/300` appears. A timeout-based exit is accepted, because external termination is known to produce non-graceful shutdown behavior.

## Environment

- Conda env: `env_isaaclab_src`
- Launcher: `upstream/IsaacLab/isaaclab.sh`
- Task: `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0`

## Canonical smoke command

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
```

The `pkill` lines clear stale Isaac Sim and Kit processes that can occasionally survive previous runs and block GPU resources. The `timeout` invocation sends `SIGINT` first (to allow a clean shutdown attempt), then escalates to `SIGKILL` after 20 s if the process is still alive.

## Known caveats

- Warp CUDA startup warnings appear during initialization, but training proceeds normally.
- Timeout-driven shutdown may end with a `carb.tasking` Mutex assertion. This is a known shutdown-path artifact and does not invalidate the run if learning iterations were reached before termination.
