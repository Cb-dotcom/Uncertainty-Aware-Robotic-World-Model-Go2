# Baseline Execution

The baseline execution check verifies that the upstream codebase still runs end-to-end on the local machine. Status: **Validated**, last confirmed in [`manifests/baseline_state_20260425_100703.txt`](../../manifests/baseline_state_20260425_100703.txt).

## What it verifies

A successful baseline run confirms that the following components work together: Isaac Sim launcher, task registration, environment configuration, scene creation, the reward, observation, and termination managers, the RSL-RL wrapper, the PPO runner, the policy and critic networks, and headless execution. Failure of any one component fails the check.

The validated task is `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0`. This is the upstream baseline task, not the full RWM pipeline; the world-model-specific paths are validated separately under [World-Model Pretraining Check](world-model-pretraining-check.md).

## Canonical command

Run from any directory inside the cloned repository:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_isaaclab_src
export OMNI_KIT_ACCEPT_EULA=YES

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
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

The `pkill` lines clear stale Isaac Sim processes from previous runs. The `timeout` invocation sends `SIGINT` first to allow a clean shutdown, then escalates to `SIGKILL` after 20 s if the process is still alive. The simpler ancestor of this command (`timeout 120s ...` without signal handling) is what the original manifest captured; the elaboration above is what is recommended for ongoing use.

## Success criterion

The run is considered successful when the log includes the line:

```text
Learning iteration 0/300
```

Subsequent iterations indicate stable training. A timeout exit (code `124`) is acceptable provided the success line was observed before termination; this is the expected outcome of the bounded canonical run. Forced shutdown may emit a `carb.tasking` Mutex assertion at exit, which is a known shutdown-path artifact and does not invalidate the run.

Expected non-blocking warnings during startup are documented in [Local Environment](../setup/local-environment.md#expected-startup-warnings).