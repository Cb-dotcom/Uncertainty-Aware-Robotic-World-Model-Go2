# World-Model Pretraining Check

The world-model pretraining check verifies that the RWM pretraining pipeline executes end-to-end on the local machine and produces well-formed dynamics checkpoints. Status: **Validated at reduced scale**. The default configuration exceeds the local 8 GB VRAM budget and is deferred to the lab workstation.

The validated task is `Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0`.

## What the task represents

`Pretrain-v0` is the world-model pretraining stage of the RWM pipeline. It differs from `Init-v0` (the baseline check) in two ways. First, it activates four world-model observation groups that the baseline does not: `system_state`, `system_action`, `system_contact`, `system_termination`. Second, it uses the `MBPOOnPolicyRunner` instead of the standard PPO runner, which adds the system-dynamics replay buffer and the dynamics model update loop. Imagination is disabled in this stage; the model is trained, the policy is not yet rolled out through it.

For the full task family and the role of each task mode, see [Task Modes](../world-model/task-modes.md).

## Default configuration: out of memory

A first attempt with the default pretraining configuration reached environment setup and entered the system-dynamics update, but failed with CUDA out-of-memory during GRU-based system-dynamics loss computation.

The code path is structurally valid; the configuration is too large for the local GPU. Default-scale pretraining is deferred to the lab workstation, where the 48 GB A6000 is expected to accommodate it.

## Reduced-scale validation command

Run from any directory inside the cloned repository:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_isaaclab_src
export OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
ISAACLAB_ROOT="$PROJECT_ROOT/upstream/IsaacLab"
UPSTREAM_RWM="$PROJECT_ROOT/upstream/robotic_world_model"
LOG_DIR="$PROJECT_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

pkill -f isaacsim || true
pkill -f "kit_" || true
sleep 2

cd "$UPSTREAM_RWM"
timeout --signal=INT --kill-after=20s 180s \
  "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
  --headless \
  --num_envs=64 \
  --max_iterations=1 \
  agent.algorithm.system_dynamics_num_mini_batches=1 \
  agent.algorithm.system_dynamics_mini_batch_size=128 \
  agent.algorithm.system_dynamics_replay_buffer_size=256 \
  agent.algorithm.system_dynamics_forecast_horizon=2 \
  | tee "$LOG_DIR/pretrain_check_${STAMP}.log"
```

## Reduced-scale settings

The Hydra overrides used in the validated run:

```text
num_envs                                = 64
max_iterations                          = 1
system_dynamics_num_mini_batches        = 1
system_dynamics_mini_batch_size         = 128
system_dynamics_replay_buffer_size      = 256
system_dynamics_forecast_horizon        = 2
```

These settings are chosen to verify execution flow within the local VRAM budget. They are not configured for representative training quality, and the resulting loss values are not used for any quantitative claim about model accuracy.

## Observation groups

The validated run loaded five observation groups. The `policy` group is the standard ANYmal-D policy observation; the four `system_*` groups are the world-model interface that distinguishes pretraining from the baseline.

| Group | Shape | Contents |
|---|---|---|
| `policy` | (48,) | Standard ANYmal-D policy observation. |
| `system_state` | (45,) | Base linear velocity, base angular velocity, projected gravity, joint position, joint velocity, joint torque. |
| `system_action` | (12,) | Previous or predicted actions. |
| `system_contact` | (8,) | Thigh contact, foot contact. |
| `system_termination` | (1,) | Base contact. |

If a future run produces different shapes or different group contents, the configuration has drifted and the diff is meaningful.

## Saved outputs

The successful run produced the following artifacts under `$PROJECT_ROOT/logs/rsl_rl/anymal_d_flat/<timestamp>_pretrain/`:

```text
model_0.pt                  ~18 MB   (initial checkpoint)
model_1.pt                  ~18 MB   (after one iteration)
params/agent.yaml                    (resolved agent config)
params/env.yaml                      (resolved env config)
```

The runner also stored Git diffs for `robotic_world_model` and `rsl_rl_rwm` under the run directory, capturing the local code state at the moment of the run.

## Required upstream fix

The pretraining run requires a small fix in `rsl_rl_rwm/rsl_rl/runners/mbpo_on_policy_runner.py` to handle the case where imagination is disabled. The fix narrative is documented in [Submodules and Forks](../development/submodules-and-forks.md#pretraining-fix). Without it, the run crashes at the end of the first iteration.

## What the validation establishes

**Validated** by this run:

- The `Pretrain-v0` task resolves correctly and creates the expected environment.
- The four world-model observation groups are active and have the expected shapes.
- The `MBPOOnPolicyRunner` instantiates and runs end-to-end.
- The system-dynamics replay buffer is populated and the dynamics model update completes without error.
- Checkpoints are written to disk in the expected format.

**Not verified** by this run:

- That the trained dynamics model has useful predictive accuracy.
- That the model would generalize beyond the small replay buffer it saw.
- That full-scale pretraining produces results matching the published paper.
- That the `Finetune-v0` stage runs (this is checkpoint-dependent and documented separately).

The reduced-scale settings are sufficient for execution validation and explicitly insufficient for quality validation.