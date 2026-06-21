# Go2 Transfer Inventory

This document tracks the Unitree Go2 inventory used to transfer the RWM / RWM-U pipeline from ANYmal-D to Go2.

The purpose of this file is to record the durable robot/task facts: asset locations, joint/body layout, actuator values, observation layout, reward/task differences, and the transfer implications for RWM.

## Current source of truth

Go2 support already exists in the pinned Isaac Lab submodule.

Relevant upstream Isaac Lab files:

```text
upstream/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py
upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/
upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py
upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py
upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py
```

Relevant RWM Go2 source-of-truth files:

```text
scripts/phase4b/go2_rwm_config/
scripts/phase4b/go2_rwm_config/agents/rsl_rl_ppo_cfg.py
scripts/phase4b/install_go2_config.py
upstream/robotic_world_model/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/go2/
```

Note: this section records the pinned upstream Isaac Lab Go2 inventory. Working-tree control edits, including the Phase 4C `feet_slide = -0.25` reward patch, are tracked separately in the Phase 4C / Submodules and Forks notes.

## Robot asset inventory

The Go2 robot configuration is defined in Isaac Lab's `unitree.py`.

Runtime-confirmed durable facts:

| Item | Go2 value |
|---|---|
| Robot | Unitree Go2 |
| Initial base height | `0.4` |
| Actuated joints | 12 |
| Body count | 19 |
| Passive/non-actuated head links | `Head_upper`, `Head_lower` |
| Default actuator effort limit | `23.5` |
| Default actuator stiffness | `25.0` |
| Default actuator damping | `0.5` |

The 19 bodies consist of the base/trunk, four leg chains, four feet, and the two passive head links. The passive head links are present in the USD/body tree but are not actuated joints.

## Joint order

The Go2 joint order is grouped by joint type, with leg order `FL`, `FR`, `RL`, `RR` inside each group.

Runtime-confirmed order:

```text
FL_hip_joint
FR_hip_joint
RL_hip_joint
RR_hip_joint
FL_thigh_joint
FR_thigh_joint
RL_thigh_joint
RR_thigh_joint
FL_calf_joint
FR_calf_joint
RL_calf_joint
RR_calf_joint
```

This matters for all indexed vectors:

```text
joint_pos
joint_vel
joint_torque
actions
system_state joint slices
state/action normalizers
world-model inputs and targets
```

## Body/contact inventory

Runtime-confirmed durable facts:

```text
body_count = 19
actuated_joints = 12
passive_head_links = Head_upper, Head_lower
```

For RWM transfer, the important contact groups are:

```text
feet: four foot bodies
thigh/calf/body contact groups: task-dependent
base_contact termination: active in the flat locomotion task
```

The Phase 4C reward experiments showed that `undesired_contacts` did not provide a useful anti-skate fix for this Go2 flat setup. The working anti-skate term is `feet_slide = -0.25`.

## Actuator block

Go2 uses the Isaac Lab actuator settings from the pinned `unitree.py`.

Durable values confirmed during Phase 4C:

```text
effort_limit = 23.5
stiffness    = 25.0
damping      = 0.5
```


## Observation and RWM system-state layout

The Go2 RWM system state is 45-dimensional.

Layout:

```text
0:3     base linear velocity
3:6     base angular velocity
6:9     projected gravity
9:21    joint positions, 12 dims
21:33   joint velocities, 12 dims
33:45   joint torques, 12 dims
```

Summary:

```text
system_state_dim = 45
action_dim       = 12
```

This matches the Go2 RWM config family and the world-model state index dictionary.

Important implication: the 45-dimensional shape matches ANYmal-D-style locomotion state structure, but the joint-index semantics differ because the robot joint order differs.

## Normalizer decision

Do not reuse ANYmal-D normalizer values — and in practice, use identity normalizers for the current Go2 RWM/RWM-U pipeline.

The original concern remains valid: state-normalizer means/stds are index-dependent. For example, `system_state[9]` refers to a specific Go2 joint, not necessarily the same physical joint as in ANYmal-D. Therefore ANYmal-D normalizer statistics must not be copied to Go2.

Resolved in Phase 4C: the Go2 RWM pipeline uses identity normalizers throughout:

```text
state_normalizer.mean  = 0
state_normalizer.std   = 1
action_normalizer.mean = 0
action_normalizer.std  = 1
```

This is position-agnostic and matches the world model actually trained during Go2 pretraining, as confirmed in each run's `params/agent.yaml`.

The invariant is consistency:

```text
finetune normalizers must match the pretrain normalizers used to train the world model.
```

Recomputing Go2-specific normalizer statistics is valid only if the world model is freshly pretrained using those same statistics and the finetune carries the same values. Pairing a recomputed finetune normalizer with an identity-trained world model breaks imagination.

## Stock Go2 flat task inventory

The pinned Isaac Lab Go2 flat task provides a usable starting point, but the stock flat reward permits a low/skating velocity-tracking optimum.

Phase 4C confirmed:

```text
stock flat Go2 can visually scoot / skate
velocity tracking alone is not enough
anti-slide shaping is required for a usable transfer baseline
```

Working Phase 4C reward decision:

```text
feet_slide = -0.25
undesired_contacts = None
```

Rejected/less useful variants:

```text
feet_slide = -0.10 + undesired_contacts
feet_slide = -0.50
```

Interpretation:

- `feet_slide = -0.10` was too weak.
- `undesired_contacts` did not materially fix the flat Go2 skating mode.
- `feet_slide = -0.50` was too strong and degraded tracking/gait quality.
- `feet_slide = -0.25` gave the best tradeoff: less skating, acceptable tracking, usable upright gait.

## RWM Go2 config family

Status as of Phase 4B–4C: created and validated.

The Go2 RWM config family is registered for:

```text
Template-Isaac-Velocity-Flat-Unitree-Go2-Init-v0
Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0
Template-Isaac-Velocity-Flat-Unitree-Go2-Baseline-v0
Template-Isaac-Velocity-Flat-Unitree-Go2-Finetune-v0
Template-Isaac-Velocity-Flat-Unitree-Go2-Visualize-v0
```

Source of truth:

```text
scripts/phase4b/go2_rwm_config/
```

Installed runtime location:

```text
upstream/robotic_world_model/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/go2/
```

The install mechanism is:

```text
/isaac-sim/python.sh scripts/phase4b/install_go2_config.py --force
```

If the installer is unavailable, a manual copy can reproduce the same state, but the installer is the preferred reproducible path.

## RWM pretrain status

A usable Go2 RWM pretrain was obtained with the Phase 4C reward patch.

Accepted pretrain:

```text
run:        2026-06-12_09-07-38_pretrain
checkpoint: model_2000.pt
reward:     feet_slide = -0.25
```

Approximate accepted metrics:

```text
base_contact         ≈ 0.02
episode_length       ≈ 982
error_vel_xy         ≈ 0.20
feet_slide reward    ≈ -0.06
visual gait          acceptable / upright / non-belly-skating
```

This is the clean single-model RWM pretrain baseline.

## Plain MBPO finetune status

A valid plain MBPO finetune was run from the accepted pretrain.

Plain MBPO negative-control run:

```text
run:        2026-06-12_11-07-01_finetune
pretrain:   2026-06-12_09-07-38_pretrain/model_2000.pt
ensemble:   1
penalty:    uncertainty_penalty_weight = -0.0
```

Important scalar confirmations:

```text
Imagination/feet_slide              = 0
Model Based/epistemic_uncertainty   = 0
Model Based/num_valid_imagination_envs = 8192
```

Final aggregate metrics showed substantial degradation relative to the pretrain:

```text
Train/mean_reward                  ≈ 8.11
Train/mean_episode_length          ≈ 607.65
Episode_Termination/base_contact   ≈ 0.30 final, often 0.38–0.50 near the end
Episode_Reward/track_lin_vel_xy_exp ≈ 0.50
Episode_Reward/feet_slide          ≈ -0.225
```

Interpretation:

A single rendered rollout can still look acceptable, but aggregate real-sim metrics show distributional instability and frequent base-contact termination. This run is a useful plain-MBPO negative control, not the final RWM-U method.

## Reward mismatch and imagination handling

The Go2 flat real rollout reward includes `feet_slide`, but the RWM imagination reward cannot compute true physical feet sliding unless an explicit imagined/proxy term is implemented.

Observed issue:

```text
KeyError: 'feet_slide'
```

Cause:

```text
real rollout reward includes feet_slide
imagination reward dictionary did not contain a valid feet_slide term
```

Runtime fix:

```python
if term not in self.imagination_reward_per_step:
    continue
```

Patched file:

```text
upstream/robotic_world_model/source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py
```

Tracked patch file should be kept under the Phase 4B/4C scripts area so the submodule edit is reproducible.

Interpretation:

```text
Episode_Reward/feet_slide is real-only.
Imagination/feet_slide is zero/skipped.
```

This is acceptable for the controlled comparison only if held constant across finetune arms. The cleaner future implementation is to explicitly split rewards into:

```text
real_env_rewards
imagination_rewards
```

and to replace true `feet_slide` with an imagination-computable proxy if needed.

## RWM-U ensemble pretrain status

The clean RWM-U comparison requires an ensemble world model.

Current ensemble pretrain:

```text
run:        2026-06-12_13-39-03_pretrain_ens5
ensemble:   5
run_name:   pretrain_ens5
penalty:    uncertainty_penalty_weight = -0.0
```

Purpose:

```text
Train the shared ensemble-5 pretrain checkpoint used by both finetune arms.
```

This run should produce:

```text
logs/rsl_rl/unitree_go2_flat/2026-06-12_13-39-03_pretrain_ens5/model_2000.pt
```

Gate before finetunes:

```text
model_2000.pt exists
base_contact is low
episode length is near timeout
feet_slide is controlled
render looks like the accepted 09-07-38 pretrain
```

Do not judge this gate from a single video only. Use aggregate metrics and a render.

## RWM-U finetune comparison plan

The primary RWM-U experiment should compare two finetune arms from the same ensemble-5 pretrain.

Shared setup:

```text
pretrain checkpoint: 2026-06-12_13-39-03_pretrain_ens5/model_2000.pt
ensemble_size:       5
feet_slide handling: real-only / skipped in imagination
normalizers:         identity
max_iterations:      same for both arms
```

Arm A — ensemble-5 without uncertainty penalty:

```text
run_name = "finetune_ens5_pen0"
uncertainty_penalty_weight = -0.0
```

Arm B — RWM-U method:

```text
run_name = "finetune_ens5_pen1"
uncertainty_penalty_weight = -1.0
```

Both arms must use exactly the same pretrain:

```python
load_run = "2026-06-12_13-39-03_pretrain_ens5"
system_dynamics_load_path = "logs/rsl_rl/unitree_go2_flat/2026-06-12_13-39-03_pretrain_ens5/model_2000.pt"
```

The earlier ensemble-1 plain MBPO run remains a supplementary baseline, not the primary ablation partner, because comparing ensemble-1/penalty-0 to ensemble-5/penalty-1 changes two variables at once.

## Evaluation protocol

Primary comparison:

```text
finetune_ens5_pen0 vs finetune_ens5_pen1
same pretrain
same ensemble
same reward/imagination handling
only uncertainty_penalty_weight differs
```

Primary metric:

```text
Episode_Termination/base_contact averaged over the last ~50 iterations
```

Secondary metrics:

```text
Train/mean_episode_length
Train/mean_reward
Metrics/base_velocity/error_vel_xy
Episode_Reward/feet_slide
Model Based/epistemic_uncertainty
Episode_Reward/track_lin_vel_xy_exp
```

Success criterion:

```text
pen1 has materially lower base_contact than pen0
pen1 keeps longer episode length
pen1 preserves upright gait in render
pen0 shows degradation/collapse or materially worse distributional stability
```

Interpretation rules:

```text
Do not judge from one video.
Do not judge from one scalar snapshot.
Use last-window averages and trajectory plots.
```

Best thesis figure:

```text
base_contact vs finetune iteration
mean_reward vs finetune iteration
mean_episode_length vs finetune iteration
Model Based/epistemic_uncertainty vs finetune iteration
```

Expected mechanism:

```text
pen0 can drift into high-uncertainty imagined regions without penalty
pen1 subtracts epistemic uncertainty from imagined reward and should avoid that drift
```

If both arms collapse:

```text
-1.0 may be too weak
or the failure is dominated by reward mismatch rather than dynamics uncertainty
next sweep: -2.0, then possibly -5.0 only if needed
```

## Closure of open items

| Open item | Status |
|---|---|
| Go2 asset and joint/body inventory | Closed. Runtime-confirmed: 12 actuated joints, 19 bodies, passive `Head_upper` and `Head_lower`, FL/FR/RL/RR grouped joint order. |
| Actuator values | Closed. Confirmed from pinned `unitree.py`: effort `23.5`, stiffness `25.0`, damping `0.5`, init height `0.4`. |
| RWM Go2 task registration | Closed. Init/Pretrain/Baseline/Finetune/Visualize task family exists and is installed from `scripts/phase4b/go2_rwm_config/`. |
| Go2 reward weights decision | Closed in Phase 4C. RWM ANYmal-D-style base weights were kept; the missing piece was anti-slide shaping, not reward-scale retuning. Stock flat Go2 and early RWM Go2 runs exposed a low/skating velocity-tracking optimum. The working fix is `feet_slide = -0.25`, with `undesired_contacts = None`. |
| `height_scanner` decision | Closed. Removed/stripped for the flat Go2 task, matching RWM ANYmal-D flat and stock IsaacLab Go2 flat behavior. |
| Normalizer decision | Closed for current pipeline. Use identity normalizers consistently across pretrain and finetune. Do not reuse ANYmal-D stats. Recomputed Go2 stats are allowed only for a fresh pretrain+finetune pair using the same stats. |
| Reward/imagination mismatch | Known and controlled. `feet_slide` is real-only and skipped in imagination. This must be held constant across pen0 and pen1. A cleaner future implementation should explicitly split real and imagination reward terms. |
| RWM-U comparison | In progress. Requires ensemble-5 pretrain followed by pen0 and pen1 finetunes from the same checkpoint. |

## Current bottom line

The transfer inventory is complete enough for Go2 RWM/RWM-U experiments.

Durable facts:

```text
Go2 joint/body/actuator inventory is known.
system_state layout is 45-dimensional.
identity normalizers are the current correct choice.
feet_slide = -0.25 is the working real-rollout anti-skate patch.
Go2 RWM task family is registered and usable.
plain MBPO has a documented aggregate instability negative control.
RWM-U requires ensemble-5 pen0 vs pen1 comparison from the same pretrain.
```

Main invariant going forward:

```text
Do not compare different pretrains when attributing the effect of uncertainty penalty.
For the main RWM-U claim, pen0 and pen1 must start from the same ensemble-5 checkpoint.
```