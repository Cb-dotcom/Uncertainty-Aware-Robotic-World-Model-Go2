# Go2 Transfer Inventory

This document tracks the initial Unitree Go2 inventory for transferring the RWM/RWM-U pipeline from ANYmal-D to Go2.

## Current source of truth

Go2 support already exists in the pinned Isaac Lab submodule.

Relevant upstream Isaac Lab files:

```text
upstream/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py
upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py
upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py
upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py
```

Existing Isaac Lab tasks:

```text
Isaac-Velocity-Flat-Unitree-Go2-v0
Isaac-Velocity-Flat-Unitree-Go2-Play-v0
Isaac-Velocity-Rough-Unitree-Go2-v0
Isaac-Velocity-Rough-Unitree-Go2-Play-v0
```

Asset configuration
Isaac Lab defines:

```python
UNITREE_GO2_CFG
```

in:

```text
upstream/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py
```

The Go2 USD path is:

```python
f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
```

Contact sensors are enabled:

```python
activate_contact_sensors=True
```

Robot base
Known base body name from the Go2 environment config:

```text
base
```

Used for:

```python
self.events.add_base_mass.params["asset_cfg"].body_names = "base"
self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
```

Feet
Known foot body pattern:

```text
.*_foot
```

Used by Go2 rewards:

```python
self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
```

Joint naming pattern
Known actuator joint regexes:

```text
.*_hip_joint
.*_thigh_joint
.*_calf_joint
```

Likely actuated joint count:

```text
12
```

Confirmed at runtime (see "Runtime confirmation" section below).

Initial pose
From UNITREE_GO2_CFG:

```python
pos = (0.0, 0.0, 0.4)
joint_pos = {
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F[L,R]_thigh_joint": 0.8,
    "R[L,R]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
}
joint_vel = {".*": 0.0}
```

Actuator model
Go2 uses a DC motor actuator model:

```python
DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit=23.5,
    saturation_effort=23.5,
    velocity_limit=30.0,
    stiffness=25.0,
    damping=0.5,
    friction=0.0,
)
```

Existing Isaac Lab Go2 flat task
The upstream flat config class is:

```python
UnitreeGo2FlatEnvCfg
```

It inherits from:

```python
UnitreeGo2RoughEnvCfg
```

Flat config changes:

```python
self.rewards.flat_orientation_l2.weight = -2.5
self.rewards.feet_air_time.weight = 0.25

self.scene.terrain.terrain_type = "plane"
self.scene.terrain.terrain_generator = None

self.scene.height_scanner = None
self.observations.policy.height_scan = None
self.curriculum.terrain_levels = None
```

Existing Isaac Lab Go2 rough task
The upstream rough config class is:

```python
UnitreeGo2RoughEnvCfg
```

Key Go2-specific changes:

```python
self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
self.actions.joint_pos.scale = 0.25
self.events.push_robot = None
self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
self.events.add_base_mass.params["asset_cfg"].body_names = "base"
self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
self.events.base_com = None
```

Reward overrides:

```python
self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
self.rewards.feet_air_time.weight = 0.01
self.rewards.undesired_contacts = None
self.rewards.dof_torques_l2.weight = -0.0002
self.rewards.track_lin_vel_xy_exp.weight = 1.5
self.rewards.track_ang_vel_z_exp.weight = 0.75
self.rewards.dof_acc_l2.weight = -2.5e-7
```

Termination override:

```python
self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
```

Existing Isaac Lab Go2 PPO config
The upstream RSL-RL classes are:

```python
UnitreeGo2RoughPPORunnerCfg
UnitreeGo2FlatPPORunnerCfg
```

Flat PPO values:

```python
max_iterations = 300
experiment_name = "unitree_go2_flat"
actor_hidden_dims = [128, 128, 128]
critic_hidden_dims = [128, 128, 128]
```

Inherited PPO values:

```python
num_steps_per_env = 24
save_interval = 50
init_noise_std = 1.0
activation = "elu"
value_loss_coef = 1.0
clip_param = 0.2
entropy_coef = 0.01
num_learning_epochs = 5
num_mini_batches = 4
learning_rate = 1.0e-3
schedule = "adaptive"
gamma = 0.99
lam = 0.95
desired_kl = 0.01
max_grad_norm = 1.0
```

RWM transfer implication
The RWM ANYmal-D manager-based template currently exists only for:

```text
upstream/robotic_world_model/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d
```

It registers:

```text
Template-Isaac-Velocity-Flat-Anymal-D-Init-v0
Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0
Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0
```

A Go2 RWM transfer should create an equivalent config family, probably:

```text
Template-Isaac-Velocity-Flat-Unitree-Go2-Init-v0
Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0
Template-Isaac-Velocity-Flat-Unitree-Go2-Finetune-v0
Template-Isaac-Velocity-Flat-Unitree-Go2-Visualize-v0
```

The first implementation should adapt from the RWM ANYmal-D config while importing Go2-specific values from Isaac Lab's existing Go2 config.

---

## Runtime confirmation (local laptop, tiny run)

Status: confirmed on local laptop before Phase 4A lab work.

A 1-iteration smoke test of the upstream Isaac Lab Go2 flat task was executed locally:

```bash
# On laptop, in repo root
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab_src
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y

cd upstream/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 1 \
    --max_iterations 1 \
    --headless \
    --logger tensorboard
```

Confirmed at runtime:

- Task `Isaac-Velocity-Flat-Unitree-Go2-v0` registered and resolved
- Go2 environment created without errors
- `num_envs = 1`
- Action shape: `(12,)`
- Policy observation shape: `(48,)`
- Actor: input `48`, output `12`
- Critic: input `48`, output `1`
- One full PPO iteration completed
- Total timesteps: `24`

Non-blocking warnings observed (also seen on workstation):

- `isaaclab_contrib` extension.toml warning (extension does not exist in pinned IsaacLab)
- Warp CUDA driver entry point warning
- `obs_groups` policy/critic deprecation warning from `rsl_rl_rwm`
- FabricManager mismatched-prototype command visualization warnings

This confirms the Go2 articulation, action manager, and observation
manager from upstream Isaac Lab work in our environment as shipped.
It does not validate Go2 PPO learning (1 iteration is not learning); it
validates the task plumbing.

## Open items before scaffolding RWM Go2 config

1. Confirm Go2 articulation joint *order* (not just names/count). Joint
   ordering must match the action vector layout the policy emits.
   Inspect the articulation at runtime to dump `joint_names` in their
   actuator order.
2. Confirm Go2 default joint position vector matches the regex-keyed
   dict in `UNITREE_GO2_CFG` once resolved to concrete joints.
3. Confirm Go2 `system_state` shape would be 45 (the ANYmal-D shape) for
   the RWM dynamics head: 3 lin_vel + 3 ang_vel + 3 gravity + 12 joint_pos
   + 12 joint_vel + 12 joint_torque. Identical to ANYmal-D if Go2 has
   12 actuated joints (which it does). Reuse possible without reshaping
   heads.
4. Decide on Go2 reward weights for the RWM Init/Pretrain stages. The
   ANYmal-D weights from the RWM repo and the upstream Go2 weights
   diverge (e.g. `track_lin_vel_xy_exp` weight is 1.0 in RWM ANYmal-D
   vs 1.5 in upstream Go2 flat). Start with the RWM ANYmal-D weights
   to keep the world model and curriculum apples-to-apples, then tune.
5. Decide whether to keep height_scanner-style observations or strip
   them as the upstream Go2 *flat* config does. The RWM ANYmal-D *flat*
   does not use a height scanner, so default to stripping it.

   ## Runtime inspection results (lab workstation)

Generated by `scripts/phase4b/inspect_go2_articulation.py`. Full JSON at
`logs/go2_inventory/<timestamp>/articulation.json` (gitignored).

### Counts

- Joints: 12
- Bodies: 19

### Joint order (actuator order)

| Idx | Name | Default (rad) | Lower (rad) | Upper (rad) |
|---:|---|---:|---:|---:|
| 0 | FL_hip_joint   | +0.10 | -1.047 | +1.047 |
| 1 | FR_hip_joint   | -0.10 | -1.047 | +1.047 |
| 2 | RL_hip_joint   | +0.10 | -1.047 | +1.047 |
| 3 | RR_hip_joint   | -0.10 | -1.047 | +1.047 |
| 4 | FL_thigh_joint | +0.80 | -1.571 | +3.491 |
| 5 | FR_thigh_joint | +0.80 | -1.571 | +3.491 |
| 6 | RL_thigh_joint | +1.00 | -0.524 | +4.538 |
| 7 | RR_thigh_joint | +1.00 | -0.524 | +4.538 |
| 8 | FL_calf_joint  | -1.50 | -2.723 | -0.838 |
| 9 | FR_calf_joint  | -1.50 | -2.723 | -0.838 |
| 10 | RL_calf_joint | -1.50 | -2.723 | -0.838 |
| 11 | RR_calf_joint | -1.50 | -2.723 | -0.838 |

Order is: all four hips, then all four thighs, then all four calves.
Within each group: FL, FR, RL, RR.

### Bodies

```text
[ 0] base
[ 1] FL_hip      [ 2] FR_hip      [ 4] RL_hip      [ 5] RR_hip
[ 6] FL_thigh    [ 7] FR_thigh    [ 9] RL_thigh    [10] RR_thigh
[11] FL_calf    [12] FR_calf     [13] RL_calf    [14] RR_calf
[15] FL_foot    [16] FR_foot     [17] RL_foot    [18] RR_foot
[ 3] Head_upper  [ 8] Head_lower
```

Body regex matches:

- `^base$` → `base`
- `.*_foot` → `FL_foot, FR_foot, RL_foot, RR_foot`
- `.*_thigh` → `FL_thigh, FR_thigh, RL_thigh, RR_thigh`
- `.*_calf` → `FL_calf, FR_calf, RL_calf, RR_calf`

### system_state shape (RWM dynamics head)

```text
base_lin_vel      : 3
base_ang_vel      : 3
projected_gravity : 3
joint_pos         : 12
joint_vel         : 12
joint_torque      : 12
--------------------- :
total             : 45
```

Matches ANYmal-D. The RWM world-model architecture (state/contact/
termination heads, GRU shape, normalizer dims) can be reused without
shape changes.

### Implications for the RWM Go2 config

1. **Reuse architecture.** Action dim 12, observation dim 48, system_state
   dim 45. All identical to ANYmal-D. `SystemDynamicsEnsemble` heads, GRU
   hidden size, normalizer shapes do not need to be modified.

2. **Do not reuse normalizer values.** The state-normalizer means/stds in
   `RslRlNormalizerCfg` are position-dependent. Index 9 of the joint_pos
   slice is `FL_calf` for Go2 but is a different joint for ANYmal-D.
   Recompute normalizer statistics from Go2 rollouts before training the
   Go2 world model.

3. **Head bodies (`Head_upper`, `Head_lower`) are passive.** They have no
   actuated joints (joint count stays 12). They do have colliders, so they
   could trigger contact-based rewards or terminations if matched by a
   broad body regex. The standard patterns are safe: `.*_thigh` does not
   match `Head_upper`, etc. Keep an explicit `^base$` for termination
   rather than anything broader.

4. **Front/rear thigh limits are asymmetric.** Front thighs:
   `[-1.571, +3.491]`. Rear thighs: `[-0.524, +4.538]`. Per-joint reset
   randomization ranges may need to respect this if randomization is
   widened beyond `(1.0, 1.0)`. For initial Pretrain, default reset is
   inside both ranges.

5. **Default pose is valid.** All twelve defaults sit inside their soft
   limits with comfortable margin. No clamping warnings expected at
   reset.

### Closure of open items

| Open item | Status |
|---|---|
| 1. Joint order at runtime | Closed — FL/FR/RL/RR by group |
| 2. Default joint position vector concrete | Closed — see table above |
| 3. system_state shape verification | Closed — 45, matches ANYmal-D |
| 4. Go2 reward weights decision | Pending — start with RWM ANYmal-D weights, tune in Phase 4C |
| 5. height_scanner decision | Pending — strip for flat (matches RWM ANYmal-D flat) |