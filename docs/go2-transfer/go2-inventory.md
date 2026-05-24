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

This still needs runtime confirmation from the articulation object.

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
