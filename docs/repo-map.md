# Repo Map

This document explains the role of each repository in the workspace and where to look for the main implementation of the Robotic World Model (RWM) pipeline. Every claim below is grounded in code paths inspected directly, not in conceptual guesses.

## Workspace structure

The top-level project is a wrapper repo around three upstream codebases under `upstream/`:

1. `upstream/robotic_world_model`
2. `upstream/rsl_rl_rwm`
3. `upstream/IsaacLab`

The top-level repo also contains `docs/` (interpretation and documentation), `manifests/` (frozen environment and version records), `notes/` (research notes), and local `scripts/` used during setup and validation.

## Repository roles

### 1. `upstream/robotic_world_model`

This is the main method repo and the first place to inspect when tracing Paper A. It contains task registration, task-specific configs, environment specializations, observation and reward definitions, and training entrypoints.

Key source areas:

- Entrypoints
    - `scripts/reinforcement_learning/rsl_rl/train.py`
    - `scripts/reinforcement_learning/model_based/...`
- Method package
    - `source/mbrl/mbrl/tasks/...`
    - `source/mbrl/mbrl/rl/rsl_rl/rl_cfg.py`
    - `source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py`

The validated baseline task and the Paper A task family are both registered in:

```
source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py
```

That file registers four task IDs, which makes this repo the main switchboard for the Paper A training phases:

- `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0`
- `Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0`
- `Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0`
- `Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0`

### 2. `upstream/rsl_rl_rwm`

This is the RL backend and world-model training backend. It contains PPO, MBPO-PPO, rollout storage, the replay buffer, the system dynamics model, and the runner logic for imagination-based training.

Key source areas:

- Algorithms
    - `rsl_rl/algorithms/ppo.py`
    - `rsl_rl/algorithms/mbpo_ppo.py`
- Runners
    - `rsl_rl/runners/on_policy_runner.py`
    - `rsl_rl/runners/mbpo_on_policy_runner.py`
- Dynamics model
    - `rsl_rl/modules/system_dynamics.py`
    - `rsl_rl/modules/architectures/rnn.py`
    - `rsl_rl/modules/architectures/mlp.py`

The top-level RSL-RL train script in `robotic_world_model` imports `OnPolicyRunner` and `MBPOOnPolicyRunner` from `rsl_rl.runners`, so once execution leaves the launcher, the method logic moves into this repo. This is where the PPO loss, the system-dynamics loss, replay-buffer-based system-dynamics training, and imagination rollout orchestration are implemented.

### 3. `upstream/IsaacLab`

This is the simulation and runtime framework layer. It provides app launching, environment classes, task infrastructure, manager abstractions, and the inherited baseline locomotion configs.

The main train script imports `AppLauncher`, `ManagerBasedRLEnvCfg`, `RslRlVecEnvWrapper`, and IsaacLab task utilities. The local ANYmal runner config in `source/mbrl/mbrl/tasks/.../agents/rsl_rl_ppo_cfg.py` inherits from `isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg`, so IsaacLab supplies part of the inherited RL baseline surface.

IsaacLab is essential for runtime, but it is mostly infrastructure rather than the Paper A novelty layer. For Paper A tracing, treat it as a dependency, not the first conceptual source.

## Execution-layer map

The practical runtime path is:

1. Launcher script in `robotic_world_model`.
2. Task registration and env/config lookup in `mbrl.tasks`.
3. Runner and algorithm instantiation in `rsl_rl_rwm`.
4. IsaacLab runtime underneath.

### Source chain

**Launcher.** `upstream/robotic_world_model/scripts/reinforcement_learning/rsl_rl/train.py`. This script parses arguments, launches Isaac Sim, loads task config via `@hydra_task_config(...)`, creates the gym env, wraps it for RSL-RL, selects a runner class, and calls `runner.learn(...)`.

**Task registration.** `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py`. This file maps task IDs to env cfg entry points and runner cfg entry points.

**Runner backend.** `upstream/rsl_rl_rwm/rsl_rl/runners/mbpo_on_policy_runner.py`. This file implements rollout collection, system-dynamics updates, imagination rollout, and PPO update scheduling.

## Paper A task family map

The Paper A code path is not a single task. It is split into four task modes.

### `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0`

Registered in `.../config/anymal_d/__init__.py` with:

- `entry_point="isaaclab.envs:ManagerBasedRLEnv"`
- env cfg: `AnymalDFlatEnvCfg_INIT`
- runner cfg: `AnymalDFlatPPORunnerCfg`

This is the baseline initialization RL path, not the full model-based Paper A pipeline. The interpretation is justified by the fact that this task does not use the custom MBRL environment and does not load the Paper A-specific pretrain or finetune runner configs.

### `Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0`

Uses:

- `entry_point="isaaclab.envs:ManagerBasedRLEnv"`
- env cfg: `AnymalDFlatEnvCfg_PRETRAIN`
- runner cfg: `AnymalDFlatPPOPretrainRunnerCfg`

This is the world-model pretraining stage. The runner cfg sets `class_name = "MBPOOnPolicyRunner"`, `history_horizon = 32`, and `system_dynamics_forecast_horizon = 8`, while imagination is disabled (`num_envs = 0`, `num_steps_per_env = 0`). The dynamics model is trained, but imagined policy rollouts are not yet used.

### `Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0`

Uses:

- Custom entry point: `ANYmalDManagerBasedMBRLEnv`
- env cfg: `AnymalDFlatEnvCfg_FINETUNE`
- runner cfg: `AnymalDFlatPPOFinetuneRunnerCfg`

This is the actual imagination-based policy training stage. The finetune config resumes from previous training, loads system dynamics, and enables imagination with `num_envs = 8192` and `num_steps_per_env = 24`. The custom MBRL env implements learned-dynamics transitions, imagined reward computation, imagined terminations, and imagination resets.

### `Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0`

Uses a custom visualize env and a pretrained world-model path. This is the visualization and inspection stage for model behavior.

## Recommended reading order

For Paper A tracing, inspect files in this order:

1. **Task switchboard.** `source/mbrl/mbrl/tasks/.../config/anymal_d/__init__.py`
2. **Task and env configs.** `.../flat_env_cfg.py`, `.../agents/rsl_rl_ppo_cfg.py`
3. **Imagination environment.** `source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py`, `.../envs/anymal_d_manager_based_mbrl_env.py`
4. **Backend algorithm.** `rsl_rl/runners/mbpo_on_policy_runner.py`, `rsl_rl/algorithms/mbpo_ppo.py`, `rsl_rl/algorithms/ppo.py`
5. **System dynamics model.** `rsl_rl/modules/system_dynamics.py`, `rsl_rl/modules/architectures/rnn.py`, `rsl_rl/modules/architectures/mlp.py`

## Summary

The validated smoke-tested `Init-v0` task is not the full Paper A method path. Paper A is implemented as a multi-stage pipeline: init baseline, pretrain world model, finetune in imagination, visualize. The method is split across `robotic_world_model` (task definition), `rsl_rl_rwm` (algorithm and model backend), and `IsaacLab` (infrastructure). These claims are grounded in task registration, runner configs, env entry points, runner code, and system dynamics wiring.
