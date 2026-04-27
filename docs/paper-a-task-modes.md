# Paper A Task Modes

This document explains the four ANYmal-D task modes used by the Paper A code path and clarifies which one corresponds to which stage of the method.

Paper A here refers to *Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics*. The content below is based on direct inspection of the task registry, local env configs, runner configs, and model-based environment code.

## Source of truth

The task family is registered in:

```
source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py
```

This file defines four gym task IDs:

- `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0`
- `Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0`
- `Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0`
- `Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0`

This registry is the primary source for the task-mode split.

## 1. `Init-v0`

**Registration.** Task ID `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0`, with:

- `entry_point="isaaclab.envs:ManagerBasedRLEnv"`
- env cfg: `flat_env_cfg:AnymalDFlatEnvCfg_INIT`
- runner cfg: `agents.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg`

**Interpretation.** This is the baseline initialization task, not the full Paper A model-based training path.

The interpretation rests on two facts. First, the task uses the generic IsaacLab `ManagerBasedRLEnv` rather than the custom `ANYmalDManagerBasedMBRLEnv`. Second, it points to `AnymalDFlatPPORunnerCfg`, which is inherited from IsaacLab tasks rather than the Paper A pretrain or finetune configurations.

In practice, this task is useful as a smoke-test baseline, an RL sanity path, and a stable reference state. It should not be confused with the actual neural-simulator training stage.

## 2. `Pretrain-v0`

**Registration.** Task ID `Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0`, with:

- `entry_point="isaaclab.envs:ManagerBasedRLEnv"`
- env cfg: `AnymalDFlatEnvCfg_PRETRAIN`
- runner cfg: `AnymalDFlatPPOPretrainRunnerCfg`

**Interpretation.** This is the world-model pretraining stage.

Three pieces of evidence support this. First, the runner class is model-based: in `source/mbrl/mbrl/tasks/.../agents/rsl_rl_ppo_cfg.py`, the pretrain config sets `class_name = "MBPOOnPolicyRunner"`, so the Paper A backend is already engaged rather than plain PPO. Second, the system dynamics model is explicitly configured via `system_dynamics = RslRlSystemDynamicsCfg(...)` with `ensemble_size = 1`, `history_horizon = 32`, architecture type `rnn`, GRU with 2 layers and hidden size 256, and state, contact, and termination heads with hidden size 128. Third, imagination is disabled in the same config, with `imagination.num_envs = 0` and `imagination.num_steps_per_env = 0`, so the policy is not yet being optimized via imagined rollouts.

In practice, this stage trains the learned dynamics model using real-environment data and system-dynamics replay-buffer updates.

## 3. `Finetune-v0`

**Registration.** Task ID `Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0`, with:

- Custom env: `ANYmalDManagerBasedMBRLEnv`
- env cfg: `AnymalDFlatEnvCfg_FINETUNE`
- runner cfg: `AnymalDFlatPPOFinetuneRunnerCfg`

**Interpretation.** This is the actual Paper A imagination-based policy optimization stage.

The finetune runner config sets `load_system_dynamics = True` and `system_dynamics_load_path = ...`, so it assumes the world model has already been trained. It also overrides `imagination.num_envs = 8192`, `imagination.num_steps_per_env = 24`, and `imagination.max_episode_length = 256`, meaning imagined rollouts are active. Finally, `ANYmalDManagerBasedMBRLEnv` implements `get_imagination_observation(...)`, `imagination_step(...)`, predicted-state parsing, reward reconstruction, and termination handling. In this stage the learned system dynamics model is used as the rollout engine.

This is the stage that best corresponds to the *neural network simulator for policy optimization* claim in Paper A.

## 4. `Visualize-v0`

**Registration.** Task ID `Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0`, with a custom visualize env and runner cfg `AnymalDFlatPPOVisualizeRunnerCfg`.

**Interpretation.** This is the qualitative inspection and visualization stage. It uses a dedicated visualize env, pretrained system dynamics, and a small scene configuration intended for inspection. This makes it a natural path for checking imagined behavior, debugging world-model rollouts, and relating visual behavior to terminal metrics.

## Supporting env config details

The env configs are defined in `source/mbrl/mbrl/tasks/.../flat_env_cfg.py`.

`AnymalDFlatEnvCfg_INIT` reverts terrain and reward settings closer to the initial locomotion baseline: the rough-terrain generator is re-enabled, air-time and torque weights are adjusted, and the flat orientation penalty is removed. This supports the interpretation that `Init-v0` is a baseline task rather than the full Paper A model-based training stage.

`ObservationsCfg_PRETRAIN` defines world-model-related observation groups: `system_state`, `system_action`, `system_contact`, and `system_termination`. This supports the interpretation that `Pretrain-v0` introduces the actual dynamics-learning interface.

## Summary table

| Task mode | Main role | Env type | Imagination active | Uses system dynamics |
|---|---|---|---|---|
| `Init-v0` | Baseline RL path, smoke test | Generic `ManagerBasedRLEnv` | No | Not in the Paper A sense |
| `Pretrain-v0` | Train world model | Generic `ManagerBasedRLEnv` | No | Yes |
| `Finetune-v0` | Train policy in imagination | Custom `ANYmalDManagerBasedMBRLEnv` | Yes | Yes |
| `Visualize-v0` | Inspect imagined behavior | Custom visualize env | Yes (inspection path) | Yes |

## Conclusion

The Paper A code path should be understood as a pipeline rather than a single task: `Init-v0` provides a stable baseline reference, `Pretrain-v0` learns the world model, `Finetune-v0` optimizes the policy in the learned simulator, and `Visualize-v0` inspects model behavior. This is supported directly by task registration, env cfg specialization, runner config specialization, and the custom MBRL environment code.
