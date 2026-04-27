# Paper A Runtime Flow

This document explains the runtime flow of the Paper A implementation, from launcher to imagination rollout. It answers where execution starts, how a task is resolved, when the world model is created, when imagination is activated, and how the learned model becomes the simulator.

## High-level flow

For the RSL-RL entry path, execution proceeds as follows:

1. The launcher script parses arguments and launches Isaac Sim.
2. The task ID resolves to an env cfg and a runner cfg.
3. The gym env is created.
4. The env is wrapped for RSL-RL.
5. The runner is instantiated.
6. The runner creates the policy, system dynamics, normalizers, and imagination storage.
7. The training loop alternates between real rollout collection, system-dynamics updates, imagination rollouts, and PPO updates.

## 1. Launcher entry point

The primary script is `upstream/robotic_world_model/scripts/reinforcement_learning/rsl_rl/train.py`. It parses CLI arguments, launches the Omniverse and Isaac Sim app via `AppLauncher`, imports `mbrl.tasks`, and calls the main training function through `@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")`.

The task string provided on the command line is the key handle that resolves the rest of the runtime graph.

## 2. Task resolution

Task registration happens in `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py`. This file maps each task ID to an `env_cfg_entry_point` and a `rsl_rl_cfg_entry_point`. For example, `Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0` maps to `AnymalDFlatEnvCfg_FINETUNE` and `AnymalDFlatPPOFinetuneRunnerCfg`.

This is where the repo decides whether execution will follow the baseline init, pretrain, finetune, or visualize mode.

## 3. Env creation

Back in `train.py`, the env is created through:

```python
env = gym.make(args_cli.task, cfg=env_cfg, ...)
```

Depending on the task mode, this creates either the generic `ManagerBasedRLEnv` or the custom `ANYmalDManagerBasedMBRLEnv`. This is the first point where the pipeline diverges between standard RL-like execution and model-based imagined execution.

## 4. RSL-RL wrapping

The env is wrapped by `RslRlVecEnvWrapper`, which makes it compatible with the RSL-RL training interface used by the backend runner. This is infrastructure rather than method novelty.

## 5. Runner selection

The train script selects the runner by `agent_cfg.class_name`. The two possible branches are `OnPolicyRunner` and `MBPOOnPolicyRunner`. For Paper A pretraining and finetuning, the runner config uses `MBPOOnPolicyRunner` (see `source/mbrl/mbrl/tasks/.../agents/rsl_rl_ppo_cfg.py`). This is the handoff from launcher and task config into the model-based backend.

## 6. Runner initialization

The core backend logic lives in `upstream/rsl_rl_rwm/rsl_rl/runners/mbpo_on_policy_runner.py`. During initialization, the runner builds the actor-critic policy, reads the `system_dynamics` and `imagination` configs, builds a `SystemDynamicsEnsemble`, builds state and action normalizers, creates `MBPOPPO`, and injects imagination-related objects into the env.

Concretely, the runner creates `SystemDynamicsEnsemble(...)` and `EmpiricalNormalization(...)` objects for state and action, loads their mean and std from config, and assigns the following attributes on `env.unwrapped`:

- `num_imagination_envs`
- `num_imagination_steps`
- `max_imagination_episode_length`
- `imagination_command_resample_interval_range`
- `imagination_state_normalizer`
- `imagination_action_normalizer`
- `system_dynamics`
- `uncertainty_penalty_weight`

This is the critical architectural bridge: the environment becomes a model-based imagination environment because the runner injects the learned model and its normalization context into it.

## 7. Real rollout collection

In `MBPOOnPolicyRunner.learn(...)`, the loop begins by collecting real environment rollouts. For each step, the policy acts on real observations, the env steps, rewards and dones and extras are recorded, and the history buffer is filled for system dynamics. The key call is:

```python
self.alg.fill_history_buffer(obs)
```

This inserts normalized `system_state` and `system_action`, plus optional extension, contact, and termination targets, into the system replay buffer. This is where the data for world-model learning is accumulated.

## 8. System dynamics update

After real rollout collection, the runner calls:

```python
self.alg.update_system_dynamics()
```

The implementation lives in `upstream/rsl_rl_rwm/rsl_rl/algorithms/mbpo_ppo.py`. It samples sequences from the `ReplayBuffer`, resets the world model, computes the system-dynamics loss, backpropagates through the system dynamics only, and updates it with its own optimizer. This cleanly separates world-model optimization from policy optimization.

## 9. Imagination activation

Imagination only occurs when `num_imagination_envs > 0` and `num_imagination_steps > 0`. It is off in pretraining and on in finetuning. The runner prepares imagination via `self.env.unwrapped.prepare_imagination()` and later calls `self.imagine()`. This is the runtime switch between pure world-model training and policy-in-imagination training.

## 10. Imagination rollout mechanics

The core logic is in `source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py` and `.../envs/anymal_d_manager_based_mbrl_env.py`. The imagination step:

1. Normalizes the rollout action.
2. Appends the action to the action history.
3. Calls `system_dynamics.forward(state_history, action_history, ...)`.
4. Denormalizes the predicted states.
5. Parses state, contacts, and terminations.
6. Computes imagined reward terms.
7. Applies terminations and timeouts.
8. Resets imagination envs if needed.
9. Returns the imagined observation to the PPO loop.

This is the exact place where the learned world model becomes the simulator. The method is not merely *predict next state offline*; it is *drive PPO through imagined transitions generated by the learned model*.

## 11. Policy update

After the imagination rollout, the runner calls `self.alg.update(imagination=True)`. The PPO implementation is in `upstream/rsl_rl_rwm/rsl_rl/algorithms/ppo.py` and uses the standard clipped surrogate loss, clipped value loss, and entropy bonus. The policy side is therefore standard PPO, but during finetuning the trajectories it learns from come from the learned simulator. That is the Paper A core claim in code form.

## 12. Runtime split by task mode

- `Init-v0`: generic RL path, no Paper A imagination pipeline.
- `Pretrain-v0`: real rollouts and system-dynamics learning, no imagination rollout.
- `Finetune-v0`: real rollouts, system-dynamics updates, imagination rollouts, PPO on imagination.
- `Visualize-v0`: custom visualization-oriented inspection path.

## Conclusion

The runtime flow shows that the Paper A implementation is a two-stage model-based RL pipeline. Stage 1 trains the world model from real trajectory data. Stage 2 uses the trained world model as an imagination simulator for PPO. This is supported directly by the launcher code, task registration, runner config, runner initialization, replay-buffer system-dynamics updates, and imagination-step environment code.
