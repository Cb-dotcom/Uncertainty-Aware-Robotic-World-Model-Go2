# Repository Structure

This page describes the role of each top-level directory and the architectural relationship between the three upstream codebases. The goal is that any file in the repository can be located in one of these layers and the reader can identify the layer's job.

For the operational concerns (how to clone with submodules, how to edit a forked submodule and update the pointer) see [Submodules and Forks](../development/submodules-and-forks.md).

## Top-level workspace

The top-level repository is a research workspace, not an application yet. It contains documentation, validation records, setup scripts, and the submodule pointers that pin the upstream code at known-good commits.

The simulation, learning, and method code lives entirely inside `upstream/`.

```text
.
├── docs/                       Published project documentation
├── manifests/                  Frozen project-state snapshots
├── scripts/                    Local setup and validation scripts
├── upstream/
│   ├── IsaacLab/               Simulation infrastructure (pinned upstream)
│   ├── robotic_world_model/    Tasks, configs, custom envs (forked)
│   └── rsl_rl_rwm/             RL backend with world-model extensions (forked)
├── .github/workflows/          GitHub Pages deployment workflow
├── .gitignore                  Files excluded from version control
├── .gitmodules                 Submodule registry
├── mkdocs.yml                  Documentation site configuration
└── README.md
```

## The three upstream layers

The simulation, method, and algorithm code is split across three repositories that form a stack. Each layer depends on the layer below it.

| Layer | Repository | Role | Used by |
|---|---|---|---|
| Simulation infrastructure | `IsaacLab` | Isaac Sim app launcher, manager-based env classes, task registry primitives, RSL-RL wrapper. | `robotic_world_model` |
| Method and tasks | `robotic_world_model` | ANYmal-D task family, environment configs, custom model-based env, training entrypoint. | The training command. Indirectly uses `rsl_rl_rwm` through the runner. |
| RL and world-model backend | `rsl_rl_rwm` | PPO and MBPO-PPO algorithms, on-policy and model-based runners, system-dynamics model, replay buffer. | `robotic_world_model` (via runner config). |

A typical training command flows top-down through the stack: the Isaac Lab launcher starts the simulator, `robotic_world_model` resolves the task and instantiates the environment and runner, and `rsl_rl_rwm` runs the rollout, system-dynamics update, and PPO loop.

## `docs/`

Published project documentation. Built with MkDocs Material and deployed to GitHub Pages by the workflow in `.github/workflows/`. The structure of this directory is documented on the [landing page](../index.md).

This is the supervisor-facing technical record of the project. It is structured for readers, not for raw notes.

## `manifests/`

Frozen snapshots of known-good project states, recorded at the moment a baseline was validated.

Each manifest is a single timestamped text file capturing:

- The conda environment name and active Python interpreter path.
- The exact upstream commit hashes, branches, and `git describe` output for the three submodules.
- The full working command used to validate the state.
- The Python package versions for the relevant scientific stack (`mbrl`, `rsl-rl-lib`, `isaaclab`, `isaaclab_rl`, `isaaclab_tasks`, `isaacsim`, `torch`).
- CUDA availability and device count.
- Known caveats observed during validation (warnings that are non-blocking, shutdown-path artifacts).
- A status marker confirming the freeze stage.

A manifest is forensic: it is sufficient to reconstruct the validated state on another machine, or to diff against a future state when something stops working. Manifests are never edited after creation. New states produce new manifest files with new timestamps.

## `scripts/`

Local setup and validation scripts. These record the operational path used to bring up the local environment and to run the canonical validation commands.

Scripts are not the scientific method itself. They support reproducibility and debugging. Polished explanations of what the scripts do live in the documentation, not in the script comments.

## `upstream/IsaacLab`

The Isaac Lab source, pinned as an unmodified submodule. This layer provides the simulator and the environment framework on which both papers are built. The project treats it as upstream infrastructure and avoids modifications.

If a future need forces a change, the policy is to fork and pin to the fork; until then, this submodule tracks an upstream tag.

## `upstream/robotic_world_model`

The Robotic World Model task and configuration code, tracked as a forked submodule. This layer is expected to change for Go2 integration and for project-specific experiment configurations.

The fork allows project-specific commits to be recorded without losing the ability to pull upstream changes when relevant.

## `upstream/rsl_rl_rwm`

The RSL-RL backend extended with world-model logic, tracked as a forked submodule. This layer is expected to change for the RWM-U activation, for offline data handling, and for any future modifications to the dynamics model or imagination loop.

The fork already carries one local change (a guard in the model-based runner's pretraining cleanup path); future changes will accumulate here.

## Key files reference

The most important files across the three upstream repositories. This is a navigation aid for readers new to the codebase. Deeper file-by-file analysis lives in [Implementation Analysis](../world-model/implementation-analysis.md).

| Repository | Path | Role |
|---|---|---|
| `robotic_world_model` | `scripts/reinforcement_learning/rsl_rl/train.py` | Training entrypoint. Parses arguments, launches Isaac Sim, resolves task, creates env and runner. |
| `robotic_world_model` | `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py` | Registers the four ANYmal-D task IDs (`Init`, `Pretrain`, `Finetune`, `Visualize`). |
| `robotic_world_model` | `.../config/anymal_d/agents/rsl_rl_ppo_cfg.py` | Per-task runner configs, including system-dynamics architecture, ensemble size, imagination settings. |
| `robotic_world_model` | `.../config/anymal_d/flat_env_cfg.py` | ANYmal-D environment configs and observation groups (`system_state`, `system_action`, `system_contact`, `system_termination`). |
| `robotic_world_model` | `source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py` | Custom model-based environment. Implements `imagination_step` and the imagined-reward reconstruction. |
| `robotic_world_model` | `.../envs/anymal_d_manager_based_mbrl_env.py` | ANYmal-D specialization of the model-based environment. |
| `rsl_rl_rwm` | `rsl_rl/runners/mbpo_on_policy_runner.py` | Model-based runner. Coordinates real rollout, system-dynamics update, imagination, and PPO. |
| `rsl_rl_rwm` | `rsl_rl/algorithms/mbpo_ppo.py` | MBPO-PPO algorithm. Assembles system-dynamics loss and orchestrates updates. |
| `rsl_rl_rwm` | `rsl_rl/algorithms/ppo.py` | Standard clipped PPO loss. |
| `rsl_rl_rwm` | `rsl_rl/modules/system_dynamics.py` | `SystemDynamicsEnsemble`, the world model. State, contact, termination prediction heads. |
| `rsl_rl_rwm` | `rsl_rl/modules/architectures/rnn.py` | GRU base used by the dynamics model in the current configuration. |
| `rsl_rl_rwm` | `rsl_rl/modules/architectures/mlp.py` | State prediction head. Residual prediction in normalized state space. |