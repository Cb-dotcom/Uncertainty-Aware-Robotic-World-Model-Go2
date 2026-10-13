# Repository Structure

This page explains how the repository is organized and how the top-level directories map onto the simulation, task, and learning layers used by the project.

It is a navigation and architecture page. It should answer:

- Where does a kind of file live?
- Which repository layer owns which responsibility?
- Where should a reader go next for operational details?

Operational concerns such as cloning submodules, editing forked submodules, runtime patches, and updating submodule pointers are documented in [Submodules and Forks](../development/submodules-and-forks.md).

## Top-level workspace

The top-level repository is a research workspace, not a single monolithic application. It contains documentation, validation records, setup scripts, local Go2 configuration sources, and submodule pointers that pin the upstream codebases at known-good commits.

The simulator, task runtime, and learning backend live under `upstream/`.

```text
.
├── docs/                       Published project documentation
├── manifests/                  Frozen project-state snapshots
├── scripts/                    Local setup, validation, install, and patch scripts
├── upstream/
│   ├── IsaacLab/               Simulation infrastructure
│   ├── robotic_world_model/    RWM task/config/runtime layer
│   └── rsl_rl_rwm/             RL and world-model backend
├── .github/workflows/          GitHub Pages deployment workflow
├── .gitignore                  Files excluded from version control
├── .gitmodules                 Submodule registry
├── mkdocs.yml                  Documentation site configuration
└── README.md
```

## The three upstream layers

The simulation, method, and algorithm code is split across three repositories that form a stack. Each layer depends on the layer below it.

| Layer | Repository | Role | Primary responsibility |
|---|---|---|---|
| Simulation infrastructure | `upstream/IsaacLab` | Isaac Sim / Isaac Lab framework | Simulator launch, manager-based environments, robot assets, task registry primitives, RSL-RL wrappers. |
| Method and task layer | `upstream/robotic_world_model` | RWM task/runtime code | Task definitions, environment configs, model-based environment wrappers, training/visualization entrypoints. |
| RL and world-model backend | `upstream/rsl_rl_rwm` | Learning algorithms | PPO, MBPO-PPO, on-policy/model-based runners, replay buffer, system-dynamics ensemble, uncertainty machinery. |

A typical training command flows through the stack in this order:

```text
train.py
  -> Isaac Lab app launcher
  -> task registry
  -> robotic_world_model task/env config
  -> MBPO/RWM runner from rsl_rl_rwm
  -> rollout, world-model update, imagination, PPO update
```

## What belongs on this page vs Submodules and Forks

This page records the architectural role of each directory.

[Submodules and Forks](../development/submodules-and-forks.md) records the operational policy:

- which upstream repositories are forked or pinned,
- how to clone with submodules,
- how to install the Go2 runtime config,
- which runtime patches currently exist,
- which edits are committed versus working-tree only,
- how to avoid losing submodule-local changes.

In short:

```text
Repository Structure = where things live and how the layers fit together.
Submodules and Forks = how those upstream repositories are tracked, patched, and maintained.
```

## `docs/`

Published project documentation. It is built with MkDocs Material and deployed to GitHub Pages by the workflow in `.github/workflows/`.

This is the supervisor-facing technical record of the project. It is structured for readers, not for raw notes.

Typical contents:

```text
docs/
├── index.md
├── project/
├── setup/
├── development/
├── world-model/
├── go2-transfer/
└── experiments/
```

The exact navigation is controlled by `mkdocs.yml`.

## `manifests/`

Frozen snapshots of known-good project states, recorded at the moment a baseline is validated.

Each manifest captures:

- conda environment name and active Python interpreter path,
- upstream commit hashes and branches for submodules,
- the full validation command,
- relevant Python package versions,
- CUDA availability and device count,
- known non-blocking warnings or caveats,
- a status marker confirming the freeze stage.

A manifest is forensic. It should be sufficient to reconstruct a validated state on another machine or to diff against a future state when something stops working.

Manifests are not edited after creation. New validated states produce new timestamped manifest files.

## `scripts/`

Local setup, validation, install, render, and patch scripts.

This directory contains project-owned code that supports reproducibility and experimentation without modifying upstream repositories directly.

Important script groups include:

```text
scripts/phase4b/go2_rwm_config/
scripts/phase4b/install_go2_config.py
scripts/phase4b/render/
```

The Go2 RWM config source of truth lives under:

```text
scripts/phase4b/go2_rwm_config/
```

That config is installed into the `robotic_world_model` runtime tree by:

```bash
/isaac-sim/python.sh scripts/phase4b/install_go2_config.py --force
```

The installed copy under `upstream/robotic_world_model/.../config/go2/` should be treated as a runtime/build artifact. Edit the source under `scripts/phase4b/go2_rwm_config/`.

## `upstream/IsaacLab`

Isaac Lab provides the simulator and environment framework.

Responsibilities:

- Isaac Sim application launch,
- manager-based environment infrastructure,
- robot asset definitions,
- stock locomotion task templates,
- task registry primitives,
- wrappers used by RSL-RL training scripts.

In this project, Isaac Lab is infrastructure. The repository-structure role is simply:

```text
IsaacLab = simulator + environment framework + stock robot/task definitions
```

Operational details about whether it is pinned, forked, or carrying temporary working-tree patches belong in [Submodules and Forks](../development/submodules-and-forks.md).

## `upstream/robotic_world_model`

`robotic_world_model` is the method/task runtime layer.

Responsibilities:

- RWM task registration,
- ANYmal-D task family,
- Go2 task family after install,
- environment configs,
- model-based environment wrappers,
- training and visualization entrypoints,
- imagined-reward reconstruction,
- RWM/RWM-U task-specific runner configs.

Important conceptual role:

```text
robotic_world_model = task/config/runtime bridge between Isaac Lab and rsl_rl_rwm
```

The project-owned Go2 task source is not edited directly here first. It lives in the main repository under:

```text
scripts/phase4b/go2_rwm_config/
```

and is installed into the runtime location:

```text
upstream/robotic_world_model/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/go2/
```

This keeps the Go2 config reproducible from the top-level workspace.

## `upstream/rsl_rl_rwm`

`rsl_rl_rwm` is the learning backend.

Responsibilities:

- PPO,
- MBPO-PPO,
- on-policy runner,
- model-based runner,
- replay buffer,
- system-dynamics model,
- world-model ensemble,
- uncertainty estimation,
- imagination rollout integration.

Important conceptual role:

```text
rsl_rl_rwm = optimization algorithms + world-model backend
```

This layer is where the distinction between plain RWM/MBPO and uncertainty-aware RWM-U becomes operational through the runner, algorithm, ensemble, and uncertainty-penalty settings.

## Key runtime flow

A typical RWM/RWM-U run uses the following flow:

```text
1. User launches train.py with a task ID.
2. Isaac Lab starts Isaac Sim and resolves the task registry entry.
3. robotic_world_model provides the task config, env config, and runner config.
4. The environment is constructed through Isaac Lab manager-based env infrastructure.
5. rsl_rl_rwm creates the runner and algorithm.
6. Real rollouts are collected in Isaac Sim.
7. The system-dynamics model is updated from real data.
8. Finetune runs may generate imagined rollouts through the learned world model.
9. PPO updates the policy using real and/or imagined data depending on the runner config.
```

For Go2, the same high-level path is used after the Go2 config family is installed.

## Important file reference

This table is a navigation aid for readers new to the codebase. Deeper file-by-file analysis belongs in [Implementation Analysis](../world-model/implementation-analysis.md).

| Repository | Path | Role |
|---|---|---|
| `robotic_world_model` | `scripts/reinforcement_learning/rsl_rl/train.py` | Training entrypoint. Parses arguments, launches Isaac Sim, resolves task, creates env and runner. |
| `robotic_world_model` | `scripts/reinforcement_learning/rsl_rl/visualize.py` | RWM/MBPO visualization entrypoint. Used for model-based checkpoints. |
| `robotic_world_model` | `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py` | Registers the ANYmal-D task IDs. |
| `robotic_world_model` | `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py` | ANYmal-D runner configs, including system-dynamics architecture, ensemble size, and imagination settings. |
| `robotic_world_model` | `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py` | ANYmal-D flat environment config and observation groups. |
| `robotic_world_model` | `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/go2/__init__.py` | Installed Go2 task registration file. Generated from `scripts/phase4b/go2_rwm_config/`. |
| `robotic_world_model` | `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py` | Installed Go2 runner configs. Generated from `scripts/phase4b/go2_rwm_config/`. |
| `robotic_world_model` | `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py` | Installed Go2 flat environment config and reward terms. Generated from `scripts/phase4b/go2_rwm_config/`. |
| `robotic_world_model` | `source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py` | Custom model-based environment. Implements imagination stepping and imagined-reward reconstruction. |
| `robotic_world_model` | `source/mbrl/mbrl/mbrl/envs/anymal_d_manager_based_mbrl_env.py` | ANYmal-D specialization of the model-based environment. |
| `rsl_rl_rwm` | `rsl_rl/runners/mbpo_on_policy_runner.py` | Model-based runner. Coordinates real rollout, system-dynamics update, imagination, and PPO. |
| `rsl_rl_rwm` | `rsl_rl/algorithms/mbpo_ppo.py` | MBPO-PPO algorithm. Assembles system-dynamics loss and orchestrates updates. |
| `rsl_rl_rwm` | `rsl_rl/algorithms/ppo.py` | Standard clipped PPO loss. |
| `rsl_rl_rwm` | `rsl_rl/modules/system_dynamics.py` | `SystemDynamicsEnsemble`, the world model. Contains state, contact, and termination prediction heads. |
| `rsl_rl_rwm` | `rsl_rl/modules/architectures/rnn.py` | GRU base used by the current dynamics model configuration. |
| `rsl_rl_rwm` | `rsl_rl/modules/architectures/mlp.py` | MLP head used for state prediction and residual prediction in normalized state space. |
| `IsaacLab` | `source/isaaclab_assets/isaaclab_assets/robots/unitree.py` | Stock Unitree robot asset definitions, including Go2 asset and actuator values. |
| `IsaacLab` | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/` | Stock Isaac Lab Go2 locomotion task configs. Used as transfer reference and control baseline. |

## Go2-specific source-of-truth rule

For Go2 RWM/RWM-U work, use this rule:

```text
Edit:
  scripts/phase4b/go2_rwm_config/

Install to:
  upstream/robotic_world_model/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/go2/

Do not hand-edit the installed copy unless debugging a temporary runtime issue.
```

This avoids drift between the top-level repository and the runtime submodule tree.