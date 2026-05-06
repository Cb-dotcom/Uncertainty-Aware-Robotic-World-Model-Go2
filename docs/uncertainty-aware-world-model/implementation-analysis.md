# Implementation Analysis: Uncertainty-Aware World Model

This page describes what the upstream codebase actually implements for the second paper (RWM-U + MOPO-PPO). It is the code-faithful counterpart to the [Paper Analysis](paper-analysis.md), which describes only what the paper says. The mapping between the two, including discrepancies and the configuration that activates each, is on the [Paper-to-Code Synthesis](paper-to-code-synthesis.md) page.

The base RWM implementation analysis is documented separately at [world-model/implementation-analysis.md](../world-model/implementation-analysis.md). This page assumes the reader is familiar with that material, in particular the `SystemDynamicsEnsemble` module structure, the dual-base GRU architecture, and the imagination loop mechanics.

## 1. Two model-based pipelines coexist in the repository

The most important structural finding for RWM-U is that the upstream repository contains **two separate model-based training pipelines**, sharing some backbone components but operating through different scripts, configs, environments, and runners. Both pipelines exist in the same checkout. They are distinguished by entry point, by whether they use Isaac Lab, and by which configuration values they activate.

### 1.1 The manager-based pipeline (online, base RWM)

This is the pipeline analyzed in detail in the [base RWM implementation analysis](../world-model/implementation-analysis.md).

Entry point: `scripts/reinforcement_learning/rsl_rl/train.py`.

Key components:

- Manager-based environment (`source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/`).
- `MBPOOnPolicyRunner` from `rsl_rl_rwm/runners/`.
- `MBPOPPO` algorithm from `rsl_rl_rwm/algorithms/`.
- Full Isaac Lab integration: the runner calls `env.step(...)` to collect fresh real-environment transitions every iteration.

Active configuration values (from `Pretrain-v0` / `Finetune-v0`):

```python
ensemble_size = 1
uncertainty_penalty_weight = -0.0
```

This pipeline implements *online* model-based RL. The world model and the policy are updated jointly using fresh data from the simulator each iteration. The uncertainty hooks (ensemble support, penalty term) are present in the code but switched off in the active configuration. This is the path used for `Init-v0`, `Pretrain-v0`, `Finetune-v0`, and `Visualize-v0` task IDs.

### 1.2 The standalone model-based pipeline (offline, RWM-U)

Entry point: `scripts/reinforcement_learning/model_based/train.py`.

Key components:

- Standalone environment classes (`scripts/reinforcement_learning/model_based/envs/base.py` and `envs/anymal_d_flat.py`).
- Inline policy training loop (`scripts/reinforcement_learning/model_based/policy_training.py`), with no separate runner class.
- Vanilla PPO from `rsl_rl_rwm/algorithms/PPO`, not `MBPOPPO`.
- No Isaac Lab dependencies (inspected by `grep`: no `isaaclab`, `IsaacLab`, `env.step`, `simulator`, or `sim.step` references were found in the offline path's scripts).

Active configuration values (from `configs/anymal_d_flat_cfg.py`):

```python
experiment_name = "offline"
ensemble_size = 5
uncertainty_penalty_weight = -1.0
dataset_root = "assets"
dataset_folder = "data"
```

This pipeline implements *offline* model-based RL. A pretrained world model checkpoint is loaded once at the start; policy training proceeds entirely through autoregressive imagination against this frozen model, with initial states sampled from a fixed dataset. No environment interaction occurs during policy training. The active values realize the paper's RWM-U configuration ($B = 5$ ensemble members, nonzero uncertainty penalty) under the sign and time-step convention documented in [paper-to-code synthesis §3.2](paper-to-code-synthesis.md#32-policy-optimization-table-s11): the shipped code's operational default corresponding to the paper's reported $\lambda = 1.0$ is `uncertainty_penalty_weight = -1.0`, with the caveat that the code also multiplies the uncertainty term by `step_dt`.

### 1.3 What the two pipelines share

The two pipelines share the backbone components in `upstream/rsl_rl_rwm/`:

- `rsl_rl/modules/system_dynamics.py`: the `SystemDynamicsEnsemble` class.
- `rsl_rl/modules/architectures/`: the GRU base and MLP heads.
- `rsl_rl/modules/actor_critic.py`: the policy and value network classes.
- `rsl_rl/algorithms/ppo.py`: the vanilla `PPO` algorithm.
- `rsl_rl/algorithms/mbpo_ppo.py`: the `MBPOPPO` algorithm (used only by the manager-based pipeline).

The `SystemDynamicsEnsemble` class is identical for both pipelines. Architectural details (shared `state_base` and `auxiliary_base` GRUs, $K$ independent heads in `nn.ModuleList`, hard-zero of epistemic uncertainty when `ensemble_size = 1`) are documented in [base RWM implementation analysis Section 4.1](../world-model/implementation-analysis.md#41-recurrent-base) and [Section 8.1](../world-model/implementation-analysis.md#81-ensemble-support). The same module is instantiated with different `ensemble_size` values in the two pipelines.

### 1.4 Algorithmic implication: MOPO-PPO is just PPO + custom env

A practical consequence of the offline pipeline using vanilla `PPO` rather than `MBPOPPO`: the "MOPO-PPO" framing in the paper is implemented mechanically as standard PPO consuming a reward stream from a custom imagination environment that applies the uncertainty penalty internally. There is no separate "MOPO-PPO algorithm class" in the codebase. The penalty is part of the env's reward function, not part of the policy update step.

This matches the paper's Algorithm 1 step 8 ("Update $\pi_\theta$ using PPO or another RL algorithm"), which treats the policy optimizer as interchangeable. The novel content of MOPO-PPO is the penalized-reward formulation in Eq. 5, not a new policy update rule.

## 2. The standalone offline pipeline: file-by-file

This section documents the components of the offline pipeline. The base-RWM analysis covers the manager-based pipeline; this section is focused on what is different for RWM-U.

### 2.1 Configuration: `configs/base_cfg.py` and `configs/anymal_d_flat_cfg.py`

The configuration is structured as a base config (defaults for any robot) plus a robot-specific override. The base config in `configs/base_cfg.py` declares the structure with conservative defaults:

```python
class EnvironmentCfg:
    uncertainty_penalty_weight: float = -0.0
    ...

class DataCfg:
    dataset_root: str = "logs/online"
    dataset_folder: str = "train"

class ModelCfg:
    ensemble_size: int = 1
```

The robot-specific config in `configs/anymal_d_flat_cfg.py` overrides these defaults to activate the RWM-U setting:

```python
experiment_name: str = "offline"
...
class EnvironmentCfg:
    uncertainty_penalty_weight: float = -1.0
    ...

class DataCfg:
    dataset_root: str = "assets"
    dataset_folder: str = "data"

class ModelCfg:
    ensemble_size: int = 5
```

The four values that distinguish RWM-U from the base configuration are therefore: ensemble size 5, uncertainty penalty weight -1.0, dataset path `assets/data/`, and the experiment name `"offline"`.

### 2.2 Entry script: `train.py`

`scripts/reinforcement_learning/model_based/train.py` is the offline pipeline's entry point. Its imports establish the dependency surface:

```python
from rsl_rl.modules import ActorCritic, SystemDynamicsEnsemble
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_obs_groups
```

No Isaac Lab imports. No `MBPOPPO` import. The `train.py` script provides a class with `_load_data`, `prepare_data`, `prepare_model`, `prepare_environment`, and `prepare_algorithm` methods that wire the pieces together. The key calls:

- `_load_data(dataset_root, dataset_folder, ...)`: loads CSV files from `assets/data/` via `pandas.read_csv`. The loader iterates files matching the dataset folder pattern, expecting fixed numeric layouts.
- `prepare_model(history_horizon, forecast_horizon, extension_dim, contact_dim, termination_dim, ensemble_size, architecture_config, ...)`: instantiates `SystemDynamicsEnsemble` with the configured ensemble size (5 for RWM-U). Optionally loads a pretrained checkpoint via `torch.load(resume_path)`.
- `prepare_environment(num_envs, max_episode_length, step_dt, reward_term_weights, uncertainty_penalty_weight, ...)`: instantiates the offline env (the standalone class from `envs/`, not the Isaac Lab env).
- `self.alg = PPO(...)` (line 245): instantiates the vanilla PPO algorithm. No `MBPOPPO`.

After construction, the script delegates to the policy training loop in `policy_training.py`.

### 2.3 Policy training loop: `policy_training.py`

The policy training loop is implemented inline in `policy_training.py` rather than through a runner class. It receives the constructed `alg: PPO`, the constructed env, and (indirectly, through the env) the system dynamics model. Its responsibilities:

- Initialize history buffers from the dataset by calling `env.prepare_imagination()`, which internally invokes `dataset.sample_batch` with the configured `init_data_ratio = 0.8` mixing.
- For each training iteration, perform `num_steps_per_env = 24` imagination steps under `torch.inference_mode()`.
- For each imagination step: call `self.alg.act(imagination_obs)` to sample actions, then `env.imagination_step(...)` to get the next predicted observation, the penalized reward, the done signal, the updated history buffers, and the epistemic uncertainty. Push the transition to PPO's storage via `self.alg.process_env_step(...)`.
- For RNN-style architectures (the active config has `architecture_config["type"] = "rnn"`), after the first step the history is truncated to the last entry (`state_history = state_history[:, -1:]`) since the GRU base maintains the temporal context internally through its hidden state.
- Log epistemic uncertainty per step: `epistemic_uncertainty[i] = uncertainty.mean(dim=0)`.
- After the imagination collection completes, call `self.alg.compute_returns(imagination_critic_obs)` once to compute GAE returns, then `self.alg.update()` for the PPO step.
- Aggregate the logged uncertainty across the rollout: `epistemic_uncertainty.mean(dim=0)` for the iteration's average.
- Log to Weights & Biases: `"Imagination/epistemic_uncertainty"`, `"Imagination/num_valid_imagination_envs"`. (The codebase uses `wandb`, not TensorBoard, in the offline pipeline.)
- Save a policy checkpoint (`policy_{it}.pt`) every `save_interval = 50` iterations.

There is no `env.step(...)` call. There is no real-environment data collection. The entire training loop runs against the frozen world model, with initial states sampled from the offline dataset.

### 2.4 Standalone environments: `envs/base.py` and `envs/anymal_d_flat.py`

The offline pipeline does not use the Isaac Lab `ManagerBasedRLEnv`. Instead it uses a custom lightweight env class in `envs/base.py` (the generic offline imagination env) extended by `envs/anymal_d_flat.py` (the ANYmal-D specialization).

Key responsibilities of the env:

- *Dataset management*: `set_dataset(...)`, `set_init_dataset(...)`, and a `dataset.sample_batch(num_envs, normalized=True)` method that draws state and action history windows for initial conditions.
- *Imagination step*: `imagination_step(actions, state_history, action_history)` is the equivalent of the Isaac Lab env's imagination_step. It calls `self.system_dynamics.forward(...)` to predict the next state, contacts, and termination, and returns the next observation, the reconstructed reward, the done signal, the updated history buffers, and the epistemic uncertainty.
- *Reward reconstruction*: same approach as the manager-based env's `_compute_imagination_reward_terms`. The active reward terms are reconstructed from the predicted state and contact signals, with the uncertainty penalty added at the end:

```python
rewards += self.uncertainty_penalty_weight * self.epistemic_uncertainty * self._step_dt
```

- *Random ensemble member assignment*: at reset time, each parallel imagination env is assigned a random ensemble member index:

```python
self.system_dynamics_model_ids = torch.randint(
    0, self.system_dynamics.ensemble_size,
    (1, self._num_envs, 1), device=self.device
)
```

This `model_ids` tensor is passed to `SystemDynamicsEnsemble.forward(...)`, which then gathers the prediction from the assigned member rather than averaging across all members.

The `model_ids` mechanism is structurally significant. The paper's Eq. 4 says the predicted observation at inference is the *mean* across ensemble members ($o'_{t+1} = \mathbb{E}_b[\mu^b_{o_{t+1}}]$). The code, when `model_ids` is provided, instead returns the prediction from a *single sampled member* per env. The ensemble mean is used only when `model_ids = None`. This is a documented paper-to-code divergence handled in the synthesis page.

## 3. The shipped offline assets

The repository ships two assets needed for end-to-end offline training:

### 3.1 The dataset: `assets/data/state_action_data_0.csv`

A single CSV file: 6.6 MB, 10,000 rows, 66 numeric columns. The schema corresponds to the manager-based pretraining observation groups concatenated in order:

| Column range | Count | Group |
|---:|---:|---|
| 1-3 | 3 | base linear velocity |
| 4-6 | 3 | base angular velocity |
| 7-9 | 3 | projected gravity |
| 10-21 | 12 | joint position (relative) |
| 22-33 | 12 | joint velocity (relative) |
| 34-45 | 12 | joint torque |
| 46-57 | 12 | system action |
| 58-65 | 8 | system contact (4 thigh + 4 foot, binary) |
| 66 | 1 | system termination (binary) |

This matches the `system_state` (45) + `system_action` (12) + `system_contact` (8) + `system_termination` (1) = 66-column layout used by `Pretrain-v0`, confirmed against Section 4.5 of the [base RWM implementation analysis](../world-model/implementation-analysis.md#45-policy-observation-versus-system-state).

At a 50 Hz control frequency, 10,000 rows represent approximately 200 seconds (3 minutes 20 seconds) of robot operation. This is sufficient for end-to-end pipeline validation but well below the 6M transitions reported in the paper's Table S10. A reproduction at paper scale would require additional dataset files following the same `state_action_data_N.csv` convention, either by generating them through the manager-based path's data export or by obtaining the full dataset from the project page.

The naming convention `state_action_data_0.csv` (zero-indexed) suggests the loader expects multiple files. The `_load_data` method in `train.py` iterates over CSV files in `assets/data/`, so dropping additional `state_action_data_1.csv`, `state_action_data_2.csv`, etc. into the directory would extend the dataset without code changes.

### 3.2 The pretrained model: `assets/models/pretrain_rnn_ens.pt`

A fresh checkout contains the Git LFS pointer until `git lfs pull` materializes the actual checkpoint file. The 133-byte ASCII pointer file records the SHA-256 hash and the actual file size:

```
version https://git-lfs.github.com/spec/v1
oid sha256:2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a
size 25054639
```

To materialize the checkpoint locally:

```bash
cd "$(git rev-parse --show-toplevel)/upstream/robotic_world_model"
git lfs install
git lfs pull
```

The 25 MB file size is consistent with a 2-base GRU (256 hidden, 2 layers each) plus 5 ensemble members of MLP heads at 128-dim, in float32 with no quantization. This is a real pretrained checkpoint, not a placeholder.

The checkpoint stores both the policy and the dynamics state from the run that produced it. After loading with `torch.load`, the dictionary has six top-level keys:

```python
['model_state_dict', 'system_dynamics_state_dict',
 'optimizer_state_dict', 'system_dynamics_optimizer_state_dict',
 'iter', 'infos']
```

The naming distinguishes the policy from the world model: `model_state_dict` holds the *policy* actor-critic weights, while `system_dynamics_state_dict` holds the *world model* (`SystemDynamicsEnsemble`) weights. Downstream tooling that only needs the dynamics, such as a held-out trajectory evaluator or a dynamics-only inference utility, must load `system_dynamics_state_dict` into `SystemDynamicsEnsemble`. Loading `model_state_dict` into `SystemDynamicsEnsemble` raises a key mismatch error (the keys are `actor.*` and `critic.*` rather than `state_base.*`, `state_heads.*`, etc.). This finding is documented in detail on the [RWM-U Execution Check §4](../validation/rwm-u-execution-check.md#4-checkpoint-key-finding-for-downstream-tooling).

After `git lfs pull`, the offline pipeline has the required static inputs for an end-to-end run: the dataset provides initial state-action histories, and the pretrained model provides the frozen dynamics model. The offline pipeline has been executed locally at reduced scale; results are documented on the [RWM-U Execution Check](../validation/rwm-u-execution-check.md) page.

## 4. Active RWM-U configuration values verified

The following values are verified from `scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py` (active overrides) and `scripts/reinforcement_learning/model_based/configs/base_cfg.py` (inherited defaults).

### 4.1 Configuration overrides in `AnymalDFlatConfig`

| Parameter | Verified value | Source |
|---|---:|---|
| `experiment_name` | `"offline"` | `anymal_d_flat_cfg.py` line 8 |
| `uncertainty_penalty_weight` | `-1.0` | `anymal_d_flat_cfg.py` line 30 |
| `dataset_root` | `"assets"` | `anymal_d_flat_cfg.py` line 36 |
| `dataset_folder` | `"data"` | `anymal_d_flat_cfg.py` line 37 |
| `batch_data_size` | `10000` | `anymal_d_flat_cfg.py` |
| `history_horizon` | `32` | `anymal_d_flat_cfg.py` |
| `forecast_horizon` | `8` | `anymal_d_flat_cfg.py` |
| `ensemble_size` | `5` | `anymal_d_flat_cfg.py` line 70 |
| `contact_dim` | `8` | `anymal_d_flat_cfg.py` |
| `termination_dim` | `1` | `anymal_d_flat_cfg.py` |
| `architecture_config.type` | `"rnn"` | `anymal_d_flat_cfg.py` |
| `architecture_config.rnn_type` | `"gru"` | `anymal_d_flat_cfg.py` |
| `architecture_config.rnn_num_layers` | `2` | `anymal_d_flat_cfg.py` |
| `architecture_config.rnn_hidden_size` | `256` | `anymal_d_flat_cfg.py` |
| `resume_path` | `"assets/models/pretrain_rnn_ens.pt"` | `anymal_d_flat_cfg.py` |
| `observation_dim` | `48` | `anymal_d_flat_cfg.py` |
| `action_dim` | `12` | `anymal_d_flat_cfg.py` |
| `learning_rate` (policy) | `1.0e-4` | `anymal_d_flat_cfg.py` |
| `entropy_coef` | `1.0e-4` | `anymal_d_flat_cfg.py` |
| `save_interval` | `50` | `anymal_d_flat_cfg.py` |
| `max_iterations` | `500` | `anymal_d_flat_cfg.py` |

### 4.2 Inherited defaults from `BaseConfig`

These values are inherited unchanged in the offline ANYmal-D configuration:

| Parameter | Inherited value | Notes |
|---|---:|---|
| `num_envs` | `8192` | Imagination env count |
| `max_episode_length` | `256` | Per-env episode length cap |
| `step_dt` | `0.02` | Time step (50 Hz) |
| `observation_noise` | `True` | Noise injection during imagination |
| `init_data_ratio` | `0.8` | 80% dataset init, 20% random init |
| `file_data_size` | `10000` | Per-file data buffer size |
| `num_eval_trajectories` | `10` | Evaluation rollouts |
| `len_eval_trajectory` | `400` | Evaluation rollout length |
| `num_steps_per_env` | `24` | Imagination steps per training iteration |
| `value_loss_coef` | `1.0` | PPO value loss weight |
| `clip_param` | `0.2` | PPO clip range $\epsilon$ |
| `num_learning_epochs` | `5` | PPO epochs per iteration |
| `num_mini_batches` | `4` | PPO minibatches per epoch |
| `gamma` | `0.99` | Discount factor |
| `lam` | `0.95` | GAE $\lambda$ |
| `desired_kl` | `0.01` | KL target for adaptive LR schedule |
| `max_grad_norm` | `1.0` | Gradient clipping norm |
| `schedule` | `"adaptive"` | Adaptive LR schedule based on KL |
| `actor_hidden_dims` | `[128, 128, 128]` | Policy network |
| `critic_hidden_dims` | `[128, 128, 128]` | Value network |
| `activation` | `"elu"` | Policy/value activation |
| `init_noise_std` | `1.0` | Initial action distribution std |

### 4.3 Reward term weights

The active reward term weights for the offline ANYmal-D configuration, from `EnvironmentConfig.reward_term_weights`:

```python
{
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 0.5,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -2.5e-5,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.5,
    "undesired_contacts": -1.0,
    "stand_still": -1.0,
    "flat_orientation_l2": -5.0,
    "dof_pos_limits": 0.0,
}
```

This is the same set of 11 nonzero-weighted reward terms documented for the manager-based pipeline ([base RWM implementation analysis, Section 5.9](../world-model/implementation-analysis.md#59-active-reward-terms)), with `dof_pos_limits` configured but inactive (weight 0). Both pipelines use the same reward weights for ANYmal-D flat locomotion.

## 5. What the offline pipeline does not include

For completeness, three things the offline pipeline does *not* implement, all of which are deferred to Phase 4 work:

1. *Dataset generation tooling.* The pipeline expects pre-existing CSV files. It does not include scripts to generate dataset files from the manager-based path's rollouts. Producing a paper-scale dataset (6M transitions) would require either running the manager-based path with data export enabled, or obtaining the original dataset.

2. *Real-world data collection.* The paper describes a safety-bounded autonomous data collection pipeline (Section A.4). The repository does not include this pipeline. Real-world data, if used, must be obtained externally and converted to the CSV format above.

3. *Simulation-to-reality transfer infrastructure.* The offline pipeline is designed for offline policy training. Deployment (sim-to-real) is a separate concern, handled outside the offline training scripts.

## 6. Summary of architectural findings

The findings most relevant for the [paper-to-code synthesis](paper-to-code-synthesis.md):

- *Mapped*: The two-pipeline structure cleanly separates online RWM (manager-based path) from offline RWM-U (standalone path). The shared `SystemDynamicsEnsemble` backbone is reused with different `ensemble_size` values.
- *Mapped*: The active offline configuration uses `ensemble_size = 5` and `uncertainty_penalty_weight = -1.0`, matching the paper's Table S10 for $B$ and mapping to the paper's $\lambda = 1.0$ under the sign and time-step convention documented in the synthesis page.
- *Mapped*: The shared-base ensemble architecture matches the paper's Section 4.1 exactly: bootstrap ensembles applied to prediction heads after a shared recurrent feature extractor.
- *Mapped*: The MOPO-PPO algorithm is implemented as vanilla PPO consuming a reward stream from an env that applies the uncertainty penalty inside its reward function. This matches Algorithm 1 in the paper, which treats the policy optimizer as interchangeable.
- *Mapped*: The dataset format (66-column CSV) matches the manager-based observation groups exactly.
- *Discrepancy noted*: The paper's Eq. 4 uses ensemble *mean* for the predicted observation. The code uses a *random per-env ensemble member* during imagination (`model_ids` mechanism), with the mean only used when `model_ids = None`.
- *Discrepancy noted*: The paper's Eq. 4 leaves the scalar reduction of vector-valued epistemic uncertainty implicit. The code reduces by summing across observation dimensions (`state_means.std(dim=0).sum(dim=1)`), not by averaging or taking a max.
- *Partially mapped*: The shipped dataset is 10,000 transitions; the paper uses 6M. The pipeline can run end-to-end with the shipped dataset, but quantitative reproduction requires more data.
- *Mapped*: The pretrained world model checkpoint is shipped via Git LFS (25 MB, requires `git lfs pull`).

These findings are consolidated into the [Paper-to-Code Synthesis](paper-to-code-synthesis.md) page with explicit per-claim status.