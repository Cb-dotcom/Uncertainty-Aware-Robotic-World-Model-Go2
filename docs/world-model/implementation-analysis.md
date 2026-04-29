# Implementation Analysis

This page is the architectural description of the upstream RWM codebase. It documents what the code is, component by component, with explicit file references. It uses the convention vocabulary from the [landing page](../index.md): claims about *what the paper says* live on the [Paper Analysis](paper-analysis.md) page, claims about *paper-to-code correspondence* live on the [Paper-to-Code Synthesis](paper-to-code-synthesis.md) page, and forensic execution evidence lives on the [validation pages](../validation/baseline-execution.md).

Source repositories referenced throughout, with the locations from the [Repository Structure](../project/repository-structure.md) page:

- `upstream/robotic_world_model/` (tasks, configs, custom envs, training entrypoint)
- `upstream/rsl_rl_rwm/` (RL backend, system-dynamics model)
- `upstream/IsaacLab/` (simulation infrastructure)

## 1. Codebase decomposition

The implementation is split across three repositories with clean separation of concerns. Their roles are documented in [Repository Structure](../project/repository-structure.md). For implementation analysis, the relevant view is which file owns which method component:

| Method component | Repository | Path |
|---|---|---|
| Training entrypoint | `robotic_world_model` | `scripts/reinforcement_learning/rsl_rl/train.py` |
| Task registration | `robotic_world_model` | `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py` |
| Per-task runner configs | `robotic_world_model` | `.../config/anymal_d/agents/rsl_rl_ppo_cfg.py` |
| Per-task env configs | `robotic_world_model` | `.../config/anymal_d/flat_env_cfg.py` |
| Custom MBRL environment | `robotic_world_model` | `source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py` |
| ANYmal-D MBRL specialization | `robotic_world_model` | `.../config/anymal_d/envs/anymal_d_manager_based_mbrl_env.py` |
| Model-based runner | `rsl_rl_rwm` | `rsl_rl/runners/mbpo_on_policy_runner.py` |
| MBPO-PPO algorithm | `rsl_rl_rwm` | `rsl_rl/algorithms/mbpo_ppo.py` |
| Standard PPO | `rsl_rl_rwm` | `rsl_rl/algorithms/ppo.py` |
| System dynamics module | `rsl_rl_rwm` | `rsl_rl/modules/system_dynamics.py` |
| GRU recurrent base | `rsl_rl_rwm` | `rsl_rl/modules/architectures/rnn.py` |
| State and auxiliary heads | `rsl_rl_rwm` | `rsl_rl/modules/architectures/mlp.py` |
| Replay buffer | `rsl_rl_rwm` | `rsl_rl/storage/` |

Throughout the rest of this page, file references are abbreviated where context makes the location unambiguous.

## 2. Task registration

The four ANYmal-D task IDs are registered in `anymal_d/__init__.py` via `gym.register(...)` calls. Each registration binds a task ID to two entry points: an environment config class and an agent (runner) config class. The four registrations are documented in [Task Modes](task-modes.md).

The most consequential aspect for implementation analysis is that the four tasks share most of their machinery. They differ in:

1. Which observation groups are active in the env config.
2. Which runner class is selected by `agent_cfg.class_name`.
3. Which dynamics model parameters and imagination settings are configured.
4. Which environment class is instantiated (generic `ManagerBasedRLEnv` versus the custom `ANYmalDManagerBasedMBRLEnv`).

The shared configuration scaffolding lives in the parent classes; the task-specific overrides live in the four `_INIT`, `_PRETRAIN`, `_FINETUNE`, `_VISUALIZE` config variants.

## 3. Pretraining configuration

The pretraining runner config `AnymalDFlatPPOPretrainRunnerCfg` is the canonical reference for what the model-based path looks like. It is defined in `rsl_rl_ppo_cfg.py`.

The class sets:

```python
class_name = "MBPOOnPolicyRunner"
```

This selects the model-based runner. Standard PPO uses `OnPolicyRunner`; this selection is what activates everything described in the rest of this page.

It also defines a `system_dynamics` configuration block that controls the dynamics model. Key fields with the values used by Pretrain-v0:

```python
system_dynamics = RslRlSystemDynamicsCfg(
    architecture_config={"type": "rnn", "rnn_type": "gru",
                         "rnn_num_layers": 2, "rnn_hidden_size": 256},
    state_head_config={"hidden_dims": [128]},
    contact_head_config={"hidden_dims": [128]},
    termination_head_config={"hidden_dims": [128]},
    history_horizon=32,
    system_dynamics_forecast_horizon=8,
    ensemble_size=1,
    ...
)
```

These values match the paper's reported configuration ($M = 32$, $N = 8$, GRU 256x256, MLP heads 128). The `ensemble_size = 1` collapses the ensemble to a single network; this is the structural hook reserved for the RWM-U extension and is documented in Section 8 below.

The pretraining config also disables imagination:

```python
imagination_cfg.num_envs = 0
imagination_cfg.num_steps_per_env = 0
```

In this stage the dynamics model is trained, but the policy is not rolled out through it. PPO updates use real-environment transitions only.

## 4. System dynamics module

`SystemDynamicsEnsemble` in `rsl_rl/modules/system_dynamics.py` is the world model. The "ensemble" naming reflects the design: the class supports an ensemble of $K$ networks indexed by member, and the active configuration uses $K = 1$.

Each ensemble member contains:

- A recurrent base (`rnn.RNNBase` in the current config, which is a 2-layer GRU with hidden size 256).
- A state prediction head (mean and standard deviation over the next observation).
- A contact prediction head (logits for binary contact targets).
- A termination prediction head (logits for binary termination targets).
- An optional extension head (not active in the ANYmal-D pretraining config).

### 4.1 Recurrent base

`rnn.py` implements a GRU base that ingests observation-action pairs sequentially and produces a hidden representation. For the current configuration:

- 2 stacked GRU layers, hidden size 256.
- Inputs at each step are concatenated normalized observation and action.
- Output at each step is the GRU hidden state, fed to the prediction heads.

The implementation supports both GRU and LSTM via the `rnn_type` parameter, but only GRU is active in the ANYmal-D config.

### 4.2 State prediction head

The state head is implemented in `mlp.py`. It produces both a mean and a standard deviation for a Gaussian distribution over the next observation.

The mean head computes:

$$
\mu_\phi = \text{MLP}_\mu(h) + x_{\text{state}}^{\text{last}}
$$

where $h$ is the GRU output, $\text{MLP}_\mu$ is the mean head's network, and $x_{\text{state}}^{\text{last}}$ is the last input state in the history window. The MLP learns a *residual* on top of the most recent state, not the next state from scratch. This is implemented in code as roughly:

```python
state_mean = self.state_mean_layers(x) + x_state_batch[:, -1]
```

This is a non-obvious implementation choice. The paper's Eq. 1 specifies the predicted observation distribution $p_\phi(\cdot \mid \ldots)$ but does not specify whether the mean is predicted directly or as a residual. The residual form is a stability choice: predicting small deltas is easier than predicting absolute states, especially when the simulator step time is short (50 Hz means $\Delta t = 0.02$ s, so per-step state changes are small).

The residual is computed in *normalized* state space, not absolute units. The predicted next state is then denormalized when consumed downstream.

### 4.3 State standard deviation

The standard deviation head produces a log standard deviation, which is then exponentiated. The log std is *bounded* between learned minimum and maximum values:

```text
log_std_predicted = clamp(log_std_head(h), log_std_min, log_std_max + delta_max)
```

The min and max log std are learned parameters, not fixed hyperparameters. The bound regularization (Section 5) penalizes the gap between min and max log std to keep them from drifting apart.

The predicted std is used to construct the Gaussian over the next observation. With reparameterization, the sampled state becomes:

$$
o' = \mu_\phi + \sigma_\phi \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

This is the standard reparameterized Gaussian sampler. The paper's Section 3.2 acknowledges the use of reparameterization for gradient flow.

### 4.4 Auxiliary heads

The contact head and termination head are MLPs producing logits. They are trained as binary classifiers (one logit per binary target):

- Contact head output: 8 logits for ANYmal-D (4 thigh contacts + 4 foot contacts).
- Termination head output: 1 logit (base contact, used as the failure signal).

At inference time, the logits pass through a sigmoid and are rounded to produce binary signals. The contact predictions are used in the imagination reward reconstruction (the reward function depends on contact patterns); the termination prediction is used to truncate imagined episodes when the model predicts the base has fallen.

The optional extension head is configured but not active in the ANYmal-D pretraining config (the corresponding `system_extension` observation group is not registered).

## 5. System dynamics loss

The total system dynamics loss is computed inside `SystemDynamicsEnsemble.compute_loss(...)` and is assembled by the MBPO-PPO algorithm in `mbpo_ppo.py`. It is a weighted sum of seven terms:

$$
L_{\text{dyn}} = w_s L_{\text{state}} + w_{\text{seq}} L_{\text{seq}} + w_b L_{\text{bound}} + w_{\text{kl}} L_{\text{kl}} + w_e L_{\text{ext}} + w_c L_{\text{contact}} + w_t L_{\text{term}}
$$

The configured weights for the ANYmal-D pretraining config are:

```text
w_s    state         = 1.0
w_seq  sequence      = 1.0
w_b    bound         = 1.0
w_kl   kl            = 0.1
w_e    extension     = 1.0
w_c    contact       = 1.0
w_t    termination   = 1.0
```

The seven terms are not all active in the current configuration. Sections 5.1 through 5.7 describe what each term computes and whether it contributes in the active path.

### 5.1 State loss

The state loss is computed via a regression helper in `system_dynamics.py`. The helper supports two modes:

- `loss_type="mse"` (mean squared error against the target).
- `loss_type="nll"` (Gaussian negative log-likelihood).

The active call path is:

```python
loss = self.compute_regression_loss(predicted, target, loss_type="mse")
```

with no override anywhere in the codebase that sets `loss_type="nll"`. This is an implementation finding worth flagging because the *architecture* predicts both a mean and a standard deviation (Section 4.2 and 4.3), which would be the natural input for an NLL loss. The current code samples from the predicted Gaussian and takes squared error against the target, which uses the std for variance scaling but not as a likelihood term.

The result: the state loss is **sampled MSE**, not Gaussian NLL. The paper's Eq. 2 specifies $L_o$ abstractly as a discrepancy measure and does not commit to either form, but the architecture's prediction of a full Gaussian implies NLL would be the canonical choice. The codebase's sampled MSE is a reasonable proxy under reparameterization but is not the same loss.

This is a **discrepancy noted** finding for the synthesis page.

### 5.2 Sequence loss

The sequence loss is also returned by the regression helper. It is a separate prediction-head output that operates over a multi-step sequence rather than a single next step.

In the current configuration, the system dynamics module has:

```python
self.prediction_type = "single"
```

set in `system_dynamics.py`. With this setting, the GRU path produces single-step predictions and the sequence prediction head is not active. The `sequence` loss term is therefore zero or near-zero in the active configuration.

The component exists in the framework for architectures that produce sequence outputs (e.g., a transformer producing all forecast steps in one forward pass). The current GRU does not.

### 5.3 Bound loss

The bound loss regularizes the gap between the learned min and max log standard deviation. It is computed roughly as:

$$
L_{\text{bound}} = \mathbb{E}\left[\log \sigma_{\max}\right] - \mathbb{E}\left[\log \sigma_{\min}\right]
$$

with the convention that both expectations are taken over batch elements and observation dimensions. The loss is positive when max log std exceeds min log std (the normal case) and is minimized by keeping them close.

This term keeps the predicted uncertainty interval from collapsing to zero (which would make the std meaningless) or growing unboundedly (which would make every prediction nearly noise).

The bound loss is active in the current configuration and contributes to training.

### 5.4 KL loss

The KL term is only active for RSSM-style architectures, where the model maintains a posterior over a latent variable and the KL divergence between prior and posterior is part of the ELBO objective.

The current ANYmal-D configuration uses `architecture_config["type"] = "rnn"`, not RSSM. The KL term is therefore zero in the active path. The weight of 0.1 in the config is preserved in case the architecture is later switched to RSSM, but for the GRU path it has no effect.

### 5.5 Extension loss

The extension loss is `nn.MSELoss()` applied to a separate auxiliary head that predicts an "extension" observation group. This head exists in the framework for tasks that have additional privileged information beyond contacts (for example, terrain geometry or external forces).

The ANYmal-D pretraining config does not register a `system_extension` observation group. The extension loss is therefore zero or inactive in the current path.

### 5.6 Contact loss

The contact loss is `nn.BCEWithLogitsLoss()` applied to the contact head's logits against binary contact targets. Targets come from the `system_contact` observation group registered in `flat_env_cfg.py`, which captures thigh and foot contacts (8 binary signals for ANYmal-D).

This term is active and contributes meaningfully to training. The contact predictions are critical because the imagination reward function depends on contact patterns (feet air time, undesired contacts, foot clearance).

### 5.7 Termination loss

The termination loss is `nn.BCEWithLogitsLoss()` applied to the termination head's logit against the binary termination target. The target comes from the `system_termination` observation group, which registers base contact as the failure signal.

This term is active. The termination prediction is used inside imagination to decide when to truncate an imagined episode, which affects how PPO computes returns.

### 5.8 Active versus inactive losses

Summary of which terms contribute in the current ANYmal-D pretraining configuration:

| Term | Status | Reason |
|---|---|---|
| State (sampled MSE) | Active | Default regression mode. |
| Sequence | Inactive (zero) | `prediction_type = "single"` for GRU path. |
| Bound | Active | Regularizes learned std bounds. |
| KL | Inactive (zero) | Only used for RSSM architectures. |
| Extension | Inactive (zero) | No `system_extension` observation group registered. |
| Contact | Active | BCE on thigh and foot contacts. |
| Termination | Active | BCE on base contact. |

Of the seven configured loss terms, four are active (state, bound, contact, termination). This is a *Mapped* observation about the implementation; the paper does not specify which auxiliary terms should be present.

## 6. Replay buffer

The system dynamics replay buffer in `rsl_rl/storage/` stores sequences of observations and actions for world-model training. The buffer interface is consumed by `MBPOOnPolicyRunner.update_system_dynamics(...)`, which samples sequences of length $M + N$ from the buffer for each training mini-batch.

The buffer stores normalized state and action sequences. Normalization is handled by `EmpiricalNormalization` modules instantiated by the runner; the buffer holds normalized values directly so the dynamics model sees normalized inputs end-to-end.

For pretraining, the buffer is populated from real-environment rollouts. For finetuning, the buffer continues to receive real-environment rollouts (real interactions are not skipped just because imagination is enabled), and additionally the imagination loop produces imagined transitions that are *not* added to the system-dynamics buffer (those go to the imagination storage for PPO).

## 7. Model-based runner

`MBPOOnPolicyRunner` in `mbpo_on_policy_runner.py` is the model-based runner. It extends the base on-policy runner with three responsibilities not present in standard PPO:

1. Constructing and owning the `SystemDynamicsEnsemble` and the state and action `EmpiricalNormalization` modules.
2. Coordinating system-dynamics updates on top of the standard rollout-and-policy-update loop.
3. Injecting model and normalizer references into the environment so the custom MBRL env can step imagined transitions.

The runner's `learn(...)` method is the training loop entry point. The temporal sequence of operations inside `learn(...)` is documented on the [Runtime Pipeline](runtime-pipeline.md) page; this page describes the runner's *structure* rather than its execution order.

The injection step (item 3) sets attributes on `env.unwrapped`:

```python
env.unwrapped.system_dynamics = self.system_dynamics
env.unwrapped.imagination_state_normalizer = self.state_normalizer
env.unwrapped.imagination_action_normalizer = self.action_normalizer
env.unwrapped.num_imagination_envs = ...
env.unwrapped.uncertainty_penalty_weight = ...
# and others
```

This is the architectural bridge that lets the custom MBRL env reach into the runner's owned state to step imagined transitions. From an engineering standpoint it is unusual (the env reaches "up" into the runner), but it allows the imagination logic to live in the env class rather than the runner class.

## 8. Uncertainty hooks

The codebase contains the structural hooks for the RWM-U extension, but they are switched off in the active ANYmal-D pretraining configuration.

### 8.1 Ensemble support

`SystemDynamicsEnsemble` accepts an `ensemble_size` parameter. With `ensemble_size > 1`, the module instantiates $K$ independent copies of the prediction heads sharing the GRU base. Each member has its own state, contact, and termination heads. The forward method returns predictions from all $K$ members.

In the current configuration:

```python
ensemble_size = 1
```

This collapses the ensemble to a single member. Epistemic uncertainty (defined as variance across ensemble means) is structurally zero because there is only one mean to compute variance over.

### 8.2 Uncertainty penalty in imagination

The custom MBRL env's `_post_imagination_step(...)` method includes a hook for an uncertainty penalty:

```python
imagined_reward -= self.uncertainty_penalty_weight * epistemic_uncertainty * step_dt
```

In the current configuration:

```python
uncertainty_penalty_weight = -0.0
```

(or equivalently 0.0). The penalty term contributes nothing to the imagined reward.

Combined with `ensemble_size = 1`, this means the active path implements RWM with no uncertainty handling. The `uncertainty_penalty_weight` and `ensemble_size` parameters are reserved for the RWM-U extension; activating them switches the configuration toward Paper 2's behavior.

This is a **discrepancy noted** finding for the synthesis page: the codebase already contains the surface area for uncertainty-aware behavior, but the active configuration does not exercise it.

### 8.3 Aleatoric versus epistemic

The state head's predicted standard deviation (Section 4.3) captures *aleatoric* uncertainty (irreducible per-prediction noise) within each ensemble member. With $K = 1$, this is the only uncertainty signal available. With $K > 1$, the variance across ensemble means provides the *epistemic* uncertainty signal that RWM-U's MOPO-PPO penalty is designed to use.

The aleatoric signal is computed and used inside the state loss (via the predicted std in the bound loss and in the reparameterized sampling), but it is not consumed by the policy as a control signal. The epistemic signal is consumed by the policy via the uncertainty penalty hook, but is structurally zero in the current configuration.

## 9. Imagination environment

The custom MBRL env `ManagerBasedMBRLEnv` (in `manager_based_mbrl_env.py`) and its ANYmal-D specialization `ANYmalDManagerBasedMBRLEnv` implement the imagination logic. The runtime sequence of an imagination step is documented on the [Runtime Pipeline](runtime-pipeline.md) page; this section describes the architectural pieces.

The env owns:

- An imagination state buffer (the rolling window of past observations used as input to the dynamics model).
- An imagination action buffer (the corresponding action history).
- References to the runner's dynamics model and normalizers (injected per Section 7).

The env's `imagination_step(...)` method performs the prediction. The `_compute_imagination_reward_terms(...)` method reconstructs the reward function from the predicted state and contact signals. The reward terms are the ones documented in the [paper's Section A.1.2](paper-analysis.md#6-reward-formulation): velocity tracking, torque penalty, action rate, feet air time, undesired contacts, flat orientation, foot clearance, joint deviation.

The reward reconstruction is a literal Python translation of the paper's reward equations applied to predicted observations rather than simulator-reported observations. This is the operational meaning of "the world model acts as a simulator": the env produces both states and reward signals from the learned model.

## 10. Checkpointing

The model-based runner's `save(...)` method serializes both policy and dynamics state to a single checkpoint file. The fields are:

```text
model_state_dict                       (policy network parameters)
system_dynamics_state_dict             (dynamics model parameters)
optimizer_state_dict                   (PPO optimizer)
system_dynamics_optimizer_state_dict   (dynamics model optimizer)
iter                                   (iteration number)
infos                                  (run metadata)
```

The dual save format is what allows the finetuning stage to load a pretrained dynamics model from a pretraining checkpoint. The finetuning runner config sets `load_system_dynamics = True` and provides a `system_dynamics_load_path`; the runner's `load(...)` method reads the dynamics state from the indicated checkpoint and the policy state from the resume run directory (which can be the same checkpoint or different).

The reduced-scale validation run on local hardware produced two checkpoint files (`model_0.pt`, `model_1.pt`), each approximately 18 MB. The size is dominated by the GRU base parameters (~250K parameters) plus the four prediction heads.

## 11. The pretraining fix

The model-based runner contained a structural bug that prevented the Pretrain-v0 task from completing. The bug was identified during the local pretraining validation and was fixed via a small project-applied patch.

### 11.1 The bug

Inside `MBPOOnPolicyRunner`, the cleanup logic at the end of an iteration includes:

```python
self.imagination_infos.clear()
```

The `self.imagination_infos` attribute is created lazily, only when imagination is enabled. The class does not initialize it in `__init__`; it is set on first use during the imagination rollout.

In the Pretrain-v0 task, imagination is disabled (`num_imagination_envs = 0`). The imagination rollout never runs, so `self.imagination_infos` is never created. The cleanup call then raises `AttributeError: 'MBPOOnPolicyRunner' object has no attribute 'imagination_infos'`.

The crash occurs at the end of the first iteration. The pretraining stage cannot complete with the unmodified upstream code.

### 11.2 The fix

The minimal fix guards the clear call with a `hasattr` check:

```python
if hasattr(self, "imagination_infos"):
    self.imagination_infos.clear()
```

This preserves the original behavior when imagination is enabled (the attribute exists, the clear runs) and adds a safe no-op when imagination is disabled (the attribute does not exist, the clear is skipped).

The fix is applied in the project fork of `rsl_rl_rwm` and is operationally documented in [Submodules and Forks](../development/submodules-and-forks.md). The behavioral change is restricted to the disabled-imagination path; the enabled-imagination path (used by Finetune-v0) is unaffected.

### 11.3 Why this matters architecturally

The bug reflects a structural inconsistency in the upstream code: a class attribute is treated as if it were initialized at construction time, but is actually created only when a downstream code path runs. The Pretrain-v0 task is the only configuration that exercises the model-based runner without imagination, so the bug is unique to this stage. The standard PPO runner (used by Init-v0) does not call into this code path; the finetune configuration always enables imagination.

The fix is in the runner rather than in the cleanup call site because the attribute itself should arguably be initialized in `__init__`. Either approach (init the attribute or guard the clear) addresses the bug; the `hasattr` guard is the smaller change.

## 12. Summary of architectural findings

The implementation analysis identifies the following findings, classified by claim type:

**Mapped** (paper concept identified in code):

- The dual-autoregressive training scheme is implemented in `system_dynamics.py` and the GRU base in `rnn.py`, matching the paper's $M = 32$, $N = 8$ configuration.
- The MBPO-PPO algorithm is implemented in `mbpo_ppo.py` and orchestrated by `mbpo_on_policy_runner.py`, matching Algorithm 1 of the paper.
- The reward reconstruction in `_compute_imagination_reward_terms(...)` matches the paper's Section A.1.2 reward equations.

**Discrepancy noted** (code diverges from paper in a documented way):

- The state loss is **sampled MSE**, not Gaussian NLL. The architecture predicts a full Gaussian (mean and std), which the paper's notation suggests would be evaluated as a likelihood, but the active call path uses MSE against a sampled point estimate. Both are reasonable implementations; the choice is not noted in the paper.
- The state mean is predicted as a **residual** in normalized state space ($\mu_\phi = \text{MLP}_\mu(h) + x_{\text{state}}^{\text{last}}$) rather than as a direct prediction of the next state. This is a stability choice not specified in the paper.
- The codebase contains **uncertainty hooks** (ensemble support, uncertainty penalty in imagined reward) that are switched off in the active configuration. These are the structural surface for the RWM-U extension.

**Partially mapped** (component exists with simplifications or scope changes):

- The system-dynamics loss has seven configured terms, of which four are active in the current GRU configuration (state, bound, contact, termination). The other three (sequence, KL, extension) are inactive due to architecture and observation-group choices.
- The pretraining stage required a **project-applied fix** to handle the disabled-imagination path; the unmodified upstream code crashes at the end of the first iteration.

These findings are the input to the [Paper-to-Code Synthesis](paper-to-code-synthesis.md) page, which aligns each paper claim with its code location and notes the discrepancies above against the paper's stated method.