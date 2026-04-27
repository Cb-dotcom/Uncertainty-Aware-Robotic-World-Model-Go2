# Paper A Loss Map

This document records the exact loss structure of the Paper A implementation and separates the policy loss, the world-model loss, and the task reward. The separation matters because the method does not optimize one single scalar objective at one level. It uses multiple nested objectives.

## Loss layers

The Paper A code path contains three conceptually different objective layers:

1. The policy optimization objective, implemented through PPO.
2. The system-dynamics training objective, implemented through the system dynamics model and replay-buffer updates.
3. The task reward function, used inside both real rollouts and imagination rollouts.

These are related but not interchangeable.

## 1. PPO policy objective

Source: `upstream/rsl_rl_rwm/rsl_rl/algorithms/ppo.py`.

The PPO update computes:

- A **surrogate loss** using the probability ratio between the new and old action log-probs and the clipped PPO surrogate.
- A **value loss** using either a clipped value loss or plain squared error, depending on config. In the current Paper A config, clipped value loss is enabled.
- An **entropy bonus**, subtracted from the total objective as `entropy_coef * entropy`.

The total PPO objective is:

```
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
```

This comes directly from the bottom half of `rsl_rl/algorithms/ppo.py`, where the code computes `ratio`, `surrogate_loss`, `value_loss`, and the total PPO loss.

## 2. System-dynamics total loss

Sources:

- `upstream/rsl_rl_rwm/rsl_rl/algorithms/mbpo_ppo.py`
- `upstream/rsl_rl_rwm/rsl_rl/modules/system_dynamics.py`

The world-model update calls `self.system_dynamics.compute_loss(...)` and forms a weighted total loss:

$$
L_{dyn} =
w_s L_{state}
+ w_{seq} L_{sequence}
+ w_b L_{bound}
+ w_{kl} L_{kl}
+ w_e L_{extension}
+ w_c L_{contact}
+ w_t L_{termination}
$$

The configured weights, taken from `source/mbrl/mbrl/tasks/.../agents/rsl_rl_ppo_cfg.py` for the pretrain config, are:

- `state = 1.0`
- `sequence = 1.0`
- `bound = 1.0`
- `kl = 0.1`
- `extension = 1.0`
- `contact = 1.0`
- `termination = 1.0`

## 3. Exact system-dynamics terms

### 3.1 `state_loss`

Sources: `SystemDynamicsEnsemble.compute_state_loss(...)` and `SystemDynamicsEnsemble.compute_regression_loss(...)`.

The active call path uses `compute_regression_loss(..., loss_type="mse")`, because `compute_state_loss(...)` calls it without overriding `loss_type` and a grep showed no other call site overrides this. So the active path is the default MSE path.

One detail matters here. The state model predicts a mean and a std, but the default MSE branch samples from the predicted Gaussian:

$$
x_{pred} = \mu + \sigma \epsilon
$$

then computes the squared error against the target. The active state regression objective is therefore **sampled MSE**, not Gaussian NLL, even though Gaussian NLL support exists in the helper. This is a critical implementation fact.

### 3.2 `sequence_loss`

Also returned by `compute_regression_loss(...)`. However, in the current active RNN configuration, the system dynamics base sets `self.prediction_type = "single"` in `rsl_rl/modules/system_dynamics.py`. So for the current GRU path, sequence prediction is not the active mode in the same way as a sequence-output architecture. This term exists in the framework but is likely inactive or zero in the current Paper A GRU configuration.

### 3.3 `bound_loss`

Source: `SystemDynamicsEnsemble.compute_bound_loss(...)`.

Defined as the mean of max log-std minus the mean of min log-std. This regularizes the learned uncertainty bounds of the state head, which predicts bounded uncertainty: min log-std is learned, max log-std is derived from min plus `exp(delta)`, and predicted log-std is softly clamped between them.

### 3.4 `kl_loss`

Source: `SystemDynamicsEnsemble.compute_state_loss(...)`.

This term is nonzero only if the architecture type is `"rssm"`. The current Paper A pretrain config sets `architecture_config["type"] = "rnn"`, so for the current GRU configuration this term is expected to be inactive or zero.

### 3.5 `extension_loss`

Source: `SystemDynamicsEnsemble.compute_extension_loss(...)`.

This uses `nn.MSELoss()`. However, in the current ANYmal pretrain observation config, `system_extension` is not active, so this term is likely inactive or zero in the current Paper A path.

### 3.6 `contact_loss`

Source: `SystemDynamicsEnsemble.compute_contact_loss(...)`.

This uses `nn.BCEWithLogitsLoss()`. Targets come from `system_contact`, which is defined in `flat_env_cfg.py` and includes thigh contact and foot contact. Contact prediction is therefore trained as a binary classification problem over contact logits.

### 3.7 `termination_loss`

Source: `SystemDynamicsEnsemble.compute_termination_loss(...)`.

This uses `nn.BCEWithLogitsLoss()`. Targets come from `system_termination`, which currently includes base contact. Termination prediction is also trained as a binary classification problem.

## 4. Effective active world-model losses in current Paper A config

Given the active config (`ensemble_size = 1`, architecture type `rnn`, no active `system_extension`, prediction type `"single"`), the currently effective losses are:

**Active:** `state_loss`, `bound_loss`, `contact_loss`, `termination_loss`.

**Likely inactive or zero:** `sequence_loss`, `kl_loss`, `extension_loss`.

This is the current best code-grounded interpretation.

## 5. State head behavior

Source: `upstream/rsl_rl_rwm/rsl_rl/modules/architectures/mlp.py`.

The state head predicts a state mean and a state std, and the mean is computed as a residual:

```
predicted_state_mean = learned_delta + last_input_state
```

The model is therefore learning a residual transition in normalized state space rather than predicting the next state from scratch. This is a major implementation detail that is not obvious from the paper summary alone.

## 6. Task reward layer

The task reward is separate from both the system-dynamics loss and the PPO loss. It appears in two places.

**Real-env reward definition.** Mostly inherited from the IsaacLab locomotion configs, with local modifications in `flat_env_cfg.py` and `rewards.py`. Local terms include `joint_pos_stand_still` and `foot_clearance`.

**Imagined reward reconstruction.** Implemented in `ANYmalDManagerBasedMBRLEnv._compute_imagination_reward_terms(...)`. It reconstructs reward terms from imagined state and contact predictions, including velocity tracking, angular velocity tracking, torque penalty, acceleration penalty, action-rate penalty, feet air time, undesired contacts, stand still, flat orientation, and dof position limits.

The learned world model does not directly output reward. Imagined reward is reconstructed from predicted state and contact signals, which is what makes the learned model function as a simulator.

## 7. Uncertainty penalty

Sources: `ManagerBasedMBRLEnv._post_imagination_step(...)` and the imagination cfg in `rsl_rl_ppo_cfg.py`.

The imagined reward pipeline includes an uncertainty penalty term:

```
uncertainty_penalty_weight * epistemic_uncertainty * step_dt
```

In the current Paper A config, `uncertainty_penalty_weight = -0.0` and `ensemble_size = 1`. Since epistemic uncertainty is only nonzero when the ensemble has multiple members, the current active Paper A path is effectively not using uncertainty as a control penalty.

This strongly supports the interpretation that the current Paper A implementation is a learned neural simulator plus PPO pipeline, rather than the uncertainty-aware control pipeline expected in the Paper B sense.

## 8. Difference between paper equation and code

The paper-level world-model loss is presented compactly as a multi-step forecast loss over observation prediction and privileged-information prediction. The code decomposes this into state loss, contact loss, termination loss, optional extension loss, bound regularization, optional KL term, and optional sequence term.

The paper discussion also suggests probabilistic prediction, but the current active path uses sampled MSE rather than Gaussian NLL. The forecast-decay factor $\alpha^k$ from the paper was not identified in the traced code path.

The implementation captures the method spirit of Paper A, but it is not a literal one-line transcription of the compact paper formula. This is a normal research-code pattern and worth documenting explicitly.

## Conclusion

The Paper A code path optimizes three nested layers: the policy, via standard clipped PPO; the world model, via a decomposed multi-term system-dynamics loss; and the task, via a manually reconstructed locomotion reward applied inside imagination. This separation is essential for understanding the repo correctly and for mapping paper claims to code behavior.

---

**Rendering note.** The block equations on this page use `$$ ... $$` and inline math uses `$ ... $`. These will only render in MkDocs if the `pymdownx.arithmatex` extension is enabled and a MathJax (or KaTeX) script is loaded. See `mkdocs-nav-snippet.md` for the required `mkdocs.yml` additions.
