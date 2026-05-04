# Paper-to-Code Synthesis

This page aligns the RWM paper's claims with their implementation in the upstream codebase, surfaces the discrepancies between the two, and serves as the cross-reference between the [Paper Analysis](paper-analysis.md) and [Implementation Analysis](implementation-analysis.md) pages.

The two source pages should be treated as authoritative for their respective domains: paper-analysis for what the paper claims, implementation-analysis for what the code does. This page neither restates them nor adds new claims; it maps one onto the other.

## How to read this page

Each row of the synthesis tables uses the [mapping convention](../index.md#conventions) from the landing page:

- **Mapped**: paper concept identified in code.
- **Partially mapped**: identified with documented gaps.
- **Discrepancy noted**: code diverges from paper in a documented way.
- **Not mapped**: no code location identified.

The discrepancies are the higher-leverage findings of the analysis. They are summarized in Section 4 of this page after the mapping tables.

## 1. Method components

| Paper concept | Paper section | Code location | Mapping status |
|---|---|---|---|
| World model $p_\phi$ | §3.1, Eq. 1 | `rsl_rl/modules/system_dynamics.py` (`SystemDynamicsEnsemble`) | Mapped |
| GRU recurrent base | §3.2, Table S7 | `rsl_rl/modules/architectures/rnn.py` | Mapped |
| State prediction head (mean and std) | §3.2 | `rsl_rl/modules/architectures/mlp.py` | Mapped (with discrepancy on residual form, see §4.1) |
| Privileged information prediction (contacts, terminations) | §3.1, Table S3 | Auxiliary heads in `system_dynamics.py` | Mapped |
| Inner autoregression (GRU update across history) | §3.2, Fig. S6 | GRU recurrent forward pass in `rnn.py` | Mapped |
| Outer autoregression (predicted observations fed back over forecast horizon) | §3.2, Fig. 2a | Forecast loop in `system_dynamics.py` `compute_loss(...)` | Mapped |
| Multi-step prediction loss (Eq. 2) | §3.2 | `compute_loss(...)` aggregation in `system_dynamics.py` | Partially mapped (discrepancy on $L_o$ form, see §4.2) |
| Reparameterization trick | §3.2 | $\mu_\phi + \sigma_\phi \cdot \epsilon$ in MLP state head | Mapped |
| Forecast decay $\alpha$ | §3.2, Eq. 2 | Per-step weighting in loss aggregation | Partially mapped (paper sets $\alpha = 1.0$, code path supports decay but config uses 1.0) |
| MBPO-PPO algorithm (Algorithm 1) | §3.3 | `mbpo_on_policy_runner.py` `learn(...)` | Mapped |
| Imagined action selection (Eq. 3) | §3.3 | Policy invocation inside `imagination_step(...)` | Mapped |
| Imagined reward computation | §3.3, §A.1.2 | `_compute_imagination_reward_terms(...)` in `manager_based_mbrl_env.py` | Mapped |
| Replay buffer $\mathcal{D}$ | §3.3 | `rsl_rl/storage/` | Mapped |
| 100-step imagination rollout | §3.3, Table S11 | `imagination_step` loop, `num_imagination_steps_per_env = 24` (per-env) × 8192 envs | Partially mapped (per-env step count differs from paper's 100; the policy still sees long autoregressive chains but the runner config splits them across envs) |
| Policy observation versus system state | Paper notation uses observation broadly | `policy` group, `system_state` group, and the imagination env's policy-observation reconstruction | Partially mapped: the dynamics model predicts `system_state`, while the policy observation is reconstructed from predicted state plus command and previous action |

## 2. Reward terms

The paper defines a reward structure for the locomotion task in Section A.1.2. The implementation does not redefine these rewards from scratch; it inherits the standard Isaac Lab locomotion reward set from `velocity_env_cfg.RewardsCfg` and applies project-specific modifications. The active code-level reward set is therefore not identical to the paper's full reward table and should be tracked separately.

The 11 reward terms active in the `Pretrain-v0` and `Finetune-v0` tasks, with their effective weights and source, are:

| Active code reward term | Effective weight | Source | Status |
|---|---:|---|---|
| `track_lin_vel_xy_exp` | `+1.0` | inherited | mapped |
| `track_ang_vel_z_exp` | `+0.5` | inherited | mapped |
| `lin_vel_z_l2` | `-2.0` | inherited | mapped |
| `ang_vel_xy_l2` | `-0.05` | inherited | mapped |
| `dof_torques_l2` | `-2.5e-5` | inherited, overridden by project | mapped |
| `dof_acc_l2` | `-2.5e-7` | inherited | mapped |
| `action_rate_l2` | `-0.01` | inherited | mapped |
| `feet_air_time` | `+0.5` | inherited, overridden by project | mapped |
| `undesired_contacts` | `-1.0` | inherited | mapped |
| `flat_orientation_l2` | `-5.0` | inherited (default 0.0), overridden by project | mapped |
| `stand_still` | `-1.0` | added by project | active in code, paper correspondence to verify |

`dof_pos_limits` is configured but inactive (weight 0.0). The `Init-v0` baseline task reverts the three flat-terrain overrides and therefore has different effective weights for `flat_orientation_l2`, `dof_torques_l2`, and `feet_air_time`.

Reward terms listed in the paper that are not in this active code set should be treated as paper-level terms until their code activation is verified.

## 3. Training and architecture parameters

| Parameter | Paper value | Inspected code value | Status |
|---|---:|---:|---|
| History horizon $M$ | 32 | 32 | matched |
| Forecast horizon $N$ | 8 | 8 | matched |
| Forecast decay $\alpha$ | 1.0 | 1.0 (default) | matched |
| Step time $\Delta t$ | 0.02 s | 0.02 s | matched |
| GRU hidden size | 256 | 256 | matched |
| GRU layers | 2 | 2 | matched |
| State mean head | 128 | `[128]` | matched at architecture level |
| State log-std head | 128 | `[128]` | matched at architecture level |
| Contact head | 128 | `[128]` | matched at architecture level |
| Termination head | 128 | `[128]` | matched at architecture level |
| Ensemble size | to verify for base RWM | 1 | active code uses single member |
| Policy learning rate | to verify | `1.0e-3` | code value verified, paper comparison pending |
| System-dynamics learning rate | to verify | `1.0e-3` | code value verified, paper comparison pending |
| System-dynamics weight decay | to verify | `0.0` | code value verified, paper comparison pending |
| System-dynamics mini-batches | to verify | `20` | code value verified |
| System-dynamics mini-batch size | to verify | `5000` | code value verified |
| System-dynamics replay buffer size | to verify | `1000` | code value verified |
| Pretrain max iterations | to verify | `2000` | code value verified |
| Finetune imagination envs | 4096 | `8192` | code value verified, paper comparison pending |
| Finetune imagination steps per env | 100 (per iteration in paper) | `24` (per env per iteration in code) | distribution-across-envs question, see §4.3 |
| Discount $\gamma$ | 0.99 | `0.99` | matched |
| PPO clip range $\epsilon$ | 0.2 | `0.2` | matched |
| PPO mini-batches | to verify | `4` | code value verified |
| Entropy coefficient | to verify | `0.005` | code value verified |

Values marked "to verify" require the paper-side check before being confirmed as matched or noted as discrepancies. The paper values that *are* listed (history horizon, forecast horizon, GRU dimensions, discount, clip range, imagination envs) are taken from the paper analysis page and remain to be reconfirmed against the paper PDF.

## 4. Discrepancies

The four substantive discrepancies between the paper and the active code path. Each is the basis for a finding worth surfacing to a supervisor.

### 4.1 State mean predicted as a residual

The paper's Eq. 1 specifies the predicted observation distribution $p_\phi(\cdot \mid \ldots)$ without prescribing whether the mean is predicted directly or as a residual on the most recent state.

The code predicts a residual:

$$
\mu_\phi = \text{MLP}_\mu(h) + x_{\text{state}}^{\text{last}}
$$

in normalized state space. The MLP head learns the per-step delta rather than the absolute next state. With $\Delta t = 0.02$ s, per-step state changes are small, so residual prediction is numerically more stable than absolute prediction. This is a sensible implementation choice not specified in the paper.

Status: **Discrepancy noted**, but the discrepancy is at the level of "the paper does not specify, the code makes a reasonable choice" rather than "the paper specifies X, the code does Y."

### 4.2 State loss is sampled MSE, not Gaussian NLL

The architecture predicts both a mean $\mu_\phi$ and a standard deviation $\sigma_\phi$ for a Gaussian over the next observation. The natural training objective for such an architecture is the Gaussian negative log-likelihood:

$$
L_{\text{NLL}} = \frac{1}{2}\left(\frac{(o_{\text{target}} - \mu_\phi)^2}{\sigma_\phi^2} + 2\log\sigma_\phi\right) + \text{const}
$$

The paper's Eq. 2 specifies $L_o$ abstractly as a discrepancy measure and does not commit to either form. The active code path uses sampled MSE:

$$
L_{\text{MSE}} = \left(o_{\text{target}} - (\mu_\phi + \sigma_\phi \cdot \epsilon)\right)^2, \quad \epsilon \sim \mathcal{N}(0, I)
$$

The two losses are related but not equivalent: under reparameterization, sampled MSE is a Monte Carlo estimator that uses the predicted std for sampling but does not penalize $\sigma_\phi$ in the same way NLL does. NLL drives the model to produce calibrated uncertainty (predicted std matches actual error magnitude); sampled MSE relies on the bound regularization term to keep std meaningful.

The codebase has a helper that supports both modes (`compute_regression_loss(loss_type="mse" | "gaussian_nll")`); the active call uses `loss_type="mse"`. Switching the configured mode would activate NLL training without further changes.

Under sampled MSE, the predicted standard deviation affects the sampled prediction. In expectation, this penalizes both mean error and large predicted variance, but it is not the same as Gaussian NLL and does not calibrate uncertainty in the same way.

Status: **Discrepancy noted**. This is the most substantive of the four.

### 4.3 Imagination scale differs from paper's stated values

Table S11 of the paper reports 4096 imagination environments and 100 imagination steps per iteration. The Finetune-v0 runner config uses 8192 imagination environments and 24 steps per environment per iteration.

These do not contradict each other directly. The paper's "100 imagination steps per iteration" likely refers to the autoregressive chain length the model is rolled out for; the codebase's `num_imagination_steps_per_env = 24` may refer to a different dimension (steps per env per PPO update batch). Without instrumenting the active loop, the relationship is ambiguous.

Status: **Partially mapped**. Worth resolving during the next code read by counting actual autoregressive rollout lengths during a Finetune-v0 run.

### 4.4 Uncertainty hooks present but inactive

The codebase contains the structural hooks for uncertainty-aware behavior:

- `ensemble_size` parameter on `SystemDynamicsEnsemble` (default 1, currently 1). With `ensemble_size > 1`, the implementation instantiates multiple prediction heads sharing the GRU recurrent base, so epistemic uncertainty in this design reflects disagreement across prediction heads rather than across fully independent dynamics networks.
- `uncertainty_penalty_weight` field on the imagination env (default `-0.0`, currently `-0.0`).
- A penalty term in the imagined reward computation: `rewards += uncertainty_penalty_weight * epistemic_uncertainty * step_dt`. With a negative weight this becomes a penalty.

The paper introduces no uncertainty machinery. With `ensemble_size = 1`, epistemic uncertainty is structurally zero; with `uncertainty_penalty_weight = -0.0`, the penalty term contributes nothing either way.

These hooks are the structural surface for the RWM-U extension (the second paper). The active configuration runs as RWM. Activating the hooks (`ensemble_size > 1`, `uncertainty_penalty_weight < 0`) is the path forward to the RWM-U + MOPO-PPO configuration. Whether the shared-base ensemble design is sufficient for RWM-U, or whether the second paper assumes fully independent ensemble members, is an open paper-to-code question to resolve during the RWM-U deep dive.

Status: **Discrepancy / extension hook noted**, with the framing that this is not a discrepancy from the RWM paper (which does not specify uncertainty handling) but the visible surface of the RWM-U extension, switched off in the active base configuration.

## 5. Inactive loss components

The system-dynamics loss has seven configured terms. Four are active in the current GRU configuration, three are inactive:

| Term | Status | Reason |
|---|---|---|
| State (sampled MSE) | Active | Default regression mode. |
| Sequence | Inactive | `prediction_type = "single"` for the GRU path. |
| Bound | Active | Regularizes learned std bounds. |
| KL | Inactive | Only used for RSSM architectures. |
| Extension | Inactive | No `system_extension` observation group registered. |
| Contact | Active | BCE on thigh and foot contacts. |
| Termination | Active | BCE on base contact. |

The inactive components are scaffolding that supports alternative architectures or task variants. They are not bugs or omissions; they are configuration-dependent code paths.

Status: **Partially mapped**. The seven-term decomposition is a code-level structure not present in the paper. The four active terms together implement the paper's Eq. 2 plus the bound regularization.

## 6. Summary

The active code path implements the paper's RWM method faithfully in structure, with documented discrepancies in two specific places: the state loss form (sampled MSE rather than Gaussian NLL) and the residual mean prediction. The reward equations from Section A.1.2 of the paper are reconstructed term-by-term inside the imagination loop. The 100-step rollout claim in the paper maps to a different per-env / per-iteration accounting in the code that would need run-time inspection to fully reconcile.

The codebase additionally contains uncertainty hooks reserved for the RWM-U extension. They are switched off in the active configuration and produce RWM behavior. Activating them is the planned path to RWM-U + MOPO-PPO; per-claim status of that activation is tracked on the [Reproduction Status](../validation/reproduction-status.md) page.