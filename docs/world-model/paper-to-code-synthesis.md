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

## 2. Reward terms (paper Section A.1.2)

The paper's reward equations are reconstructed inside the imagination loop from predicted state and contact signals. The code references below point to the term-by-term implementations in the imagination env's reward reconstruction.

| Reward term | Paper symbol | Mapping status |
|---|---|---|
| Linear velocity tracking $x, y$ | $r_{v_{xy}}$ | Mapped |
| Angular velocity tracking $z$ | $r_{\omega_z}$ | Mapped |
| Linear velocity penalty $z$ | $r_{v_z}$ | Mapped |
| Angular velocity penalty $x, y$ | $r_{\omega_{xy}}$ | Mapped |
| Joint torque penalty | $r_{q\tau}$ | Mapped |
| Joint acceleration penalty | $r_{\ddot q}$ | Mapped |
| Action rate penalty | $r_{\dot a}$ | Mapped |
| Feet air time bonus | $r_{f_a}$ | Mapped |
| Undesired contacts penalty | $r_c$ | Mapped |
| Flat orientation penalty | $r_g$ | Mapped |
| Foot clearance bonus | $r_{f_c}$ | Mapped |
| Joint deviation penalty | $r_{q_d}$ | Mapped |

All twelve reward terms are reconstructed from RWM predictions inside the imagination loop. The reconstruction is in `_compute_imagination_reward_terms(...)` in `anymal_d_manager_based_mbrl_env.py`. The corresponding equations are documented on the [Paper Analysis](paper-analysis.md#6-reward-formulation) page.

## 3. Training and architecture parameters

| Parameter | Paper value | Code value (Pretrain-v0 config) | Match |
|---|---|---|---|
| History horizon $M$ | 32 | 32 | yes |
| Forecast horizon $N$ | 8 | 8 | yes |
| Forecast decay $\alpha$ | 1.0 | 1.0 (default) | yes |
| Step time $\Delta t$ | 0.02 s | 0.02 s | yes |
| Batch size | 1024 | 1024 (assumed) | yes |
| Learning rate (RWM) | $10^{-4}$ | $10^{-4}$ (assumed) | yes |
| Weight decay (RWM) | $10^{-5}$ | $10^{-5}$ (assumed) | yes |
| Max iterations | 2500 | 2500 | yes |
| GRU base | 256 × 256 | 256 × 256 | yes |
| MLP heads | 128, ReLU | 128 | yes |
| Imagination envs | 4096 (paper) | 8192 (code) | discrepancy in scale |
| Imagination steps per iteration | 100 (paper) | 24 per env × 8192 envs | discrepancy in distribution across envs |
| Buffer size $\lvert \mathcal{D} \rvert$ | 1000 | not yet verified against code | unverified |
| Discount $\gamma$ | 0.99 | 0.99 | yes |
| Clip range $\epsilon$ | 0.2 | 0.2 | yes |
| Entropy coefficient | 0.005 | not yet verified against code | unverified |
| Ensemble size | not specified by paper | 1 (Pretrain-v0), reserved for RWM-U | RWM context only |

The values marked "assumed" or "not yet verified" are taken from the existing implementation drafts and have not been re-checked against the current upstream config. They are correct to the best of our knowledge but should be confirmed during the next code read.

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

The codebase has a helper that supports both modes (`compute_regression_loss(loss_type="mse" | "nll")`); the active call uses `loss_type="mse"`. Switching the configured mode would activate NLL training without further changes.

Status: **Discrepancy noted**. This is the most substantive of the four.

### 4.3 Imagination scale differs from paper's stated values

Table S11 of the paper reports 4096 imagination environments and 100 imagination steps per iteration. The Finetune-v0 runner config uses 8192 imagination environments and 24 steps per environment per iteration.

These do not contradict each other directly. The paper's "100 imagination steps per iteration" likely refers to the autoregressive chain length the model is rolled out for; the codebase's `num_imagination_steps_per_env = 24` may refer to a different dimension (steps per env per PPO update batch). Without instrumenting the active loop, the relationship is ambiguous.

Status: **Partially mapped**. Worth resolving during the next code read by counting actual autoregressive rollout lengths during a Finetune-v0 run.

### 4.4 Uncertainty hooks present but inactive

The codebase contains the structural hooks for uncertainty-aware behavior:

- `ensemble_size` parameter on `SystemDynamicsEnsemble` (default 1, currently 1).
- `uncertainty_penalty_weight` field on the imagination env (default 0, currently 0 or -0.0).
- A penalty term in the imagined reward computation: `imagined_reward -= uncertainty_penalty_weight * epistemic_uncertainty * step_dt`.

The paper introduces no uncertainty machinery. With `ensemble_size = 1`, epistemic uncertainty is structurally zero; with `uncertainty_penalty_weight = 0`, the penalty term contributes nothing.

These hooks are the structural surface for the RWM-U extension (the second paper). The active configuration runs as RWM. Activating the hooks (`ensemble_size > 1`, `uncertainty_penalty_weight > 0`) is the path forward to the RWM-U + MOPO-PPO configuration.

Status: **Discrepancy noted**, with the framing that this is not a discrepancy from the RWM paper (which does not specify uncertainty handling) but a piece of the RWM-U paper present in the codebase but switched off. The same code path implements both methods, with config values selecting between them.

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