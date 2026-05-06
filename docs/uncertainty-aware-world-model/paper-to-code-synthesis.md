# Paper-to-Code Synthesis: Uncertainty-Aware World Model

This page maps the claims of the [RWM-U paper analysis](paper-analysis.md) to the implementation documented in the [RWM-U implementation analysis](implementation-analysis.md). It records what is *Mapped* (paper claim faithfully realized in the active code path), *Partially mapped* (realized in structure but with quantitative or formulation differences), and *Discrepancy noted* (paper and code disagree in a way that matters).

This page also documents the relationship between the two pipelines in the codebase. The base RWM paper-to-code synthesis is in [world-model/paper-to-code-synthesis.md](../world-model/paper-to-code-synthesis.md), describing the manager-based pipeline. This page covers the standalone offline pipeline, which is the codebase's RWM-U implementation.

## How to read this page

Three frames stack to give the synthesis its shape:

1. *Two pipelines in one repository*. The codebase contains both an online MBPO-PPO pipeline (the manager-based path, used for base RWM) and an offline MOPO-PPO pipeline (the standalone path, used for RWM-U). Both share `SystemDynamicsEnsemble` and other backbone modules.

2. *Active configuration distinguishes the methods*. The same backbone code runs as RWM (with `ensemble_size = 1` and `uncertainty_penalty_weight = -0.0`) in the manager-based pipeline, and as RWM-U (with `ensemble_size = 5` and `uncertainty_penalty_weight = -1.0`) in the offline pipeline. The configuration values are the load-bearing distinction.

3. *Sign and time-step convention*. The paper writes the penalty as $\tilde r = r - \lambda u$ with $\lambda > 0$ and no explicit time-step factor. The code writes the same penalty as `rewards += weight * uncertainty * dt` with `weight < 0` and an explicit multiplication by `step_dt = 0.02 s`. Two conventions stack: (a) the negative-weight sign convention maps paper $\lambda > 0$ to code `weight < 0`, so paper $\lambda = 1.0$ corresponds to code `weight = -1.0`; and (b) the code multiplies the uncertainty term by `step_dt`, matching how all other reward terms in the env are time-integrated. The paper writes Eq. 5 as a discrete per-step penalty without explicitly showing time-step scaling. Therefore paper $\lambda = 1.0$ maps to code `uncertainty_penalty_weight = -1.0` only under both the sign convention and the code's time-step scaling. This is documented once here and labeled "Mapped under sign and time-step convention" throughout.

## 1. Method components

| RWM-U paper concept | Code location | Status |
|---|---|---|
| Offline RL formulation (POMDP, fixed dataset $\mathcal{D}$) | `scripts/reinforcement_learning/model_based/train.py` loads CSV from `assets/data/`; no `env.step` calls in the path | Mapped |
| Bootstrap ensemble of prediction heads on shared GRU base (Sec. 4.1) | `SystemDynamicsEnsemble._init_networks()`: shared `state_base` + `auxiliary_base`, $K$ heads per base in `nn.ModuleList` | Mapped |
| Per-member Gaussian prediction (Eq. 3, $o'^b \sim \mathcal{N}(\mu^b, \sigma^{2,b})$) | `MLPStateHead` outputs mean and standard deviation per member | Mapped |
| Predicted observation as ensemble mean (Eq. 4 left) | `output_state_means = state_means.mean(dim=0)` when `model_ids is None`; per-env sampled member when `model_ids` provided | Discrepancy noted: imagination uses trajectory sampling rather than ensemble-mean dynamics |
| Epistemic uncertainty as variance across ensemble means (Eq. 4 right) | `state_means.std(dim=0).sum(dim=1)` if `ensemble_size > 1` else hard zero | Partially mapped: code uses standard deviation summed across observation dimensions, not variance |
| Aleatoric uncertainty (mean of predicted standard deviations) | `state_stds.mean(dim=0).sum(dim=1)` | Mapped |
| Penalized reward (Eq. 5, $\tilde r = r - \lambda u$) | `rewards += uncertainty_penalty_weight * epistemic_uncertainty * step_dt` in offline env's reward function | Mapped under sign and time-step convention |
| Algorithm 1 step 3 (one-shot offline world model training) | `prepare_model(...)` with `torch.load(resume_path)` to load pretrained checkpoint; no further dynamics updates during policy training | Mapped |
| Algorithm 1 steps 4-9 (offline policy optimization loop) | `policy_training.py` runs imagination steps + PPO updates; no Isaac Lab calls | Mapped |
| Initial states sampled from $\mathcal{D}$ (step 5) | `dataset.sample_batch(num_envs, normalized=True)` in offline env's reset path | Mapped |
| 100-step autoregressive rollout (Table S11) | `policy_training.py` imagination loop runs `num_steps_per_env = 24` per iteration in the active offline config | Discrepancy in scale (24 vs 100) |
| Policy update via PPO (step 8) | `from rsl_rl.algorithms import PPO`, `self.alg = PPO(...)` in `train.py` line 245 | Mapped |
| MOPO-PPO as a distinct algorithm class | No separate class; MOPO-PPO is implemented as vanilla PPO + custom env that applies the uncertainty penalty inside the reward function | Mapped (matches paper Algorithm 1, which treats policy optimizer as interchangeable) |

## 2. Architecture parameters

The paper's Table S7 (RWM-U architecture) and Table S8 (policy and value architecture) are documented on the [paper analysis](paper-analysis.md#71-rwm-u-architecture-table-s7) page. The active code architecture matches.

| Component | Paper (Table S7/S8) | Inspected code value | Status |
|---|---|---|---|
| World model base | GRU 256, 256 | 2-layer GRU, hidden size 256 | Mapped |
| World model heads | MLP 128, ReLU | MLP `[128]`, ReLU activation | Mapped |
| Policy network | MLP 128, 128, 128, ELU | MLP 128, 128, 128, ELU | Mapped |
| Value network | MLP 128, 128, 128, ELU | MLP 128, 128, 128, ELU | Mapped |

Architecture is verified via the shared `SystemDynamicsEnsemble` and `ActorCritic` modules in `rsl_rl_rwm/modules/`. The offline pipeline imports these unchanged.

## 3. Training and runtime parameters

The paper's Table S10 (RWM-U training) and Table S11 (MOPO-PPO training) are documented on the [paper analysis](paper-analysis.md#73-rwm-u-training-parameters-table-s10) page. The configured values from `configs/anymal_d_flat_cfg.py` and the inherited defaults from `configs/base_cfg.py` and `rsl_rl_rwm` map as follows.

### 3.1 World model training (Table S10)

The active ANYmal-D offline configuration sets `resume_path = "assets/models/pretrain_rnn_ens.pt"`, so the world model is loaded from the shipped pretrained checkpoint and is not trained again during the offline run. The world-model training parameters below therefore apply only to a from-scratch retrain (achieved by overriding `resume_path = None` in the model-architecture config) or as a description of the regime that produced the shipped checkpoint.

| Parameter | Paper value | Inspected code value | Status |
|---|---:|---:|---|
| Step time $\Delta t$ | 0.02 s | 0.02 s | Mapped |
| Max iterations | 2500 | not separately configured for dynamics in offline pipeline | Not applicable: world model is loaded, not trained |
| Learning rate | $1\times10^{-4}$ | not separately configured for dynamics in offline pipeline | Not applicable: world model is loaded, not trained |
| Weight decay | $1\times10^{-5}$ | not separately configured for dynamics in offline pipeline | Not applicable: world model is loaded, not trained |
| Batch size | 1024 | `batch_data_size = 10000`, `file_data_size = 10000` (data buffer sizes, not dynamics minibatch) | Not applicable: world model is loaded, not trained |
| Ensemble size $B$ | 5 | 5 | Mapped (offline pipeline) |
| History horizon $M$ | 32 | 32 | Mapped |
| Forecast horizon $N$ | 8 | 8 | Mapped |
| Forecast decay $\alpha$ | 1.0 | not configurable; uniform mean | Equivalent to $\alpha = 1.0$ |
| Transitions in training data | 6M | 10K shipped (`state_action_data_0.csv`) | Discrepancy in scale: shipped dataset is 0.17% of paper-scale |
| Approximate training hours | ~1h on RTX 4090 | not measured | Not yet measured |
| Number of seeds | 5 | not measured | Not yet measured |

The architecture parameters under `ModelArchitectureConfig` in `configs/anymal_d_flat_cfg.py` match the paper exactly: `history_horizon = 32`, `forecast_horizon = 8`, `ensemble_size = 5`, `contact_dim = 8`, `termination_dim = 1`, with `architecture_config = {"type": "rnn", "rnn_type": "gru", "rnn_num_layers": 2, "rnn_hidden_size": 256, ...}` and head shapes `[128]`. The base config defaults to MLP architecture; the ANYmal-D override switches to GRU.

### 3.2 Policy optimization (Table S11)

Verified from `configs/base_cfg.py` (defaults) and `configs/anymal_d_flat_cfg.py` (ANYmal-D overrides). Parameter names in the table are the paper's; code field names are noted in the third column.

| Parameter | Paper value | Inspected code value | Status |
|---|---:|---:|---|
| Imagination environments | 4096 | 8192 (`EnvironmentConfig.num_envs`) | Discrepancy in scale (factor 2 above paper) |
| Imagination steps per iteration | 100 | 24 (`PolicyTrainingConfig.num_steps_per_env`) | Discrepancy in scale (factor 4 below paper) |
| Uncertainty penalty weight $\lambda$ | 1.0 | `-1.0` (`EnvironmentConfig.uncertainty_penalty_weight`) | Mapped under sign and time-step convention |
| Step time $\Delta t$ | 0.02 s | 0.02 s (`EnvironmentConfig.step_dt`) | Mapped |
| Buffer size $\lvert\mathcal{D}\rvert$ | 1000 | n/a in offline pipeline | The offline pipeline does not use a system-dynamics replay buffer; the world model is frozen at training start |
| Max iterations | 2500 | 500 (`PolicyTrainingConfig.max_iterations` in ANYmal-D config) | Discrepancy in scale (factor 5 below paper) |
| Learning rate (policy) | 0.001 | 1.0e-4 (`PolicyAlgorithmConfig.learning_rate` in ANYmal-D config) | Discrepancy: code value is 10x lower than paper |
| Weight decay (policy) | 0.0 | not configurable (PPO does not expose weight decay) | Effectively mapped: PPO optimizer does not apply weight decay |
| Learning epochs | 5 | 5 (`PolicyAlgorithmConfig.num_learning_epochs`) | Mapped |
| Mini-batches | 4 | 4 (`PolicyAlgorithmConfig.num_mini_batches`) | Mapped |
| KL divergence target | 0.01 | 0.01 (`PolicyAlgorithmConfig.desired_kl`) | Mapped |
| Discount $\gamma$ | 0.99 | 0.99 (`PolicyAlgorithmConfig.gamma`) | Mapped |
| Clip range $\epsilon$ | 0.2 | 0.2 (`PolicyAlgorithmConfig.clip_param`) | Mapped |
| Entropy coefficient | 0.005 | 1.0e-4 (`PolicyAlgorithmConfig.entropy_coef` in ANYmal-D config) | Discrepancy: code value is 50x lower than paper |
| Number of seeds | 5 | not measured | Not yet measured |

Three additional values from `PolicyAlgorithmConfig` not in Table S11 but verified in code: `value_loss_coef = 1.0`, `lam = 0.95` (GAE $\lambda$), `max_grad_norm = 1.0`, `schedule = "adaptive"` (adaptive learning-rate schedule based on KL divergence). All inherited from `base_cfg.py`.

The `PolicyTrainingConfig` in ANYmal-D also overrides `save_interval = 50` (defaults to 200), so a checkpoint is written every 50 iterations.

The `DataConfig` in ANYmal-D sets `init_data_ratio = 0.8` (inherited from base): when imagination resets, 80% of envs initialize their state-action history from the loaded dataset, and the remaining 20% initialize from random samples. The mixture is passed through `dataset.sample_batch(num_envs, normalized=True)` and the random init path. This implements the "initialize imagination from dataset" semantics of Algorithm 1 step 5.

The three meaningful discrepancies between configured offline values and paper Table S11 (max iterations 500 vs 2500, policy learning rate $10^{-4}$ vs $10^{-3}$, entropy coefficient $10^{-4}$ vs $5\times 10^{-3}$) are values the codebase ships with as defaults for the ANYmal-D offline configuration. Whether the paper's Table S11 numbers were used to produce the shipped pretrained model and the paper's reported results, or whether the shipped configuration is a downscaled "demo" configuration meant to run on smaller hardware, cannot be determined from the code alone.

## 4. Discrepancies

Three substantive discrepancies between the paper and the active offline code path. Each is the basis for a finding worth surfacing.

### 4.1 Imagination uses trajectory sampling, not ensemble-mean dynamics

Eq. 4 of the paper specifies that the predicted next observation at inference is the *mean across ensemble members*: $o'_{t+1} = \mathbb{E}_b[\mu^b]$. The paper's narrative implies a single ensemble-mean trajectory rolled forward $T$ steps.

The code, however, samples a single ensemble member per imagination environment at reset time:

```python
self.system_dynamics_model_ids = torch.randint(
    0, self.system_dynamics.ensemble_size,
    (1, self._num_envs, 1), device=self.device
)
```

When `forward(...)` is called with `model_ids != None`, the prediction is gathered from the assigned member rather than averaged across members:

```python
output_state_means = torch.gather(state_means, 0, model_ids.repeat(1, 1, self.state_dim)).squeeze(0)
```

The ensemble *mean* is used only when `model_ids` is `None` (e.g., during evaluation).

In the manager-based pipeline (online RWM), the ensemble has one member, so `model_ids = 0` always and the mean and the single member coincide. In the offline pipeline (RWM-U), `model_ids` is randomly drawn from $\{0, \ldots, B - 1\}$ per env at reset (and per env when an episode terminates and is reset). This means different parallel imagination envs see different transition dynamics in the same imagined state, the ensemble mean is never used at imagination time when `ensemble_size > 1`, and the diversity across envs acts as a kind of structured exploration even before the policy adds its own action noise.

This is the trajectory-sampling pattern used by PETS (Chua et al., 2018): a different ensemble member is assigned to each parallel rollout, and the policy is updated on the mixture of trajectories. The pattern is not without precedent in deep MBRL, but it is not what RWM-U's Eq. 4 specifies.

The epistemic uncertainty signal $u$ (variance across all member means) is computed from the full ensemble regardless of which member is sampled for the dynamics prediction, so the penalty term itself is unaffected. Only the imagined trajectory dynamics differ. The penalty acts on a per-env trajectory whose dynamics come from a single sampled member, with the disagreement signal computed across all members at every step.

Status: **Discrepancy noted**. Resolution requires either (a) modifying the env to use ensemble mean at imagination time, matching Eq. 4 literally, or (b) accepting that the implementation chooses trajectory sampling and documenting this as a deliberate departure. The PETS-style choice is plausible enough as an implementation decision that interpretation (b) is the safer default for a reproduction effort.

### 4.2 Epistemic uncertainty is summed standard deviation, not variance

Eq. 4 of the paper defines epistemic uncertainty as $u = \mathrm{Var}_b[\mu^b]$, which is observation-vector-valued (one variance per observation dimension). For a scalar reward penalty, this vector must be reduced.

The code computes:

```python
epistemic_uncertainty = state_means.std(dim=0).sum(dim=1) if self.ensemble_size > 1 \
                       else torch.zeros(output_state_means.shape[0], device=self.device)
```

The code uses the square root of the ensemble variance, then sums across dimensions. Two reductions, both away from the paper's definition:

1. *Standard deviation, not variance*: `std(dim=0)` is $\sqrt{\mathrm{Var}_b[\mu^b]}$, not $\mathrm{Var}_b[\mu^b]$.
2. *Sum across dimensions*: `.sum(dim=1)` reduces the vector-valued quantity to a scalar by summation.

Both reductions affect the penalty's magnitude relative to a literal reading of Eq. 4. The paper's $\lambda = 1.0$ is empirically tuned against whatever reduction the authors used in their implementation, and the code's `weight = -1.0` is similarly tuned against the standard-deviation-summed reduction. The two are not directly interchangeable as paper-vs-code, but both are working values for their respective reduction conventions.

Status: **Discrepancy noted**. The vector-to-scalar reduction is implicit in the paper and made concrete in the code. The choice of standard deviation versus variance is an additional, independent divergence. For reproduction work, the paper's $\lambda$ values cannot be plugged into the code's penalty without unit conversion.

## 5. Discrepancies inherited from the base RWM analysis

The following discrepancies from the [base RWM paper-to-code synthesis](../world-model/paper-to-code-synthesis.md) also apply to the offline pipeline, since both pipelines share the same backbone code:

- *State mean predicted as a residual* (RWM synthesis §4.1): the predicted mean is $o'_{t+k} = o_t + \Delta$, with $\Delta$ produced by the head. Active in both pipelines.
- *State loss is sampled MSE, not Gaussian NLL* (RWM synthesis §4.2): the active world model training uses sampled MSE; `gaussian_nll` is implemented but inactive. Affects the offline pipeline if it pretrains the world model from scratch (rather than loading the shipped checkpoint).
- *Forecast decay $\alpha$ is not a configurable parameter* (RWM synthesis §3): the loss aggregation uses a uniform mean, equivalent to $\alpha = 1.0$ but not parameterized.

These are tracked in the base RWM synthesis page and need not be re-documented here.

## 6. Configuration that activates RWM-U behavior

For reference, the configuration changes required to switch the codebase between online RWM and offline RWM-U:

**Online RWM (manager-based pipeline, `Pretrain-v0` / `Finetune-v0`)**:

```python
# from upstream/robotic_world_model/source/mbrl/.../rsl_rl_ppo_cfg.py
ensemble_size = 1
uncertainty_penalty_weight = -0.0
```

Run via:
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
```

**Offline RWM-U (standalone pipeline, `experiment_name = "offline"`)**:

```python
# from scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py
ensemble_size = 5
uncertainty_penalty_weight = -1.0
dataset_root = "assets"
dataset_folder = "data"
```

Run via:
```bash
python scripts/reinforcement_learning/model_based/train.py
# (after ensuring `git lfs pull` has fetched the pretrained model)
```

The two pipelines are independent at the script level. They do not share entry points, configs, or runtime classes; they share only the backbone modules in `rsl_rl_rwm`.

## 7. Active versus inactive surface area

A condensed view of what is active in each pipeline, for quick reference:

| Surface | Manager-based pipeline | Standalone pipeline |
|---|---|---|
| Entry script | `scripts/reinforcement_learning/rsl_rl/train.py` | `scripts/reinforcement_learning/model_based/train.py` |
| Algorithm | `MBPOPPO` (dynamics + policy) | `PPO` (policy only) |
| Runner | `MBPOOnPolicyRunner` | inline loop in `policy_training.py` |
| Environment | Isaac Lab `ManagerBasedRLEnv` (real or imagined) | Standalone offline env (no Isaac Lab) |
| Dataset | Online replay buffer (filled from real env) | Fixed CSV files in `assets/data/` |
| World model updates during training | Yes, alternating with policy updates | No, model frozen at start |
| Real environment interaction | Yes, every iteration | No, never |
| `ensemble_size` | 1 (uncertainty disabled) | 5 (uncertainty active) |
| `uncertainty_penalty_weight` | -0.0 (no penalty contribution) | -1.0 (paper's $\lambda = 1.0$) |
| Use case | Base RWM reproduction, Pretrain-v0/Finetune-v0 | RWM-U reproduction, offline policy training |

## 8. Reproduction targets and current status

The five paper claims from the [paper analysis](paper-analysis.md#10-summary-of-paper-claims) map to the following reproduction status:

| Paper claim | Code-side target | Status |
|---|---|---|
| Epistemic uncertainty correlates with prediction error (Fig. 3) | Run offline pipeline with shipped pretrained model; log epistemic uncertainty during autoregressive rollout against reference trajectory windows from the shipped dataset | Qualitatively validated at reduced scale: Pearson 0.7014 step-averaged, ensemble-mean rollout, 32 rollout windows, 200-step horizon. See [RWM-U Execution Check §5](../validation/rwm-u-execution-check.md#5-figure-3-style-uncertainty-error-evaluation). |
| Penalty coefficient $\lambda = 1.0$ optimum (Fig. 4) | Sweep `uncertainty_penalty_weight` over $\{0.0, -1.0, -5.0\}$, compare imagined reward across runs | Penalty mechanism qualitatively validated at reduced scale: monotonic effect on imagined reward (6.53 → 1.82 → -8.21). The optimum $\lambda$ itself was not assessed because that requires evaluation under fixed unpenalized reward (simulator pass or hardware), which was out of scope for the local check. See [RWM-U Execution Check §6](../validation/rwm-u-execution-check.md#6-uncertainty-penalty-lambda-sweep). |
| MOPO-PPO outperforms MBPO/MBPO-PPO/MOPO/CQL (Fig. 5) | Requires implementing or accessing the four baselines; not feasible without significant additional infrastructure | Not in current scope |
| Hardware deployment on ANYmal D and Unitree G1 (Sec. 5.4) | Requires hardware access, deployment infrastructure, and the safety-bounded data collection pipeline (not in repository) | Out of scope for this project |
| Best result 0.91 ± 0.03 with 800K sim + 200K real (Table 1) | Requires the full 1M-transition dataset and policy evaluation infrastructure | Not in current scope |

The first two claims have been advanced to qualitative validation at reduced scale through the local execution work documented on the [RWM-U Execution Check](../validation/rwm-u-execution-check.md) page. The detailed reproduction status entries are in the [reproduction status](../validation/reproduction-status.md) page.

## 9. Summary

The codebase implements RWM-U + MOPO-PPO in a separate pipeline from the base RWM implementation, with the same backbone modules but distinct entry scripts, configs, environments, and runtime structure. The offline pipeline is inspected to be simulator-free, ships a working dataset (10K transitions) and a Git-LFS-backed pretrained model (25 MB), and uses the key RWM-U activation values (`ensemble_size = 5`, `uncertainty_penalty_weight = -1.0`) under the documented sign and time-step convention. Several runtime-scale and PPO hyperparameters in the active offline configuration differ from the paper's Table S10/S11 (notably `num_envs = 8192` vs paper 4096, `num_steps_per_env = 24` vs paper 100, `max_iterations = 500` vs paper 2500, policy `learning_rate = 1e-4` vs paper 1e-3, and `entropy_coef = 1e-4` vs paper 5e-3); these are tracked in §3.2. The pipeline has been executed end-to-end at reduced scale on local hardware ([RWM-U Execution Check](../validation/rwm-u-execution-check.md)).

Two discrepancies remain between paper and code: trajectory sampling vs ensemble-mean dynamics at imagination time (§4.1), and standard-deviation-summed vs variance reduction for the epistemic signal (§4.2). Both are noted, neither is blocking. The reduced-scale uncertainty-error evaluation showed that the trajectory-sampling vs ensemble-mean choice does not materially change the uncertainty-error coupling at this scale.

The primary reproduction targets within current project scope have been qualitatively validated at reduced scale: epistemic uncertainty correlates with prediction error in the rollout-step direction (Pearson 0.7014 step-averaged), and the penalty mechanism produces the expected monotonic effect on imagined reward across $\lambda \in \{0, 1, 5\}$. Hardware deployment claims, paper-scale dataset reproduction, and full baseline comparisons remain out of scope.
