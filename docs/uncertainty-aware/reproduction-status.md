# Reproduction Status

This page is the per-claim ledger for the project. It separates *execution claims* (whether the code has been run end-to-end on local or lab hardware) from *mapping claims* (whether the paper's content has been identified in the code). The two axes are independent and use the [convention vocabulary](../index.md#conventions) from the landing page.

For the procedural and architectural detail behind each row, follow the cross-references.

## Execution claims

What has been executed, on which hardware, with what outcome.

| Claim | Status | Evidence | Required to advance |
|---|---|---|---|
| Local Isaac Lab and RWM stack runs headlessly. | Validated | Baseline task reaches PPO learning iterations (`manifests/baseline_state_*.txt`). | None for local validation. |
| Baseline ANYmal-D task `Init-v0` executes locally. | Validated | See [Baseline Execution](baseline-execution.md). | None. |
| RWM pretraining pipeline executes end-to-end at reduced scale on local hardware and writes checkpoints. | Validated at reduced scale | Reduced-scale Pretrain-v0 run produced `model_0.pt`, `model_1.pt`. See [World-Model Pretraining Check](world-model-pretraining-check.md). | Lab-scale pretraining for full validation. |
| Default RWM pretraining configuration runs on local hardware. | Not verified | Default config exceeds 8 GB VRAM during system-dynamics loss. | Lab workstation with sufficient VRAM. |
| RWM pretraining at the paper's full scale ($M=32$, $N=8$, batch 1024, 2500 iterations) on lab hardware. | Not verified | Lab access not yet exercised. | Lab workstation execution. |
| Trained dynamics model has useful predictive accuracy on held-out trajectories. | Not verified | Reduced-scale run was not configured for predictive performance. | Full pretrain plus held-out forecast evaluation. |
| Imagination-based finetuning (`Finetune-v0`) runs end-to-end. | Not verified | Upstream config requires a pretrained dynamics checkpoint that does not exist locally. See [Checkpoint and Finetune Status](checkpoint-and-finetune-status.md). | Generate or obtain a valid dynamics checkpoint and a project-specific finetune config. |
| MBPO-PPO produces a policy whose performance matches the paper's reported `0.90 ± 0.04` velocity-tracking reward. | Not verified | No finetuning has been run. | Finetune execution and benchmark evaluation. |
| RWM-U offline pipeline identified in upstream code. | Validated | Located at `scripts/reinforcement_learning/model_based/`, separate from the manager-based pipeline. Inspected by grep: no direct Isaac Lab imports, simulator calls, or `env.step(...)` calls were found in the standalone offline path. See [Uncertainty-Aware Implementation Analysis](../uncertainty-aware-world-model/implementation-analysis.md). | None for identification. |
| RWM-U dataset (`assets/data/state_action_data_0.csv`) is present and matches the expected schema. | Validated | 10,000 rows × 66 columns (45 state + 12 action + 8 contact + 1 termination), confirmed matches `Pretrain-v0` observation groups. | None for identification. |
| RWM-U pretrained world model (`assets/models/pretrain_rnn_ens.pt`) is materialized locally. | Validated | After `git lfs pull`, the 133-byte LFS pointer was replaced with the actual 25,054,639-byte PyTorch checkpoint archive. The checkpoint contains five top-level keys; dynamics evaluation must use `system_dynamics_state_dict` (not `model_state_dict`, which is the policy actor-critic state). See [RWM-U Execution Check §1, §4](rwm-u-execution-check.md#1-git-lfs-materialization-of-the-pretrained-world-model). | None. |
| RWM-U offline pipeline runs end-to-end with shipped assets at reduced scale. | Validated at reduced scale | Completed 500 policy-training iterations with the shipped 10K-transition dataset and the materialized pretrained world model under the active default `uncertainty_penalty_weight = -1.0`. Successful active-default runs reached final imagined rewards of 1.47 and 1.82 (the two complete runs at this weight; the difference is stochastic). Policy checkpoints saved to `upstream/robotic_world_model/logs/offline/anymal_d_flat/`. See [RWM-U Execution Check §3](rwm-u-execution-check.md#3-end-to-end-offline-training-run). | Lab-scale run for paper-scale comparison. |
| Uncertainty-error correlation (paper Fig. 3) reproduced at reduced scale. | Qualitatively validated at reduced scale | 32 rollout windows from the shipped 10,000-step trajectory, 200-step autoregressive horizon. Pearson correlation between epistemic uncertainty and L1 prediction error: 0.7014 step-averaged (ensemble-mean rollout), 0.6898 step-averaged (sampled-member rollout). Paper does not report a Pearson value, so this is an independent measurement supporting the paper's qualitative claim. Aleatoric uncertainty stayed near zero throughout. See [RWM-U Execution Check §5](rwm-u-execution-check.md#5-figure-3-style-uncertainty-error-evaluation). | Paper-scale reproduction is out of current scope. |
| Penalty coefficient $\lambda$ behaves monotonically (paper Fig. 4 mechanism). | Qualitatively validated at reduced scale | Sweep over code weights `{0.0, -1.0, -5.0}` (paper $\lambda \in \{0.0, 1.0, 5.0\}$) produced final imagined rewards `{6.53, 1.82, -8.21}`. Penalty mechanism is operationally active and increasingly conservative. See [RWM-U Execution Check §6](rwm-u-execution-check.md#6-uncertainty-penalty-lambda-sweep). | Determining the optimum $\lambda$ requires policy evaluation under fixed unpenalized reward (simulator pass or hardware), which is out of current scope. |
| RWM-U + MOPO-PPO matches paper's reported real-world performance (Table 1). | Not verified | Requires the full 6M-transition dataset and hardware deployment infrastructure. | Out of current project scope. |
| Zero-shot deployment of trained policy on hardware. | Not verified | No trained policy has been deployed. | Trained policy plus hardware availability. |

## Mapping claims

What has been identified in the code as a counterpart to a paper claim.

| Paper claim | Status | Cross-reference |
|---|---|---|
| RWM world model architecture (GRU base, MLP heads). | Mapped | [Implementation Analysis §4](../world-model/implementation-analysis.md#4-system-dynamics-module). |
| Dual-autoregressive training (inner + outer autoregression). | Mapped | [Implementation Analysis §4 to §5](../world-model/implementation-analysis.md). |
| Multi-step prediction loss (Eq. 2). | Partially mapped | [Synthesis §1](../world-model/paper-to-code-synthesis.md#1-method-components); discrepancy on $L_o$ form. |
| MBPO-PPO algorithm (Algorithm 1). | Mapped | [Implementation Analysis §7](../world-model/implementation-analysis.md#7-model-based-runner). |
| Imagined action selection (Eq. 3). | Mapped | [Implementation Analysis §9](../world-model/implementation-analysis.md#9-imagination-environment). |
| Autoregressive evaluation under noise injection (paper Fig. 3b). | Mapped | `evaluate_system_dynamics(...)` rolls out 100 trajectories of length 400 with noise scales `[0.1, 0.2, 0.4, 0.5, 0.8]`, matching the noise levels in the paper figure. See [Implementation Analysis §5.10](../world-model/implementation-analysis.md#510-autoregressive-evaluation-procedure). |
| Reward function (paper Section A.1.2). | Partially mapped | [Synthesis §2](../world-model/paper-to-code-synthesis.md#2-reward-terms); 11 reward terms active in code (10 inherited from upstream Isaac Lab plus `stand_still` added by project), with three weights overridden by the project. The paper-to-code term-by-term correspondence remains to be verified. |
| Policy observation versus system-state distinction. | Mapped | Local Pretrain-v0 run shows separate `policy` (48-dim) and `system_state` (45-dim) observation groups; the dynamics model predicts `system_state`, while the policy observation is reconstructed during imagination from predicted state, command, and previous action. See [Implementation Analysis §4.5](../world-model/implementation-analysis.md#45-policy-observation-versus-system-state). |
| State mean predicted directly (paper does not specify) versus residual prediction (code). | Discrepancy noted | [Synthesis §4.1](../world-model/paper-to-code-synthesis.md#41-state-mean-predicted-as-a-residual). |
| State loss as Gaussian NLL (architecturally implied) versus sampled MSE (active code). | Discrepancy noted | [Synthesis §4.2](../world-model/paper-to-code-synthesis.md#42-state-loss-is-sampled-mse-not-gaussian-nll). |
| Imagination autoregressive horizon: 100 steps per iteration (paper) versus 24-step rollout (code). | Discrepancy in scale | [Synthesis §4.3](../world-model/paper-to-code-synthesis.md#43-imagination-autoregressive-horizon-is-shorter-than-the-paper). |
| Uncertainty handling (RWM-U paper, not RWM paper) hooks present in code but switched off. | Discrepancy noted | [Synthesis §4.4](../world-model/paper-to-code-synthesis.md#44-uncertainty-hooks-present-but-inactive). The hooks are switched off in the manager-based pipeline; they are switched on in the standalone offline pipeline (`ensemble_size = 5`, `uncertainty_penalty_weight = -1.0`). |
| RWM-U bootstrap ensemble architecture (paper Sec. 4.1: shared GRU base, ensembled heads). | Mapped | `SystemDynamicsEnsemble._init_networks()` instantiates one `state_base` and one `auxiliary_base` GRU, with $K$ heads per base in `nn.ModuleList`. See [Uncertainty-Aware Implementation Analysis §1.3](../uncertainty-aware-world-model/implementation-analysis.md#13-what-the-two-pipelines-share). |
| RWM-U epistemic uncertainty signal (paper Eq. 4: variance across ensemble means). | Partially mapped | Code computes `state_means.std(dim=0).sum(dim=1)`, which is standard deviation summed across observation dimensions rather than variance. See [Uncertainty-Aware Synthesis §4.2](../uncertainty-aware-world-model/paper-to-code-synthesis.md#42-epistemic-uncertainty-is-summed-standard-deviation-not-variance). |
| RWM-U penalized reward (paper Eq. 5: $\tilde r = r - \lambda u$). | Mapped under sign and time-step convention | Code applies `rewards += weight * uncertainty * dt` with `weight = -1.0`, equivalent to paper's $\lambda = 1.0$ under the negative-weight sign convention and the code's per-step time-scaling convention. See [Uncertainty-Aware Synthesis §3.2](../uncertainty-aware-world-model/paper-to-code-synthesis.md#32-policy-optimization-table-s11). |
| Algorithm 1 (offline policy optimization with RWM-U). | Mapped | Standalone pipeline at `scripts/reinforcement_learning/model_based/`: pretrained model loaded, no further updates, vanilla `PPO` consumes a custom env that applies the uncertainty penalty inside its reward function. See [Uncertainty-Aware Implementation Analysis §1.4](../uncertainty-aware-world-model/implementation-analysis.md#14-algorithmic-implication-mopo-ppo-is-just-ppo--custom-env). |
| RWM-U imagination uses ensemble mean (paper Eq. 4 left). | Discrepancy noted | Code samples a random ensemble member per imagination env at reset (`model_ids` mechanism); ensemble mean is used only when `model_ids = None`. See [Uncertainty-Aware Synthesis §4.1](../uncertainty-aware-world-model/paper-to-code-synthesis.md#41-imagination-uses-trajectory-sampling-not-ensemble-mean-dynamics). |

## Notes

The execution claims marked "Not verified" are not failures; they are blocked on lab access, lab access plus checkpoint generation, or hardware deployment that is later in the project. Most of the RWM-U execution claims that were "Not verified" at Phase 3 documentation time have since been advanced to **Validated at reduced scale** or **Qualitatively validated at reduced scale** through the work documented on the [RWM-U Execution Check](rwm-u-execution-check.md) page.

The mapping claims marked "Discrepancy noted" are findings to surface to a supervisor as part of a faithful reproduction effort, not as defects in the codebase.

One execution claim qualifies as an active blocker:

- The dynamics checkpoint dependency for `Finetune-v0`, documented in [Checkpoint and Finetune Status](checkpoint-and-finetune-status.md). This concerns the manager-based pipeline, not the offline RWM-U pipeline.

The RWM-U pipeline now has its own paper analysis ([here](../uncertainty-aware-world-model/paper-analysis.md)), implementation analysis ([here](../uncertainty-aware-world-model/implementation-analysis.md)), paper-to-code synthesis ([here](../uncertainty-aware-world-model/paper-to-code-synthesis.md)), cross-paper bridge ([here](../world-model/relationship-to-uncertainty-aware.md)), and reduced-scale execution validation ([here](rwm-u-execution-check.md)).
