# Relationship Between RWM and RWM-U

This page is the bridge between the two papers and the two pipelines this project reproduces. It is intended for a reader who already knows what each paper does individually and wants to understand how they fit together: what is shared, what is different, what changes if you switch from one to the other, and what this means for the project's roadmap.

The detailed paper analyses are at [world-model/paper-analysis.md](paper-analysis.md) (RWM) and [uncertainty-aware-world-model/paper-analysis.md](../uncertainty-aware-world-model/paper-analysis.md) (RWM-U). The detailed implementation analyses are at [world-model/implementation-analysis.md](implementation-analysis.md) and [uncertainty-aware-world-model/implementation-analysis.md](../uncertainty-aware-world-model/implementation-analysis.md). This page summarizes their relationship without repeating the contents.

## 1. The two papers, in one sentence each

*Robotic World Model (RWM)*: a self-supervised, autoregressive neural network learns to simulate robot dynamics from interaction data, accurate enough that policies trained inside the learned model deploy zero-shot on hardware. Training is *online*: real environment interactions are collected at every iteration and used to update both the world model and the policy.

*Uncertainty-Aware Robotic World Model (RWM-U)*: an extension that lets policy training run *offline*, against a fixed dataset, by attaching ensemble-based uncertainty estimation to the world model and using a MOPO-style penalty $\tilde r = r - \lambda u$ to discourage the policy from exploiting model errors in regions the dataset doesn't cover.

The papers are not alternatives. RWM-U builds on RWM. The dual-autoregressive training scheme, the GRU architecture, the reward formulation, and the imagined-rollout framework are shared. RWM-U adds two things: an ensemble mechanism for quantifying when the model is uncertain, and a reward penalty that uses that uncertainty signal.

## 2. What stays the same between RWM and RWM-U

The following components are identical between the two papers and identical between the two pipelines in the codebase:

- *World model architecture*: a 2-layer GRU with hidden size 256 as the recurrent base, plus MLP heads (128-dim) for state mean, state log-std, and auxiliary outputs (contacts, terminations). This is the structure documented in the [base RWM implementation analysis Section 4](implementation-analysis.md#4-system-dynamics-module).

- *Dual-autoregressive training scheme*: the inner autoregression is the GRU consuming the history window step by step; the outer autoregression is the loop that feeds predicted observations back as inputs to predict further into the future. Both papers describe this in their respective Section 3.2 / Eq. 1-2.

- *Multi-step prediction loss*: the loss aggregates per-step prediction discrepancy across the forecast horizon $N$, weighted by $\alpha^k$. With $\alpha = 1.0$ (used by both papers), this is a uniform mean.

- *Privileged auxiliary prediction*: contact and termination predictions are produced alongside the next-state prediction. They are not used by the policy but are needed for the imagination env's reward reconstruction.

- *Imagination-based policy optimization framework*: the policy is rolled forward through the world model to produce synthetic trajectories, on which a policy gradient method (PPO in both papers) is applied.

- *Reward formulation*: the locomotion reward terms (linear velocity tracking, angular velocity tracking, joint torque penalty, action rate, feet air time, etc., as enumerated in Section A.1.2 of both papers).

- *Backbone code modules*: the `SystemDynamicsEnsemble`, `ActorCritic`, and `PPO` classes in `rsl_rl_rwm/` are reused unchanged by both pipelines.

This is not coincidence. RWM-U was built deliberately as an extension: it should reduce to RWM when its uncertainty machinery is disabled, and the codebase reflects this by having both pipelines depend on the same backbone.

## 3. What changes between RWM and RWM-U

Three things change, all on the data and policy-optimization side rather than on the world-model architecture side.

### 3.1 Ensemble vs single network

RWM trains a single dynamics network. RWM-U trains $B$ ensemble members, with $B = 5$ in the paper. The architecture choice the paper makes is to share the recurrent base across members and ensemble only the prediction heads (paper Section 4.1). This is a computational simplification compared to a fully independent ensemble of $B$ separate networks: only the heads are duplicated, not the GRU. The shared-base, multi-head structure is documented as the architecture of `SystemDynamicsEnsemble` in [base RWM implementation analysis Section 8.1](implementation-analysis.md#81-ensemble-support).

The ensemble produces a useful new signal: epistemic uncertainty, defined as the variance across ensemble means. In regions where the dataset has heavy coverage, the members converge to the same prediction and the variance is small. In regions where the data is sparse, the members extrapolate differently and the variance is large. This signal is the operational output that RWM-U adds beyond what RWM produces.

### 3.2 Online vs offline training

RWM trains the world model *online*: at each iteration, fresh real-environment interactions are collected, the world model is updated on those interactions, and the policy is updated against rollouts in the freshly updated model. The interaction loop keeps the model close to reality.

RWM-U trains the world model *offline*: the model is trained once on a fixed dataset, then frozen. Policy training proceeds entirely against the frozen model, with no further data collection. The model cannot be corrected mid-training.

This is a more demanding setting. Without online correction, the policy can drive the model into regions where it has no training data, and the model's predictions there will be wrong. The penalty term in MOPO-PPO is the mechanism that makes this regime workable: it discourages the policy from venturing into regions of high epistemic uncertainty.

### 3.3 Penalized reward

RWM uses the standard reward $r$ inherited from the locomotion task. RWM-U uses a penalized reward $\tilde r = r - \lambda u$, where $u$ is the epistemic uncertainty and $\lambda$ is a tunable coefficient. With $\lambda = 1.0$ on ANYmal D and Unitree G1 (paper's reported optimum), the penalty is calibrated to balance exploration against model reliability.

## 4. What the codebase implements for each

The codebase contains both implementations, in two separate pipelines.

### 4.1 RWM (manager-based pipeline)

The manager-based pipeline at `scripts/reinforcement_learning/rsl_rl/train.py` implements online RWM. The base RWM implementation analysis covers it in detail. Key facts for this page:

- Active configuration: `ensemble_size = 1`, `uncertainty_penalty_weight = -0.0`. The uncertainty hooks exist but contribute nothing.
- Real-environment interaction at every iteration via Isaac Lab's `ManagerBasedRLEnv`.
- World model updates alternate with policy updates (the MBPO-PPO loop).
- This is the path used for the project's `Init-v0`, `Pretrain-v0`, and `Finetune-v0` task validations.

### 4.2 RWM-U (standalone pipeline)

The standalone pipeline at `scripts/reinforcement_learning/model_based/train.py` implements offline RWM-U. The [RWM-U implementation analysis](../uncertainty-aware-world-model/implementation-analysis.md) covers it in detail. Key facts for this page:

- Active configuration: `ensemble_size = 5`, `uncertainty_penalty_weight = -1.0` (paper's $\lambda = 1.0$ under the documented sign and time-step convention). The uncertainty mechanism is fully active.
- No Isaac Lab dependencies. No `env.step` calls. The pipeline is purely dataset-driven.
- World model is frozen at training start (loaded from a pretrained checkpoint via `torch.load`).
- Vanilla PPO is used as the policy optimizer; the uncertainty penalty is applied inside the env's reward function.
- Ships a 10K-transition dataset (`assets/data/state_action_data_0.csv`) and a 25 MB pretrained model checkpoint via Git LFS (`assets/models/pretrain_rnn_ens.pt`).

### 4.3 The bridge

The two pipelines are mostly independent at the script level (separate entry points, configs, runners, environments). They share the `rsl_rl_rwm` backbone modules: `SystemDynamicsEnsemble`, `ActorCritic`, the PPO algorithm, the architecture classes. The same ensemble class runs as RWM (with `ensemble_size = 1`, epistemic uncertainty hard-zeroed) in the manager-based pipeline and as RWM-U (with `ensemble_size = 5`, full uncertainty signal) in the standalone pipeline.

This shared-backbone, separate-pipeline structure mirrors the conceptual relationship between the papers: one method is an extension of the other, the architecture is the same, and the difference is in the training regime and the uncertainty handling.

## 5. Implications for project scope

For this project (a Robotics MSc thesis reproducing both papers and porting toward Unitree Go2), the relationship between the two papers structures the work in stages.

### 5.1 Stage 1: base RWM

Reproduce the manager-based pipeline. This is *online* training, requires Isaac Lab, requires lab-scale compute for full reproduction, but works at reduced scale on a single GPU for sanity checking. The deliverable is a trained RWM model that produces accurate autoregressive rollouts and a fine-tuned policy that performs the locomotion task in simulation.

The project's Phase 1 and Phase 2 cover this stage at the documentation level. Lab-scale validation work is deferred to Phase 4 once the documentation is settled.

### 5.2 Stage 2: RWM-U validation

With base RWM understood, the next stage is to validate the offline pipeline. This is the natural extension: same architecture, same backbone, different training regime. Two reduced-scale validation goals are within reach without lab access:

1. *Uncertainty-error correlation*: run the offline pipeline with the shipped pretrained model, log epistemic uncertainty over an autoregressive rollout, and compare against actual prediction error on reference trajectory windows from the shipped dataset. This is a reduced-scale validation of paper Figure 3's qualitative claim. **Completed**: Pearson correlation between epistemic uncertainty and prediction error of 0.7014 (step-averaged, ensemble-mean rollout, 32 rollout windows, 200-step horizon). See [RWM-U Execution Check §5](../validation/rwm-u-execution-check.md#5-figure-3-style-uncertainty-error-evaluation).

2. *Penalty coefficient sweep*: run the offline pipeline with several values of `uncertainty_penalty_weight`, log imagined reward, and confirm the qualitative shape of paper Figure 4. **Completed for the penalty mechanism**: monotonic suppression of imagined reward across $\lambda \in \{0.0, 1.0, 5.0\}$ (final imagined rewards 6.53 → 1.82 → -8.21). The optimum $\lambda$ itself was not assessed because that requires evaluation under fixed unpenalized reward, which is out of current scope. See [RWM-U Execution Check §6](../validation/rwm-u-execution-check.md#6-uncertainty-penalty-lambda-sweep).

Both goals ran on a 4060 with the shipped 10K-transition dataset. Absolute performance numbers do not match the paper's 6M-transition results, and that is expected: the local check validates the code path, not the paper's quantitative outcomes.

### 5.3 Stage 3: Go2 transfer

The final stage is to port the pipeline from ANYmal D to Unitree Go2. This requires:

- A Go2 robot model (USD or URDF) and matching environment configuration in Isaac Lab.
- Action interface verification: Go2 uses joint position commands at a different gain scaling than ANYmal D's PD controller.
- Reward weight retuning for Go2's morphology (different leg lengths, mass distribution, foot contact geometry).
- Observation normalization recomputation, since the running statistics from ANYmal D do not transfer.
- Either a Go2 dataset (collected via simulation rollouts of a baseline policy) or pretraining of the Go2 world model from scratch.

The Go2 transfer is the project's primary research contribution: extending the RWM-U pipeline to a different robot platform than the one the original papers validate on.

## 6. What the project does not address

For completeness, three things this project does not address that are part of the broader RWM-U story:

1. *Hardware deployment infrastructure*: the safety-bounded autonomous data collection pipeline described in paper Section A.4 is not in the repository and not in the project scope.

2. *Real-world data collection*: no real Go2 data collection is planned for this project. Go2 transfer work uses simulation-collected data only.

3. *Comparative baselines*: the paper's Figure 5 compares MOPO-PPO against CQL, MBPO, MBPO-PPO, and MOPO. Reproducing these baselines is not in scope; the project focuses on the MOPO-PPO + RWM-U path itself.

## 7. The minimum claim the project will be able to make

At the end of the planned work, the project should be able to make the following claim with documentation backing:

> *We reproduced the base RWM implementation, identified and documented its discrepancies with the published paper, validated its core components at reduced scale, and extended the analysis to the uncertainty-aware variant (RWM-U + MOPO-PPO). We then characterized the implementation requirements for porting the pipeline to a different robot platform (Unitree Go2), with a documented mapping of the changes needed in robot model, action interface, reward weights, and observation normalization.*

This claim is well-bounded, defensible, and matches the work the project can complete with the available compute and time budget. Larger claims (paper-scale reproduction, hardware deployment, novel algorithmic contributions) are deliberately out of scope.

The relationship between RWM and RWM-U documented on this page is the architectural foundation of that claim. Both papers operate on the same backbone, the same task, and the same observation space. The project's ability to reproduce one informs its ability to extend the other.
