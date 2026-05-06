# RWM-U Execution Check

The RWM-U execution check verifies that the offline RWM-U pipeline (`scripts/reinforcement_learning/model_based/`) executes end-to-end on local hardware and that its core components behave as the paper describes. Status: **Validated at reduced scale**. The check covers four things: Git LFS materialization of the pretrained world model, dataset loader behavior on the shipped CSV, full 500-iteration offline policy training, and a Figure-3-style uncertainty-versus-error evaluation plus a $\lambda$ sweep.

These results are reduced-scale qualitative validation, not paper-scale reproduction. The paper's reported numbers come from a 6M-transition dataset and hardware deployment; the local validation uses the shipped 10K-transition dataset and a 25 MB pretrained checkpoint, with no hardware. The local results validate the code path, dataset format, uncertainty mechanism, and penalty behavior.

## 1. Git LFS materialization of the pretrained world model

The pretrained world-model checkpoint at `assets/models/pretrain_rnn_ens.pt` is shipped through Git LFS. A fresh clone of the repository does not include the actual checkpoint; only the 133-byte LFS pointer file is present. The pointer records the SHA-256 hash and the actual file size:

```
version https://git-lfs.github.com/spec/v1
oid sha256:2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a
size 25054639
```

Materialization required two steps. First, Git LFS itself was not installed (`git: 'lfs' is not a git command`); installing the `git-lfs` package resolved this. Second, `git lfs pull` from inside `upstream/robotic_world_model/` downloaded the actual file:

```
Downloading LFS objects: 100% (3/3), 88 MB
```

After materialization, the file is a real PyTorch checkpoint:

```
assets/models/pretrain_rnn_ens.pt: Zip archive data (25,054,639 bytes)
```

The checkpoint contains six top-level keys:

```python
['model_state_dict', 'system_dynamics_state_dict',
 'optimizer_state_dict', 'system_dynamics_optimizer_state_dict',
 'iter', 'infos']
```

The naming is informative: `model_state_dict` holds the *policy* actor-critic weights from the run that produced the checkpoint, while `system_dynamics_state_dict` holds the *world model* weights. This split is significant for downstream tooling (see Section 4 below).

## 2. Dataset loader validation

The shipped dataset at `assets/data/state_action_data_0.csv` was loaded directly by the offline pipeline. The loader's own output confirmed the schema:

```
[Motion Loader] Loaded 1 data trajectories of 10000 steps from state_action_data_0.csv.
[Motion Loader] Total number of data: 10000 / 10000
[Motion Loader] State dim: 45 | Action dim: 12 | Extension dim: 0 | Contact dim: 8 | Termination dim: 1
```

This matches the 66-column layout predicted from the configuration analysis (45 state + 12 action + 0 extension + 8 contact + 1 termination), and confirms that the dataset and the configured architecture agree on dimensionality.

The 10,000-transition shipped dataset is sufficient for end-to-end pipeline validation. It is not paper-scale: paper Table S10 reports 6 million transitions for world-model training. A reproduction at paper scale would require either generating additional dataset files following the same `state_action_data_N.csv` convention or obtaining the original dataset.

## 3. End-to-end offline training run

The standalone offline training entry point is `scripts/reinforcement_learning/model_based/train.py`. The first run failed with a Python-environment issue, not a model or pipeline issue:

```
ModuleNotFoundError: No module named 'wandb'
```

`wandb` is the offline pipeline's logging backend (the manager-based pipeline uses TensorBoard; the offline pipeline uses Weights & Biases). After installing `wandb`, the pipeline started cleanly and produced the expected initialization sequence:

```
[Prepare Model] Loading model from assets/models/pretrain_rnn_ens.pt.
[Motion Loader] Loaded 1 data trajectories of 10000 steps from state_action_data_0.csv.
[Train Policy] Training policy for 500 iterations.
```

A default-weight run completed 500 iterations. Two complete runs were produced under the active default `uncertainty_penalty_weight = -1.0`: the initial end-to-end validation run ended with final imagined reward 1.47, and the matching $\lambda$-sweep run (Section 6) ended with final imagined reward 1.82. The difference is expected stochastic variation across PPO training runs (different random seeds, different sequences of imagined trajectories). The validation point of this section is completion, checkpoint writing, and functional reward and penalty flow, not the exact final imagined reward.

A separate run with `uncertainty_penalty_weight = 0.0` (no penalty) reached:

```
Learning iteration 499/500
Mean reward: 6.53
```

Policy checkpoints were written to `upstream/robotic_world_model/logs/offline/anymal_d_flat/<timestamp>/` with files named `policy_<iter>.pt`. The completion of 500 iterations using the shipped dataset and the materialized pretrained world model confirms that the offline pipeline runs end-to-end without Isaac Lab and without real-environment interaction during policy training.

The "Mean reward" reported during training is the *imagined* reward computed inside the offline env, including the uncertainty penalty contribution for nonzero $\lambda$. It is the training-time reward signal, not a measurement of policy performance on a real or simulated task. Interpreting these numbers as comparative policy quality requires evaluation under a fixed unpenalized reward (which we did not perform).

## 4. Checkpoint key finding for downstream tooling

A first attempt at the Figure-3-style evaluation (Section 5) failed because the evaluator script tried to load the wrong checkpoint key into the world-model class:

```
Unexpected key(s): actor..., critic...
Missing key(s): state_base..., state_heads..., auxiliary_heads...
```

The error confirms the key naming: `model_state_dict` is the *policy* actor-critic state, while `system_dynamics_state_dict` is the *world model* state. After patching the evaluator to load `system_dynamics_state_dict` into `SystemDynamicsEnsemble`, strict loading succeeded:

```
[MODEL] strict load_state_dict OK
```

This is a finding worth recording for any downstream tooling that loads from `pretrain_rnn_ens.pt`: dynamics evaluation must use `system_dynamics_state_dict`. The other key, `model_state_dict`, belongs to a policy that was trained alongside the dynamics model and is not the dynamics weights.

## 5. Figure-3-style uncertainty-error evaluation

A reduced-scale evaluator was run to test the central qualitative claim of paper Section 5.1: that epistemic uncertainty tracks autoregressive prediction error. The setup:

- 32 rollout windows sampled from the shipped 10,000-step trajectory.
- History horizon $M = 32$ (the model conditions on 32 ground-truth observations).
- Autoregressive rollout horizon of 200 steps (matching paper Figure 3's rollout length).
- Both rollout modes evaluated: ensemble-mean dynamics and per-trajectory sampled-member dynamics (the latter matching the offline pipeline's actual behavior with `model_ids` set, see [paper-to-code synthesis §4.1](../uncertainty-aware-world-model/paper-to-code-synthesis.md#41-imagination-uses-trajectory-sampling-not-ensemble-mean-dynamics)).

Output directory: `logs/rwmu_fig3_uncertainty_error_fixed_20260505_200032/`. Generated artifacts:

- `summary.json`
- `uncertainty_error_rollout.csv`
- `uncertainty_error_by_step_mean.png`
- `uncertainty_error_scatter_mean.png`
- `uncertainty_error_by_step_sampled_member.png`
- `uncertainty_error_scatter_sampled_member.png`

### 5.1 Numerical results

| Quantity | Ensemble-mean rollout | Sampled-member rollout |
|---|---:|---:|
| Pearson (epistemic, L1 error), all points | 0.2946 | 0.2892 |
| Pearson (epistemic, L1 error), step-averaged | 0.7014 | 0.6898 |
| Final-step mean L1 sum | 215.54 | 223.29 |
| Final-step mean epistemic uncertainty | 1.9938 | 2.6162 |
| Final-step mean aleatoric uncertainty | 0.00265 | 0.00264 |

### 5.2 Interpretation

Two correlations are reported, and the gap between them matters. The all-points correlation of approximately 0.29 measures whether higher uncertainty *at any rollout step* coincides with higher prediction error *at that same step* across all trajectories. The step-averaged correlation of approximately 0.70 first averages epistemic uncertainty and prediction error across trajectories at each step, then correlates the two averaged sequences over the rollout horizon.

The interpretation: the model's uncertainty grows with prediction horizon in a way that closely tracks how prediction error grows with horizon (the step-averaged trend), while at any given step there is substantial trajectory-level variance (the lower all-points correlation). This is consistent with a well-calibrated *per-horizon* uncertainty estimator that has noisy per-trajectory behavior. Both numbers are valid measurements; both should be quoted.

The aleatoric uncertainty stays near zero throughout. This matches the paper's qualitative description in Section 5.1: epistemic uncertainty is the useful signal for autoregressive prediction error in this setting, while aleatoric uncertainty (the model's predicted output noise) is small.

This is Figure-3-style reduced-scale validation, not a reproduction of paper Figure 3. The paper does not report a Pearson value for Figure 3; the figure is a visual demonstration of the trend. The local numbers above are independent measurements that support the paper's qualitative claim under reduced-scale conditions.

The two rollout modes give nearly identical results, suggesting that the trajectory-sampling discrepancy between paper and code (documented in [paper-to-code synthesis §4.1](../uncertainty-aware-world-model/paper-to-code-synthesis.md#41-imagination-uses-trajectory-sampling-not-ensemble-mean-dynamics)) does not materially change the uncertainty-error coupling at this scale.

## 6. Uncertainty-penalty $\lambda$ sweep

A reduced sweep was run with three penalty values, all other configuration kept at the active offline defaults. The sign convention maps paper $\lambda$ to code `uncertainty_penalty_weight` as documented in [paper-to-code synthesis §3.2](../uncertainty-aware-world-model/paper-to-code-synthesis.md#32-policy-optimization-table-s11): paper $\lambda > 0$ corresponds to code `weight < 0`, with an additional `step_dt` multiplication.

| Code weight | Paper-equivalent $\lambda$ | Final mean imagined reward |
|---:|---:|---:|
| 0.0 | 0.0 | 6.53 |
| -1.0 | 1.0 | 1.82 |
| -5.0 | 5.0 | -8.21 |

The -5.0 run was interrupted once and rerun successfully to completion at `logs/offline/anymal_d_flat/2026-05-05_20-10-23_None`.

### 6.1 Interpretation

The reward decreases monotonically as the penalty magnitude increases, from positive imagined reward at $\lambda = 0$, through reduced positive reward at $\lambda = 1$, to strongly negative imagined reward at $\lambda = 5$. This confirms that the uncertainty penalty is operationally active in the offline pipeline and that its magnitude controls how much of the original reward the penalty subtracts.

The reported "Final mean imagined reward" is the *training-time imagined* reward, which already includes the penalty contribution. It is not the policy's performance under the unpenalized task. A negative imagined reward at high $\lambda$ does not necessarily mean the policy is bad on the underlying task; it means the imagined trajectories accumulate enough epistemic-uncertainty contribution that the penalty dominates the task reward inside the imagination loop.

The $\lambda$ sweep confirms the penalty mechanism is active and increasingly conservative as expected. It does not establish that paper-equivalent $\lambda = 1$ is the optimum, because the paper's optimum is determined by *deployment performance under unpenalized reward* (paper Figure 4), which we did not measure. A real validation of $\lambda = 1$ as optimal would require evaluating each trained policy on a fixed unpenalized reward, either in an Isaac Lab simulator pass or on hardware. Both are out of scope for this reduced-scale check.

## 7. What this validation does and does not establish

What this validation establishes:

- The Git LFS pretrained checkpoint is a real, loadable PyTorch model and contains both policy and dynamics state.
- The shipped dataset is structurally consistent with the active configuration's expected dimensions.
- The standalone offline pipeline runs end-to-end without Isaac Lab and without real environment interaction during policy training.
- The uncertainty mechanism is functional: ensemble disagreement is computed, an epistemic signal is produced, and it correlates with prediction error in the rollout-step direction.
- The penalty is operationally connected: changing `uncertainty_penalty_weight` produces the expected monotonic effect on imagined reward.

What this validation does not establish:

- It does not reproduce paper Figure 3. The paper's figure is at full scale on a different evaluation set.
- It does not reproduce paper Figure 4. We did not evaluate trained policies on an unpenalized task, so we cannot make claims about the optimum $\lambda$.
- It does not validate hardware deployment. No policy was deployed.
- It does not validate the paper's quantitative performance numbers (e.g., the 0.91 ± 0.03 best result in paper Table 1). The shipped dataset is 0.17% of paper-scale.

## 8. Repository state after the runs

`git status --short` at both the project root and the `upstream/robotic_world_model` submodule was clean after all runs completed. The active offline configuration was restored to `uncertainty_penalty_weight: float = -1.0`, the intended RWM-U default. Generated runtime outputs went to ignored directories (`logs/`, `outputs/`).

## 9. Evidence artifacts

The following paths contain the evidence for this validation:

```
Project logs:
  logs/rwmu_offline_complete_20260505_194656.log
  logs/rwmu_fig3_uncertainty_error_fixed_20260505_200032/

Lambda sweep logs:
  logs/rwmu_lambda_sweep_20260505_200241/
    lambda_0p0.log
    lambda_neg_1p0.log
    lambda_neg_5p0_rerun.log

RWM-U output policies:
  upstream/robotic_world_model/logs/offline/anymal_d_flat/2026-05-05_19-46-59_None/
  upstream/robotic_world_model/logs/offline/anymal_d_flat/2026-05-05_20-10-23_None/
```

The `summary.json` file in the Figure-3 evaluation directory contains the exact Pearson, L1, and uncertainty values quoted in Section 5.1. The policy checkpoint directories contain `policy_499.pt` and intermediate checkpoints saved at the configured interval.

## 10. Status

This validation closes the "RWM-U identified" status and replaces it with **Validated at reduced scale**. The status changes recorded on the [reproduction status](reproduction-status.md) page reflect this transition. Two further validation goals remain out of scope for the local reduced-scale work: paper-scale dataset reproduction (requires the original 6M-transition dataset or its generation), and hardware deployment validation (requires Go2 hardware access plus the safety-bounded data collection pipeline that is not in the repository).
