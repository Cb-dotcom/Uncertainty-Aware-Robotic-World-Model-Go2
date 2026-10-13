# [Uncertainty-Aware Robotic World Model for Go2](https://cb-dotcom.github.io/Uncertainty-Aware-Robotic-World-Model-Go2/)

A research workspace for reproducing, analyzing, and extending the ETH Zurich Robotic World Model line of work toward Unitree Go2 integration.

## Project goal

Model-based reinforcement learning offers a path to sample-efficient robotic control, but its practical value depends on whether the learned dynamics model is accurate and reliable enough over long horizons to drive a usable policy. The two ETH Zurich papers this project builds on, taken together, define a coherent program: first construct a world model that supports stable autoregressive prediction over long horizons, then make that world model uncertainty-aware so it can be used in fully offline settings on real hardware.

This project takes the published implementation, validates it locally and on the lab workstation, dissects it at the level of paper-to-code correspondence, and carries the offline pipeline through to a contribution on a different quadruped embodiment, the Unitree Go2.

## Research foundation

### Paper 1: Robotic World Model (RWM + MBPO-PPO)

*Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics* (Li, Krause, Hutter, ETH Zurich, arXiv:2501.10100) introduces:

1. **A self-supervised, dual-autoregressive training scheme** for a GRU-based dynamics model that predicts future observations and privileged signals such as contacts from a window of past observation-action pairs. Inner autoregression updates GRU hidden states across the history horizon. Outer autoregression feeds the model's own predictions back into the input over a forecast horizon, training the model on the input distribution it will face at deployment. The objective is a decay-weighted multi-step prediction loss.
2. **MBPO-PPO**, a policy optimization framework that uses the trained world model as an imagination simulator. The replay buffer stores real transitions, the world model is updated from this buffer, imagined rollouts are generated for hundreds of steps, and PPO updates the policy on those imagined trajectories.
3. **Zero-shot hardware deployment** of policies trained entirely in the learned simulator, demonstrated on ANYmal-D and Unitree G1, improving on SHAC and DreamerV3 baselines on long-horizon prediction error and policy quality.

The paper's stated limitations include reliance on online environment interaction to keep the world model accurate, and the absence of any explicit uncertainty handling. These are the points the second paper addresses.

### Paper 2: Uncertainty-Aware Robotic World Model (RWM-U + MOPO-PPO)

*Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots* (Li, Krause, Hutter, ETH Zurich, arXiv:2504.16680) extends the framework to the offline setting, where the policy must be trained entirely from a fixed dataset with no further environment interaction. It has two contributions:

1. **RWM-U** augments RWM with bootstrap-ensemble uncertainty estimation. Each ensemble member shares a GRU feature extractor but maintains its own observation prediction head, predicting a Gaussian over the next observation. Variance within a member captures aleatoric uncertainty, variance across ensemble means captures epistemic uncertainty, and both propagate through the autoregressive rollout.
2. **MOPO-PPO** adapts the MOPO uncertainty-penalty objective to PPO, penalizing the reward by the epistemic uncertainty estimate scaled by a coefficient that controls the penalty strength. The technical difficulty is that this penalty must be propagated over the same long episodic rollouts used in MBPO-PPO, rather than the one-step rollouts of standard MOPO.

The reported result is a 0.91 plus/minus 0.03 normalized score on ANYmal-D from a fully offline mixed dataset (800K simulation plus 200K real transitions), surpassing the 0.88 plus/minus 0.01 of an online model-free baseline trained purely in simulation. The paper reports this as the first demonstration of uncertainty-penalized offline MBRL deployed on physical hardware.

### How the two papers compose

The relationship between the two papers maps onto two separate pipelines in the upstream codebase. The manager-based pipeline (`scripts/reinforcement_learning/rsl_rl/`) implements online RWM with MBPO-PPO using Isaac Lab and `MBPOOnPolicyRunner`. The standalone pipeline (`scripts/reinforcement_learning/model_based/`) implements offline RWM-U with MOPO-PPO, loads a pretrained world model and a fixed dataset, and uses vanilla PPO without Isaac Lab. Both pipelines share backbone components (`SystemDynamicsEnsemble`, `ActorCritic`, the PPO algorithm) but have separate entry scripts, configs, environments, and runtime structure. The active configurations select between the methods: the manager-based pipeline runs with `ensemble_size = 1` and `uncertainty_penalty_weight = -0.0` (base RWM), and the standalone pipeline runs with `ensemble_size = 5` and `uncertainty_penalty_weight = -1.0` (RWM-U). Recognizing the two-pipeline structure is one of the higher-leverage findings of the paper-to-code analysis.

## Current state

The project has moved from reproduction into a contribution, and is ongoing. The offline RWM-U and MOPO-PPO pipeline has been exercised end-to-end on both ANYmal-D (as a validation gate) and the Unitree Go2 (the contribution target). On the online side, MBPO-PPO reproduces the paper at its operating budget on Go2 (best-checkpoint tracking around 0.92 against the paper's 0.90); the documented model-exploitation regime appears only when the loop is run far past that budget, which is the behaviour the paper sidesteps by early checkpoint selection. Two root-cause bugs were found and fixed along the way: a policy that was optimized against an untrained world model, and a normalizer inconsistency that froze the imagined rollout. The headline result concerns uncertainty calibration: on a light, contact-dominated quadruped, broadening the offline dataset improves world-model prediction while flattening the ensemble disagreement that the MOPO penalty relies on, so the bottleneck is calibration of the uncertainty signal rather than data volume. Two control experiments bound what is achievable: a reward-support proxy reduces sliding only by freezing locomotion, and warm-starting from a competent policy is locally non-destructive but yields no measurable refinement. The path to a deployable, hardware-ready offline policy therefore runs through the uncertainty architecture (independent or diversity-regularized ensembles, better-calibrated estimators, and forward-kinematics reward support) rather than through more data. The full analysis is on the documentation site.

## Documentation

The full project documentation is published at
**[cb-dotcom.github.io/Uncertainty-Aware-Robotic-World-Model-Go2](https://cb-dotcom.github.io/Uncertainty-Aware-Robotic-World-Model-Go2/)**.

It contains the project status, hardware and environment setup notes, the phase-by-phase validation record, the paper-to-code analysis for both papers, and the development roadmap.

## Progress

- [x] Local Isaac Lab and Robotic World Model stack installed and runs headlessly.
- [x] Baseline ANYmal-D initialization task reaches PPO learning iterations on local hardware.
- [x] Reduced-scale world-model pretraining executes end-to-end and produces a checkpoint.
- [x] Task modes, runtime pipeline, and loss decomposition mapped at the code level.
- [x] Paper-to-code synthesis pages for both RWM and RWM-U completed.
- [x] Cross-paper relationship page (RWM to RWM-U) completed.
- [x] Migration to the lab workstation completed and validated.
- [x] Online RWM and MBPO-PPO reproduced at the paper's budget on Go2 (tracking ~0.92 vs 0.90); the gait scoot fixed; the documented model-exploitation regime characterized only when the loop is run past that budget.
- [x] Pivot to the offline RWM-U and MOPO-PPO method as the project direction.
- [x] Offline Go2 pipeline and real-environment evaluation harness built.
- [x] Bug 1 fixed: policy was being optimized against an untrained world model.
- [x] Bug 2 fixed: normalizer inconsistency that froze the imagined rollout.
- [x] ANYmal fitter gate: the standalone fitter converges and strict-loads the reference module.
- [x] Exploit-rollout uncertainty diagnostic built and run on real policy-failure traces.
- [x] Headline finding established: coverage improves prediction while flattening the MOPO disagreement signal (calibration, not volume), with the prediction-error versus disagreement decoupling replicated across two traces.
- [x] Stage-Mixed experiment: PPO-replay diversity decorrelates the ensemble heads (cosine 0.965 to 0.850) but does not beat the curated model on selectivity.

### Remaining

- [ ] Lift the disagreement ceiling and recalibrate the signal by fitting an ANYmal world model with our own multi-member trainer as the gating check, then move to independent or diversity-regularized ensemble trunks and a better-calibrated estimator.
- [ ] Restore observable reward support and adopt warm-start as the deployment recipe: add forward-kinematics targets so the foot-slide term is penalizable in imagination, and validate a morphology-aware reward online.
- [ ] Deploy the resulting offline RWM-U policy on Go2 hardware.
- [ ] Contribute and publish the result.

## Repository structure

```text
.
├── docs/                       Published project documentation
├── lab/                        Lab-workstation setup and run configuration
├── logs/                       Training and evaluation run outputs
├── manifests/                  Frozen environment and state records
├── scripts/                    Local setup and validation scripts
├── upstream/
│   ├── IsaacLab/               Isaac Lab source, kept aligned with upstream
│   ├── robotic_world_model/    Task, environment, and config code
│   └── rsl_rl_rwm/             RSL-RL backend with world-model extensions
├── .gitignore
├── .gitmodules
├── mkdocs.yml                  Documentation site configuration
├── README.md
└── requirements-lab.txt        Lab-workstation Python dependencies
```

## Cloning

The upstream codebases are tracked as Git submodules. Clone with:

```bash
git clone --recurse-submodules https://github.com/Cb-dotcom/Uncertainty-Aware-Robotic-World-Model-Go2.git
```

The offline RWM-U pipeline ships its pretrained world model as a Git LFS object. After cloning, run:

```bash
cd upstream/robotic_world_model
git lfs install
git lfs pull
```

to materialize the model file (`assets/models/pretrain_rnn_ens.pt`, 25 MB).

## License and attribution

The original *Robotic World Model* and *Uncertainty-Aware Robotic World Model* methods, code, and the ANYmal-D and Unitree G1 results are due to Chenhao Li, Andreas Krause, and Marco Hutter at ETH Zurich. This repository is a reproduction and extension workspace and does not claim authorship of the original methods. Upstream code retains its original license. Project-specific additions (documentation, scripts, configuration) are released under the license declared at the repository root.