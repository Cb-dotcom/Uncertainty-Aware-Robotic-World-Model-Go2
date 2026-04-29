# [Uncertainty-Aware Robotic World Model for Go2](https://cb-dotcom.github.io/Uncertainty-Aware-Robotic-World-Model-Go2/)

A research workspace for reproducing, analyzing, and extending the ETH Zurich Robotic World Model line of work toward Unitree Go2 integration.

## Project goal

Model-based reinforcement learning offers a path to sample-efficient robotic control, but its practical value depends on whether the learned dynamics model is accurate and reliable enough over long horizons to drive a usable policy. The two ETH Zurich papers this project builds on, taken together, define a coherent program: first construct a world model that supports stable autoregressive prediction over long horizons; then make that world model uncertainty-aware so it can be used in fully offline settings on real hardware.

This project takes the published implementation, validates it locally on ANYmal-D, dissects it at the level of paper-to-code correspondence, and prepares the pipeline for transfer to a different quadruped embodiment, the Unitree Go2.

## Research foundation

### Paper 1: Robotic World Model (RWM + MBPO-PPO)

*Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics* (Li, Krause, Hutter, ETH Zurich, arXiv:2501.10100) introduces:

1. **A self-supervised, dual-autoregressive training scheme** for a GRU-based dynamics model that predicts future observations and privileged signals (such as contacts) from a window of past observation-action pairs. *Inner* autoregression updates GRU hidden states across the history horizon. *Outer* autoregression feeds the model's own predictions back into the input over a forecast horizon, training the model on the input distribution it will face at deployment. The objective is a decay-weighted multi-step prediction loss.
2. **MBPO-PPO**, a policy optimization framework that uses the trained world model as an imagination simulator. The replay buffer stores real transitions, the world model is updated from this buffer, imagined rollouts are generated for hundreds of steps, and PPO updates the policy on those imagined trajectories.
3. **Zero-shot hardware deployment** of policies trained entirely in the learned simulator, demonstrated on ANYmal-D and Unitree G1, beating SHAC and DreamerV3 baselines on long-horizon prediction error and policy quality.

The paper's stated limitations include reliance on online environment interaction to keep the world model accurate, and the absence of any explicit uncertainty handling. These are exactly what the second paper addresses.

### Paper 2: Uncertainty-Aware Robotic World Model (RWM-U + MOPO-PPO)

*Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots* (Li, Krause, Hutter, ETH Zurich, arXiv:2504.16680) extends the framework to the offline setting, where the policy must be trained entirely from a fixed dataset with no further environment interaction. Two contributions:

1. **RWM-U** augments RWM with bootstrap-ensemble uncertainty estimation. Each ensemble member shares a GRU feature extractor but maintains its own observation prediction head, predicting a Gaussian over the next observation. Variance within a member captures aleatoric uncertainty, variance across ensemble means captures epistemic uncertainty, and both propagate consistently through the autoregressive rollout.
2. **MOPO-PPO** adapts the MOPO uncertainty-penalty objective to PPO, modifying the reward as `r̃ = r − λ · u` where `u` is the epistemic uncertainty estimate and `λ` controls the penalty strength. The technical hard part is that this penalty must be propagated over the same 100-step episodic rollouts used in MBPO-PPO, rather than the one-step rollouts of standard MOPO.

The reported result is a 0.91 ± 0.03 normalized score on ANYmal-D from a fully offline mixed dataset (800K simulation + 200K real transitions), surpassing the 0.88 ± 0.01 of an online model-free baseline trained purely in simulation. The paper reports this as the first demonstration of uncertainty-penalized offline MBRL deployed on physical hardware.

### How the two papers compose

The relationship between the two papers maps directly onto the codebase. The current upstream implementation already exposes uncertainty hooks (an `uncertainty_penalty_weight` in the imagination reward, an `ensemble_size` parameter on the dynamics model), but these are wired off in the validated configuration: `ensemble_size = 1` collapses the ensemble to a single network, and `uncertainty_penalty_weight = 0` removes the MOPO-style penalty term. In other words, the codebase as published implements Paper 1 in its active configuration but reserves the structural surface for the Paper 2 extension. Recognizing this is one of the higher-leverage findings of the paper-to-code analysis.

## Documentation

The full project documentation is published at
**[cb-dotcom.github.io/Uncertainty-Aware-Robotic-World-Model-Go2](https://cb-dotcom.github.io/Uncertainty-Aware-Robotic-World-Model-Go2/)**.

It contains the project status, hardware and environment setup notes, validation results, the paper-to-code analysis, and the development roadmap.

## Progress

- [x] Local Isaac Lab and Robotic World Model stack installed and runs headlessly.
- [x] Baseline ANYmal-D initialization task reaches PPO learning iterations on local hardware.
- [x] Reduced-scale world-model pretraining executes end-to-end and produces a checkpoint.
      Full default configuration exceeds local GPU memory and is deferred to lab hardware.
- [x] Task modes, runtime pipeline, and loss decomposition mapped at the code level.
- [ ] Paper-to-code synthesis pages for both RWM and RWM-U completed.
- [ ] Imagination-based finetuning executed locally with a usable pretrained checkpoint.
- [ ] Full-scale pretraining on lab workstation, validated against Paper 1 prediction accuracy.
- [ ] MBPO-PPO finetuning executed and benchmarked.
- [ ] RWM-U + MOPO-PPO configuration activated and validated against Paper 2.
- [ ] ANYmal-D to Unitree Go2 embodiment transfer in simulation.
- [ ] Go2 deployment on real hardware.
- [ ] Research contribution direction identified.

For per-claim reproduction status, see the [Reproduction Status](https://cb-dotcom.github.io/Uncertainty-Aware-Robotic-World-Model-Go2/validation/reproduction-status/) page.

## Repository structure

```text
.
├── docs/                       Published project documentation
├── manifests/                  Frozen environment and state records
├── scripts/                    Local setup and validation scripts
├── upstream/
│   ├── IsaacLab/               Isaac Lab source, kept aligned with upstream
│   ├── robotic_world_model/    Task, environment, and config code
│   └── rsl_rl_rwm/             RSL-RL backend with world-model extensions
├── .github/workflows/          
├── .gitignore                  
├── .gitmodules                 
├── mkdocs.yml                  Documentation site configuration
└── README.md
```

## Cloning

The upstream codebases are tracked as Git submodules. Clone with:

```bash
git clone --recurse-submodules https://github.com/Cb-dotcom/Uncertainty-Aware-Robotic-World-Model-Go2.git
```

## License and attribution

The original *Robotic World Model* and *Uncertainty-Aware Robotic World Model* methods, code, and the ANYmal-D and Unitree G1 results are due to Chenhao Li, Andreas Krause, and Marco Hutter at ETH Zurich. This repository is a reproduction and extension workspace and does not claim authorship of the original methods. Upstream code retains its original license; project-specific additions (documentation, scripts, configuration) are released under the license declared at the repository root.