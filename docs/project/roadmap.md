# Roadmap

This roadmap describes the phase plan for the project. It is the structural counterpart to the README's progress checklist: the checklist tracks *which milestones are done*, the roadmap describes *what each phase consists of and how phases depend on each other*.

The current phase is marked at the top. Phases are sequential by default; some may overlap when independent work streams allow.

**Current phase: P1 (RWM understanding and validation), in progress. Local validation complete; lab work pending.**

## P0: Local setup and baseline validation

*Completed.*

The local environment was brought up, the upstream baseline was executed headlessly, and the project state was frozen in `manifests/`. The baseline `Init-v0` task reaches PPO learning iterations on the local laptop.

This phase established the working reference state from which all later phases proceed.

## P0.5: Repository organization and reproducibility

*In progress, mostly complete.*

The top-level repository was created, the documentation site was deployed via GitHub Pages and MkDocs Material, the upstream dependencies were converted to submodules with forks for the two repositories expected to receive modifications, and the documentation was structured around project use rather than raw notes.

What remains: ensure all submodule pointers are pushed and reproducible, and verify the clone-with-submodules workflow from a fresh checkout on a new machine. The latter is part of the lab migration.

## P1: RWM understanding and validation

*Current phase, in progress.*

The goal is to understand the base RWM pipeline before extending toward RWM-U or transferring to Go2.

What has been completed:

- Reduced-scale `Pretrain-v0` validated locally (see [World-Model Pretraining Check](../validation/world-model-pretraining-check.md)).
- The four task modes have been enumerated and their roles documented (see [Task Modes](../world-model/task-modes.md)).
- The runtime pipeline has been traced from launcher to checkpoint (see [Runtime Pipeline](../world-model/runtime-pipeline.md)).
- The RWM paper has been analyzed in full, with equations, figures, and quantitative results (see [Paper Analysis](../world-model/paper-analysis.md)).
- The upstream codebase has been analyzed architecturally (see [Implementation Analysis](../world-model/implementation-analysis.md)).
- The paper and code have been mapped against each other and four discrepancies have been documented (see [Paper-to-Code Synthesis](../world-model/paper-to-code-synthesis.md)).

What remains in this phase:

- A second-pass code read to verify the parameter values and code paths cited in the implementation analysis against the current upstream code (some values are taken from earlier drafts and marked as assumed in the synthesis tables).
- Final pass on documentation polish.

## P1.5: Full RWM training on lab hardware

*Planned. Blocked on lab access.*

The goal is to execute default-scale `Pretrain-v0` on the lab workstation, generate a usable dynamics checkpoint, and then run `Finetune-v0` against it.

The expected sequence:

1. Compare the lab Docker Isaac Lab installation against the project's pinned submodule. Resolve any version mismatch.
2. Reproduce the local baseline check on lab hardware.
3. Reproduce the reduced-scale pretraining check on lab hardware.
4. Run default-scale pretraining (target: paper-reported settings, $M=32$, $N=8$, full batch size, 2500 iterations, 5 seeds).
5. Generate the dynamics checkpoint required by `Finetune-v0`.
6. Write a project-specific finetune config that points to the locally generated checkpoint (see [Checkpoint and Finetune Status](../validation/checkpoint-and-finetune-status.md)).
7. Execute finetuning and benchmark policy quality against the paper's reported `0.90 ± 0.04` velocity-tracking reward.

This phase produces the first real reproduction result: a policy trained inside the learned world model that performs at paper-reported levels.

## P2: RWM-U analysis and validation

*Planned.*

The goal is to bring the second paper (RWM-U + MOPO-PPO) to the same depth of analysis as RWM, then exercise the codebase paths that activate the uncertainty handling.

The analysis side mirrors P1: a paper analysis page, an implementation analysis page (focused on what changes when the uncertainty hooks are activated), and a synthesis page. These will live under a parallel `uncertainty-aware/` folder.

The validation side requires:

- Setting `ensemble_size > 1` in the runner config and validating that the ensemble construction works as expected.
- Activating `uncertainty_penalty_weight > 0` and verifying the imagined reward includes the penalty term.
- Running the offline RWM-U + MOPO-PPO configuration on a fixed dataset and reproducing the paper's `0.91 ± 0.03` mixed-dataset result.

This phase requires a working RWM dynamics checkpoint from P1.5 as a starting point.

## P3: Go2 simulation transition

*Planned.*

The goal is to port the pipeline from ANYmal-D to Unitree Go2, in simulation.

The expected work:

- Identify the Go2 robot asset for Isaac Lab and verify availability.
- Map the embodiment differences: joint count, joint ordering, action dimensions, observation terms, contact sensors, termination conditions, default joint configuration, normalization constants.
- Adapt the reward function (the equations in [Paper Analysis §6](../world-model/paper-analysis.md#6-reward-formulation) need per-Go2 weight tuning; ANYmal-D and Unitree G1 already use slightly different weights in the paper, suggesting Go2 will need its own).
- Create Go2 versions of the four task modes (`Init`, `Pretrain`, `Finetune`, `Visualize`).
- Run the validation sequence (baseline, reduced-scale pretrain, full pretrain, finetune) for the Go2 configuration.

This phase produces the first end-to-end demonstration of the pipeline on Go2 in simulation.

## P4: Go2 hardware deployment

*Planned.*

The goal is to deploy a trained Go2 policy on real hardware, mirroring the paper's zero-shot deployment claim for ANYmal-D and Unitree G1.

This phase introduces concerns not present in earlier phases: hardware safety, sensor calibration, sim-to-real gap analysis, recovery procedures, and the limits the RWM paper itself acknowledges around online learning on hardware. Detailed planning for this phase will be documented when P3 is sufficiently complete.

## P5: Research contribution

*Planned. Form intentionally not fixed.*

The contribution phase begins after the previous phases provide a working, understood, validated RWM and RWM-U pipeline on Go2 in both simulation and on hardware. The form of the contribution is not chosen in advance because the right contribution depends on what the earlier phases reveal: which limitations are most pressing, which extensions are most feasible given the available infrastructure, which open questions in the research arc admit a tractable answer with the project's resources.

Candidate directions are noted in early conversations but are not committed to this roadmap.