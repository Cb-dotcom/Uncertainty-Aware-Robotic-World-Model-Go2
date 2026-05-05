# Current Status

This page is the executive summary of where the project stands. It is intentionally short and consists almost entirely of pointers. The detailed content lives on the validation, implementation, and synthesis pages; this page only summarizes the state at the level a supervisor needs.

The README progress checklist is the primary place tracking *what is done* (with checkboxes). This page describes the *current state* in narrative form. The two are not redundant: the checklist tracks discrete milestones, this page describes the technical reality at the present moment.

## Summary

The project is in the documentation, validation, and code-understanding phase. The local Isaac Lab and RWM stack is operational. The baseline ANYmal-D path, a reduced-scale RWM pretraining path, and the offline RWM-U pipeline all run end-to-end on the local laptop. Full-scale RWM pretraining and imagination-based finetuning are planned for the lab workstation; the lab access is the next significant unblock for the RWM track. The RWM-U track has been validated at reduced scale locally.

## What works locally

The local laptop runs three validated paths:

- The baseline `Init-v0` task reaches PPO learning iterations. Procedure and success criterion: [Baseline Execution](../validation/baseline-execution.md).
- The `Pretrain-v0` task runs end-to-end at reduced scale and produces dynamics checkpoints. The default configuration exceeds the local 8 GB VRAM and is deferred to lab hardware. Procedure and reduced-scale settings: [World-Model Pretraining Check](../validation/world-model-pretraining-check.md).
- The standalone offline RWM-U pipeline (`scripts/reinforcement_learning/model_based/`) runs end-to-end at reduced scale, completing 500 policy-training iterations against the shipped 10K-transition dataset and the materialized 25 MB pretrained world model. A Figure-3-style uncertainty-error evaluation and a $\lambda$ sweep have been performed at reduced scale. Procedure, results, and scope boundaries: [RWM-U Execution Check](../validation/rwm-u-execution-check.md).

The reduced-scale RWM pretraining run required a small fix in the model-based runner. The fix is documented on [Submodules and Forks](../development/submodules-and-forks.md#rsl_rl_rwm-pretraining-fix) and the architectural detail is on [Implementation Analysis §11](../world-model/implementation-analysis.md#11-the-pretraining-fix).

## What is blocked

`Finetune-v0` (manager-based pipeline) has not been executed because the upstream config expects a pretrained dynamics checkpoint that does not exist locally and a policy run to resume from. The path forward is to generate the checkpoint via lab-scale `Pretrain-v0` and then write a project-specific finetune config pointing at it. Detail: [Checkpoint and Finetune Status](../validation/checkpoint-and-finetune-status.md).

Default-scale `Pretrain-v0` is blocked on local VRAM and is deferred to the lab workstation. The lab workstation has the hardware capacity but has not yet been used for project work.

## Paper-to-code analysis

Both papers have been read in full and the upstream codebase has been traced component by component. The findings are split across two parallel sets of three pages each, plus a cross-paper bridge.

For RWM (first paper):

- [Paper Analysis](../world-model/paper-analysis.md) describes the method as the paper presents it, with the equations, figures, and quantitative results.
- [Implementation Analysis](../world-model/implementation-analysis.md) describes the manager-based pipeline architecturally.
- [Paper-to-Code Synthesis](../world-model/paper-to-code-synthesis.md) aligns the two and surfaces four discrepancies: state mean predicted as a residual, state loss using sampled MSE rather than Gaussian NLL, imagination scale differing in the per-env / per-iteration accounting, and uncertainty hooks present but inactive in the manager-based pipeline.

For RWM-U (second paper):

- [Paper Analysis](../uncertainty-aware-world-model/paper-analysis.md) describes the method as the paper presents it.
- [Implementation Analysis](../uncertainty-aware-world-model/implementation-analysis.md) describes the standalone offline pipeline architecturally, including the active configuration values, the dataset format, and the pretrained model shipping mechanism.
- [Paper-to-Code Synthesis](../uncertainty-aware-world-model/paper-to-code-synthesis.md) aligns the two and surfaces two RWM-U-specific discrepancies (trajectory sampling vs ensemble-mean dynamics, standard-deviation-summed vs variance reduction for the epistemic signal) plus a sign and time-step convention that maps the paper's $\lambda$ to the code's negative weight.

The [Relationship to Uncertainty-Aware RWM](../world-model/relationship-to-uncertainty-aware.md) page bridges the two: what stays the same between RWM and RWM-U, what changes, what the codebase implements for each, and what this means for project staging.

## Reproduction ledger

The full per-claim status of execution and mapping claims is on the [Reproduction Status](../validation/reproduction-status.md) page. Of the sixteen execution claims tracked there, nine have positive validation status: five validated, two validated at reduced scale, and two qualitatively validated at reduced scale. The remaining seven are blocked on lab access, hardware availability, or are out of current project scope. Of the seventeen mapping claims, twelve are mapped (eight fully, three partially mapped, one mapped under sign and time-step convention), four are discrepancies noted, and one is a discrepancy in scale.

## What is next

The Phase 4 work splits into two natural tracks that can run independently:

- *Lab-scale RWM*: take the manager-based pipeline to lab hardware. Full-scale `Pretrain-v0` (paper hyperparameters, 2500 iterations, batch 1024), generate a real dynamics checkpoint, then run `Finetune-v0` end-to-end. This unblocks the `Finetune-v0` checkpoint dependency and validates the manager-based path at paper scale.
- *Go2 transfer prep*: the project's primary research contribution. Acquire a Go2 USD/URDF, set up an Isaac Lab Go2 flat-locomotion environment, retune reward weights, recompute observation normalization, and prepare for both online RWM and offline RWM-U runs on Go2.

The dependency chain from there is documented on [Roadmap](roadmap.md).