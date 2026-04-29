# Current Status

This page is the executive summary of where the project stands. It is intentionally short and consists almost entirely of pointers. The detailed content lives on the validation, implementation, and synthesis pages; this page only summarizes the state at the level a supervisor needs.

The README progress checklist is the primary place tracking *what is done* (with checkboxes). This page describes the *current state* in narrative form. The two are not redundant: the checklist tracks discrete milestones, this page describes the technical reality at the present moment.

## Summary

The project is in the documentation, validation, and code-understanding phase. The local Isaac Lab and RWM stack is operational. The baseline ANYmal-D path and a reduced-scale RWM pretraining path both run end-to-end on the local laptop. Full-scale pretraining and imagination-based finetuning are planned for the lab workstation; the lab access is the next significant unblock.

## What works locally

The local laptop runs both validated paths:

- The baseline `Init-v0` task reaches PPO learning iterations. Procedure and success criterion: [Baseline Execution](../validation/baseline-execution.md).
- The `Pretrain-v0` task runs end-to-end at reduced scale and produces dynamics checkpoints. The default configuration exceeds the local 8 GB VRAM and is deferred to lab hardware. Procedure and reduced-scale settings: [World-Model Pretraining Check](../validation/world-model-pretraining-check.md).

The reduced-scale pretraining run required a small fix in the model-based runner. The fix is documented on [Submodules and Forks](../development/submodules-and-forks.md#rsl_rl_rwm-pretraining-fix) and the architectural detail is on [Implementation Analysis §11](../world-model/implementation-analysis.md#11-the-pretraining-fix).

## What is blocked

`Finetune-v0` has not been executed because the upstream config expects a pretrained dynamics checkpoint that does not exist locally and a policy run to resume from. The path forward is to generate the checkpoint via lab-scale `Pretrain-v0` and then write a project-specific finetune config pointing at it. Detail: [Checkpoint and Finetune Status](../validation/checkpoint-and-finetune-status.md).

Default-scale `Pretrain-v0` is blocked on local VRAM and is deferred to the lab workstation. The lab workstation has the hardware capacity but has not yet been used for project work.

## Paper-to-code analysis

The RWM paper has been read in full and the upstream codebase has been traced component by component. The findings are split across three pages:

- [Paper Analysis](../world-model/paper-analysis.md) describes the method as the paper presents it, with the equations, figures, and quantitative results.
- [Implementation Analysis](../world-model/implementation-analysis.md) describes the upstream codebase architecturally.
- [Paper-to-Code Synthesis](../world-model/paper-to-code-synthesis.md) aligns the two and surfaces four discrepancies: state mean predicted as a residual, state loss using sampled MSE rather than Gaussian NLL, imagination scale differing in the per-env / per-iteration accounting, and uncertainty hooks present but inactive (the structural surface for RWM-U).

The RWM-U paper has been read but its dedicated analysis (paper, implementation, synthesis under a separate folder) is planned for after the RWM reproduction is further along.

## Reproduction ledger

The full per-claim status of execution and mapping claims is on the [Reproduction Status](../validation/reproduction-status.md) page. Of the eleven execution claims tracked there, three are currently validated (local baseline, reduced-scale pretraining produces checkpoints, the headless local stack) and the rest are blocked on lab access or downstream stages of the pipeline.

## What is next

The next concrete unblock is lab workstation access for full-scale RWM pretraining. The dependency chain from there is documented on [Roadmap](roadmap.md).