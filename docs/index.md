# Project Documentation

This is the technical documentation for a research workspace that reproduces, analyzes, and extends the ETH Zurich Robotic World Model pipeline for quadruped locomotion, with the eventual target of Unitree Go2 integration.

The project builds on two papers by Li, Krause, and Hutter (ETH Zurich):

1. Robotic World Model (RWM) with the MBPO-PPO policy optimizer, trained online with environment interaction.
2. Uncertainty-Aware Robotic World Model (RWM-U) with the MOPO-PPO policy optimizer, trained fully offline with ensemble-based uncertainty penalization.

A single upstream codebase implements both. The default configuration runs as RWM with MBPO-PPO; setting `ensemble_size > 1` and `uncertainty_penalty_weight > 0` activates the RWM-U + MOPO-PPO path. The project will exercise both.

For high-level project status and the milestone checklist, see the [repository README](https://github.com/Cb-dotcom/Uncertainty-Aware-Robotic-World-Model-Go2). This documentation site is the technical companion to that README.

## Reading paths

Different readers will want different starting points.

- For project status and direction: [Current Status](project/current-status.md), then [Roadmap](project/roadmap.md), then [Reproduction Status](validation/reproduction-status.md) for the per-claim verification ledger.
- For setup and reproduction: [Local Environment](setup/local-environment.md), then [Hardware and System Specs](setup/hardware-and-system-specs.md), then [Baseline Execution](validation/baseline-execution.md) for the canonical command and what it verifies.
- For paper-to-code understanding: [Paper Analysis](world-model/paper-analysis.md) for the method as the paper presents it, [Implementation Analysis](world-model/implementation-analysis.md) for what the code does, and [Paper-to-Code Synthesis](world-model/paper-to-code-synthesis.md) for the merged view including discrepancies.

## Documentation layout

- **Project**: goal, repository layout, status, roadmap.
- **Setup**: local environment, hardware specs, lab workstation migration.
- **Validation**: what has been executed locally and what has been verified.
- **Robotic World Model**: paper, code, synthesis, task structure, runtime flow.
- **Development**: submodule and fork strategy.

## Conventions

The documentation makes two kinds of claims, each with its own vocabulary.

Execution claims describe whether code runs as expected:

- **Validated**: a concrete command was executed, the expected outcome was observed, and the run is cited.
- **Structurally understood**: the code path has been traced and is consistent with its expected role, but no execution has been run end-to-end.
- **Not verified**: the claim has neither been executed nor traced sufficiently to make a judgment.

Mapping claims describe whether code corresponds to what the paper states:

- **Mapped**: the paper concept has been identified in the code, with a file path or symbol reference.
- **Partially mapped**: the concept is identified but the code differs from the paper in scope or simplification, and the difference is documented.
- **Discrepancy noted**: the code diverges from the paper in a meaningful way (for example, a different loss formulation or an inactive component) and the divergence is documented.
- **Not mapped**: no code location has been identified yet for the paper concept.

The [Reproduction Status](validation/reproduction-status.md) page tracks execution claims. The [Paper-to-Code Synthesis](world-model/paper-to-code-synthesis.md) page tracks mapping claims and discrepancies.

Naming convention used throughout: RWM and MBPO-PPO refer to the first paper's method and policy optimizer; RWM-U and MOPO-PPO refer to the uncertainty-aware extension and its policy optimizer.