# Project Documentation

This is the technical documentation for a research workspace that reproduces, analyzes, and extends the ETH Zurich Robotic World Model pipeline for quadruped locomotion, with the eventual target of Unitree Go2 integration.

The project builds on two papers by Li, Krause, and Hutter (ETH Zurich):

1. Robotic World Model (RWM) with the MBPO-PPO policy optimizer, trained online with environment interaction.
2. Uncertainty-Aware Robotic World Model (RWM-U) with the MOPO-PPO policy optimizer, trained fully offline with ensemble-based uncertainty penalization.

The original project's source code and files contain two model-based training pipelines that share backbone components but have separate entry scripts, configs, environments, and runners. The manager-based pipeline implements online RWM with MBPO-PPO, whereas the standalone `model_based/` pipeline implements offline RWM-U with MOPO-PPO. The project exercises both. The two-pipeline structure is documented in detail in [Uncertainty-Aware Implementation Analysis §1](uncertainty-aware-world-model/implementation-analysis.md#1-two-model-based-pipelines-coexist-in-the-repository).

The milestone checklist of this project will be at [repository README](https://github.com/Cb-dotcom/Uncertainty-Aware-Robotic-World-Model-Go2).
This documentation is for the in depth technical side. 

## Reading paths

Different readers will want different starting points. Be assured, all roads lead to rome, you will all end up happily in the results page. 

- For project status and direction: [Current Status](project/current-status.md), then [Roadmap](project/roadmap.md), then [Reproduction Status](validation/reproduction-status.md) for the per-claim verification ledger.
- For setup and reproduction: [Local Environment](setup/local-environment.md), then [Hardware and System Specs](setup/hardware-and-system-specs.md), then [Baseline Execution](validation/baseline-execution.md) for the canonical command and what it verifies.
- For paper-to-code understanding (RWM): [Paper Analysis](world-model/paper-analysis.md) for the method as the paper presents it, [Implementation Analysis](world-model/implementation-analysis.md) for what the code does, and [Paper-to-Code Synthesis](world-model/paper-to-code-synthesis.md) for the merged view including discrepancies.
- For paper-to-code understanding (RWM-U): [Paper Analysis](uncertainty-aware-world-model/paper-analysis.md), [Implementation Analysis](uncertainty-aware-world-model/implementation-analysis.md), [Paper-to-Code Synthesis](uncertainty-aware-world-model/paper-to-code-synthesis.md), and [Relationship to Uncertainty-Aware RWM](world-model/relationship-to-uncertainty-aware.md) for how the two papers compose.
- For execution evidence: [Baseline Execution](validation/baseline-execution.md), [World-Model Pretraining Check](validation/world-model-pretraining-check.md), and [RWM-U Execution Check](validation/rwm-u-execution-check.md) for the reduced-scale offline pipeline validation.

## Documentation layout

- **Project**: goal, repository layout, status, roadmap.
- **Setup**: local environment, hardware specs, lab workstation migration.
- **Validation**: what has been executed locally and what has been verified.
- **Robotic World Model**: paper, code, synthesis, task structure, runtime flow, and the cross-paper bridge to RWM-U.
- **Uncertainty-Aware Robotic World Model**: paper, code, synthesis for the offline RWM-U + MOPO-PPO pipeline.
- **Development**: submodule and fork strategy.

## Conventions

The documentation makes two kinds of claims, each with its own vocabulary.

Execution claims describe whether code runs as expected:

- **Validated**: a concrete command was executed, the expected outcome was observed, and the run is cited.
- **Validated at reduced scale**: a concrete run completed end-to-end with reduced hyperparameters or smaller-than-paper-scale assets, sufficient to confirm the code path but not the paper's quantitative results.
- **Qualitatively validated at reduced scale**: a paper claim has been supported in trend or sign at reduced scale, without reproducing the paper's quantitative numbers.
- **Structurally understood**: the code path has been traced and is consistent with its expected role, but no execution has been run end-to-end.
- **Not verified**: the claim has neither been executed nor traced sufficiently to make a judgment.

Mapping claims describe whether code corresponds to what the paper states:

- **Mapped**: the paper concept has been identified in the code, with a file path or symbol reference.
- **Partially mapped**: the concept is identified but the code differs from the paper in scope or simplification, and the difference is documented.
- **Discrepancy noted**: the code diverges from the paper in a meaningful way (for example, a different loss formulation or an inactive component) and the divergence is documented.
- **Not mapped**: no code location has been identified yet for the paper concept.

The [Reproduction Status](validation/reproduction-status.md) page tracks execution claims. The [Paper-to-Code Synthesis](world-model/paper-to-code-synthesis.md) page (RWM) and the [Paper-to-Code Synthesis](uncertainty-aware-world-model/paper-to-code-synthesis.md) page (RWM-U) track mapping claims and discrepancies for their respective papers.

Naming convention used throughout: RWM and MBPO-PPO refer to the first paper's method and policy optimizer; RWM-U and MOPO-PPO refer to the uncertainty-aware extension and its policy optimizer.