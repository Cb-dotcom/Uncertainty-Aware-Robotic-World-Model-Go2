---
hide:
  - navigation
---

<div class="rwm-hero" markdown="1">
# Uncertainty-Aware Robotic World Model for Go2
<p>Technical documentation for reproducing, analyzing, and extending the ETH Zurich Robotic World Model pipeline, with the target platform being the Unitree Go2. This site is the in-depth engineering and validation record for the project.</p>
<div class="rwm-badges">
  <span class="rwm-badge">RWM · MBPO-PPO (online)</span>
  <span class="rwm-badge">RWM-U · MOPO-PPO (offline)</span>
  <span class="rwm-badge">Isaac Lab · Unitree Go2</span>
</div>
</div>

The project builds on two papers by Li, Krause, and Hutter (ETH Zurich):

1. **Robotic World Model (RWM)** with the MBPO-PPO policy optimizer, trained online with environment interaction.
2. **Uncertainty-Aware Robotic World Model (RWM-U)** with the MOPO-PPO policy optimizer, trained fully offline with ensemble-based uncertainty penalization.

The upstream codebase contains two model-based training pipelines that share backbone components but have separate entry scripts, configs, environments, and runners. The manager-based pipeline implements online RWM with MBPO-PPO; the standalone `model_based/` pipeline implements offline RWM-U with MOPO-PPO. This project exercises both. The two-pipeline structure is documented in [Uncertainty-Aware Implementation Analysis §1](uncertainty-aware-world-model/implementation-analysis.md#1-two-model-based-pipelines-coexist-in-the-repository).

<div class="rwm-keyresult" markdown="1">
**Headline result.** On a light, contact-dominated quadruped, broadening the offline dataset improves world-model prediction while flattening the ensemble disagreement that MOPO relies on. The bottleneck is uncertainty calibration, not data volume. See [Phase 4G](validation/phase-4g-uncertainty-calibration.md) for the full analysis and [Current Status](project/current-status.md) for the one-page summary.
</div>

## Start here

<div class="grid cards" markdown>

-   __Project status and direction__

    ---

    Where the work stands today, the phase roadmap, and the per-claim verification ledger.

    [Current Status](project/current-status.md) · [Roadmap](project/roadmap.md) · [Reproduction Status](validation/reproduction-status.md)

-   __Setup and reproduction__

    ---

    Local environment, hardware specifications, and the canonical baseline command with what it verifies.

    [Local Environment](setup/local-environment.md) · [Hardware and System Specs](setup/hardware-and-system-specs.md) · [Baseline Execution](validation/baseline-execution.md)

-   __Method: Robotic World Model (RWM)__

    ---

    The online RWM and MBPO-PPO method, the code that implements it, and the merged paper-to-code view.

    [Paper Analysis](world-model/paper-analysis.md) · [Implementation Analysis](world-model/implementation-analysis.md) · [Synthesis](world-model/paper-to-code-synthesis.md)

-   __Method: Uncertainty-Aware RWM (RWM-U)__

    ---

    The offline RWM-U and MOPO-PPO method, its implementation, and how the two papers compose.

    [Paper Analysis](uncertainty-aware-world-model/paper-analysis.md) · [Implementation Analysis](uncertainty-aware-world-model/implementation-analysis.md) · [Relationship](world-model/relationship-to-uncertainty-aware.md)

-   __Validation phases (4A to 4G)__

    ---

    The full experimental record: lab bring-up, the Go2 gait fix, the offline pivot, the two root-cause bugs, and the uncertainty-calibration finding.

    [Phase 4G: headline result](validation/phase-4g-uncertainty-calibration.md) · [Engineering Findings](validation/engineering-findings.md)

-   __Go2 transfer and development__

    ---

    The Go2 asset and task inventory, plus the submodule and fork strategy.

    [Go2 Inventory](go2-transfer/go2-inventory.md) · [Submodules and Forks](development/submodules-and-forks.md)

</div>

The repository and milestone checklist are on [GitHub](https://github.com/Cb-dotcom/Uncertainty-Aware-Robotic-World-Model-Go2).

## Documentation layout

- **Project**: goal, repository layout, current status, roadmap.
- **Setup**: local environment, hardware specifications, lab workstation migration.
- **Validation**: what has been executed and verified, phase by phase, plus the engineering-findings ledger.
- **Robotic World Model**: paper, code, synthesis, task structure, runtime flow, and the bridge to RWM-U.
- **Uncertainty-Aware Robotic World Model**: paper, code, and synthesis for the offline RWM-U and MOPO-PPO pipeline.
- **Development**: submodule and fork strategy.

## Conventions

The documentation makes two kinds of claims, each with its own vocabulary.

Execution claims describe whether code runs as expected:

- **Validated**: a concrete command was executed, the expected outcome was observed, and the run is cited.
- **Validated at reduced scale**: a run completed end-to-end with reduced hyperparameters or smaller-than-paper assets, sufficient to confirm the code path but not the paper's quantitative results.
- **Qualitatively validated at reduced scale**: a paper claim is supported in trend or sign at reduced scale, without reproducing the paper's numbers.
- **Structurally understood**: the code path has been traced and is consistent with its expected role, but no end-to-end execution has been run.
- **Not verified**: the claim has neither been executed nor traced sufficiently to make a judgment.

Mapping claims describe whether code corresponds to what the paper states:

- **Mapped**: the paper concept is identified in the code, with a file path or symbol reference.
- **Partially mapped**: the concept is identified but the code differs in scope or simplification, and the difference is documented.
- **Discrepancy noted**: the code diverges from the paper in a meaningful way, and the divergence is documented.
- **Not mapped**: no code location has been identified yet for the paper concept.

The [Reproduction Status](validation/reproduction-status.md) page tracks execution claims. The Paper-to-Code Synthesis pages for [RWM](world-model/paper-to-code-synthesis.md) and [RWM-U](uncertainty-aware-world-model/paper-to-code-synthesis.md) track mapping claims and discrepancies.

Naming convention used throughout: RWM and MBPO-PPO refer to the first paper's method and policy optimizer; RWM-U and MOPO-PPO refer to the uncertainty-aware extension and its policy optimizer.
