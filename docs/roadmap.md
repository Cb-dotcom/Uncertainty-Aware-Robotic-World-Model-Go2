# Roadmap

**Current phase:** P0 (Setup and repository documentation), in progress.

## Phase overview

### P0 — Setup and repository documentation

*In progress.*

Goals for this phase:

- Validate the full setup running locally.
- Define a clean top-level repository policy.
- Create supervisor-facing documentation.
- Prepare a clean first Git history.
- Document portability assumptions before moving to in-depth analysis.

### P1 — Paper-to-code mapping

*Next.*

Goals:

- Map both related papers to the implementation stack.
- Identify the world-model modules.
- Identify the uncertainty logic.
- Identify the runner and training path.
- Identify the losses and where they are encoded.
- Distinguish conceptual paper structure from practical code structure.

### P2 — Controlled reproduction and understanding

Goals:

- Determine whether the current code reproduces the claimed method or an adapted practical variant.
- Validate the role of upstream RSL-RL and Isaac Lab abstractions.
- Identify which results are likely reproducible on the available hardware.

### P3 — Go2 transfer planning

Goals:

- Map embodiment assumptions from ANYmal-D to Go2.
- Identify required interface, observation, action, reward, and asset changes.
- Assess the implications of sim-only versus real-robot work.

### P4 — Contribution phase

Goals:

- Choose the strongest feasible contribution after the baseline is properly understood.
- Possible directions include robustness, embodiment transfer, compute efficiency, or a justified spiking recurrent extension.
