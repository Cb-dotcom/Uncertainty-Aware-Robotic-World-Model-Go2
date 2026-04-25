# Roadmap

## Phase overview

### P0: Setup & Repository documentation
Status: in progress

Current goals:
- validated full setup running,
- define clean top-level repository policy,
- create supervisor-facing documentation,
- prepare a clean first Git history,
- document portability assumptions before moving to in depth analysis.

### P1: Paper-to-code mapping
Status: next

Planned goals:
- map both related papers to the implementation stack,
- identify world-model modules,
- identify uncertainty logic,
- identify runner/training path,
- identify losses and where they are encoded,
- distinguish conceptual paper structure from practical code structure.

### P2: Controlled reproduction understanding
Planned goals:
- inspect whether current code reproduces the claimed method or an adapted practical variant,
- validate the role of upstream RSL-RL and Isaac abstractions,
- identify which results are likely reproducible on current hardware.

### P3: Go2 transfer planning
Planned goals:
- map embodiment assumptions from ANYmal-D to Go2,
- identify required interface, observation, action, reward, and asset changes,
- assess sim-only versus real-robot implications.

### P4: Contribution phase
Planned goals:
- choose the strongest feasible contribution after the baseline is properly understood,
- possible directions include robustness, embodiment transfer, compute efficiency, or a justified spiking recurrent extension.