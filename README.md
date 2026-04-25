# Go2 Integration of an Uncertainty-Aware Robotic World Model

## Project scope
This repository is the research workspace for reproducing and extending the ETH Zurich robotic world model stack toward a future Unitree Go2 integration.

The intended long-term direction is:
1. reproduce and understand the current uncertainty-aware robotic world model stack,
2. validate the untouched upstream baseline locally,
3. map paper concepts to code,
4. port the pipeline toward Go2,
5. only then evaluate possible research contributions.

## Current status
The local software stack has been recovered and a bounded baseline validation has been completed.

Validated so far:
- source-based Isaac Lab workspace is operational,
- upstream `robotic_world_model` is locally runnable,
- the untouched ANYmal-D initialization baseline launches successfully,
- the canonical bounded smoke test reaches learning iterations,
- baseline caveats and shutdown-path behavior have been classified.

Not yet completed:
- full paper-to-code mapping,
- full scientific reproduction of the paper claims,
- Go2 integration,
- dataset adaptation,
- world-model internals analysis,
- spiking extension work.

## Validated baseline
Current validated baseline:
- task: `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0`
- launcher: `upstream/IsaacLab/isaaclab.sh`
- environment: `env_isaaclab_src`

A bounded smoke run is considered successful if:
- environment setup completes, and
- `Learning iteration 0/300` appears.

A timeout-based exit is currently accepted for smoke validation because external termination is known to produce non-graceful shutdown behavior.

See:
- `docs/baseline_status.md`
- `docs/baseline_smoke_command.md`


## Next phase
The next technical phase is paper-to-code mapping:
- identify where the world model is implemented,
- identify where uncertainty is represented,
- identify the practical training path relative to the papers,
- separate what is inherited from Isaac/RSL-RL versus what is novel to the RWM stack.

## Workspace layout
- `docs/` — supervisor-facing and project-facing documentation
- `manifests/` — frozen environment / state records
- `notes/` — working technical notes
- `upstream/` — local upstream dependencies (not tracked here)
- `logs/` — local execution/debug logs (not tracked here)
- `scripts/` — local setup/debug scripts (not tracked here)\n