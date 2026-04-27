# Go2 Integration of an Uncertainty-Aware Robotic World Model

## Project scope

This repository is the research workspace for reproducing and extending the ETH Zurich robotic world model stack toward a future Unitree Go2 integration.

The intended long-term direction is:

1. Reproduce and understand the current uncertainty-aware robotic world model stack.
2. Validate the untouched upstream baseline locally.
3. Map paper concepts to code.
4. Port the pipeline toward Go2.
5. Only then evaluate possible research contributions.

## Current status

The local software stack has been recovered and a bounded baseline validation has been completed.

Validated:

- Source-based Isaac Lab workspace is operational.
- Upstream `robotic_world_model` is locally runnable.
- The untouched ANYmal-D initialization baseline launches successfully.
- The canonical bounded smoke test reaches learning iterations.
- Baseline caveats and shutdown-path behavior have been classified.

Not yet completed:

- Full paper-to-code mapping.
- Full scientific reproduction of the paper claims.
- Go2 integration.
- Dataset adaptation.
- World-model internals analysis.
- Research contribution.

## Validated baseline

The current validated baseline is:

- Task: `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0`
- Launcher: `upstream/IsaacLab/isaaclab.sh`
- Environment: `env_isaaclab_src`

A bounded smoke run is considered successful when environment setup completes and the log line `Learning iteration 0/300` appears. A timeout-based exit is currently accepted for smoke validation, because external termination is known to produce non-graceful shutdown behavior.

The full procedure, command, and known caveats are documented in `baseline.md`.

## Next phase

The next technical phase is paper-to-code mapping. See `roadmap.md` for the full phase plan.

## Workspace layout

- `docs/` — supervisor-facing and project-facing documentation
- `manifests/` — frozen environment and state records
- `notes/` — working technical notes
- `upstream/` — local upstream dependencies (not tracked in this repo)
- `logs/` — local execution and debug logs (not tracked in this repo)
- `scripts/` — local setup and debug scripts (not tracked in this repo)
