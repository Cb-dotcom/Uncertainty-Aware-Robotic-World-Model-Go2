# Go2 Integration of an Uncertainty-Aware Robotic World Model

Research workspace for reproducing and extending the ETH Zurich robotic world model stack toward a future Unitree Go2 integration.

## Documentation

The full project documentation lives in `docs/`. If you have MkDocs installed, run `mkdocs serve` from the repo root to render it locally. Key entry points:

- `docs/index.md` — project scope, current status, and validated baseline.
- `docs/baseline.md` — the canonical smoke command and what counts as a successful run.
- `docs/repo-map.md` — where each part of the pipeline lives across the three upstream repos.
- `docs/roadmap.md` — phase plan (P0 to P4).
- `docs/paper-a-*.md` — paper-to-code mapping for *Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics*.

## Workspace layout

- `docs/` — project documentation
- `manifests/` — frozen environment and state records
- `notes/` — working technical notes
- `upstream/` — local upstream dependencies (not tracked in this repo)
- `logs/` — local execution and debug logs (not tracked in this repo)
- `scripts/` — local setup and debug scripts (not tracked in this repo)
