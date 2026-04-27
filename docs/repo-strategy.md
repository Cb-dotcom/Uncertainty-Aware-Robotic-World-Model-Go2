# Repository Strategy

This document defines the policy for what this repository tracks, how upstream dependencies are managed, and how branches are used.

## Scope of this repository

This repository is the top-level research workspace for the project. It tracks:

- Project documentation.
- Validated manifests.
- Research notes.
- Progress summaries.
- Original project-specific modifications.

It does not track upstream source trees, generated logs, or local scripts. Those live under `upstream/`, `logs/`, and `scripts/` and are gitignored.

## Upstream dependencies

The following upstream codebases are kept as local clones under `upstream/`:

- `robotic_world_model`
- `IsaacLab`
- `rsl_rl_rwm`

They will be promoted to Git submodules once project-specific modifications to upstream code become necessary. Until then, keeping them as plain local clones avoids the friction of submodule pinning while no upstream changes are being made.

## Branching policy

The `main` branch should remain the clean validated reference state. Feature work happens on topic branches and is merged back only after a smoke run still succeeds.

A lab- or machine-specific branch should only be created if a target machine forces genuine machine-specific divergence (for example, a different Isaac Lab installation path that cannot be parameterized via environment variables). Configuration that varies by machine and can be expressed in environment variables, paths, or local config files should not produce a separate branch.
