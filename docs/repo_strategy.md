# Repository Strategy

## Purpose of this repository
This repository is the top-level research workspace for the project.

Its role is to track:
- project documentation,
- validated manifests,
- research notes,
- progress summaries,
- original project-specific modifications.

## Upstream dependency policy
The following are kept as local upstream clones under `upstream/`:
- `robotic_world_model`
- `IsaacLab`
- `rsl_rl_rwm`

They will be included as submodules once relevant modifications will be needed. 

For now, `main` should remain the clean validated reference branch.