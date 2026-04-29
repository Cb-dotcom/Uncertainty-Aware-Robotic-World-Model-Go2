# Local Environment

This page describes the software environment used for local validation work. The exact package versions, upstream commit hashes, and CUDA state are recorded per-validation in `manifests/`; this page does not duplicate those values.

## Environment overview

The local environment is a conda environment named `env_isaaclab_src` running Python 3.11 on Ubuntu 22.04 with an NVIDIA RTX 4060 Laptop GPU. Isaac Lab is installed from source rather than from a binary distribution, which is why the environment name carries the `_src` suffix.

The active conda environment is activated with:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_isaaclab_src
export OMNI_KIT_ACCEPT_EULA=YES
```

The `OMNI_KIT_ACCEPT_EULA` export is required before launching Isaac Sim. It is not optional, and it is not persisted across shell sessions; it must be set before each set of runs (or sourced from a setup script).

The local hardware specifications are documented separately under [Hardware and System Specs](hardware-and-system-specs.md).

## Canonical state

Each validation run produces a manifest file under `manifests/` capturing the exact state at the moment of validation: conda env, Python interpreter path, upstream commit hashes for the three submodules, the working command, package versions for the scientific stack (`mbrl`, `rsl-rl-lib`, `isaaclab`, `isaaclab_rl`, `isaaclab_tasks`, `isaacsim`, `torch`), CUDA availability, and known caveats observed during the run.

Manifests are the source of truth for "what was running when X was validated." This page describes what the environment *is*; manifests describe what the environment *was* at a specific successful validation. When in doubt, trust the manifest.

The structure of the manifest format is documented in [Repository Structure](../project/repository-structure.md#manifests).

## Expected startup warnings

Isaac Sim and Warp emit several startup warnings on this machine that are non-blocking. The two with identifiable causes:

- **Warp CUDA `cuDeviceGetUuid`**: a driver compatibility warning. Training proceeds normally; Isaac Lab uses `cuda:0`.
- **Intel iGPU skipped**: expected on a laptop with hybrid graphics. Isaac Sim selects the discrete NVIDIA GPU and ignores the Intel one.

Other warnings appear during Isaac Sim startup (deprecated `omni.isaac.dynamic_control`, FabricManager mismatched-prototype messages, modules-loaded-before-SimulationApp messages). These are tracked but have not blocked any validated run.

If a warning category appears that is *not* in the above list and the run does not reach the expected validation marker, treat it as a real failure rather than noise.

## Local scope

The local environment is sized for setup validation, code reading, reduced-scale execution checks, and documentation work. Full-scale RWM pretraining and imagination-based finetuning are deferred to lab hardware. The reasoning is documented under [Hardware and System Specs](hardware-and-system-specs.md).