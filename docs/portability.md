# Portability

This document summarizes how portable the current validated setup is to a different machine, such as a lab workstation.

## Validated configuration

The current setup has been validated locally on:

- Ubuntu 22.04
- ROS 2 Humble installed system-wide (not required by the pipeline itself, but present)
- NVIDIA RTX 4060 Laptop GPU
- Python 3.11 environment
- Source-based Isaac Lab workflow
- Upstream `robotic_world_model` launched via `IsaacLab/isaaclab.sh`

## Target machine assumptions

A target machine should provide:

- Ubuntu Linux (other distributions are not validated).
- An NVIDIA GPU.
- A compatible NVIDIA driver and runtime for Isaac Sim and Warp.
- Sufficient RAM and disk for Isaac-related dependencies.
- Permission to create local Conda environments and install dependencies.

## Notes

The pipeline is GPU-bound and Isaac-Sim-bound. Headless execution works on laptop GPUs, but training-scale runs (large `num_envs`, full imagination horizons) are expected to need a workstation-class GPU.

For repository policy on machine-specific branching, see `repo-strategy.md`.
