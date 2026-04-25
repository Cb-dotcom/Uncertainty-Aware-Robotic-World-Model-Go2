# Portability Notes

## Purpose
These notes summarize how portable the current validated setup is to another machine, such as a lab workstation.

## Current validated machine
Validated locally on:
- Ubuntu 22.04
- ROS 2 Humble present on system
- NVIDIA RTX 4060 Laptop GPU
- Python 3.11 environment
- source-based Isaac Lab workflow
- upstream `robotic_world_model` launched via `IsaacLab/isaaclab.sh`

## Important portability assumptions
A target machine should ideally provide:
- Ubuntu Linux,
- NVIDIA GPU,
- compatible NVIDIA driver/runtime for Isaac Sim and Warp,
- enough RAM and disk for Isaac-related dependencies,
- permission to create local environments and install dependencies.

A lab-specific branch should only be created if the lab machine forces genuine machine-specific divergence.