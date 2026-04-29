# Hardware and System Specs

This page is the authoritative record of the hardware available to the project and which workloads each machine is sized for. Other pages reference this one rather than restating roles.

## Specs

The project uses two machines: a local laptop for setup, code reading, reduced-scale validation, and documentation, and a lab workstation for full-scale training and benchmark reproduction.

|  | Local laptop | Lab workstation |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | Ubuntu 24.04.02 LTS |
| CPU | 13th Gen Intel Core i7 (16 threads) | AMD Ryzen Threadripper PRO 7975WX (workstation-class) |
| RAM | 16 GB | 128 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | NVIDIA RTX A6000 |
| VRAM | 8 GB | 48 GB |
| Display | Hybrid Intel iGPU + discrete NVIDIA | Headless |

A pre-built Isaac Sim Docker environment is available on the lab workstation. The migration procedure and the version-compatibility check are documented in [Lab Workstation Notes](lab-workstation-notes.md).

## Workload requirements

Which workloads each machine can support, in practice.

| Workload | Local laptop | Lab workstation |
|---|---|---|
| Environment setup and validation | yes | yes |
| Code reading and tracing | yes | no |
| Documentation work | yes | no |
| Figures, videos, GUI inspection | yes | no |
| Baseline execution check | yes | yes |
| Reduced-scale RWM pretraining | yes | yes |
| Default RWM pretraining | no | yes |
| Imagination-based finetuning | no | yes |
| RWM-U with full ensemble | no | yes |
| Long training campaigns | no | yes |
| Benchmark reproduction | no | yes |

The local laptop's ceiling is set by VRAM. The default RWM pretraining configuration computes system-dynamics losses over a GRU-based dynamics model with batches that exceed 8 GB. Reduced-scale pretraining (small batch, small replay buffer, short forecast horizon) fits within the local budget and is sufficient for verifying that the pipeline runs end-to-end.

The lab workstation has 6× the VRAM and 8× the system RAM of the local laptop. Default-configuration training is expected to fit, including the imagination workload at the configured 8192 environments. The lab workstation is headless and is not used for code reading, documentation, or GUI-bound work; those remain local-only.

## Storage

Lab-workstation storage is sized to accommodate the codebase, submodules, the Docker environment, training logs, checkpoints, and experiment outputs over multiple training campaigns. A reasonable initial allocation is approximately 300 GB.

If long training campaigns are run with frequent checkpointing, or if multiple experiment configurations are kept in parallel, the storage requirement grows past this baseline.