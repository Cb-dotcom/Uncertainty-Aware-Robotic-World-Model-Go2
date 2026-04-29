# Checkpoint and Finetune Status

The finetuning stage of the RWM pipeline has not been executed. Status: **Not verified**, blocked by missing artifacts. The code path is **Structurally understood**, but execution requires a pretrained dynamics checkpoint and a policy run to resume from, and neither exists locally.

## The finetuning stage

The finetuning task is `Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0`. It is registered with a custom model-based environment, `ANYmalDManagerBasedMBRLEnv`, which uses the learned dynamics model as an imagination simulator. The runner is `MBPOOnPolicyRunner`, the same runner used for pretraining, but with imagination enabled.

The finetuning runner config inherits from pretraining and adds three things:

```text
resume                              = True
load_system_dynamics                = True
system_dynamics_warmup_iterations   = 500
run_name                            = "finetune"
```

It also enables imagination at lab-scale settings:

```text
num_imagination_envs                = 8192
num_imagination_steps_per_env       = 24
max_imagination_episode_length      = 256
```

These imagination settings are not feasible on the local laptop, which is one reason finetuning is a lab-workstation task. See [Hardware and System Specs](../setup/hardware-and-system-specs.md).

## Missing artifacts

Two artifacts are required for the upstream finetuning configuration to load. Neither exists locally.

The system-dynamics checkpoint is hardcoded in the upstream config to:

```text
logs/rsl_rl/anymal_d_flat/2025-11-04_14-31-20_pretrain_rnn/model_5000.pt
```

This file is not present in the local logs directory. The dated path suggests this is a vestige of the upstream authors' own training run rather than a downloadable artifact; no upstream release containing it has been identified. If a future check finds that an upstream artifact does exist, the path should be retargeted accordingly.

The policy resume reference is also pinned to a specific timestamp:

```text
load_run = "2025-11-04_09-59-00"
```

The runner will attempt to resume policy state from this run directory. The directory does not exist locally either.

## Path forward

The recommended path has two stages.

The first stage is to generate the prerequisite checkpoints. A full-scale RWM pretraining run on the lab workstation produces a dynamics checkpoint at the path determined by the run timestamp. The same run also writes the policy state, which becomes the resume target. Until the lab pretraining run has been executed, finetuning cannot proceed.

The second stage is to write a project-specific finetune configuration that points to the locally generated checkpoint paths rather than the hardcoded upstream values. This configuration lives in the project fork of `robotic_world_model` (under `agents/rsl_rl_ppo_cfg.py` for the ANYmal-D family). It can also be the place where lab-appropriate imagination settings are tuned, if the default 8192 environments turn out to be too aggressive for the available batch size.

The unblock dependency is therefore:

1. Lab-scale RWM pretraining produces a usable dynamics checkpoint.
2. Project-specific finetune config points to that checkpoint and any associated policy resume directory.
3. Finetuning runs.

The full-scale RWM pretraining is itself blocked on lab access; see [Hardware and System Specs](../setup/hardware-and-system-specs.md) for the workload split rationale and [World-Model Pretraining Check](world-model-pretraining-check.md) for what was already validated locally at reduced scale.