# Phase 4A: Lab Workstation Validation

Status: complete.

This document records the validation of the RWM / RWM-U pipelines on the
shared lab workstation (`k8s-worker-node-2`) inside the
`rwmu-cogar-cb` Docker container.

## Summary

| Step | Status | Notes |
|---|---|---|
| Lab environment sanity (9/9 checks) | PASS | After fixing PYTHONPATH namespace-package bug |
| Reduced ANYmal-D Init-v0 | PASS | 64 envs, 5 iterations |
| Reduced ANYmal-D Pretrain-v0 | PASS | 256 envs, 100 iterations, dynamics checkpoint produced |
| Reduced ANYmal-D Finetune-v0 | PASS | Local blocker resolved; MBPO imagination loop verified |
| RWM-U offline reproduction | PASS | 500 iterations, final mean reward 1.96 (local: 1.82) |
| Default-scale ANYmal-D Pretrain-v0 | PASS | 4096 envs, 20 iterations, no OOM |

No milestone failed. No OOM observed at any tested scale. The bounded
default-scale attempt fits comfortably on the lab GPU.

## Environment

- Host: `k8s-worker-node-2`
- OS (container): Ubuntu 22.04 Jammy
- GPU: NVIDIA RTX 6000 Ada Generation (49 GB VRAM)
- Driver: 570.211.01
- CUDA (container view): 12.8
- Isaac Sim: 5.1 (kit log path `/isaac-sim/kit/logs/Kit/Isaac-Sim/5.1/`)
- Isaac Lab: v2.3.0 / commit `3c6e67b`
- torch: 2.7.0+cu128
- Container image: `rwmu-cogar-cb:latest`
- Repo commit at validation: `49c2724` (head of `main`)

## Repo runtime environment

The session uses the editable `mbrl` install. The PYTHONPATH must **not**
include `$RWM/source`, because doing so promotes `source/mbrl/` to a
namespace package and shadows the PEP 660 editable finder.

```bash
unset PYTHONPATH
export PROJECT_ROOT="/workspace/Uncertainty-Aware-Robotic-World-Model-Go2"
export RWM="$PROJECT_ROOT/upstream/robotic_world_model"
export RSL="$PROJECT_ROOT/upstream/rsl_rl_rwm"
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$RSL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
```

See `docs/setup/lab-workstation-notes.md` for the rationale.

## Step 1: Lab environment sanity check

Result: 9 passed, 0 failed.

```
[1/9] nvidia-smi PASS
[2/9] torch sees GPU (47.4 GB) PASS
[3/9] isaacsim import PASS
[4/9] isaaclab import PASS
[5/9] omni.client import PASS
[6/9] project package imports (rsl_rl, mbrl) PASS
[7/9] data science extras PASS
[8/9] pretrained checkpoint and dataset present PASS
[9/9] RWM train.py --help PASS
```

## Step 2: Reduced ANYmal-D Init-v0

Command:

```bash
cd "$RWM"
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
 --task Template-Isaac-Velocity-Flat-Anymal-D-Init-v0 \
 --num_envs 64 \
 --max_iterations 5 \
 --headless --logger tensorboard
```

Result:
- 5 iterations complete in ~2 seconds wall clock
- ~3,750 steps/s
- mean reward -2.07 (meaningless at 5 iters; reported for trace continuity)
- Log dir: `logs/rsl_rl/anymal_d_flat/<timestamp>/`

## Step 3: Reduced ANYmal-D Pretrain-v0

Command:

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
 --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
 --num_envs 256 \
 --max_iterations 100 \
 --headless --logger tensorboard
```

Result:
- 100 iterations in 2:43 wall clock, ~3,700 steps/s
- PPO policy collapses by iter 100 (mean episode length 19,
 base_contact rate 1.0). Expected at this scale; not a Pretrain
 failure. The world model is the trained artifact here, not the policy.
- Dynamics losses cleanly decreasing:

 | Loss | step 0 | step 50 | step 99 |
 |---|---|---|---|
 | state_loss | 287.4 | 42.9 | 32.8 |
 | contact_loss | 0.48 | 0.07 | 0.04 |

- Checkpoints saved at iters 0, 50, 100. Each ~18 MB, combined-format
 matching the RWM-U LFS checkpoint structure:
 - `model_state_dict` (PPO actor-critic)
 - `system_dynamics_state_dict` (world model)
 - `optimizer_state_dict`
 - `system_dynamics_optimizer_state_dict`
 - `iter`, `infos`

- Checkpoint produced and used as input for Step 4:
 `logs/rsl_rl/anymal_d_flat/2026-05-28_19-31-54_pretrain/model_100.pt`

## Step 4: Reduced ANYmal-D Finetune-v0 (the local blocker)

Upstream `AnymalDFlatPPOFinetuneRunnerCfg` hardcodes paths to the
original author's machine. These were patched in-place for this run,
then the config was restored. See `scripts/phase4a/patch_finetune_config.py`.

Patched values:

- `resume = True` -> `False`
- `load_run = "2025-11-04_09-59-00"` -> `""`
- `system_dynamics_load_path = "logs/.../2025-11-04_14-31-20_pretrain_rnn/model_5000.pt"`
 -> path to the Step 3 `model_100.pt`
- `system_dynamics_warmup_iterations = 500` -> `50`

Command:

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
 --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
 --num_envs 256 \
 --max_iterations 100 \
 --headless --logger tensorboard
```

Result:

- 100 iterations in 2:42 wall clock
- MBPO imagination loop confirmed firing. Tensorboard contains the
 `Imagination/*` namespace which is absent in Init-v0 and Pretrain-v0
 runs. Tags include `Imagination/track_lin_vel_xy_exp`,
 `Imagination/uncertainty`, `Model Based/epistemic_uncertainty`,
 `Model Based/num_valid_imagination_envs`,
 `Perf/imagination collection time`,
 `Train/mean_reward_imagination`.
- Runtime config (`logs/.../params/agent.yaml`) confirms the patch
 landed: `resume: false`, `load_system_dynamics: true`,
 `system_dynamics_load_path: .../model_100.pt`,
 `system_dynamics_warmup_iterations: 50`.
- Policy did not collapse: mean episode length 285, base_contact rate
 0.98 (high but episodes are long, unlike Pretrain's 19-step collapse).

Note: `Model Based/epistemic_uncertainty` is being computed and logged
in base RWM with `ensemble_size = 1`. With a single ensemble member this
signal is effectively noise (no spread), and
`uncertainty_penalty_weight = 0.0` means it doesn't feed back into the
reward. This refines the earlier Phase 2 statement that uncertainty
hooks are completely inactive in base RWM. They are computed and
logged; they are just not used.

## Step 5: RWM-U offline reproduction

Command:

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/model_based/train.py
```

Result:

- 500 iterations in 1:19 wall clock
- ~1.26 M synthetic steps/s (pure tensor throughput, no Isaac Sim)
- Final mean reward 1.96 (local laptop: 1.82, within seed variance)
- Final mean episode length 256
- Output: `logs/offline/anymal_d_flat/2026-05-28_19-48-14_None/policy_*.pt`

This validates the offline pipeline on lab hardware. The shipped LFS
checkpoint `assets/models/pretrain_rnn_ens.pt` reproduces.

## Step 6: Default-scale Pretrain-v0 (bounded attempt)

Default `num_envs = 4096` (inherited from the upstream Isaac Lab
velocity base env at `velocity_env_cfg.py:294`). The RWM
`AnymalDFlatEnvCfg` does not override this value, so default Pretrain
runs at 4096 envs.

Command:

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
 --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
 --max_iterations 20 \
 --headless --logger tensorboard
```

(No `--num_envs` flag; inherits 4096.)

Result:

- 20 iterations in 35 seconds wall clock
- ~57,000 steps/s (15x the 256-env throughput)
- Iteration time ~1.72s, basically the same as 256-env Pretrain. The
 GPU is not saturated even at 4096 envs.
- No OOM. No NaN. Clean exit.
- Policy did not collapse: mean episode length grows 13 -> 402 over 20
 iterations; base_contact rate 0.09 at iter 19 (vs 1.0 in the
 256-env / 100-iter Pretrain run). Default-scale PPO is more stable
 due to richer parallel sampling.
- GPU memory after exit: 52 MB (no leak)

The OOM binary search from the Phase 4A plan was not needed; the
default scale fits well within the 49 GB GPU.

## Known issues observed during Phase 4A

1. **cuDNN RNN non-contiguous warning** (3 occurrences per training
 run): `RNN module weights are not part of single contiguous chunk of
 memory. This means they need to be compacted at every call, possibly
 greatly increasing memory usage. To compact weights again call
 flatten_parameters().` Triggered inside the system-dynamics GRU
 forward. Performance penalty per call. Not blocking. Fix would be
 a `flatten_parameters()` call in the world model's forward.

2. **`Visuals/Command/velocity_*` mismatched prototypes** warning from
 `omni.physx.fabric.plugin`. Cosmetic, headless mode.

3. **`obs_groups` warning** about missing `policy`/`critic` keys.
 Upstream `rsl_rl_rwm` deprecation warning. Current fallback behavior
 is correct.

4. **VLLM idle process holding GPU.** Before Phase 4A could begin, an
 unrelated VLLM EngineCore (PID 1723768) on the host was holding
 44 GB VRAM at idle (P8, 0% util, 24W). Process was visible only from
 the host (`nvidia-smi` outside the container), not from inside.
 Released before validation runs. Risk for shared workstations.

## What this validation does and does not establish

Validated:

- The Docker container reproduces the local working environment
- Both training entry points (`rsl_rl/train.py` and
 `model_based/train.py`) function on lab hardware
- The full base RWM pipeline runs end-to-end at reduced scale
- The Finetune-v0 local blocker is resolved given the patch
- Default-scale ANYmal-D Pretrain fits on the lab GPU and runs at
 ~57k steps/s
- The RWM-U offline pipeline reproduces the previously validated reward

Not validated:

- Paper-scale reproduction. No multi-thousand-iteration run was
 attempted. Phase 4A is a pipeline check, not a metric reproduction.
- Long-horizon learning curves. Pretrain ran 100 iters, Finetune 100
 iters, default Pretrain 20 iters.
- Uncertainty-error coupling at lab scale. The local Figure-3-style
 validation stands; lab repeat was not in Phase 4A scope.

## Next phase

Phase 4B - Go2 inventory and environment scaffolding. See
`docs/go2-transfer/go2-inventory.md`.