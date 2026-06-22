# Phase 4B: Go2 Inventory and RWM Config Scaffolding

Status: complete.

Goal: produce a working Go2 task family in the RWM manager-based pipeline,
matching the structure of the ANYmal-D RWM tasks. Prove at minimum scale
that Init-v0, Pretrain-v0, and Baseline-v0 boot, train one or more
iterations, and produce the expected observation/reward/loss structure.

Phase 4B does not include serious training. That is Phase 4C.

## Summary

| Step | Status | Notes |
|---|---|---|
| Go2 inventory (joint order, default pose, body names) | PASS | scripts/phase4b/inspect_go2_articulation.py |
| Go2 system_state shape verification | PASS | 45-dim, matches ANYmal-D |
| Go2 RWM config scaffolds (Init/Pretrain/Baseline) | PASS | scripts/phase4b/go2_rwm_config/ |
| Install script (overlay into submodule) | PASS | scripts/phase4b/install_go2_config.py |
| Init-v0 task registers and boots | PASS | 5 iters, 64 envs |
| Pretrain-v0 task registers and boots | PASS | 20 iters, 256 envs, world-model losses decrease |
| Baseline-v0 task registers and boots | PASS | 5 iters, 256 envs, PPO params match RWM paper |

## Environment

Same as Phase 4A. Workstation `k8s-worker-node-2`, container `rwmu-cogar-cb`,
RTX 6000 Ada (49 GB VRAM), Isaac Sim 5.1, Isaac Lab v2.3.0 / commit `3c6e67b`.

## Repo layout

The upstream `robotic_world_model` submodule is gitignored at the
parent-repo level. Therefore the Go2 RWM config cannot live directly
inside the submodule and stay tracked. The source of truth lives at:

```text
scripts/phase4b/
├── inspect_go2_articulation.py # diagnostic
├── install_go2_config.py # overlay into submodule
└── go2_rwm_config/
 ├── __init__.py # registers Init/Pretrain/Baseline
 ├── flat_env_cfg.py # env classes
 └── agents/
 ├── __init__.py
 └── rsl_rl_ppo_cfg.py # PPO and MBPO runner cfgs
```

`install_go2_config.py` copies the `go2_rwm_config/` tree into the
runtime location at:

```text
upstream/robotic_world_model/source/mbrl/mbrl/tasks/
manager_based/locomotion/velocity/config/go2/
```

To use on a new machine: clone, run sanity check, run
`/isaac-sim/python.sh scripts/phase4b/install_go2_config.py`, then run
training. To modify the config: edit in `scripts/phase4b/go2_rwm_config/`,
re-run install with `--force`.

## Step 1: Go2 articulation inventory

Inspector spawned a single Go2 via `UNITREE_GO2_CFG` in a minimal Isaac Lab
scene, dumped joint and body data.

Key findings (full data in `docs/go2-transfer/go2-inventory.md`):

- 12 joints in order: FL/FR/RL/RR hip, then FL/FR/RL/RR thigh, then FL/FR/RL/RR calf
- 19 bodies: 1 base, 4 hips, 4 thighs, 4 calves, 4 feet, 2 head bodies
 (`Head_upper`, `Head_lower`), heads are passive (no joint)
- system_state shape: 45 (3 lin_vel + 3 ang_vel + 3 gravity + 12 q + 12 q_dot + 12 tau)
- Matches ANYmal-D system_state dimensions exactly, world-model
 architecture can be reused without reshape

Front and rear thigh joints have different position limits
(`[-1.571, +3.491]` vs `[-0.524, +4.538]`). Not blocking; relevant for
reset randomization design.

## Step 2: RWM config scaffold

Three Gym task IDs registered:

```text
Template-Isaac-Velocity-Flat-Unitree-Go2-Init-v0
Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0
Template-Isaac-Velocity-Flat-Unitree-Go2-Baseline-v0
```

Env class hierarchy (mirrors RWM ANYmal-D):

```text
UnitreeGo2RoughEnvCfg (upstream Isaac Lab)
 └── UnitreeGo2FlatEnvCfg (RWM ANYmal-D-style reward weights, flat terrain)
 ├── UnitreeGo2FlatEnvCfg_INIT (reverts to rough terrain, lighter penalties)
 ├── UnitreeGo2FlatEnvCfg_PRETRAIN (adds system_* observation groups)
 └── UnitreeGo2FlatEnvCfg_BASELINE (no system_* groups; default observations only)
```

Runner class hierarchy:

```text
UnitreeGo2FlatPPORunnerCfg (upstream Isaac Lab)
 ├── UnitreeGo2FlatPPOPretrainRunnerCfg (class_name = MBPOOnPolicyRunner, world-model cfg)
 └── UnitreeGo2FlatPPOBaselineRunnerCfg (class_name = OnPolicyRunner, RWM PPO hyperparameters)
```

Init-v0 reuses the upstream `UnitreeGo2FlatPPORunnerCfg` (upstream Go2 PPO
params, not RWM paper params). Acceptable because Init-v0 is a warm-up
task; its PPO params are not part of any reported comparison.

### Observation groups

Init-v0 and Baseline-v0 use only the default `policy` group (48-dim).
Pretrain-v0 adds four world-model groups:

| Group | Shape | Contents |
|---|---|---|
| policy | 48 | base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions |
| system_state | 45 | base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, joint_torque |
| system_action | 12 | last_action |
| system_contact | 8 | 4 thigh contacts + 4 foot contacts (regex `.*_thigh`, `.*_foot`) |
| system_termination | 1 | base contact (regex `base`) |

Contact body name regex differs from ANYmal-D: Go2 uses lowercase
(`.*_thigh`, `.*_foot`) where ANYmal-D used uppercase (`.*THIGH`, `.*FOOT`).

### RWM PPO hyperparameters (Baseline-v0 runner)

Applied via `__post_init__` overrides on `UnitreeGo2FlatPPOBaselineRunnerCfg`,
verified at runtime against `params/agent.yaml`:

```yaml
class_name: OnPolicyRunner
num_steps_per_env: 24
max_iterations: 2000
init_noise_std: 1.0
actor_hidden_dims: [128, 128, 128]
critic_hidden_dims: [128, 128, 128]
activation: elu
algorithm.class_name: PPO
algorithm.num_learning_epochs: 5
algorithm.num_mini_batches: 4
algorithm.learning_rate: 0.001
algorithm.entropy_coef: 0.005
algorithm.gamma: 0.99
algorithm.lam: 0.95
algorithm.clip_param: 0.2
algorithm.desired_kl: 0.01
algorithm.schedule: adaptive
algorithm.value_loss_coef: 1.0
algorithm.max_grad_norm: 1.0
```

These match RWM ANYmal-D paper values exactly.

## Step 3: Boot tests

### Init-v0 (5 iters, 64 envs)

- 5 iterations complete in ~3 seconds
- Action manager: 12-dim joint_pos
- Observation policy: 48
- 11 reward terms loaded (10 base + stand_still)
- Episode length growing 10 -> 30; base_contact terminations rare

### Pretrain-v0 (20 iters, 256 envs)

- 20 iterations complete in ~30 seconds
- All 5 observation groups present with correct shapes
- 13 `System Dynamics/*` tensorboard tags written; zero `Imagination/*` tags
 (correct for Pretrain, not Finetune)
- Dynamics losses cleanly decreasing:

 | Loss | step 0 | step 19 | reduction |
 |---|---|---|---|
 | state_loss | 1099.79 | 187.38 | 5.9x |
 | contact_loss | 0.343 | 0.154 | 2.2x |
 | termination_loss | 0.1044 | 0.0003 | 347x |

- Checkpoints saved at iter 0 and 20, ~18 MB each, combined-format
 (PPO + dynamics state) matching ANYmal-D format

### Baseline-v0 (5 iters, 256 envs)

- 5 iterations complete in ~3 seconds
- Single observation group (`policy`, 48-dim), no system_* groups
- `class_name: OnPolicyRunner` (standard PPO, not MBPO) confirmed
- All RWM PPO hyperparameters verified in saved `agent.yaml`
- ~21k steps/s at 256 envs (faster than RWM tasks due to no world-model loop)

## Open items (carried into Phase 4C and beyond)

| Item | Status | Why deferred |
|---|---|---|
| Recompute Go2 state/action normalizer from rollouts | Pending | Need a real policy distribution first; placeholder zeros/ones works for boot |
| Tune Go2 reward weights | Pending | Start with copied RWM ANYmal-D weights; tune after Baseline-v0 result is in |
| Strip vs keep height_scanner | Stripped (matches RWM ANYmal-D flat) | Default for flat-terrain RWM transfer |
| Scaffold Go2 Finetune-v0 | Pending | Need Pretrain-v0 checkpoint from real training first |
| Scaffold Go2 Visualize-v0 | Pending | Not needed until policies exist worth visualizing |

## What this validation establishes / does not

Establishes:

- Go2 articulation is fully understood (joint order, defaults, bodies, system_state shape)
- Go2 RWM env config can be inserted as an overlay despite the gitignored upstream submodule
- Both training entry points (Init, Pretrain, Baseline) register, build the env, and complete iterations
- The Pretrain world-model loop trains on Go2 data (losses decrease over 20 iters)
- The Baseline runner uses RWM paper PPO hyperparameters, not upstream Go2 defaults

Does not establish:

- That any of these policies actually learn to walk (that is Phase 4C)
- That the world model converges over many iterations (Phase 4C Pretrain)
- That the normalizers are well-fitted (Phase 4D / pre-Pretrain refinement)
- That hardware deployment is feasible (Phase 4H)

## Next phase

Phase 4C, real training. First step: Baseline-v0 at default 4096 envs,
2000 iterations, RWM paper PPO hyperparameters. Establishes the
comparison baseline against which RWM and RWM-U Go2 results will be
reported.