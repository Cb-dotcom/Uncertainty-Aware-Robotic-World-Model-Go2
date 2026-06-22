# Submodules and Forks

This page explains how the project handles the three upstream codebases.

It answers operational questions:

- Which upstream repositories are pinned or forked?
- Where should project-specific edits live?
- Which runtime patches are currently required?
- What can be reset safely, and what must be preserved?

Architectural explanations of *why* a change exists belong in [Implementation Analysis](../world-model/implementation-analysis.md). This page only explains where changes live and how to maintain them.

## Rules of thumb

Keep these rules in mind before editing anything under `upstream/`.

| Rule | Meaning |
|---|---|
| Do not edit `upstream/` first if there is a project-owned source file. | For Go2 RWM configs, edit `scripts/phase4b/go2_rwm_config/`, then install. |
| Treat installed Go2 configs as build artifacts. | The installed copy under `upstream/robotic_world_model/.../config/go2/` is generated from `scripts/phase4b/go2_rwm_config/`. |
| Runtime patches can disappear after a submodule reset. | Anything not committed to a fork or tracked as a patch file must be re-applied and verified. |
| Use `/isaac-sim/python.sh`, not `python3`, inside the container. | The container may not have `python3` or `python` on `PATH`. |
| Do not judge submodule cleanliness from the top-level repo alone. | Check inside each submodule when debugging. |

## Submodule strategy

| Repository | Tracking | Role | Policy |
|---|---|---|---|
| `upstream/IsaacLab` | Pinned upstream | Simulator and Isaac Lab infrastructure | Avoid project commits here. Temporary local render/control edits may exist but are not deliverables. |
| `upstream/robotic_world_model` | Forked submodule | RWM task/runtime layer | Receives project runtime patches and installed Go2 configs. Go2 config source still lives in the main repo. |
| `upstream/rsl_rl_rwm` | Forked submodule | RL, MBPO, world-model, and uncertainty backend | Receives project changes to runners, algorithms, or world-model logic. |

Current committed upstream-code change:

```text
rsl_rl_rwm:
 rsl_rl/runners/mbpo_on_policy_runner.py
 pretraining cleanup guard
```

Current required runtime patches:

```text
robotic_world_model:
 manager_based_mbrl_env.py imagination-reward skip guard
 visualize.py render/headless guards

IsaacLab:
 play.py render pins
 stock-Go2 reward-control edits
```

## Cloning

Fresh clone:

```bash
git clone --recurse-submodules https://github.com/Cb-dotcom/Uncertainty-Aware-Robotic-World-Model-Go2.git
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

To check submodule status:

```bash
git submodule status --recursive
```

To inspect changes inside a submodule:

```bash
cd upstream/robotic_world_model
git status --short
```

## Go2 config install mechanism

The Go2 RWM config source of truth is in the main repository:

```text
scripts/phase4b/go2_rwm_config/
```

Install it into the runtime tree with:

```bash
/isaac-sim/python.sh scripts/phase4b/install_go2_config.py --force
```

The installer writes to:

```text
upstream/robotic_world_model/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/go2/
```

That installed location contains the runtime Go2 task family:

```text
Init
Pretrain
Baseline
Finetune
Visualize
```

It also contains the Go2 manager-based MBRL environment and reward configuration, including:

```text
feet_slide = -0.25
```

Important rule:

```text
Edit:
 scripts/phase4b/go2_rwm_config/

Install to:
 upstream/robotic_world_model/.../config/go2/

Do not hand-edit the installed copy except for temporary debugging.
```

Container note:

```text
Use:
 /isaac-sim/python.sh scripts/phase4b/install_go2_config.py --force

Do not use:
 python3 scripts/phase4b/install_go2_config.py --force
```

If the installer appears to fail only because `python3` is missing, the fix is to run it through `/isaac-sim/python.sh`, not to hand-copy files.

## Project-applied changes

### 1. `rsl_rl_rwm`: pretraining cleanup guard

Status:

```text
Committed to project fork.
```

File:

```text
upstream/rsl_rl_rwm/rsl_rl/runners/mbpo_on_policy_runner.py
```

Problem:

`Pretrain-v0` runs with imagination disabled. The runner cleanup path previously called:

```python
self.imagination_infos.clear()
```

unconditionally. But `self.imagination_infos` only exists when imagination is enabled, so pretraining could fail with:

```text
AttributeError
```

Fix:

```python
if hasattr(self, "imagination_infos"):
 self.imagination_infos.clear()
```

This is the only project change currently committed directly to an upstream-code fork.

More detail: [Implementation Analysis §11](../world-model/implementation-analysis.md#11-the-pretraining-fix).

### 2. `robotic_world_model`: imagination-reward skip guard

Status:

```text
Runtime patch.
Required for every Go2 finetune.
Should be kept as a tracked patch file.
```

File:

```text
upstream/robotic_world_model/source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py
```

Problem:

Go2 real rollouts use:

```text
feet_slide
```

to discourage skating. But `feet_slide` depends on per-foot world-frame velocity, which the current world model does not predict. Therefore it cannot be computed correctly in imagined rollouts.

Before the patch, the imagination reward code assumed every real reward term existed in the imagination reward dictionary. Go2 finetune crashed at the first imagination step with:

```text
KeyError: 'feet_slide'
```

Patch:

```python
if term not in self.imagination_reward_per_step:
 continue
term_value = self.imagination_reward_per_step[term]
```

Meaning:

```text
Real rollout:
 feet_slide active

Imagined rollout:
 unsupported reward terms skipped
```

This keeps Go2 training running while making the reward mismatch explicit.

Verification command:

```bash
docker exec -i rwmu-cogar-cb bash -lc '
grep -n "if term not in self.imagination_reward_per_step" \
 /workspace/Uncertainty-Aware-Robotic-World-Model-Go2/upstream/robotic_world_model/source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py \
 || echo "MISSING, reapply skip patch before finetune"
'
```

Clean future version:

```text
Split rewards explicitly:
 real_env_rewards
 imagination_rewards
```

rather than relying on a missing-key skip.

### 3. `robotic_world_model`: render guards

Status:

```text
Runtime patch.
Tracked as a patch file.
```

File:

```text
upstream/robotic_world_model/scripts/reinforcement_learning/rsl_rl/visualize.py
```

Tracked patch:

```text
scripts/phase4b/render/visualize_render.patch
```

Purpose:

- allow rendering when `init_imagination_history` is not present,
- avoid a headless lighting import crash,
- make RWM/MBPO checkpoint visualization usable in the container.

Patch style note:

```text
Apply with:
 patch -p1 --forward

Do not use:
 git apply
```

because the patch uses bare `@@` headers.

### 4. `IsaacLab`: render pins

Status:

```text
Local working-tree patch.
Not a project deliverable.
```

File:

```text
upstream/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py
```

Purpose:

- render one robot,
- force a simple forward command,
- use a follow camera.

These pins are for stock/plain-PPO visualization. RWM/MBPO checkpoints use `visualize.py`, not `play.py`, because `play.py` does not support `MBPOOnPolicyRunner`.

### 5. `IsaacLab`: stock-Go2 reward control edits

Status:

```text
Local working-tree patch.
Control experiment only.
Not the project Go2 reward source.
```

Files:

```text
upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py
upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py
```

Purpose:

The stock Go2 task was used as a control to diagnose the skating/scooting failure mode.

Validated control setting:

```text
feet_slide = -0.25
undesired_contacts = None
feet_air_time = 0.25
```

Important:

```text
The project Go2 reward lives in:
 scripts/phase4b/go2_rwm_config/

not in the stock IsaacLab Go2 files.
```

## Runtime patch summary

| Patch | Location | Status | Required for |
|---|---|---|---|
| Pretraining cleanup guard | `rsl_rl_rwm/rsl_rl/runners/mbpo_on_policy_runner.py` | committed to fork | RWM pretrain robustness |
| Imagination-reward skip guard | `robotic_world_model/.../manager_based_mbrl_env.py` | runtime patch | every Go2 finetune |
| `visualize.py` render guards | `robotic_world_model/.../visualize.py` | runtime patch, tracked patch file | RWM/MBPO rendering |
| `play.py` render pins | `IsaacLab/.../play.py` | local working-tree patch | stock/plain-PPO renders |
| Stock-Go2 reward-control edits | `IsaacLab/.../config/go2/` | local working-tree patch | control experiments only |

## Before launching Go2 finetune

Check these before every Go2 finetune run.

### 1. Go2 config installed

```bash
/isaac-sim/python.sh scripts/phase4b/install_go2_config.py --force
```

### 2. Correct finetune checkpoint path

Check:

```text
scripts/phase4b/go2_rwm_config/agents/rsl_rl_ppo_cfg.py
```

Important fields:

```python
load_run = "..."
system_dynamics_load_path = "logs/rsl_rl/unitree_go2_flat/.../model_2000.pt"
run_name = "..."
```

For the ensemble-5 RWM-U comparison, both finetune arms must point to the same pretrain checkpoint:

```python
load_run = "2026-06-12_13-39-03_pretrain_ens5"
system_dynamics_load_path = "logs/rsl_rl/unitree_go2_flat/2026-06-12_13-39-03_pretrain_ens5/model_2000.pt"
```

### 3. Skip guard present

```bash
docker exec -i rwmu-cogar-cb bash -lc '
grep -n "if term not in self.imagination_reward_per_step" \
 /workspace/Uncertainty-Aware-Robotic-World-Model-Go2/upstream/robotic_world_model/source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py \
 || echo "MISSING, reapply skip patch before finetune"
'
```

### 4. No accidental stale runtime copy

If behavior does not match the source config, reinstall:

```bash
/isaac-sim/python.sh scripts/phase4b/install_go2_config.py --force
```

Then confirm the installed file:

```bash
grep -nE "load_run|system_dynamics_load_path|run_name|ensemble_size|uncertainty_penalty_weight" \
 upstream/robotic_world_model/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py
```

## Current action item

The highest-priority patch to formalize is:

```text
robotic_world_model:
 source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py
 imagination-reward skip guard
```

Reason:

```text
Without it, Go2 finetune crashes on KeyError: 'feet_slide'.
```

This patch should be tracked under `scripts/phase4b/`, alongside `visualize_render.patch`, so RWM-U reproduction does not depend on an invisible runtime edit.