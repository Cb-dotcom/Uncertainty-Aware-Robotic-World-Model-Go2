# Submodules and Forks

This page documents the operational handling of the upstream codebases. It covers which repositories are tracked as submodules, which are forked versus pinned upstream, and the workflow for editing a fork.

The architectural details of the project's modifications to upstream code (what the change is and why) live on the [Implementation Analysis](../world-model/implementation-analysis.md) page; this page records the operational aspect (where the change lives, how the fork is maintained).

## Strategy

| Repository | Tracking | Reason |
|---|---|---|
| `upstream/IsaacLab` | Pinned upstream | Infrastructure dependency. Avoid modifying. |
| `upstream/robotic_world_model` | Forked submodule | Expected to receive Go2-related and project-specific changes. |
| `upstream/rsl_rl_rwm` | Forked submodule | Already carries one project-applied change; expected to receive more. |

Isaac Lab is large, actively maintained upstream, and likely available pre-installed on lab machines (e.g., inside the lab Docker image). The project policy is to avoid touching it unless a specific need forces a change. If that happens, Isaac Lab can be promoted to a fork at that time.

The two RSL-RL world-model repositories are research code with smaller communities and are expected to require project-specific modifications during the work. They are therefore tracked as forks from the outset.

## Cloning

For a fresh clone:

```bash
git clone --recurse-submodules https://github.com/Cb-dotcom/Uncertainty-Aware-Robotic-World-Model-Go2.git
```

For an existing clone where submodules were not initialized:

```bash
git submodule update --init --recursive
```

## Editing a forked submodule

The standard workflow when modifying upstream code:

1. Make the change inside the submodule directory.
2. Commit and push inside the submodule. The fork on GitHub now carries the new commit.
3. Return to the top-level repository.
4. Commit the updated submodule pointer (the SHA in the parent repo's index advances).
5. Push the top-level repository.

Concrete example for `rsl_rl_rwm`:

```bash
cd upstream/rsl_rl_rwm
git add rsl_rl/runners/mbpo_on_policy_runner.py
git commit -m "Guard imagination_infos clear when imagination is disabled"
git push

cd ../..
git add upstream/rsl_rl_rwm
git commit -m "Update rsl_rl_rwm submodule pointer"
git push
```

This two-step commit pattern is what keeps the top-level repository's submodule pointers in sync with the fork's commits. Pulling someone else's update (or your own from another machine) requires re-running `git submodule update --init --recursive`.

## Pulling upstream updates

To bring the latest commits from the original upstream repositories into a fork (rather than working from the fork's branches), the standard pattern is to add an `upstream` remote in the fork and merge or rebase from it:

```bash
cd upstream/rsl_rl_rwm
git remote add upstream <original-rsl-rl-rwm-url>
git fetch upstream
git merge upstream/main   # or rebase, depending on workflow
```

This is operational housekeeping rather than a frequent need; the project pins specific commits and updates upstream deliberately.

## Project-applied changes

### `rsl_rl_rwm`: pretraining fix

The model-based runner contained a bug that prevented Pretrain-v0 from completing. The runner's cleanup logic called `self.imagination_infos.clear()` unconditionally, but the attribute is only created when imagination is enabled. With imagination disabled (as in Pretrain-v0), the attribute does not exist and the call raised `AttributeError`.

The fix guards the call with a `hasattr` check:

```python
if hasattr(self, "imagination_infos"):
    self.imagination_infos.clear()
```

The architectural detail (why the bug exists, why the fix is structurally sound, why it does not affect finetuning) is on [Implementation Analysis §11](../world-model/implementation-analysis.md#11-the-pretraining-fix).

The change is committed in the project fork of `rsl_rl_rwm` to `rsl_rl/runners/mbpo_on_policy_runner.py`. It is the only project-applied change to upstream code at this point.

### `robotic_world_model`: no changes yet

The fork exists but currently tracks upstream without modifications. Project-specific configurations (for example, the planned project-specific Finetune-v0 config that points to locally generated checkpoints) will be added under this fork.