# Lab Workstation Notes

These notes document the lab Docker/workstation setup. 

## Current lab target

Observed lab host:

```text
OS: Ubuntu 24.04.2
GPU: NVIDIA RTX 6000 Ada Generation (49 GB VRAM)
Driver: 570.211.01
CUDA shown by nvidia-smi: 12.8
Container image: rwmu-cogar-cb:latest
Container name: rwmu-cogar-cb
Host workspace: ~/workspace/rwmu-cogar-cb
Container workspace: /workspace
Isaac Python: /isaac-sim/python.sh
Isaac Sim version (kit log path): /isaac-sim/kit/logs/Kit/Isaac-Sim/5.1/
```

## Entering the container

```bash
# On lab host
docker exec -it rwmu-cogar-cb bash

# If the container is stopped:
docker start rwmu-cogar-cb
docker exec -it rwmu-cogar-cb bash
```

## Runtime pattern

Inside the container, use Isaac Sim Python:

```bash
/isaac-sim/python.sh
```

Do not use system Python for RWM/Isaac Lab runs.

Use this environment pattern:

```bash
# Inside the container, before running training
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

This applies to **both** entry points:
- `scripts/reinforcement_learning/rsl_rl/train.py` (online base RWM)
- `scripts/reinforcement_learning/model_based/train.py` (offline RWM-U)

The offline pipeline does **not** require any extra `model_based` path on
PYTHONPATH. The standalone offline scripts resolve their own imports
relative to the cwd inside `$RWM`.

### Do not add `$RWM/source` to PYTHONPATH

The `mbrl` package is installed editably via PEP 660. The editable install
exposes `mbrl` as a regular package (resolved to `source/mbrl/mbrl/`) via
a custom MetaPathFinder registered through:

```text
$SITE_PACKAGES/__editable__.mbrl-0.1.0.pth
$SITE_PACKAGES/__editable___mbrl_0_1_0_finder.py
```

If `$RWM/source` is on PYTHONPATH, Python finds `source/mbrl/` first as
an implicit namespace package (because the directory has no
`__init__.py` of its own). The namespace package wins because explicit
PYTHONPATH entries are checked before custom MetaPathFinders. The result:

```text
mbrl.__file__       == None
mbrl.__path__       == ['.../source/mbrl']
import mbrl         OK
import mbrl.tasks   FAILS, or worse, imports incomplete state
```

Diagnostic: `import mbrl; print(mbrl.__file__)`. A correctly installed
`mbrl` prints the absolute path to `source/mbrl/mbrl/__init__.py`. A
namespace-package shadow prints `None`. The `lab/sanity-check.sh` test
[6/10] catches this.

## Container compatibility fixes

The Isaac Sim 5.0 base image required several fixes during lab bring-up.

### Torch vendored packaging symlink

Torch resolved from:

```text
/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch
```

The file:

```text
torch/_vendor/packaging/_structures.py
```

was a broken symlink to a missing `omni.isaac.core_archive` packaging file. The Dockerfile now replaces that single broken symlink with the valid file from:

```text
/isaac-sim/kit/python/lib/python3.11/site-packages/packaging/_structures.py
```

### `pkg_resources` / `flatdict`

Isaac Lab base install can fail when `flatdict==4.0.1` builds without `pkg_resources`. The Dockerfile and setup script keep:

```text
setuptools<81
flatdict==4.0.1
```

### `omni.client`

`isaaclab.utils.assets` imports `omni.client`, but `/isaac-sim/python.sh`
did not include its extension path by default. The required path is:

```text
/isaac-sim/kit/extscore/omni.client.lib
```

The Dockerfile and scripts now add this to `PYTHONPATH`.

### `isaaclab_contrib`

The pinned Isaac Lab submodule has these editable packages:

```text
isaaclab
isaaclab_assets
isaaclab_rl
isaaclab_tasks
isaaclab_mimic
```

It does not contain:

```text
isaaclab_contrib
```

## Sanity check

After cloning and installing the repo in the container, run:

```bash
cd /workspace/Uncertainty-Aware-Robotic-World-Model-Go2
bash lab/sanity-check.sh
```

The sanity check verifies:

```text
[1] nvidia-smi works
[2] torch sees CUDA
[3] isaacsim imports
[4] isaaclab imports
[5] omni.client imports
[6] mbrl resolves to real package (not namespace) and mbrl.mbrl.envs.mdp imports
[7] rsl_rl imports
[8] data science extras import
[9] RWM-U checkpoint and dataset exist
[10] RWM train.py --help works
```

Expected: `10 passed, 0 failed`.

## Phase 4A validation summary

See `docs/validation/phase-4a-lab-validation.md` for the full record.
Headline: all six Phase 4A milestones passed (sanity, Init, Pretrain,
Finetune, RWM-U offline, default-scale Pretrain). No OOM at any tested
scale.