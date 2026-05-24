# Lab Workstation Notes

These notes document the lab Docker/workstation setup used for Phase 4A validation of the RWM/RWM-U project.

## Current lab target

Observed lab host:

```text
Host: k8s-worker-node-2
User: ter-ws-3
OS: Ubuntu 24.04.2
GPU: NVIDIA RTX 6000 Ada Generation
Driver: 570.211.01
CUDA shown by nvidia-smi: 12.8
Container image: rwmu-cogar-cb:latest
Container name: rwmu-cogar-cb
Host workspace: ~/workspace/rwmu-cogar-cb
Container workspace: /workspace
Isaac Python: /isaac-sim/python.sh
```

## Runtime pattern

Inside the container, use Isaac Sim Python:

```
/isaac-sim/python.sh
```

Do not use system Python for RWM/Isaac Lab runs.

Use this environment pattern:

```
export PROJECT_ROOT="/workspace/Uncertainty-Aware-Robotic-World-Model-Go2"
export RWM="$PROJECT_ROOT/upstream/robotic_world_model"
export RSL="$PROJECT_ROOT/upstream/rsl_rl_rwm"

export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$RWM/source:$RSL:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

For the standalone offline RWM-U scripts, also include:

```
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$RWM/source:$RSL:$RWM/scripts/reinforcement_learning/model_based:$PYTHONPATH"
```

## Container compatibility fixes

The Isaac Sim 5.0 base image required several fixes during lab bring-up.

### Torch vendored packaging symlink

Torch resolved from:

```
/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch
```

The file:

```
torch/_vendor/packaging/_structures.py
```

was a broken symlink to a missing `omni.isaac.core_archive` packaging file. The Dockerfile now replaces that single broken symlink with the valid file from:

```
/isaac-sim/kit/python/lib/python3.11/site-packages/packaging/_structures.py
```

### `pkg_resources` / `flatdict`

Isaac Lab base install can fail when `flatdict==4.0.1` builds without `pkg_resources`. The Dockerfile and setup script keep:

```
setuptools<81
flatdict==4.0.1
```

### `omni.client`

`isaaclab.utils.assets` imports `omni.client`, but `/isaac-sim/python.sh` did not include its extension path by default. The required path is:

```
/isaac-sim/kit/extscore/omni.client.lib
```

The Dockerfile and scripts now add this to `PYTHONPATH`.

### `isaaclab_contrib`

The pinned Isaac Lab submodule has these editable packages:

```
isaaclab
isaaclab_assets
isaaclab_rl
isaaclab_tasks
isaaclab_mimic
```

It does not contain:

```
isaaclab_contrib
```

## Sanity check

After cloning and installing the repo in the container, run:

```
cd /workspace/Uncertainty-Aware-Robotic-World-Model-Go2
bash lab/sanity-check.sh
```

The sanity check verifies:

```
nvidia-smi works
torch sees CUDA
isaacsim imports
isaaclab imports
omni.client imports
mbrl imports
rsl_rl imports
data science extras import
RWM-U checkpoint and dataset exist
RWM train.py --help works
```