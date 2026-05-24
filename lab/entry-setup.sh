#!/usr/bin/env bash
# entry-setup.sh
# Runs INSIDE the container, once, after the first `docker run`.
# Clones the project repo, initializes submodules, installs editable packages.
#
# Usage (inside container):
#   cd /workspace
#   bash /workspace/entry-setup.sh
#
# Idempotent: safe to re-run if it fails partway through (it skips work already done).

set -euo pipefail

REPO_URL="https://github.com/Cb-dotcom/Uncertainty-Aware-Robotic-World-Model-Go2.git"
REPO_DIR="Uncertainty-Aware-Robotic-World-Model-Go2"
ISAAC_SIM_PIP="/isaac-sim/python.sh -m pip"

# Required because omni.client lives in kit/extscore in the Isaac Sim 5.0 image,
# but /isaac-sim/python.sh does not include it on sys.path by default.
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:${PYTHONPATH:-}"

cd /workspace

# ---------------------------------------------------------------------------
# 1. Clone the project repo (if not already cloned)
# ---------------------------------------------------------------------------
if [ ! -d "${REPO_DIR}" ]; then
    echo "==> Cloning ${REPO_URL}"
    git clone --recurse-submodules "${REPO_URL}"
else
    echo "==> Repo already exists; updating"
    cd "${REPO_DIR}"
    git pull
    git submodule sync
    git submodule update --init --recursive
    cd /workspace
fi

cd "${REPO_DIR}"

# ---------------------------------------------------------------------------
# 2. Pull LFS objects (3 files, ~89 MB) in the robotic_world_model submodule
# ---------------------------------------------------------------------------
echo "==> Pulling Git LFS objects in upstream/robotic_world_model"
(cd upstream/robotic_world_model && git lfs pull)

# Sanity check that LFS files came down (not just pointer files)
if [ -f upstream/robotic_world_model/assets/models/pretrain_rnn_ens.pt ]; then
    SIZE_BYTES=$(stat -c%s upstream/robotic_world_model/assets/models/pretrain_rnn_ens.pt)
    if [ "${SIZE_BYTES}" -lt 1000000 ]; then
        echo "WARNING: pretrain_rnn_ens.pt is suspiciously small (${SIZE_BYTES} bytes)."
        echo "         LFS pull may have failed. Continuing, but this checkpoint is needed for Finetune-v0."
    else
        echo "    LFS checkpoint OK: ${SIZE_BYTES} bytes"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Install Isaac Lab subpackages editably (from the submodule)
# ---------------------------------------------------------------------------
echo "==> Installing Isaac Lab compatibility dependencies"
${ISAAC_SIM_PIP} install "setuptools<81" flatdict==4.0.1

echo "==> Installing Isaac Lab subpackages (editable)"
${ISAAC_SIM_PIP} install -e upstream/IsaacLab/source/isaaclab
${ISAAC_SIM_PIP} install -e upstream/IsaacLab/source/isaaclab_assets
${ISAAC_SIM_PIP} install -e upstream/IsaacLab/source/isaaclab_rl
${ISAAC_SIM_PIP} install -e upstream/IsaacLab/source/isaaclab_tasks
${ISAAC_SIM_PIP} install -e upstream/IsaacLab/source/isaaclab_mimic

# ---------------------------------------------------------------------------
# 4. Install project-specific submodules editably
# ---------------------------------------------------------------------------
echo "==> Installing project submodules (mbrl, rsl_rl_rwm) editably"
${ISAAC_SIM_PIP} install -e upstream/robotic_world_model/source/mbrl
${ISAAC_SIM_PIP} install -e upstream/rsl_rl_rwm

# ---------------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------------
echo
echo "==> Entry setup complete."
echo
echo "Repo location:    /workspace/${REPO_DIR}"
echo "Next step:        bash /workspace/${REPO_DIR}/lab/sanity-check.sh"
