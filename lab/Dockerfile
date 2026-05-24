# Dockerfile for the Uncertainty-Aware Robotic World Model thesis project
# Base: NVIDIA Isaac Sim 5.0.0 (Ubuntu 22.04, includes Isaac Sim + Omniverse Kit)
# Adds: Python project dependencies, system tools, but NOT the project code
#       (project code is cloned at runtime via entry-setup.sh)

FROM nvcr.io/nvidia/isaac-sim:5.0.0

# Accept NVIDIA license non-interactively (already required for image use)
ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        tmux \
        vim \
        less \
        wget \
        ca-certificates \
        build-essential \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python package layer — load-bearing pins from the working laptop env
# ---------------------------------------------------------------------------
# Isaac Sim's bundled Python is at /isaac-sim/python.sh, but the base image
# also exposes a usable `pip` in PATH. We install into the Isaac Sim Python
# so imports match the runtime that Isaac Lab expects.

# Use Isaac Sim's pip explicitly
ENV ISAAC_SIM_PIP=/isaac-sim/python.sh

# Upgrade pip, but keep setuptools old enough to provide pkg_resources.
# Some Isaac Lab dependencies, notably flatdict, still expect pkg_resources.
RUN $ISAAC_SIM_PIP -m pip install --upgrade pip wheel "setuptools<81"

# Core ML / RL stack (matched to laptop pip freeze)
RUN $ISAAC_SIM_PIP -m pip install \
        torch==2.7.0+cu128 \
        torchvision==0.22.0+cu128 \
        torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

RUN $ISAAC_SIM_PIP -m pip install \
        triton==3.3.0 \
        numpy==1.26.0 \
        gymnasium==1.2.1 \
        tensordict==0.12.1 \
        hydra-core==1.3.2 \
        omegaconf==2.3.0 \
        warp-lang==1.12.1 \
        flatdict==4.0.1

# Data science / logging extras
RUN $ISAAC_SIM_PIP -m pip install \
        wandb \
        tensorboard \
        pandas \
        scipy \
        matplotlib \
        h5py \
        safetensors \
        rich


# ---------------------------------------------------------------------------
# Isaac Sim 5.0 container compatibility fixes
# ---------------------------------------------------------------------------
# In this base image, torch resolves from omni.isaac.ml_archive/pip_prebundle.
# After installing torch 2.7.0+cu128, its vendored packaging/_structures.py can
# remain a broken symlink to a non-existent omni.isaac.core_archive file.
# Replacing that single broken symlink with the working packaging shim from
# Isaac Sim's site-packages.
RUN TORCH_STRUCT="/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py" \
    && GOOD_STRUCT="/isaac-sim/kit/python/lib/python3.11/site-packages/packaging/_structures.py" \
    && if [ -L "$TORCH_STRUCT" ] && ! [ -e "$TORCH_STRUCT" ]; then \
        cp -P "$TORCH_STRUCT" "${TORCH_STRUCT}.broken_symlink_backup"; \
        rm -f "$TORCH_STRUCT"; \
        cp "$GOOD_STRUCT" "$TORCH_STRUCT"; \
    fi

# Isaac Lab utilities import omni.client, which lives in kit/extscore but is not
# on /isaac-sim/python.sh sys.path by default in this container.
ENV PYTHONPATH=/isaac-sim/kit/extscore/omni.client.lib:${PYTHONPATH}

# ---------------------------------------------------------------------------
# Sanity: surface what got installed
# ---------------------------------------------------------------------------
RUN $ISAAC_SIM_PIP -m pip list | grep -iE "^(torch|isaacsim|numpy|gymnasium|tensordict|hydra|omegaconf|warp)" \
    || true

# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------
WORKDIR /workspace

# Entry point: keep it as bash for interactive use
ENTRYPOINT ["/bin/bash"]
