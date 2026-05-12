#!/usr/bin/env bash
# docker-run.sh
# Starts (or attaches to) the rwmu-cogar container with:
#   - GPU passthrough (--gpus all)
#   - Persistent Isaac Sim shader/asset caches (so first run isn't re-done every time)
#   - Project workspace mounted from host
#   - Network access for pip/git
#
# Behavior:
#   - If container does not exist: creates it, drops you into bash inside /workspace
#   - If container exists and is stopped: starts it, attaches
#   - If container exists and is running: attaches to existing session

set -euo pipefail

IMAGE_NAME="rwmu-cogar"
IMAGE_TAG="latest"
CONTAINER_NAME="rwmu-cogar"

# --- Host paths (adjust if lab admin specifies a different location) ---
HOST_WORKSPACE="${HOME}/workspace/rwmu-cogar"
HOST_CACHE_ROOT="${HOME}/docker/isaac-sim"

# Create host directories if they do not exist
mkdir -p "${HOST_WORKSPACE}"
mkdir -p "${HOST_CACHE_ROOT}"/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data,documents}

# --- Check if container already exists ---
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Container exists; either start it or attach
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "==> Container ${CONTAINER_NAME} is already running; attaching."
        docker exec -it "${CONTAINER_NAME}" bash
    else
        echo "==> Container ${CONTAINER_NAME} exists but is stopped; starting."
        docker start -ai "${CONTAINER_NAME}"
    fi
    exit 0
fi

# --- First run: create container ---
echo "==> Creating new container: ${CONTAINER_NAME}"
echo "    Workspace mount: ${HOST_WORKSPACE} -> /workspace"
echo "    Cache mount root: ${HOST_CACHE_ROOT}"
echo

docker run \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --network=host \
    -it \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -v "${HOST_WORKSPACE}":/workspace \
    -v "${HOST_CACHE_ROOT}/cache/kit":/isaac-sim/kit/cache:rw \
    -v "${HOST_CACHE_ROOT}/cache/ov":/root/.cache/ov:rw \
    -v "${HOST_CACHE_ROOT}/cache/pip":/root/.cache/pip:rw \
    -v "${HOST_CACHE_ROOT}/cache/glcache":/root/.cache/nvidia/GLCache:rw \
    -v "${HOST_CACHE_ROOT}/cache/computecache":/root/.nv/ComputeCache:rw \
    -v "${HOST_CACHE_ROOT}/logs":/root/.nvidia-omniverse/logs:rw \
    -v "${HOST_CACHE_ROOT}/data":/root/.local/share/ov/data:rw \
    -v "${HOST_CACHE_ROOT}/documents":/root/Documents:rw \
    --workdir /workspace \
    "${IMAGE_NAME}:${IMAGE_TAG}"
