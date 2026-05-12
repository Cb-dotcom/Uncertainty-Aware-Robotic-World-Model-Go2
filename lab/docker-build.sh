#!/usr/bin/env bash
# docker-build.sh
# Builds the rwmu-cogar image from the Dockerfile in the current directory.
#
# Run from the directory containing the Dockerfile.
# Build time: ~15-30 min on first build (mostly downloading torch + isaacsim base).

set -euo pipefail

IMAGE_NAME="rwmu-cogar"
IMAGE_TAG="latest"

echo "==> Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "    (this will pull nvcr.io/nvidia/isaac-sim:5.0.0 if not already cached)"
echo

# --progress=plain prints each step's output so you see what's happening
docker build \
    --progress=plain \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile \
    .

echo
echo "==> Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
echo
echo "Image size:"
docker images "${IMAGE_NAME}:${IMAGE_TAG}"
