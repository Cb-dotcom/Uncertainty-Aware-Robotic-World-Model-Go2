# Lab Workstation Setup — Playbook

Step-by-step from "I am sitting at the workstation" to "I have a working container with the project ready to run."

## Before you arrive at the workstation

Make sure you have:
- **NGC API key** (generated from https://ngc.nvidia.com → user menu → Setup → Generate API Key). Have it saved somewhere accessible.
- **The four files in this directory**: `Dockerfile`, `docker-build.sh`, `docker-run.sh`, `entry-setup.sh`, `sanity-check.sh`. Either bring them on a USB stick, push them to a temporary branch of your repo, or paste from this conversation.

## Questions for the lab admin (ask before you start)

These determine whether the playbook below "just works" or needs adjustment:

1. Is **Docker** installed on the workstation, and is my user in the `docker` group? Without this I cannot run `docker` without `sudo`.
2. Is the **NVIDIA Container Toolkit** installed? Required for `--gpus all` to work.
3. Is `nvcr.io` accessible from the workstation (no firewall block)?
4. Is there a **standard project workspace path** I should use (e.g., `/data/charbel/`, `/mnt/projects/charbel/`)? Default in `docker-run.sh` is `$HOME/workspace/rwmu-cogar/`.
5. **Disk quota** in my home directory and the suggested workspace path? I'll need ~150-200 GB.
6. Anything specific about how you want **long-running training sessions** to be managed (tmux from inside, systemd, screen)?

## Step 1: Verify the host machine

Open a terminal on the workstation. Run:

```bash
# What GPU and driver
nvidia-smi

# What Docker version
docker --version
docker compose version

# Can my user run Docker without sudo?
docker ps

# Is the NVIDIA Container Toolkit working?
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If `docker ps` fails with a permissions error, ask the admin to add you to the `docker` group:
```bash
sudo usermod -aG docker $USER
# then log out and log back in
```

If the last command (the CUDA test image) fails, the NVIDIA Container Toolkit is not configured. Ask the admin.

## Step 2: Log in to NGC (one-time, persists across sessions)

```bash
docker login nvcr.io
```

When prompted:
- **Username**: `$oauthtoken` (literally type the dollar sign and the word)
- **Password**: your NGC API key

You should see `Login Succeeded`. Credentials are stored in `~/.docker/config.json` and reused automatically afterwards.

## Step 3: Place the build files on the workstation

Create a build directory and put the four artifact files there. Example:

```bash
mkdir -p ~/rwmu-cogar-build
cd ~/rwmu-cogar-build
# Copy or paste the four files into this directory:
#   Dockerfile, docker-build.sh, docker-run.sh, entry-setup.sh, sanity-check.sh

# Make the shell scripts executable
chmod +x docker-build.sh docker-run.sh entry-setup.sh sanity-check.sh
```

## Step 4: Build the image

This takes 15-30 minutes the first time. The slowest steps are pulling the Isaac Sim 5.0 base (~15 GB) and installing torch with CUDA 12.8.

```bash
cd ~/rwmu-cogar-build
./docker-build.sh
```

At the end you should see something like:
```
==> Build complete: rwmu-cogar:latest
REPOSITORY    TAG       IMAGE ID       CREATED         SIZE
rwmu-cogar    latest    abc123def456   1 minute ago    ~30GB
```

If the build fails partway through, fix the issue and re-run — Docker caches each step, so subsequent builds skip already-completed layers.

## Step 5: Start the container

```bash
cd ~/rwmu-cogar-build
./docker-run.sh
```

This creates the container `rwmu-cogar`, mounts the host workspace, drops you into bash inside `/workspace`. You should see a root prompt like `root@<hostname>:/workspace#`.

If the admin specified a different workspace path, edit `docker-run.sh` and change `HOST_WORKSPACE` near the top before running. To re-start with the new path, you would first need to remove the container: `docker rm rwmu-cogar`.

## Step 6: Inside the container — clone repo and install editable packages

Inside the container:

```bash
# Inside container, you should be at /workspace
ls /workspace   # should be empty initially

# Copy or paste entry-setup.sh into the container.
# Easiest: it's already on the host at ~/rwmu-cogar-build/entry-setup.sh.
# Open a SECOND terminal on the workstation (not inside the container) and:
docker cp ~/rwmu-cogar-build/entry-setup.sh rwmu-cogar:/workspace/

# Back in the first terminal (inside container):
bash /workspace/entry-setup.sh
```

This will:
1. Clone your project repo with submodules (~5-10 min depending on network).
2. Pull the 89 MB of Git LFS files in the `robotic_world_model` submodule.
3. Install the six Isaac Lab subpackages editably.
4. Install the project's `mbrl` and `rsl_rl_rwm` editably.

Should finish in 10-15 minutes total.

## Step 7: Sanity check

Inside the container:

```bash
# Copy sanity-check.sh in the same way:
# (from a host terminal): docker cp ~/rwmu-cogar-build/sanity-check.sh rwmu-cogar:/workspace/

# Inside container:
bash /workspace/sanity-check.sh
```

You want to see **8 PASS, 0 FAIL**:

```
PASS: nvidia-smi works inside container
PASS: torch sees GPU
PASS: isaacsim imports
PASS: isaaclab imports
PASS: mbrl imports
PASS: rsl_rl imports
PASS: data science extras importable
PASS: pretrained checkpoint present and non-trivial
```

If anything fails, stop and debug. Common issues:
- `nvidia-smi` fails inside container → Container Toolkit issue, ask admin.
- `torch.cuda.is_available()` returns False → driver/CUDA version mismatch.
- `isaaclab` import fails → editable install didn't take, re-run `pip install -e upstream/IsaacLab/source/isaaclab`.
- LFS checkpoint missing or tiny → `cd upstream/robotic_world_model && git lfs pull` again.

## Step 8: Ready to run experiments

You're now at the workstation with the container running, the repo cloned, everything installed and verified. The next phase is the **Phase 4A floor checks** — running the actual project tests (RWM-U offline, Init-v0, reduced Pretrain-v0, reduced Finetune-v0, default-scale attempt).

Open a fresh tmux session inside the container for the first long run:

```bash
# Inside container
tmux new -s rwmu-phase4
cd /workspace/Uncertainty-Aware-Robotic-World-Model-Go2

# Then run the Phase 4A checklist you have, starting with the RWM-U offline test
# (highest probability of success, fewest dependencies).
```

## Detach / re-attach later

To leave the container running but detach from it:
- `Ctrl-P Ctrl-Q` (Docker's default detach sequence)
- Or, if you're inside a tmux session inside the container: `Ctrl-B D` to detach tmux, then `Ctrl-P Ctrl-Q` to detach Docker.

To re-attach:
```bash
./docker-run.sh   # detects existing container and attaches
```

## To remove and rebuild from scratch (if needed)

```bash
docker stop rwmu-cogar
docker rm rwmu-cogar
docker rmi rwmu-cogar:latest
# host workspace and caches at ~/workspace/rwmu-cogar and ~/docker/isaac-sim persist
# delete them too if you want a completely fresh start
```

## Disk usage rough estimate

| Component | Size |
|---|---|
| Isaac Sim 5.0 base image | ~15 GB |
| Project image layers (torch, etc.) | ~10-15 GB |
| Project repo + submodules | ~3-5 GB |
| Isaac Sim shader/asset caches (grows with use) | 10-30 GB |
| Training logs, checkpoints, replay dumps | 50-100 GB (varies) |
| **Total budget** | **150-200 GB** |

## If something goes wrong

Common failure paths and quick diagnostics:

1. **`docker login nvcr.io` fails**: check the username is exactly `$oauthtoken` (with the dollar). Check API key is correct.
2. **`docker pull nvcr.io/nvidia/isaac-sim:5.0.0` fails with "manifest not found"**: NGC sometimes restricts image access. Try logging out and back in, or check that your NGC account has access to the Isaac Sim catalog.
3. **`--gpus all` fails**: NVIDIA Container Toolkit not installed or misconfigured. Admin task.
4. **Build fails on `pip install torch`**: check network. The `--index-url https://download.pytorch.org/whl/cu128` requires outbound HTTPS to pytorch.org.
5. **`entry-setup.sh` fails on `pip install -e upstream/IsaacLab/...`**: re-check that the submodule was cloned. Run `git submodule update --init --recursive` from inside the repo.
6. **`import isaaclab` raises ImportError about Isaac Sim modules not found**: the editable install used the system pip instead of Isaac Sim's pip. Re-install using `/isaac-sim/python.sh -m pip install -e ...`.
