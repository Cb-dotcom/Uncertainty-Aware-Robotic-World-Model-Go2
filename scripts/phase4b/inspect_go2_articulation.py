#!/usr/bin/env python3
"""
inspect_go2_articulation.py

Spawn a single Unitree Go2 in a minimal Isaac Lab scene, step once,
and dump the articulation's joint order, default joint positions,
joint/body names, contact bodies, and joint limits to JSON.

This is a Phase 4B diagnostic. It exists to resolve the open items
in docs/go2-transfer/go2-inventory.md:
  1. Joint order at runtime (not just regex-keyed names)
  2. Default joint position vector resolved from UNITREE_GO2_CFG
  3. system_state shape verification (45 expected if 12 joints)

Run inside the lab container:

    cd /workspace/Uncertainty-Aware-Robotic-World-Model-Go2
    unset PYTHONPATH
    export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib"
    export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y
    /isaac-sim/python.sh scripts/phase4b/inspect_go2_articulation.py

Output:
  logs/go2_inventory/<timestamp>/articulation.json
  Human-readable summary to stdout.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Isaac Sim must be bootstrapped before any isaaclab import. AppLauncher is
# the standard pattern.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

# Use the headless app. No rendering, no GUI.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Now safe to import the rest.
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

# Upstream Isaac Lab Go2 asset
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


# ---------------------------------------------------------------------------
# Minimal scene with ground + light + a single Go2.
# ---------------------------------------------------------------------------
@configclass
class Go2InspectionSceneCfg(InteractiveSceneCfg):
    """Single Go2 on a flat ground plane."""

    ground = sim_utils.GroundPlaneCfg()
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))

    # Resolve the Go2 cfg into our scene. The prim path must use the
    # standard env-regex namespace so InteractiveScene replicates it
    # correctly (even though we only have 1 env here).
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


def regex_to_indices(joint_names: list[str], pattern: str) -> list[int]:
    """Match a joint name regex against the articulation's joint list."""
    rx = re.compile(pattern)
    return [i for i, name in enumerate(joint_names) if rx.match(name)]


def regex_to_body_indices(body_names: list[str], pattern: str) -> list[int]:
    rx = re.compile(pattern)
    return [i for i, name in enumerate(body_names) if rx.match(name)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: logs/go2_inventory/<timestamp>/",
    )
    args = parser.parse_args()

    # Build sim
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0", dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Build a tiny scene with 1 environment
    scene_cfg = Go2InspectionSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    # Step once so all buffers are populated
    actions = torch.zeros((1, 12), device=sim.device)
    scene["robot"].set_joint_position_target(actions)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    robot: Articulation = scene["robot"]
    data = robot.data

    # ---- Extract everything ---------------------------------------------
    joint_names: list[str] = list(robot.joint_names)
    body_names: list[str] = list(robot.body_names)

    n_joints = len(joint_names)
    n_bodies = len(body_names)

    # Default positions / velocities
    # data.default_joint_pos shape: (num_envs, num_joints)
    default_joint_pos = data.default_joint_pos[0].cpu().tolist()
    default_joint_vel = data.default_joint_vel[0].cpu().tolist()

    # Joint limits — soft limits (clipped to physical). Shape: (num_envs, num_joints, 2)
    joint_pos_limits = data.joint_pos_limits[0].cpu().tolist()
    joint_vel_limits = data.joint_vel_limits[0].cpu().tolist()
    joint_effort_limits = data.joint_effort_limits[0].cpu().tolist()

    # Body indices for the regex patterns in the upstream Go2 cfg
    hip_idx = regex_to_indices(joint_names, r".*_hip_joint")
    thigh_idx = regex_to_indices(joint_names, r".*_thigh_joint")
    calf_idx = regex_to_indices(joint_names, r".*_calf_joint")

    foot_body_idx = regex_to_body_indices(body_names, r".*_foot")
    thigh_body_idx = regex_to_body_indices(body_names, r".*_thigh")
    calf_body_idx = regex_to_body_indices(body_names, r".*_calf")
    base_body_idx = regex_to_body_indices(body_names, r"^base$")

    out = {
        "robot": "Unitree Go2",
        "source_cfg": "isaaclab_assets.robots.unitree.UNITREE_GO2_CFG",
        "usd": "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        "counts": {
            "num_joints": n_joints,
            "num_bodies": n_bodies,
        },
        "joints": {
            "names_in_actuator_order": joint_names,
            "default_joint_pos": default_joint_pos,
            "default_joint_vel": default_joint_vel,
            "pos_lower_limit": [lo for lo, _ in joint_pos_limits],
            "pos_upper_limit": [hi for _, hi in joint_pos_limits],
            "vel_limit": joint_vel_limits,
            "effort_limit": joint_effort_limits,
            "indices_by_regex": {
                ".*_hip_joint": hip_idx,
                ".*_thigh_joint": thigh_idx,
                ".*_calf_joint": calf_idx,
            },
        },
        "bodies": {
            "names": body_names,
            "indices_by_regex": {
                "^base$": base_body_idx,
                ".*_foot": foot_body_idx,
                ".*_thigh": thigh_body_idx,
                ".*_calf": calf_body_idx,
            },
            "names_by_regex": {
                "^base$": [body_names[i] for i in base_body_idx],
                ".*_foot": [body_names[i] for i in foot_body_idx],
                ".*_thigh": [body_names[i] for i in thigh_body_idx],
                ".*_calf": [body_names[i] for i in calf_body_idx],
            },
        },
        "system_state_shape_check": {
            "expected_for_anymal_d_compatible": 45,
            "computed": 3 + 3 + 3 + n_joints + n_joints + n_joints,
            "components": {
                "base_lin_vel": 3,
                "base_ang_vel": 3,
                "projected_gravity": 3,
                "joint_pos": n_joints,
                "joint_vel": n_joints,
                "joint_torque": n_joints,
            },
            "matches_anymal_d": (3 + 3 + 3 + 3 * n_joints) == 45,
        },
    }

    # ---- Write JSON ------------------------------------------------------
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path.cwd() / "logs" / "go2_inventory" / ts
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "articulation.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ---- Print human-readable summary ------------------------------------
    sep = "=" * 72
    print()
    print(sep)
    print("  Go2 articulation inventory")
    print(sep)
    print()
    print(f"Joints: {n_joints}    Bodies: {n_bodies}")
    print()
    print("Joint names (in actuator order):")
    for i, (name, dpos, lo, hi) in enumerate(zip(
        joint_names,
        default_joint_pos,
        out["joints"]["pos_lower_limit"],
        out["joints"]["pos_upper_limit"],
    )):
        print(f"  [{i:2d}] {name:24s}  default={dpos:+.4f}  limits=[{lo:+.3f}, {hi:+.3f}]")
    print()
    print("Body names:")
    for i, name in enumerate(body_names):
        print(f"  [{i:2d}] {name}")
    print()
    print("Body groups by regex:")
    for pat in ["^base$", ".*_foot", ".*_thigh", ".*_calf"]:
        names = out["bodies"]["names_by_regex"][pat]
        print(f"  {pat:14s} -> {names}")
    print()
    print("System state shape check:")
    sscheck = out["system_state_shape_check"]
    print(f"  expected (ANYmal-D-compatible): {sscheck['expected_for_anymal_d_compatible']}")
    print(f"  computed (with Go2 joint count): {sscheck['computed']}")
    print(f"  matches ANYmal-D layout:         {sscheck['matches_anymal_d']}")
    print()
    print(f"JSON written to: {out_path}")
    print(sep)

    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()