"""Resolve and print the feet_air_time contact-sensor bodies for the Go2 env."""
import argparse
from isaaclab.app import AppLauncher          # <-- match play.py (or omni.isaac.lab.app)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg     # <-- match play.py
import isaaclab_tasks  # noqa: F401               # <-- match play.py
import mbrl.tasks.manager_based.locomotion.velocity.config.go2  # noqa: F401

def banner(m): print(f"\n===AIRTIME=== {m}", flush=True)

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg)
u = env.unwrapped
banner(f"task = {args.task}")

try:
    cs = u.scene["contact_forces"]
    banner(f"contact_forces sensor tracks {len(cs.body_names)} bodies:")
    for i, n in enumerate(cs.body_names):
        print(f"    [{i}] {n}", flush=True)
except Exception as e:
    banner(f"no 'contact_forces' sensor ({e!r}); sensors = {list(u.scene.sensors.keys())}")

rm = u.reward_manager
banner(f"reward term names: {rm.active_terms}")

try:
    term = rm.get_term_cfg("feet_air_time")
    sc = term.params.get("sensor_cfg", None)
    banner(f"feet_air_time threshold = {term.params.get('threshold')}")
    if sc is None:
        banner("feet_air_time has NO sensor_cfg param  <-- unexpected")
    else:
        banner(f"sensor_cfg.name       = {sc.name}")
        banner(f"sensor_cfg.body_ids   = {sc.body_ids!r}")
        banner(f"sensor_cfg.body_names = {sc.body_names!r}")
        bn = sc.body_names
        if isinstance(sc.body_ids, slice):
            banner("VERDICT: body_ids is slice(None) -> matches ALL bodies, not just feet  <-- BUG")
        elif not bn:
            banner("VERDICT: ZERO bodies resolved -> dead term  <-- BUG")
        elif len(bn) == 4 and all("foot" in x.lower() for x in bn):
            banner(f"VERDICT: OK -> 4 foot bodies resolved: {bn}")
        else:
            banner(f"VERDICT: resolved {len(bn)} bodies, inspect: {bn}  <-- CHECK")
except ValueError as e:
    banner(f"no reward term named 'feet_air_time' ({e!r}); see term names above")

env.close()
simulation_app.close()
