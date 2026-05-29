#!/usr/bin/env python3
"""
install_go2_config.py

Copies the tracked Go2 RWM config tree from
    scripts/phase4b/go2_rwm_config/
into the gitignored submodule path at
    upstream/robotic_world_model/source/mbrl/mbrl/tasks/manager_based/
    locomotion/velocity/config/go2/

Required because the upstream/ submodule itself is gitignored at the
parent-repo level, so we cannot track our additions inside it. Source
of truth lives in scripts/phase4b/go2_rwm_config/; this script syncs
it into the runtime location.

Run inside the lab container, from the project root:

    cd /workspace/Uncertainty-Aware-Robotic-World-Model-Go2
    /isaac-sim/python.sh scripts/phase4b/install_go2_config.py

By default refuses to overwrite an existing go2/ directory in the
submodule. Use --force to overwrite. Use --uninstall to remove the
installed copy.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_DST_REL = (
    "upstream/robotic_world_model/source/mbrl/mbrl/tasks/"
    "manager_based/locomotion/velocity/config/go2"
)
SRC_REL = "scripts/phase4b/go2_rwm_config"


def resolve_paths() -> tuple[Path, Path]:
    """Resolve src and dst paths from the current working directory."""
    cwd = Path.cwd()
    src = cwd / SRC_REL
    dst = cwd / DEFAULT_DST_REL
    return src, dst


def do_install(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        raise SystemExit(f"ERROR: source tree not found at {src}")
    if not src.is_dir():
        raise SystemExit(f"ERROR: source path is not a directory: {src}")

    if dst.exists():
        if not force:
            raise SystemExit(
                f"ERROR: destination already exists at {dst}\n"
                "Use --force to overwrite, or --uninstall first."
            )
        shutil.rmtree(dst)
        print(f"Removed existing: {dst}")

    shutil.copytree(src, dst)
    print(f"Installed: {src} -> {dst}")
    print()
    print("Tree:")
    for path in sorted(dst.rglob("*")):
        rel = path.relative_to(dst.parent)
        print(f"  {rel}")
    print()
    print("To activate the Go2 tasks, import 'mbrl.tasks' in your script")
    print("(the RWM train.py does this automatically). The new task IDs are:")
    print("  Template-Isaac-Velocity-Flat-Unitree-Go2-Init-v0")
    print("  Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0")


def do_uninstall(dst: Path) -> None:
    if not dst.exists():
        print(f"Nothing to uninstall: {dst} does not exist")
        return
    shutil.rmtree(dst)
    # Also drop any __pycache__ inside the parent that referenced our package
    parent_pycache = dst.parent / "__pycache__"
    print(f"Removed: {dst}")
    if parent_pycache.exists():
        # Be careful: only purge our own cached entry, leave others alone
        for p in parent_pycache.glob("go2*.pyc"):
            p.unlink()
            print(f"Removed cached: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing go2/ directory in the submodule.",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the installed go2/ directory from the submodule.",
    )
    args = parser.parse_args()

    src, dst = resolve_paths()
    print(f"Source:      {src}")
    print(f"Destination: {dst}")
    print()

    if args.uninstall:
        do_uninstall(dst)
        return

    do_install(src, dst, force=args.force)


if __name__ == "__main__":
    main()