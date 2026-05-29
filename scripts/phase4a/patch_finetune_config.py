#!/usr/bin/env python3
"""
Patch (or restore) the upstream AnymalDFlatPPOFinetuneRunnerCfg block in:

  source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/
  agents/rsl_rl_ppo_cfg.py

The upstream Finetune config hardcodes paths to the original author's
machine. Running Finetune-v0 on any new machine requires patching:

  resume                              True   -> False
  load_run                            <str>  -> "" (unused when resume=False)
  load_system_dynamics                True   -> True (kept)
  system_dynamics_load_path           <str>  -> path to local dynamics ckpt
  system_dynamics_warmup_iterations   500    -> small value (default 50)

The script creates a `.backup_phase4a` next to the config before
patching. Use `--restore` to revert.

Usage:

  # Patch
  python scripts/phase4a/patch_finetune_config.py \\
      --dyn-ckpt /workspace/.../logs/.../model_100.pt \\
      [--warmup-iters 50]

  # Restore
  python scripts/phase4a/patch_finetune_config.py --restore

The script must be invoked with the same RWM repo as the working
directory parent (i.e. $RWM is set), or via the `--cfg-path` flag.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

DEFAULT_CFG_REL = (
    "source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/"
    "config/anymal_d/agents/rsl_rl_ppo_cfg.py"
)
BACKUP_SUFFIX = ".backup_phase4a"


def resolve_cfg_path(arg_cfg_path: str | None) -> Path:
    if arg_cfg_path:
        return Path(arg_cfg_path).resolve()
    rwm = os.environ.get("RWM")
    if rwm:
        return (Path(rwm) / DEFAULT_CFG_REL).resolve()
    # Fall back to current dir
    return (Path.cwd() / DEFAULT_CFG_REL).resolve()


def patch_finetune_block(text: str, dyn_ckpt: str, warmup_iters: int) -> str:
    """
    Locate the Finetune class block and replace four fields. Returns the new
    file text. Raises SystemExit if patterns do not match.
    """
    pattern = re.compile(
        r"(class AnymalDFlatPPOFinetuneRunnerCfg\([^)]+\):.*?)"
        r"(?=\n@configclass|\nclass\s+\w|\Z)",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        raise SystemExit("ERROR: Could not locate AnymalDFlatPPOFinetuneRunnerCfg block")

    block = m.group(1)
    original_block = block

    block = re.sub(
        r"^(\s*)resume\s*=\s*True\b",
        r"\1resume = False",
        block,
        flags=re.MULTILINE,
    )
    block = re.sub(
        r'^(\s*)load_run\s*=\s*".*?"',
        r'\1load_run = ""',
        block,
        flags=re.MULTILINE,
    )
    block = re.sub(
        r'^(\s*)system_dynamics_load_path\s*=\s*".*?"',
        rf'\1system_dynamics_load_path = "{dyn_ckpt}"',
        block,
        flags=re.MULTILINE,
    )
    block = re.sub(
        r"^(\s*)system_dynamics_warmup_iterations\s*=\s*\d+",
        rf"\1system_dynamics_warmup_iterations = {warmup_iters}",
        block,
        flags=re.MULTILINE,
    )

    if block == original_block:
        raise SystemExit(
            "ERROR: No substitutions applied. The Finetune block did not match "
            "expected patterns. Was the upstream config refactored?"
        )

    return text[: m.start(1)] + block + text[m.end(1) :]


def do_patch(cfg_path: Path, dyn_ckpt: str, warmup_iters: int) -> None:
    if not cfg_path.exists():
        raise SystemExit(f"ERROR: config not found at {cfg_path}")

    # Verify checkpoint actually exists
    ckpt = Path(dyn_ckpt)
    if not ckpt.exists():
        raise SystemExit(f"ERROR: dynamics checkpoint not found at {dyn_ckpt}")

    backup = cfg_path.with_suffix(cfg_path.suffix + BACKUP_SUFFIX)
    if backup.exists():
        raise SystemExit(
            f"ERROR: backup already exists at {backup}. "
            "Run with --restore first, or remove it manually."
        )

    shutil.copy2(cfg_path, backup)
    print(f"Backed up: {cfg_path} -> {backup}")

    src = cfg_path.read_text()
    new_src = patch_finetune_block(src, str(ckpt.resolve()), warmup_iters)
    cfg_path.write_text(new_src)
    print(f"Patched: {cfg_path}")
    print()
    print("Patched Finetune block:")
    print("-" * 72)
    pattern = re.compile(
        r"class AnymalDFlatPPOFinetuneRunnerCfg.*?"
        r"(?=\n@configclass|\nclass\s+\w|\Z)",
        re.DOTALL,
    )
    m = pattern.search(new_src)
    if m:
        # Trim to first 12 lines to keep output short
        block_lines = m.group(0).splitlines()[:12]
        print("\n".join(block_lines))
    print("-" * 72)


def do_restore(cfg_path: Path) -> None:
    backup = cfg_path.with_suffix(cfg_path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        raise SystemExit(f"ERROR: no backup found at {backup}")
    shutil.copy2(backup, cfg_path)
    backup.unlink()
    print(f"Restored: {backup} -> {cfg_path} (backup removed)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dyn-ckpt",
        help="Path to the dynamics checkpoint (e.g. logs/.../pretrain/model_100.pt)",
    )
    p.add_argument(
        "--warmup-iters",
        type=int,
        default=50,
        help="Value for system_dynamics_warmup_iterations (default: 50)",
    )
    p.add_argument(
        "--cfg-path",
        help=(
            "Explicit path to rsl_rl_ppo_cfg.py. "
            f"Default: $RWM/{DEFAULT_CFG_REL}"
        ),
    )
    p.add_argument(
        "--restore",
        action="store_true",
        help="Restore the config from the .backup_phase4a file and remove the backup.",
    )
    args = p.parse_args()

    cfg_path = resolve_cfg_path(args.cfg_path)

    if args.restore:
        do_restore(cfg_path)
        return

    if not args.dyn_ckpt:
        p.error("--dyn-ckpt is required unless --restore is passed")

    do_patch(cfg_path, args.dyn_ckpt, args.warmup_iters)


if __name__ == "__main__":
    main()