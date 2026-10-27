"""Write `start_observation_distribution.json` for a finished run.

`ManipulationRunner` writes the file beside the checkpoints when a run starts; a run launched
before it did (S7-8 and earlier) has none. This rebuilds the env the run trained on from the
selectors in the environment -- the ones its `submit_*.sh` exported -- and writes the file with
the runner's own function, so it is the same statistic: the actor's observation per term, over
the training environments right after a reset (`num_initial_envs` = the run's `num_envs`).

The cfg it builds is dumped and diffed with the run's `params/env.yaml` (`num_envs` and `seed`
are read from there), so a selector that is wrong shows up as differing lines. Nothing is
written while there are any, unless `--allow-diff`.

  SPAWN=WC TARGET=WC CMD_OBS=cmd REWARD=R-C-strict+M1+M2+M3+M6+J+M8 DR=DR-scene SLEW=stack \\
  BASELINE=B GAINS=G20 EFFORT=CTP TOOL_SPEED=none LAG=FP+JS ACTION=abs EPISODE_S=55 \\
    uv run python scripts/write_start_distribution.py <run dir>

`--start-config` also writes `deployment_start_config.json`, the runner's other file.
"""

from __future__ import annotations

import argparse
import difflib
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.os import dump_yaml
from pal_mjlab.tasks.manipulation.deployment import (
  write_deployment_start_config,
  write_start_observation_distribution,
)
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("run_dir", type=Path, help="the run's directory (has params/env.yaml)")
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--num-envs", type=int, default=None, help="default: the run's")
  p.add_argument("--allow-diff", action="store_true")
  p.add_argument("--overwrite", action="store_true")
  p.add_argument("--start-config", action="store_true")
  return p.parse_args()


def _run_value(env_yaml: str, pattern: str) -> int:
  return int(re.search(pattern, env_yaml, re.MULTILINE).group(1))


def main() -> None:
  args = parse_args()
  run_dir = args.run_dir.resolve()
  env_yaml = (run_dir / "params" / "env.yaml").read_text()
  seed = _run_value(env_yaml, r"^seed: (\d+)")
  run_envs = _run_value(env_yaml, r"^  num_envs: (\d+)")

  out = run_dir / "start_observation_distribution.json"
  if out.exists() and not args.overwrite:
    raise SystemExit(f"{out} exists (--overwrite to replace it)")

  cfg = kangaroo_pick_place_env_cfg(play=False)
  cfg.seed = seed
  cfg.scene.num_envs = run_envs
  # The run's dump was taken once the scene was built, which copies both into the terrain cfg
  # (`Scene._add_terrain`); do the same so the diff shows only what the selectors changed.
  cfg.scene.terrain.num_envs = run_envs
  cfg.scene.terrain.env_spacing = cfg.scene.env_spacing
  with tempfile.TemporaryDirectory() as tmp:
    regenerated = Path(tmp) / "env.yaml"
    dump_yaml(regenerated, asdict(cfg))
    diff = [
      line
      for line in difflib.unified_diff(
        env_yaml.splitlines(),
        regenerated.read_text().splitlines(),
        "run",
        "regenerated",
        lineterm="",
        n=0,
      )
      if not line.startswith(("---", "+++", "@@"))
    ]
  print(
    f"env.yaml (run) vs regenerated from the selectors: {len(diff)} differing lines"
  )
  for line in diff[:40]:
    print("  ", line)
  if diff and not args.allow_diff:
    raise SystemExit("the selectors do not rebuild the run's cfg; nothing written")

  cfg.scene.num_envs = num_envs = args.num_envs or run_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    path = write_start_observation_distribution(env, out)
    print(f"wrote {path} ({num_envs} envs, {args.device})")
    if args.start_config:
      path = write_deployment_start_config(
        env, run_dir / "deployment_start_config.json"
      )
      print(f"wrote {path}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
