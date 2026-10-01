"""Regenerate the env cfg from the selectors in the environment and diff it with a run dump (params/env.yaml)."""
import difflib, os, sys
from pathlib import Path
from dataclasses import asdict
from mjlab.utils.os import dump_yaml
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import kangaroo_pick_place_env_cfg
cfg = kangaroo_pick_place_env_cfg(play=False)
cfg.scene.num_envs = 2048
dump_yaml(Path(sys.argv[2]), asdict(cfg))
a = [l for l in open(sys.argv[1]).read().splitlines()]
b = [l for l in open(sys.argv[2]).read().splitlines()]
d = [l for l in difflib.unified_diff(a, b, "run", "regen", lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
print("differing lines:", len(d))
for l in d[:40]: print("  ", l)
