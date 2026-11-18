"""Record joint targets of scripted pick-place episodes (G20) via rollout's hook."""

import sys

import numpy as np

sys.path.insert(0, "scripts")
import scripted_pick_place as spp

out = sys.argv[1]
rec = {"t": [], "phase": [], "obj": []}
names = {}


def hook(phase, env):
  r = env.scene["robot"]
  if not names:
    names["j"] = list(r.joint_names)
    o = env.scene["object"] if "object" in env.scene.entities else None
    names["mass"] = (
      float(o.data.model.body_mass[0, o.indexing.body_ids].sum())
      if o is not None
      else -1
    )
  rec["t"].append(r.data.joint_pos_target.cpu().numpy().copy())
  rec["phase"].append(phase)


orig = spp.rollout


def wrapped(*a, **k):
  k["hook"] = hook
  return orig(*a, **k)


spp.rollout = wrapped
sys.argv = ["x", "rollout", "--gains", "G20", "--num-envs", "10", "--device", "cuda:0"]
spp.main()
np.savez(
  out,
  targets=np.stack(rec["t"]),
  phase=np.array(rec["phase"]),
  joint_names=np.array(names["j"]),
  cube_mass=names.get("mass", -1),
)
print(
  "saved",
  out,
  np.stack(rec["t"]).shape,
  "mass",
  names.get("mass"),
  "phases",
  sorted(set(rec["phase"])),
)
