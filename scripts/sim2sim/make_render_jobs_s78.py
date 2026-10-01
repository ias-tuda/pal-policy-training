"""Jobs file for `render_session.sh render` (pal-policy-deployment): one PAL panel and one MuJoCo panel per
episode of an S7-8 sim2sim run, 3x faster than real time, paths as the container sees them.

  python3 make_render_jobs_s78.py <label> <run_tag> <model_name> <n_episodes> <panel_dir> > jobs.txt

  label       "ABS-a s2" (video titles)
  run_tag     the PAL run under /runs/pick_place/
  model_name  the exported model under /catkin_ws/src/models/
  panel_dir   where the panels go, e.g. /runs/panels_s78/ABS-a_s2 (container path)
"""

import shlex
import sys

label, tag, model, n, out = sys.argv[1:6]
n = int(n)
m = f"/catkin_ws/src/models/{model}"
for k in range(n):
  for kind, src in (
    ("pal", f"--pal /runs/pick_place/{tag}"),
    ("mj", f"--mujoco {m}/mujoco_reference.npz"),
  ):
    who = "PAL physics simulator" if kind == "pal" else "MuJoCo (training)"
    args = (
      f"{src} --episode {k} --model-dir {m} --world t --out {out}/{kind}_ep{k}.mp4 "
      f"--title {shlex.quote(f'{label} (S7-8, 5k its): {who}')} "
      f"--subtitle {shlex.quote(f'episode {k}')} --speed 3 --hold-end 1.0"
    )
    print(args)
