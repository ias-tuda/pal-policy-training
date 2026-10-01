"""Jobs file for `render_session.sh render` (pal-policy-deployment): one PAL panel and one MuJoCo panel per
episode of an S7-8 sim2sim run, paths as the container sees them. Real time by default.

  python3 make_render_jobs_s78.py <label> <run_tag> <model_name> <n_episodes> <panel_dir> [kinds] [speed] > jobs.txt

  label       "ABS-a s2" (video titles)
  run_tag     the PAL run under /runs/pick_place/
  model_name  the exported model under /catkin_ws/src/models/
  panel_dir   where the panels go, e.g. /runs/panels_s78/ABS-a_s2 (container path)
  kinds       "pal,mj" (default), "pal" or "mj"
  speed       simulated seconds per video second (default 1: real time); 0.12 s of RViz settling per frame
  t_end       last simulated second rendered (default 10)
"""

import shlex
import sys

label, tag, model, n, out = sys.argv[1:6]
kinds = (sys.argv[6] if len(sys.argv) > 6 else "pal,mj").split(",")
speed = sys.argv[7] if len(sys.argv) > 7 else "1"
t_end = sys.argv[8] if len(sys.argv) > 8 else "10"
n = int(n)
m = f"/catkin_ws/src/models/{model}"
for k in range(n):
  for kind, src in (
    ("pal", f"--pal /runs/pick_place/{tag}"),
    ("mj", f"--mujoco {m}/mujoco_reference.npz"),
  ):
    if kind not in kinds:
      continue
    who = "PAL physics simulator" if kind == "pal" else "MuJoCo (training)"
    args = (
      f"{src} --episode {k} --model-dir {m} --world t --out {out}/{kind}_ep{k}.mp4 "
      f"--title {shlex.quote(f'{label} (S7-8, 5k its): {who}')} "
      f"--subtitle {shlex.quote(f'episode {k}')} --speed {speed} --t-end {t_end} --settle 0.12 --hold-end 1.0"
    )
    print(args)
