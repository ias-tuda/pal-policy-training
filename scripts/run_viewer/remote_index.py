"""List every training run on the cluster as JSON (stdlib only; piped over ssh).

Usage: python3 - PROJECT_ROOT < remote_index.py
Maps logs/rsl_rl/<experiment>/<run> folders to their W&B run the same way
scripts/download_wandb_run.sh does: same run name, start times within ten minutes.
"""

import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
import time

root = pathlib.Path(sys.argv[1])

wandb_runs = {}
for meta_path in (root / "wandb").glob("run-*-*/files/wandb-metadata.json"):
  try:
    meta = json.loads(meta_path.read_text())
    args = meta.get("args", [])
    run_name = args[args.index("--agent.run-name") + 1]
    started = dt.datetime.fromisoformat(meta["startedAt"].replace("Z", "+00:00"))
  except (ValueError, IndexError, KeyError, OSError):
    continue
  tasks = [a for a in args if a.startswith("Mjlab-")]
  wandb_runs.setdefault(run_name, []).append(
    {
      "id": meta_path.parent.parent.name.rsplit("-", 1)[-1],
      "started": started.astimezone().replace(tzinfo=None),
      "task": tasks[0] if len(tasks) == 1 else None,
      "commit": (meta.get("git") or {}).get("commit"),
      "args": args,
    }
  )

head = subprocess.run(
  ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()

runs = []
now = time.time()
for exp_dir in sorted((root / "logs/rsl_rl").iterdir()):
  if not exp_dir.is_dir():
    continue
  for run_dir in exp_dir.iterdir():
    if not run_dir.is_dir():
      continue
    try:
      stamp = dt.datetime.strptime(run_dir.name[:19], "%Y-%m-%d_%H-%M-%S")
    except ValueError:
      continue
    run_name = run_dir.name[20:]
    checkpoints, newest = [], 0.0
    for path in run_dir.glob("model_*.pt"):
      m = re.fullmatch(r"model_(\d+)\.pt", path.name)
      if m:
        checkpoints.append(int(m.group(1)))
        newest = max(newest, path.stat().st_mtime)
    if not checkpoints:
      continue
    videos = sorted(
      int(m.group(1))
      for p in (run_dir / "videos/viewer").glob("model_*.mp4")
      if (m := re.fullmatch(r"model_(\d+)\.mp4", p.name))
    )
    match = None
    for cand in wandb_runs.get(run_name, []):
      dist = abs((cand["started"] - stamp).total_seconds())
      if dist <= 600 and (match is None or dist < match[0]):
        match = (dist, cand)
    info = match[1] if match else {}
    seed = None
    args = info.get("args", [])
    if "--agent.seed" in args:
      seed = args[args.index("--agent.seed") + 1]
    runs.append(
      {
        "key": str(run_dir.relative_to(root)),
        "experiment": exp_dir.name,
        "name": run_name,
        "started": stamp.isoformat(),
        "wandb_id": info.get("id"),
        "task": info.get("task"),
        "commit": info.get("commit"),
        "seed": seed,
        "checkpoints": sorted(checkpoints),
        "videos": videos,
        "active": now - newest < 30 * 60,
      }
    )

runs.sort(key=lambda r: r["started"], reverse=True)
json.dump({"head": head, "runs": runs}, sys.stdout)
