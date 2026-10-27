"""Local web tool to browse cluster training runs and watch checkpoint videos.

  python3 scripts/run_viewer/server.py [--port 8765]

then open http://localhost:8765. It lists every run under logs/rsl_rl on the cluster,
shows its training curves, and plays one video per checkpoint. A missing video is
rendered by a short SLURM job (render_video.py) and copied down when the job ends.
A job still pending after --local-after seconds (the queue is full) is cancelled and
rendered on this machine instead, after copying its checkpoints down.
Videos are cached in wandb/<run>/videos/viewer/, next to what
scripts/download_wandb_run.sh downloads. Labels, stars and jobs are kept in
wandb/run_viewer/state.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shlex
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REMOTE = "kienle@mn.ias.informatik.tu-darmstadt.de"
REMOTE_PROJECT = "/home/kienle/OopsieData/pal_mjlab_kangaroo_talos"
REMOTE_TOOL_DIR = f"{REMOTE_PROJECT}/logs/run_viewer"

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LOCAL_ROOT = PROJECT / "wandb"
STATE_DIR = LOCAL_ROOT / "run_viewer"
STATE_FILE = STATE_DIR / "state.json"
CURVE_DIR = STATE_DIR / "curves"
LOCAL_LOG_DIR = STATE_DIR / "local"
LOCAL_PYTHON = PROJECT / ".venv" / "bin" / "python"

SSH_OPTS = [
  "-o", "BatchMode=yes",
  "-o", "ControlMaster=auto",
  "-o", f"ControlPath={Path.home()}/.ssh/cm-run-viewer-%r@%h:%p",
  "-o", "ControlPersist=10m",
]  # fmt: skip
SBATCH_OPTS = [
  "--partition=main",
  "--gres=gpu:1",
  "--cpus-per-task=4",
  "--mem-per-cpu=4096",
  "--time=01:00:00",
  "--exclude=dgx-station",
]
# Checkpoints requested within this many seconds for the same run share one job.
BATCH_DELAY = 3.0
POLL_INTERVAL = 20.0
# A job still pending this long is rendered locally instead (set by --local-after).
local_after = 10.0
local_gpu = threading.Semaphore(1)  # one local render at a time

lock = threading.RLock()
index: dict = {"head": None, "runs": [], "fetched": None}
state: dict = {"labels": {}, "starred": [], "marks": {}, "jobs": []}
pending: dict[str, dict] = {}  # run key -> {"iterations": set, "episodes": int, "timer"}


def log(msg: str) -> None:
  print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def ssh(command: str, *, stdin: str | None = None, timeout: float = 120) -> str:
  result = subprocess.run(
    ["ssh", *SSH_OPTS, REMOTE, command],
    input=stdin,
    capture_output=True,
    text=True,
    timeout=timeout,
  )
  if result.returncode != 0:
    raise RuntimeError(result.stderr.strip() or f"ssh exited {result.returncode}")
  return result.stdout


def save_state() -> None:
  STATE_DIR.mkdir(parents=True, exist_ok=True)
  tmp = STATE_FILE.with_suffix(".tmp")
  tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
  tmp.replace(STATE_FILE)


def load_state() -> None:
  if STATE_FILE.exists():
    state.update(json.loads(STATE_FILE.read_text()))


def local_videos(key: str) -> list[int]:
  folder = LOCAL_ROOT / key / "videos" / "viewer"
  return sorted(
    int(m.group(1))
    for p in folder.glob("model_*.mp4")
    if (m := re.fullmatch(r"model_(\d+)\.mp4", p.name))
  )


def valid_key(key: str) -> bool:
  with lock:
    return any(r["key"] == key for r in index["runs"])


def find_run(key: str) -> dict:
  with lock:
    for run in index["runs"]:
      if run["key"] == key:
        return run
  raise KeyError(f"unknown run {key!r}")


def refresh_index() -> None:
  script = (HERE / "remote_index.py").read_text()
  data = json.loads(
    ssh(f"python3 - {shlex.quote(REMOTE_PROJECT)}", stdin=script, timeout=180)
  )
  with lock:
    index.update(data, fetched=time.time())


def index_payload() -> dict:
  with lock:
    runs = [dict(r, local_videos=local_videos(r["key"])) for r in index["runs"]]
    return {
      "head": index["head"],
      "fetched": index["fetched"],
      "runs": runs,
      "labels": state["labels"],
      "starred": state["starred"],
      "marks": state["marks"],
    }


def curves(key: str, refresh: bool) -> dict:
  run = find_run(key)
  cache = CURVE_DIR / (key.replace("/", "__") + ".json")
  # Finished runs never change; refresh live ones after ten minutes.
  if cache.exists() and not refresh:
    age = time.time() - cache.stat().st_mtime
    if not run["active"] or age < 600:
      return json.loads(cache.read_text())
  script = (HERE / "remote_curves.py").read_text()
  remote_run = f"{REMOTE_PROJECT}/{key}"
  raw = ssh(
    f"{REMOTE_PROJECT}/.venv/bin/python - {shlex.quote(remote_run)} 600",
    stdin=script,
    timeout=300,
  )
  CURVE_DIR.mkdir(parents=True, exist_ok=True)
  cache.write_text(raw)
  return json.loads(raw)


def pick_tag(tags, wanted: str | None) -> str | None:
  """The UI's default scalar: the requested one, else the first success-like tag."""
  tags = sorted(tags)
  if wanted in tags:
    return wanted
  for pattern in (r"episode_success", r"success", r"^Train/mean_reward$"):
    for tag in tags:
      if re.search(pattern, tag, re.I):
        return tag
  return tags[0] if tags else None


def best_and_latest(key: str, tag: str | None, window: float) -> dict:
  """Best checkpoint by the moving-average curve (as the chart shows it) and the latest."""
  run = find_run(key)
  latest = run["checkpoints"][-1]
  tag = pick_tag(curves(key, False), tag)
  if tag is None:
    return {"tag": None, "best": None, "latest": latest}
  points = curves(key, False)[tag]
  smooth = []
  for step, _ in points:
    near = [v for s, v in points if abs(s - step) <= window / 2]
    smooth.append((step, sum(near) / len(near)))
  best = max(
    run["checkpoints"],
    key=lambda it: min(smooth, key=lambda p: abs(p[0] - it))[1],
  )
  return {"tag": tag, "best": best, "latest": latest}


def render_best_and_latest(keys, tag, window, episodes) -> dict:
  def plan(key):
    try:
      picked = best_and_latest(key, tag, window)
      run = find_run(key)
      have = set(run["videos"]) | set(local_videos(key))
      wanted = sorted({picked["best"], picked["latest"]} - {None} - have)
      result = request_render(key, wanted, episodes) if wanted else {"queued": []}
      return key, {**picked, **result, "existing": sorted(have & {picked["best"], picked["latest"]})}
    except Exception as exc:
      return key, {"error": str(exc)}

  # Curves come over ssh (a few seconds each), so plan the runs in parallel.
  with ThreadPoolExecutor(max_workers=6) as pool:
    return dict(pool.map(plan, keys))


def rsync_down(key: str, subdir: str, patterns: list[str]) -> None:
  """Copy the files of <run>/<subdir> matching the patterns from the cluster."""
  target = LOCAL_ROOT / key / subdir
  target.mkdir(parents=True, exist_ok=True)
  source = f"{REMOTE}:{REMOTE_PROJECT}/{key}/{subdir}/"
  filters = [f"--include={p}" for p in patterns] + ["--exclude=*"]
  result = subprocess.run(
    ["rsync", "-a", "-e", shlex.join(["ssh", *SSH_OPTS]), *filters, source, f"{target}/"],
    capture_output=True,
    text=True,
    timeout=300,
  )
  if result.returncode not in (0, 23):  # 23: some requested files do not exist
    raise RuntimeError(result.stderr.strip())


def fetch_videos(key: str, iterations: list[int] | None = None) -> list[int]:
  """Copy rendered videos of a run from the cluster (all of them by default)."""
  if iterations is not None:
    patterns = [f"model_{i}.mp4" for i in iterations]
  else:
    patterns = ["model_*.mp4"]
  rsync_down(key, "videos/viewer", patterns)
  return local_videos(key)


# --------------------------------------------------------------------------- jobs


def request_render(key: str, iterations: list[int], episodes: int) -> dict:
  run = find_run(key)
  if not run.get("task"):
    raise ValueError("this run has no W&B metadata, so its task id is unknown")
  missing = set(iterations) - set(run["checkpoints"])
  if missing:
    raise ValueError(f"checkpoints not on the cluster: {sorted(missing)}")
  with lock:
    busy = {
      i
      for job in state["jobs"]
      if job["key"] == key and job["status"] in ("SUBMITTING", "PENDING", "RUNNING")
      for i in job["iterations"]
    }
    wanted = set(iterations) - busy
    if not wanted:
      return {"queued": [], "already": sorted(busy & set(iterations))}
    entry = pending.setdefault(key, {"iterations": set(), "episodes": episodes})
    entry["iterations"] |= wanted
    entry["episodes"] = max(entry["episodes"], episodes)
    if entry.get("timer"):
      entry["timer"].cancel()
    entry["timer"] = threading.Timer(BATCH_DELAY, submit_pending, args=(key,))
    entry["timer"].daemon = True
    entry["timer"].start()
  return {"queued": sorted(wanted), "already": sorted(busy & set(iterations))}


def submit_pending(key: str) -> None:
  with lock:
    entry = pending.pop(key, None)
  if not entry:
    return
  run = find_run(key)
  iterations = sorted(entry["iterations"])
  job = {
    "id": None,
    "key": key,
    "name": run["name"],
    "iterations": iterations,
    "episodes": entry["episodes"],
    "status": "SUBMITTING",
    "submitted": time.time(),
    "log": "",
  }
  with lock:
    state["jobs"].append(job)
  try:
    ssh(
      f"mkdir -p {REMOTE_TOOL_DIR}/slurm && cat > {REMOTE_TOOL_DIR}/render_video.py",
      stdin=(HERE / "render_video.py").read_text(),
    )
    render = shlex.join(
      [
        "python",
        f"{REMOTE_TOOL_DIR}/render_video.py",
        run["task"],
        f"{REMOTE_PROJECT}/{key}",
        "--checkpoints",
        *map(str, iterations),
        "--episodes",
        str(entry["episodes"]),
      ]
    )
    wrap = (
      f"cd {REMOTE_PROJECT} && export PATH={REMOTE_PROJECT}/.venv/bin:$PATH "
      f"MUJOCO_GL=egl PYTHONUNBUFFERED=1 && {render}"
    )
    out = ssh(
      shlex.join(
        [
          "sbatch",
          "--parsable",
          f"--job-name=render-{run.get('wandb_id') or 'run'}",
          f"--output={REMOTE_TOOL_DIR}/slurm/%j.out",
          *SBATCH_OPTS,
          f"--wrap={wrap}",
        ]
      )
    )
    with lock:
      job["id"] = out.strip().split(";")[0]
      job["status"] = "PENDING"
    log(f"submitted job {job['id']}: {key} {iterations}")
    if local_after >= 0:
      # Check this job right at the deadline instead of waiting for the next poll.
      timer = threading.Timer(local_after + 1, check_pending, args=(job,))
      timer.daemon = True
      timer.start()
  except Exception as exc:
    with lock:
      job["status"] = "FAILED"
      job["log"] = f"submission failed: {exc}"
    log(f"submission failed for {key}: {exc}")
  with lock:
    save_state()


def poll_jobs() -> None:
  while True:
    time.sleep(POLL_INTERVAL)
    try:
      poll_once()
    except Exception:
      traceback.print_exc()


def poll_once() -> None:
  with lock:
    live = [
      j
      for j in state["jobs"]
      if j["id"] and not j.get("local") and j["status"] in ("PENDING", "RUNNING")
    ]
  if not live:
    return
  ids = ",".join(j["id"] for j in live)
  queue = ssh(f"squeue -h -j {ids} -o '%i %T %R' 2>/dev/null || true")
  queued = {}
  for line in queue.splitlines():
    parts = line.split(None, 2)
    if len(parts) >= 2:
      queued[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
  logs = ssh(
    " ; ".join(
      f"echo '@@{j['id']}'; tail -n 12 {REMOTE_TOOL_DIR}/slurm/{j['id']}.out 2>/dev/null || true"
      for j in live
    )
  )
  tails = dict(re.findall(r"@@(\d+)\n(.*?)(?=@@\d+\n|\Z)", logs, re.S))
  finished = [j for j in live if j["id"] not in queued]
  final = {}
  if finished:
    acct = ssh(
      f"sacct -n -X -P -o JobID,State -j {','.join(j['id'] for j in finished)} || true"
    )
    final = dict(line.split("|", 1) for line in acct.splitlines() if "|" in line)
  for job in live:
    with lock:
      if job.get("local"):  # diverted by check_pending while this poll ran
        continue
      job["log"] = tails.get(job["id"], job["log"]).strip()
      if job["id"] in queued:
        job["status"], job["reason"] = queued[job["id"]]
        waited = time.time() - job["submitted"]
        if job["status"] == "PENDING" and local_after >= 0 and waited > local_after:
          divert_to_local(job)
        continue
      job["status"] = (final.get(job["id"]) or "COMPLETED").split()[0]
      job["finished"] = time.time()
    try:
      fetch_videos(job["key"], job["iterations"])
    except Exception as exc:
      log(f"fetch after job {job['id']} failed: {exc}")
    with lock:
      for run in index["runs"]:
        if run["key"] == job["key"]:
          run["videos"] = sorted(set(run["videos"]) | set(local_videos(job["key"])))
    log(f"job {job['id']} {job['status']}")
  with lock:
    # Keep the list short: drop finished jobs older than a day.
    cutoff = time.time() - 86400
    state["jobs"] = [j for j in state["jobs"] if j.get("finished", time.time()) > cutoff]
    save_state()


def check_pending(job: dict) -> None:
  """Render a job locally if it is still waiting in the cluster queue."""
  try:
    queue = ssh(f"squeue -h -j {job['id']} -o '%T %R' 2>/dev/null || true").split(None, 1)
  except Exception as exc:
    log(f"queue check of job {job['id']} failed: {exc}")
    return
  with lock:
    if queue and queue[0] == "PENDING" and job["status"] == "PENDING" and not job.get("local"):
      job["reason"] = queue[1].strip() if len(queue) > 1 else ""
      divert_to_local(job)
      save_state()


def divert_to_local(job: dict) -> None:
  """Cancel a job stuck in the cluster queue and render it on this machine. Holds lock."""
  try:
    ssh(f"scancel {job['id']}")
  except Exception as exc:
    log(f"scancel {job['id']} failed, keeping it queued: {exc}")
    return
  log(f"job {job['id']} pending for {time.time() - job['submitted']:.0f}s, rendering locally")
  job["id"] = f"local-{job['id']}"
  job["local"] = True
  job["reason"] = "waiting for the local GPU"
  job["log"] = "cluster queue full, rendering locally"
  threading.Thread(target=render_locally, args=(job,), daemon=True).start()


def render_locally(job: dict) -> None:
  key, iterations = job["key"], job["iterations"]
  tail: collections.deque[str] = collections.deque(maxlen=12)
  try:
    with local_gpu:
      with lock:
        job["status"], job["reason"] = "RUNNING", ""
        job["log"] = "copying checkpoints from the cluster"
      rsync_down(key, ".", [f"model_{i}.pt" for i in iterations])
      LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
      command = [
        str(LOCAL_PYTHON),
        str(HERE / "render_video.py"),
        find_run(key)["task"],
        str(LOCAL_ROOT / key),
        "--checkpoints",
        *map(str, iterations),
        "--episodes",
        str(job["episodes"]),
      ]
      env = dict(os.environ, MUJOCO_GL="egl", PYTHONUNBUFFERED="1")
      with (LOCAL_LOG_DIR / f"{job['id']}.out").open("w") as out:
        proc = subprocess.Popen(
          command,
          cwd=PROJECT,
          env=env,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
        )
        for line in proc.stdout:
          out.write(line)
          tail.append(line)
          with lock:
            job["log"] = "".join(tail).strip()
        proc.wait()
      status = "COMPLETED" if proc.returncode == 0 else "FAILED"
  except Exception as exc:
    tail.append(f"local render failed: {exc}")
    status = "FAILED"
  with lock:
    job["status"] = status
    job["log"] = "".join(tail).strip()
    job["finished"] = time.time()
    for run in index["runs"]:
      if run["key"] == key:
        run["videos"] = sorted(set(run["videos"]) | set(local_videos(key)))
    save_state()
  log(f"job {job['id']} {status}")


# --------------------------------------------------------------------------- http


class Handler(BaseHTTPRequestHandler):
  def log_message(self, fmt, *args):  # quiet: only errors
    pass

  def send_json(self, payload, status=HTTPStatus.OK):
    body = json.dumps(payload).encode()
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(body)

  def body(self) -> dict:
    length = int(self.headers.get("Content-Length") or 0)
    return json.loads(self.rfile.read(length) or b"{}")

  def do_GET(self):
    url = urlparse(self.path)
    query = {k: v[0] for k, v in parse_qs(url.query).items()}
    try:
      if url.path in ("/", "/index.html"):
        body = (HERE / "index.html").read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
      elif url.path == "/api/index":
        if query.get("refresh") == "1" or index["fetched"] is None:
          refresh_index()
        self.send_json(index_payload())
      elif url.path == "/api/curves":
        self.send_json(curves(query["key"], query.get("refresh") == "1"))
      elif url.path == "/api/jobs":
        with lock:
          jobs = list(state["jobs"])
          queued = {k: sorted(v["iterations"]) for k, v in pending.items()}
        self.send_json({"jobs": jobs, "batching": queued})
      elif url.path.startswith("/video/"):
        self.send_video(unquote(url.path[len("/video/") :]))
      else:
        self.send_error(HTTPStatus.NOT_FOUND)
    except Exception as exc:
      traceback.print_exc()
      self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

  def do_POST(self):
    url = urlparse(self.path)
    try:
      data = self.body()
      if url.path == "/api/render":
        result = request_render(
          data["key"], [int(i) for i in data["iterations"]], int(data.get("episodes", 1))
        )
        self.send_json(result)
      elif url.path == "/api/render_best_latest":
        keys = [k for k in data["keys"] if valid_key(k)]
        self.send_json(
          render_best_and_latest(
            keys, data.get("tag"), float(data.get("smooth", 100)), int(data.get("episodes", 1))
          )
        )
      elif url.path == "/api/fetch":
        if not valid_key(data["key"]):
          raise KeyError(data["key"])
        its = data.get("iterations")
        self.send_json({"local_videos": fetch_videos(data["key"], its)})
      elif url.path == "/api/state":
        with lock:
          for field in ("labels", "marks"):
            for key, value in (data.get(field) or {}).items():
              if value in (None, "", {}):
                state[field].pop(key, None)
              else:
                state[field][key] = value
          if "starred" in data:
            state["starred"] = list(data["starred"])
          save_state()
        self.send_json({"ok": True})
      else:
        self.send_error(HTTPStatus.NOT_FOUND)
    except (KeyError, ValueError) as exc:
      self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    except Exception as exc:
      traceback.print_exc()
      self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

  def send_video(self, rel: str):
    # rel = <run key>/<iteration>
    key, _, iteration = rel.rpartition("/")
    if not valid_key(key) or not iteration.isdigit():
      self.send_error(HTTPStatus.NOT_FOUND)
      return
    path = LOCAL_ROOT / key / "videos" / "viewer" / f"model_{iteration}.mp4"
    if not path.is_file():
      self.send_error(HTTPStatus.NOT_FOUND)
      return
    size = path.stat().st_size
    start, end = 0, size - 1
    match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
    if match:
      if match.group(1):
        start = int(match.group(1))
        if match.group(2):
          end = min(int(match.group(2)), size - 1)
      elif match.group(2):
        start = max(0, size - int(match.group(2)))
      self.send_response(HTTPStatus.PARTIAL_CONTENT)
      self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    else:
      self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", "video/mp4")
    self.send_header("Accept-Ranges", "bytes")
    self.send_header("Content-Length", str(end - start + 1))
    self.send_header("Cache-Control", "max-age=3600")
    self.end_headers()
    with path.open("rb") as f:
      f.seek(start)
      remaining = end - start + 1
      try:
        while remaining > 0:
          chunk = f.read(min(1 << 20, remaining))
          if not chunk:
            break
          self.wfile.write(chunk)
          remaining -= len(chunk)
      except (BrokenPipeError, ConnectionResetError):
        pass


def main() -> None:
  parser = argparse.ArgumentParser(description="Browse cluster runs and checkpoint videos.")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument(
    "--local-after",
    type=float,
    default=local_after,
    help="render locally once a job has been pending this many seconds (-1: never)",
  )
  args = parser.parse_args()
  globals()["local_after"] = args.local_after

  load_state()
  # Jobs submitted by an earlier server session keep being tracked.
  with lock:
    for job in state["jobs"]:
      if job["status"] == "SUBMITTING":
        job["status"] = "FAILED"
        job["log"] = "server stopped during submission"
      elif job.get("local") and job["status"] in ("PENDING", "RUNNING"):
        job["status"] = "FAILED"
        job["log"] = "server stopped during the local render"
  log("indexing cluster runs ...")
  refresh_index()
  log(f"{len(index['runs'])} runs")
  threading.Thread(target=poll_jobs, daemon=True).start()
  server = ThreadingHTTPServer((args.host, args.port), Handler)
  log(f"open http://{args.host}:{args.port}")
  server.serve_forever()


if __name__ == "__main__":
  main()
