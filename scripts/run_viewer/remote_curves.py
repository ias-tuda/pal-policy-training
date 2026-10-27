"""Print a run's TensorBoard scalars as JSON, thinned to at most MAX_POINTS per tag.

Usage (project venv, for tensorboard): python - RUN_DIR [MAX_POINTS] < remote_curves.py
"""

import json
import pathlib
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = pathlib.Path(sys.argv[1])
max_points = int(sys.argv[2]) if len(sys.argv) > 2 else 600

curves = {}
for events in sorted(run_dir.glob("events.out.tfevents.*")):
  acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
  acc.Reload()
  for tag in acc.Tags()["scalars"]:
    points = curves.setdefault(tag, {})
    for event in acc.Scalars(tag):
      points[event.step] = event.value

out = {}
for tag, points in curves.items():
  steps = sorted(points)
  # Average consecutive points into bins instead of subsampling, so noise does not
  # masquerade as a peak.
  stride = max(1, -(-len(steps) // max_points))
  out[tag] = [
    [chunk[len(chunk) // 2], round(sum(points[s] for s in chunk) / len(chunk), 5)]
    for chunk in (steps[i : i + stride] for i in range(0, len(steps), stride))
  ]
json.dump(out, sys.stdout)
