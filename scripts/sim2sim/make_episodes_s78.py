"""Episode spec for S7-8's WC-ABS-c in PAL's simulator: cubes stratified over the whole wide area
(Claudius's, x 0.40-0.60, y -0.04..0.26 in the env frame), a goal in the same area at least 10 cm
from the cube (what `TARGET=WC` draws), yaw uniform over a cube's symmetry (+-45 deg).

  python3 make_episodes_s78.py > episodes_s78_wc.json

The format is `episodes_s7.json`'s: x, y, yaw in the env frame for the MuJoCo reference; dx, dy, dyaw
from the spawn centre and goal_dx, goal_dy from the goal centre for the node (here both the area's centre).
"""

import json
import math
import random

X, Y = (0.40, 0.60), (-0.04, 0.26)
CENTRE = ((X[0] + X[1]) / 2, (Y[0] + Y[1]) / 2)
NX, NY = 4, 3
MIN_SEP = 0.10
MARGIN = 0.02  # keep the cube / goal this far inside the area's edge, as a tracked cube would be

rng = random.Random(0)
cells = [(i, j) for i in range(NX) for j in range(NY)]
rng.shuffle(cells)
episodes = []
for k, (i, j) in enumerate(cells):
  x0 = X[0] + MARGIN + (X[1] - X[0] - 2 * MARGIN) * i / NX
  x1 = X[0] + MARGIN + (X[1] - X[0] - 2 * MARGIN) * (i + 1) / NX
  y0 = Y[0] + MARGIN + (Y[1] - Y[0] - 2 * MARGIN) * j / NY
  y1 = Y[0] + MARGIN + (Y[1] - Y[0] - 2 * MARGIN) * (j + 1) / NY
  x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
  yaw = rng.uniform(-math.pi / 4, math.pi / 4)
  while True:
    gx = rng.uniform(X[0] + MARGIN, X[1] - MARGIN)
    gy = rng.uniform(Y[0] + MARGIN, Y[1] - MARGIN)
    if math.hypot(gx - x, gy - y) >= MIN_SEP + 0.02:
      break
  episodes.append(
    {
      "id": k,
      "x": round(x, 4),
      "y": round(y, 4),
      "yaw": round(yaw, 4),
      "dx": round(x - CENTRE[0], 4),
      "dy": round(y - CENTRE[1], 4),
      "dyaw": round(yaw, 4),
      "goal_dx": round(gx - CENTRE[0], 4),
      "goal_dy": round(gy - CENTRE[1], 4),
      "cube_goal_distance": round(math.hypot(gx - x, gy - y), 4),
    }
  )
print(
  json.dumps(
    {
      "description": "Episode specs for S7-8 WC-ABS-c: a 4 x 3 stratified draw of the cube over the whole "
      "wide area (env frame x 0.40-0.60, y -0.04..0.26, yaw +-pi/4), the goal uniform in the same area "
      "at least 12 cm from the cube (>= 10 cm is the training's own bound), both 2 cm inside the edge. dx, dy, "
      "dyaw, goal_dx, goal_dy are from the area's centre (0.50, 0.11) for the node. Robot starts exactly at its "
      "default pose, table nominal. make_episodes_s78.py, seed 0.",
      "spawn_center_env_xy": list(CENTRE),
      "goal_center_env_xy": list(CENTRE),
      "episodes": episodes,
    },
    indent=1,
  )
)
