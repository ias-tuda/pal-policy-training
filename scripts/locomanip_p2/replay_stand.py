"""P2a: replay scripted pick-place arm targets (G20, left arm + jaw) on the locomanip
policy while it stands. Each env loops one of the recorded episodes with a random phase."""

import argparse
import sys
import types

import numpy as np
import torch

sys.path.insert(0, "scripts")
import locomanip_eval as ev

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--rec", required=True)
p.add_argument("--speed", type=float, default=1.0)
p.add_argument("--num-envs", type=int, default=64)
p.add_argument("--duration-s", type=float, default=30.0)
p.add_argument("--device", default="cuda:0")
p.add_argument("--video", default=None)
a = p.parse_args()
ev.TASK_ID = a.task
env = ev._make(a.num_envs, a.device, arm_hold=True, seed=0)
policy = ev._policy(env, a.checkpoint, a.device)
raw = env.unwrapped
robot = raw.scene["robot"]
cmd = raw.command_manager.get_term("twist")
arms = raw.action_manager.get_term("arms")
d = np.load(a.rec)
rn = list(d["joint_names"])
T = d["targets"]  # (steps, eps, joints)
names = arms._target_names
center = arms._center[0].cpu().numpy()
cols = [
  rn.index(n) if n.startswith(("arm_left", "gripper_left")) else -1 for n in names
]
eps = []


def ramp(a0, b0, k):
  return [
    a0 + (b0 - a0) * (s**3 * (10 - 15 * s + 6 * s * s)) for s in np.linspace(0, 1, k)
  ]


for e in range(T.shape[1]):
  tr = np.tile(center, (T.shape[0], 1))
  for j, c in enumerate(cols):
    if c >= 0:
      tr[:, j] = T[:, e, c]
  idx = np.arange(0, len(tr), a.speed)
  tr = tr[idx.astype(int)]
  seq = ramp(center, tr[0], 50) + list(tr) + ramp(tr[-1], center, 50) + [center] * 50
  eps.append(np.stack(seq))
L = max(len(x) for x in eps)
eps = [np.concatenate([x, np.tile(center, (L - len(x), 1))]) for x in eps]
traj = torch.tensor(np.stack(eps), dtype=torch.float32, device=a.device)  # (E, L, n)
n = a.num_envs
ep_of = torch.arange(n, device=a.device) % traj.shape[0]
k = torch.randint(0, L, (n,), device=a.device)


def process_actions(self, actions):
  global k
  self._target = traj[ep_of, k]
  self._processed_actions = self._target
  k = (k + 1) % L


arms.process_actions = types.MethodType(process_actions, arms)
obs = env.get_observations()
start = robot.data.root_link_pos_w[:, :2].clone()
failed = torch.zeros(n, dtype=torch.bool, device=a.device)
peak = torch.zeros(n, device=a.device)
frames = []
with torch.no_grad():
  for _ in range(int(a.duration_s / raw.step_dt)):
    cmd.vel_command_b[:] = 0.0
    obs, _, dones, _ = env.step(policy(obs))
    failed |= dones.bool()
    peak = torch.maximum(peak, (robot.data.root_link_pos_w[:, :2] - start).norm(dim=-1))
drift = (robot.data.root_link_pos_w[:, :2] - start).norm(dim=-1)
ok = ~failed
print(
  f"speed x{a.speed}: upright {ok.float().mean().item():.3f}  drift end mean {100 * drift[ok].mean().item():.1f} cm max {100 * drift[ok].max().item():.1f} cm  peak max {100 * peak[ok].max().item():.1f} cm  (episode {L * raw.step_dt:.1f} s, {a.duration_s:.0f} s)"
)
