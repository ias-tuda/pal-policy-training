"""Kangaroo Kugel-in-mug task registration."""

from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.tasks.manipulation.grasp.kangaroo import recipe_ppo_runner_cfg
from pal_mjlab.tasks.manipulation.kugel.kugel_env_cfg import mug_kugel_env_cfg
from pal_mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner

#: As the coaster: the grasp and then some.
MUG_KUGEL_ITERATIONS = 8_000

#: "Put the Kugel in the mug", from scratch on the standing grasp recipe.
register_mjlab_task(
  task_id="Mjlab-MugKugel-Standing-Pal-Kangaroo-Grippers",
  env_cfg=mug_kugel_env_cfg(),
  play_env_cfg=mug_kugel_env_cfg(play=True),
  rl_cfg=recipe_ppo_runner_cfg("kangaroo_mug_kugel", MUG_KUGEL_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: ``-HandOff``: the hand touching the mug costs ``KUGEL_HAND_MUG_PENALTY`` a second. Same
#: experiment and observations, so the first runs load with ``--agent.resume``.
register_mjlab_task(
  task_id="Mjlab-MugKugel-Standing-HandOff-Pal-Kangaroo-Grippers",
  env_cfg=mug_kugel_env_cfg(hand_off_mug=True),
  play_env_cfg=mug_kugel_env_cfg(play=True, hand_off_mug=True),
  rl_cfg=recipe_ppo_runner_cfg("kangaroo_mug_kugel", MUG_KUGEL_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: ``-HandOff-Tracker``: and the actor sees the Kugel as the robot's tracker reports it,
#: published every 120-200 ms, 180-280 ms old, lost for up to 2 s
#: (``KUGEL_TRACKER_*``). Fine-tunes the first runs with ``--agent.resume``.
register_mjlab_task(
  task_id="Mjlab-MugKugel-Standing-HandOff-Tracker-Pal-Kangaroo-Grippers",
  env_cfg=mug_kugel_env_cfg(hand_off_mug=True, real_tracker=True),
  play_env_cfg=mug_kugel_env_cfg(play=True, hand_off_mug=True, real_tracker=True),
  rl_cfg=recipe_ppo_runner_cfg("kangaroo_mug_kugel", MUG_KUGEL_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)
