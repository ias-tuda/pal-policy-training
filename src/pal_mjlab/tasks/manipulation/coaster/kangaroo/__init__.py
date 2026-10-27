"""Kangaroo mug-on-coaster task registration."""

import itertools

from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.tasks.manipulation.coaster.coaster_env_cfg import (
  cube_stack_release_env_cfg,
  mug_coaster_env_cfg,
)
from pal_mjlab.tasks.manipulation.grasp.kangaroo import recipe_ppo_runner_cfg
from pal_mjlab.tasks.manipulation.mug.kangaroo import _settled_ppo_runner_cfg
from pal_mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner

#: As the mug tasks: the grasp and then some.
MUG_COASTER_ITERATIONS = 8_000

#: "Put the mug on the coaster", from scratch on the standing grasp recipe.
register_mjlab_task(
  task_id="Mjlab-MugCoaster-Standing-Pal-Kangaroo-Grippers",
  env_cfg=mug_coaster_env_cfg(),
  play_env_cfg=mug_coaster_env_cfg(play=True),
  rl_cfg=recipe_ppo_runner_cfg("kangaroo_mug_coaster", MUG_COASTER_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: ``-Release``: the set-down survives the contact's flicker, holding at the goal stops
#: paying a second after it, and the jaw letting go pays. The recipe runner, entropy
#: bonus and all, to train from scratch; ``-Release-Settled`` without it, to fine-tune.
#: ``-Release-Replay``: and 30% of the episodes start from a set-down state the policy
#: reached, so the release is practised as soon as one is.
for (_replay, _suffix), _rl in itertools.product(
  ((False, ""), (True, "-Replay")),
  (("", recipe_ppo_runner_cfg), ("-Settled", _settled_ppo_runner_cfg)),
):
  _rl_suffix, _rl_cfg = _rl
  register_mjlab_task(
    task_id=f"Mjlab-MugCoaster-Standing-Release{_suffix}{_rl_suffix}"
    "-Pal-Kangaroo-Grippers",
    env_cfg=mug_coaster_env_cfg(release=True, replay=_replay),
    play_env_cfg=mug_coaster_env_cfg(play=True, release=True, replay=_replay),
    rl_cfg=_rl_cfg("kangaroo_mug_coaster", MUG_COASTER_ITERATIONS),
    runner_cls=ManipulationOnPolicyRunner,
  )

#: The cube stack with the ``-Release`` ladder (``cube_stack_release_env_cfg``), here as it
#: is built on this task's. The stack's experiment and iterations, so its first-generation
#: runs, which never let go, load as a warm start with ``--agent.resume``; the settled
#: runner, to fine-tune.
register_mjlab_task(
  task_id="Mjlab-Stack-Standing-Place-Release-Replay-Pal-Kangaroo-Grippers",
  env_cfg=cube_stack_release_env_cfg(replay=True),
  play_env_cfg=cube_stack_release_env_cfg(play=True),
  rl_cfg=_settled_ppo_runner_cfg("kangaroo_stack_standing", 5_000),
  runner_cls=ManipulationOnPolicyRunner,
)
