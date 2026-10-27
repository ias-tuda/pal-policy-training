"""Kangaroo mug task registration, all on the standing grasp recipe and from scratch."""

import itertools

from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.tasks.manipulation.grasp.kangaroo import recipe_ppo_runner_cfg
from pal_mjlab.tasks.manipulation.mug.mug_env_cfg import (
  RECEIVER_GRASP_REWARD,
  mug_handover_env_cfg,
  mug_turn_env_cfg,
  mug_upright_env_cfg,
)
from pal_mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner

#: Iterations each trains for: more than the grasp's 5000, as each is the grasp and then
#: some, and the handover most, as it is two grasps.
MUG_UPRIGHT_ITERATIONS = 8_000
MUG_TURN_ITERATIONS = 8_000
MUG_HANDOVER_ITERATIONS = 10_000

#: One arm; the waist and both arms (``-Bimanual``); both arms with the waist held stiff
#: (``-BimanualArms``). ``-Lower`` pays for bringing the upright mug down to the table.
for (_upper_body, _bimanual, _scope), _lower in itertools.product(
  (
    (False, False, ""),
    (True, False, "-Bimanual"),
    (False, True, "-BimanualArms"),
  ),
  (False, True),
):
  _kwargs = {"upper_body": _upper_body, "bimanual": _bimanual, "lower": _lower}
  register_mjlab_task(
    task_id=f"Mjlab-MugUpright-Standing{_scope}{'-Lower' if _lower else ''}"
    "-Pal-Kangaroo-Grippers",
    env_cfg=mug_upright_env_cfg(**_kwargs),
    play_env_cfg=mug_upright_env_cfg(play=True, **_kwargs),
    rl_cfg=recipe_ppo_runner_cfg("kangaroo_mug_upright", MUG_UPRIGHT_ITERATIONS),
    runner_cls=ManipulationOnPolicyRunner,
  )

register_mjlab_task(
  task_id="Mjlab-MugTurn-Standing-Pal-Kangaroo-Grippers",
  env_cfg=mug_turn_env_cfg(),
  play_env_cfg=mug_turn_env_cfg(play=True),
  rl_cfg=recipe_ppo_runner_cfg("kangaroo_mug_turn", MUG_TURN_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)


def _settled_ppo_runner_cfg(experiment_name: str, max_iterations: int):
  """The recipe runner without the entropy bonus and with a lower noise cap.

  The turn's entropy bonus drove the action std to the recipe's 0.5 cap, by iteration
  4000, and at that noise the turn-and-release was lost again. Meant for fine-tuning a
  checkpoint that already turns.
  """
  cfg = recipe_ppo_runner_cfg(experiment_name, max_iterations)
  cfg.algorithm.entropy_coef = 0.0
  cfg.actor.distribution_cfg = {
    **cfg.actor.distribution_cfg,
    "std_range": (1.0e-3, 0.4),
  }
  return cfg


#: ``-Release`` pays the release in steps (``TURN_RELEASING_REWARD``) and trains without
#: the entropy bonus, to fine-tune a turning checkpoint.
register_mjlab_task(
  task_id="Mjlab-MugTurn-Standing-Release-Pal-Kangaroo-Grippers",
  env_cfg=mug_turn_env_cfg(release=True),
  play_env_cfg=mug_turn_env_cfg(play=True, release=True),
  rl_cfg=_settled_ppo_runner_cfg("kangaroo_mug_turn", MUG_TURN_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: ``-Release-Regrasp`` keeps paying the turn made in the jaw after it lets go, so the mug
#: can be turned in several grasps, on top of ``-Release``. The settled runner, both to
#: fine-tune and from scratch: with the entropy bonus the std ran to its cap and the turn
#: was lost.
register_mjlab_task(
  task_id="Mjlab-MugTurn-Standing-Release-Regrasp-Pal-Kangaroo-Grippers",
  env_cfg=mug_turn_env_cfg(release=True, regrasp=True),
  play_env_cfg=mug_turn_env_cfg(play=True, release=True, regrasp=True),
  rl_cfg=_settled_ppo_runner_cfg("kangaroo_mug_turn", MUG_TURN_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: MugUpright ``-Lower-Release``: the jaw letting go of the stood mug pays, holding it
#: there stops paying after a second, and the release's rocking no longer undoes the
#: set-down. One arm, on the settled runner like the turn's ``-Release``: the ``-Lower``
#: runs' action std sat at the recipe's 0.5 cap.
register_mjlab_task(
  task_id="Mjlab-MugUpright-Standing-Lower-Release-Pal-Kangaroo-Grippers",
  env_cfg=mug_upright_env_cfg(lower=True, release=True),
  play_env_cfg=mug_upright_env_cfg(play=True, lower=True, release=True),
  rl_cfg=_settled_ppo_runner_cfg("kangaroo_mug_upright", MUG_UPRIGHT_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: ``-HandsOff``: the turned mug and the release's withdrawal pay only with the hand off
#: the mug; ``-Near`` spawns it 30-70 deg off, what one grasp turns. Settled runner, to
#: fine-tune ``-Release-Regrasp``.
for _near in (False, True):
  _kwargs = {"release": True, "regrasp": True, "hands_off": True, "near": _near}
  register_mjlab_task(
    task_id="Mjlab-MugTurn-Standing-Release-Regrasp"
    f"{'-Near' if _near else ''}-HandsOff-Pal-Kangaroo-Grippers",
    env_cfg=mug_turn_env_cfg(**_kwargs),
    play_env_cfg=mug_turn_env_cfg(play=True, **_kwargs),
    rl_cfg=_settled_ppo_runner_cfg("kangaroo_mug_turn", MUG_TURN_ITERATIONS),
    runner_cls=ManipulationOnPolicyRunner,
  )

#: MugUpright ``-Lower-Release-Overhand``: holding the mug upright and lowering it pay
#: only with the fingers level or down, from where it can be set down.
register_mjlab_task(
  task_id="Mjlab-MugUpright-Standing-Lower-Release-Overhand-Pal-Kangaroo-Grippers",
  env_cfg=mug_upright_env_cfg(lower=True, release=True, overhand=True),
  play_env_cfg=mug_upright_env_cfg(play=True, lower=True, release=True, overhand=True),
  rl_cfg=_settled_ppo_runner_cfg("kangaroo_mug_upright", MUG_UPRIGHT_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: ``-Overhand-Facing``: the lift gated on the fingers too, and the lying mug's top toward
#: the robot, the headings the left arm stands it up from fingers down.
register_mjlab_task(
  task_id="Mjlab-MugUpright-Standing-Lower-Release-Overhand-Facing-Pal-Kangaroo-Grippers",
  env_cfg=mug_upright_env_cfg(lower=True, release=True, overhand=True, facing=True),
  play_env_cfg=mug_upright_env_cfg(
    play=True, lower=True, release=True, overhand=True, facing=True
  ),
  rl_cfg=_settled_ppo_runner_cfg("kangaroo_mug_upright", MUG_UPRIGHT_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: ``-Overhand-Facing-View``: ``-Facing`` with the left arm kept off the torso camera's
#: line of sight to the mug.
register_mjlab_task(
  task_id="Mjlab-MugUpright-Standing-Lower-Release-Overhand-Facing-View-Pal-Kangaroo-Grippers",
  env_cfg=mug_upright_env_cfg(
    lower=True, release=True, overhand=True, facing=True, view=True
  ),
  play_env_cfg=mug_upright_env_cfg(
    play=True, lower=True, release=True, overhand=True, facing=True, view=True
  ),
  rl_cfg=_settled_ppo_runner_cfg("kangaroo_mug_upright", MUG_UPRIGHT_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: ``-Overhand-Facing-Still-Blind``: ``-Facing`` with the arm held still once done, as
#: the table837 fine-tune trained, and no mug pose after the grasp.
register_mjlab_task(
  task_id="Mjlab-MugUpright-Standing-Lower-Release-Overhand-Facing-Still-Blind-Pal-Kangaroo-Grippers",
  env_cfg=mug_upright_env_cfg(
    lower=True, release=True, overhand=True, facing=True, still=True, blind=True
  ),
  play_env_cfg=mug_upright_env_cfg(
    play=True,
    lower=True,
    release=True,
    overhand=True,
    facing=True,
    still=True,
    blind=True,
  ),
  rl_cfg=_settled_ppo_runner_cfg("kangaroo_mug_upright", MUG_UPRIGHT_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: ``-Overhand-Away``: ``-Facing`` with the lying mug's top pointing away from the robot,
#: so it turns up toward the body.
register_mjlab_task(
  task_id="Mjlab-MugUpright-Standing-Lower-Release-Overhand-Away-Pal-Kangaroo-Grippers",
  env_cfg=mug_upright_env_cfg(lower=True, release=True, overhand=True, away=True),
  play_env_cfg=mug_upright_env_cfg(
    play=True, lower=True, release=True, overhand=True, away=True
  ),
  rl_cfg=_settled_ppo_runner_cfg("kangaroo_mug_upright", MUG_UPRIGHT_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: The settled runner's std floored at ``STAGE_MIN_STD``: from scratch without the
#: entropy bonus the std fell to ~0.1 by iteration 4000, before the upright policies had
#: once set the mug down.
STAGE_MIN_STD = 0.1


def _stage_ppo_runner_cfg(experiment_name: str, max_iterations: int):
  cfg = _settled_ppo_runner_cfg(experiment_name, max_iterations)
  cfg.actor.distribution_cfg = {
    **cfg.actor.distribution_cfg,
    "std_range": (STAGE_MIN_STD, cfg.actor.distribution_cfg["std_range"][1]),
  }
  return cfg


#: ``-Overhand-Facing-View-Scratch``: ``-Facing-View`` to train from scratch, on the
#: settled runner with the std floored at ``STAGE_MIN_STD``; the settled runner's own
#: 1e-3 floor is for fine-tuning a policy that already stands the mug up.
register_mjlab_task(
  task_id="Mjlab-MugUpright-Standing-Lower-Release-Overhand-Facing-View-Scratch-Pal-Kangaroo-Grippers",
  env_cfg=mug_upright_env_cfg(
    lower=True, release=True, overhand=True, facing=True, view=True
  ),
  play_env_cfg=mug_upright_env_cfg(
    play=True, lower=True, release=True, overhand=True, facing=True, view=True
  ),
  rl_cfg=_stage_ppo_runner_cfg("kangaroo_mug_upright", MUG_UPRIGHT_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: ``-Stage``: the rungs above the grasp as one progress shaping term, the end state the
#: only per-step reward above the grasp ladder (``mug_upright_progress``,
#: ``mug_turn_progress``); for training from scratch.
register_mjlab_task(
  task_id="Mjlab-MugUpright-Standing-Stage-Pal-Kangaroo-Grippers",
  env_cfg=mug_upright_env_cfg(stage=True),
  play_env_cfg=mug_upright_env_cfg(play=True, stage=True),
  rl_cfg=_stage_ppo_runner_cfg("kangaroo_mug_upright", MUG_UPRIGHT_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MugTurn-Standing-Stage-Pal-Kangaroo-Grippers",
  env_cfg=mug_turn_env_cfg(stage=True),
  play_env_cfg=mug_turn_env_cfg(play=True, stage=True),
  rl_cfg=_stage_ppo_runner_cfg("kangaroo_mug_turn", MUG_TURN_ITERATIONS),
  runner_cls=ManipulationOnPolicyRunner,
)

#: MugHandover and its reward variants, every combination: ``-Grasp10`` pays the
#: receiver's grasp 10 rather than 30, ``-AlignedRelease`` pays the giver's release only
#: as far as the receiver's jaw sits squarely on the mug, ``-BodyOnly`` rules out the
#: handle.
for _grasp10 in (False, True):
  for _aligned in (False, True):
    for _body_only in (False, True):
      _kwargs = {
        "receiver_grasp_reward": 10.0 if _grasp10 else RECEIVER_GRASP_REWARD,
        "aligned_release": _aligned,
        "handle": not _body_only,
      }
      _suffix = (
        ("-Grasp10" if _grasp10 else "")
        + ("-AlignedRelease" if _aligned else "")
        + ("-BodyOnly" if _body_only else "")
      )
      register_mjlab_task(
        task_id=f"Mjlab-MugHandover-Standing{_suffix}-Pal-Kangaroo-Grippers",
        env_cfg=mug_handover_env_cfg(**_kwargs),
        play_env_cfg=mug_handover_env_cfg(play=True, **_kwargs),
        rl_cfg=recipe_ppo_runner_cfg("kangaroo_mug_handover", MUG_HANDOVER_ITERATIONS),
        runner_cls=ManipulationOnPolicyRunner,
      )

#: ``-Handle``: the handover simplified. The mug spawns with its handle toward the robot;
#: the left hand takes the handle, the right the body. The receiver's grasp pays 10, as
#: in ``-Grasp10``, the best of the variants.
#: ``-ReleaseDelta``: the release pays for opening the jaw, not for holding it open; the
#: ``-Handle`` policies learned to open it and stay hooked through the handle.
#: ``-Upright``: a held mug tipping no longer ends the episode; the lift and the
#: handed-over reward fall to 0 with its tilt instead.
#: ``-Centre``: the receiver is drawn to the centre of the mug body; it took it high,
#: beside the giver's hand.
#: ``-FreeTilt``: as ``-Upright``, without the tilt scaling; the mug is empty.
for _release_delta, _upright, _centre, _free_tilt in (
  (False, False, False, False),
  (True, False, False, False),
  (True, True, False, False),
  (True, False, True, False),
  (True, True, True, False),
  (True, False, True, True),
):
  _kwargs = {
    "receiver_grasp_reward": 10.0,
    "giver_handle": True,
    "release_delta": _release_delta,
    "upright": _upright,
    "receiver_centre": _centre,
    "free_tilt": _free_tilt,
  }
  _suffix = (
    ("-ReleaseDelta" if _release_delta else "")
    + ("-Upright" if _upright else "")
    + ("-FreeTilt" if _free_tilt else "")
    + ("-Centre" if _centre else "")
  )
  register_mjlab_task(
    task_id=f"Mjlab-MugHandover-Standing-Handle{_suffix}-Pal-Kangaroo-Grippers",
    env_cfg=mug_handover_env_cfg(**_kwargs),
    play_env_cfg=mug_handover_env_cfg(play=True, **_kwargs),
    rl_cfg=recipe_ppo_runner_cfg("kangaroo_mug_handover", MUG_HANDOVER_ITERATIONS),
    runner_cls=ManipulationOnPolicyRunner,
  )
