#!/bin/bash
# Pick-and-place v0 training on the IAS cluster.
#
# One job is one seed. A multi-seed campaign is N of these, not one multi-GPU job:
# the resource block below is single-GPU and is user-specified -- do not change it
# without asking (CLAUDE.md, *Job submission -- IAS cluster*).
#
#   sbatch --export=ALL,SEED=1,NUM_ENVS=256,MAX_ITERATIONS=200,RUN_TAG=c0 \
#          --job-name=pp_c0_s1 \
#          --output=$HOME/repo/humanoids/pal_mjlab_kangaroo_talos/slurm_logs/pp_c0_s1_%j.log \
#          --time=04:00:00 \
#          scripts/pick_place.sbatch
#
# The log directory must exist before submitting, and --job-name / --output must be
# passed as CLI flags: #SBATCH lines are parsed before shell expansion, so a ${SEED}
# in one produces a file literally named '${SEED}'.
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G

set -euo pipefail

# A batch step runs a non-login shell, so ~/.profile is never sourced and uv -- which
# installs itself to ~/.local/bin -- is not on PATH. Without this the job dies at exec
# with status 127 after the commit guard has already passed, which is a confusing way to
# find out. (Job 157122 did exactly that.)
export PATH="$HOME/.local/bin:$PATH"

# Overridable so a separate checkout (e.g. the S5.1 alignment's `-align`) can run without
# touching the main one; the commit guard below still pins the code.
REPO="${REPO:-$HOME/repo/humanoids/pal_mjlab_kangaroo_talos}"
TASK=Mjlab-PickPlace-Pal-Kangaroo-Grippers

# The point of the guard is that the job refuses to run rather than silently training
# the wrong code. Bump it deliberately, with a note saying what changed.
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT must be set to the full 40-char SHA}"

SEED="${SEED:?SEED must be set, e.g. sbatch --export=ALL,SEED=1 ...}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
RUN_TAG="${RUN_TAG:-c1}"
# D10: one task id; the arm and the rung are selected by these two variables, which
# `pick_place/kangaroo/env_cfg.py` reads at registration and validates against the names in
# `rewards.py` / the DR ladder. They must be *exported*, not just set: a value supplied via
# --export is already in the environment, but a default assigned here would otherwise be a
# plain shell variable and the training process would silently get R-A/DR0.
REWARD="${REWARD:-R-A}"
DR="${DR:-DR0}"
# The PPO recipe, the same way (S5.1 item 9, `rl_cfg.PPO_RECIPES`). Its default is C1r/C2r's
# recipe, not C1's; PPO=PPO-A gives back what C0-C2 ran.
PPO="${PPO:-PPO-stack}"
# The arm's speed bound, the same way (plan S7): a factor in (0, 1] on the URDF slew limit.
# 0.5 is speed bound A (FA); 1.0 is every run before S7. It goes into the run name and the
# tags, so an FA curve can never be mistaken for an F one.
# Plan S7-8: SLEW=stack is the stack task's speeds (arm 1-3 at 1/16 of the URDF speed, arm 4-7 at 1/8,
# the jaw 0.03 rad per step), in place of a factor.
SLEW="${SLEW:-1.0}"
# Plan S7-2, the same way (`pick_place/kangaroo/env_cfg.py`): BASELINE (S71 | B -- table height,
# episode length, target box, observations, jaw slew), SPAWN (S71 | W1 | W2; a W box brings
# P2's wider action scales) and TOOL_SPEED (none | V, the tool speed bound). The defaults are
# every run up to S7-1. All three go into the run name and the tags.
BASELINE="${BASELINE:-S71}"
SPAWN="${SPAWN:-S71}"
TOOL_SPEED="${TOOL_SPEED:-none}"
# The plant, the same way (plan S8): a gain code from `kangaroo_constants.TABLETOP_GAINS`. G0 is
# every run before S8 (SysID gains, gravity compensation on); G15/G20/G20s are stiff pelvis/arm
# gains with no compensation. It goes into the run name and the tags like the others.
GAINS="${GAINS:-G0}"
# The driven arm's effort bound, the same way (plan S7-4, `pick_place_env_cfg.EFFORT_BOUNDS`):
# none (every run before S7-4) | C (targets clipped to the joint range) | CTP (+ the effort limit
# lowered to 0.7 x effort, mirrored as a target band, + the effort hinge).
EFFORT="${EFFORT:-none}"
# The real robot's latencies, the same way (plan S7-6, `tabletop_env_cfg.LATENCY_MODELS`): none
# (every run before S7-6) | FP (the FoundationPose cube tracker, 5-8 steps on the cube pose) | JS
# (the jaw servo's lag behind its command) | FP+JS. In the run name and tags only when set, so
# every earlier run's name still reads the same.
LAG="${LAG:-none}"
export REWARD DR PPO SLEW BASELINE SPAWN TOOL_SPEED GAINS EFFORT LAG
LAG_NAME=""
LAG_TAG=""
if [ "$LAG" != "none" ]; then
  LAG_NAME="_lag${LAG}"
  LAG_TAG=",\"lag${LAG}\""
fi
# The cube out of the camera's view (plan S7-7, `mdp/occlusion.py`): OCC is the visibility model
# (none | arm | hand), OBJ_OBS what the actor sees while the cube is lost (live | hold | zero |
# estimate | estimate+ | snap); OCC_FLAG=0/1 overrides whether the actor gets the visibility flag.
# Unset is no occlusion, every run to S7-6; in the run name and tags only when set.
OCC="${OCC:-none}"
OBJ_OBS="${OBJ_OBS:-live}"
export OCC OBJ_OBS
[ -n "${OCC_FLAG:-}" ] && export OCC_FLAG
OCC_NAME=""
OCC_TAG=""
if [ "$OCC" != "none" ] || [ "$OBJ_OBS" != "live" ]; then
  OCC_NAME="_occ${OCC}_obj${OBJ_OBS}${OCC_FLAG:+_flag${OCC_FLAG}}"
  OCC_TAG=",\"occ${OCC}\",\"obj${OBJ_OBS}\""
fi

# Plan S7-8 (`pick_place/kangaroo/env_cfg.py`): ACTION (abs | delta -- the action sets or moves the
# commanded target), CMD_OBS (act | cmd -- what the actor's `actions` term shows; unset is act for abs
# and cmd for delta, which must show it), TARGET (B | WC -- the target from B's small box, or from
# the same wide area as the cube, >= 10 cm apart; SPAWN=WC is the cube's half) and EPISODE_S (the
# episode length in s; unset is the baseline's). Unset is S7-7's cfg; in the run name and the tags
# only when set, so every earlier run's name still reads the same.
ACTION="${ACTION:-abs}"
TARGET="${TARGET:-B}"
export ACTION TARGET
[ -n "${CMD_OBS:-}" ] && export CMD_OBS
[ -n "${EPISODE_S:-}" ] && export EPISODE_S
S78_NAME=""
S78_TAG=""
if [ "$ACTION" != "abs" ]; then
  S78_NAME="${S78_NAME}_act${ACTION}"
  S78_TAG="${S78_TAG},\"act${ACTION}\""
fi
if [ -n "${CMD_OBS:-}" ]; then
  S78_NAME="${S78_NAME}_cmdobs${CMD_OBS}"
  S78_TAG="${S78_TAG},\"cmdobs${CMD_OBS}\""
fi
if [ "$TARGET" != "B" ]; then
  S78_NAME="${S78_NAME}_tgt${TARGET}"
  S78_TAG="${S78_TAG},\"tgt${TARGET}\""
fi
if [ -n "${EPISODE_S:-}" ]; then
  S78_NAME="${S78_NAME}_ep${EPISODE_S}"
  S78_TAG="${S78_TAG},\"ep${EPISODE_S}\""
fi

# Plan S8 (`pick_place/kangaroo/env_cfg.py`, `pick_place_env_cfg.OBJECT_SHAPES`): SHAPE is what is picked up --
# cube (every run up to S7-8) | ball (a 6 cm sphere) | ball-dr (the ball with its size and rolling friction
# drawn) | mix (a cube or a ball, 50/50 per episode). In the run name and the tags only when set, so every
# earlier run's name still reads the same.
SHAPE="${SHAPE:-cube}"
export SHAPE
S8_NAME=""
S8_TAG=""
if [ "$SHAPE" != "cube" ]; then
  S8_NAME="_shape${SHAPE}"
  S8_TAG=",\"shape${SHAPE}\""
fi

# Plan S9 (`pick_place/kangaroo/env_cfg.py`, `pick_place_env_cfg.GOALS` / `GATES`): GOAL is where the cube goes --
# table (every run up to S8) | stack (on a blue base cube in B's target box) | box | box-ny | box-rim | box-dr (plan
# S10: into an open box standing in B's target box) -- and GATE what may hold the jaw shut --
# none | stack (the lock: no opening while the held cube is not aligned, low and slow over the base). In the run name
# and the tags only when set, so every earlier run's name still reads the same.
GOAL="${GOAL:-table}"
GATE="${GATE:-none}"
export GOAL GATE
S9_NAME=""
S9_TAG=""
if [ "$GOAL" != "table" ]; then
  S9_NAME="${S9_NAME}_goal${GOAL}"
  S9_TAG="${S9_TAG},\"goal${GOAL}\""
fi
if [ "$GATE" != "none" ]; then
  S9_NAME="${S9_NAME}_gate${GATE}"
  S9_TAG="${S9_TAG},\"gate${GATE}\""
fi

# Plan S7-9 (`pick_place/kangaroo/env_cfg.py`): CRITIC_OBS is what the critic sees -- base (every run up
# to S8) | q (also M6's latched release grade and the release latch, PQ; critic only, and a resume widens
# the loaded critic for them). The other S7-9 codes are REWARD (+PXY, +PT) and TOOL_SPEED (PV). In the run
# name and the tags only when set, so every earlier run's name still reads the same.
CRITIC_OBS="${CRITIC_OBS:-base}"
export CRITIC_OBS
S79_NAME=""
S79_TAG=""
if [ "$CRITIC_OBS" != "base" ]; then
  S79_NAME="_critic${CRITIC_OBS}"
  S79_TAG=",\"critic${CRITIC_OBS}\""
fi
# Plan S10.1 (`pick_place/box.py`): BOX_OBS is what the actor sees of the box -- goal (the goal point
# computed from the box, every S10 / DEP1 run) | pose (the box's position and its yaw as (cos 4psi,
# sin 4psi); a resume widens the loaded actor and critic). The box on the wide area is TARGET=WC with a
# box GOAL. In the run name and the tags only when set.
BOX_OBS="${BOX_OBS:-goal}"
export BOX_OBS
if [ "$BOX_OBS" != "goal" ]; then
  S79_NAME="${S79_NAME}_boxobs${BOX_OBS}"
  S79_TAG="${S79_TAG},\"boxobs${BOX_OBS}\""
fi
# Plan S10.1: CKPT_SPAWN names the spawn whose arm the RESUME checkpoint was trained with when its action
# scales are not this run's (W2 box checkpoints on SPAWN=WC): the loaded actor is converted to this run's
# scales (`runner.rescale_actions`). In the run name and the tags only when set.
if [ -n "${CKPT_SPAWN:-}" ]; then
  export CKPT_SPAWN
  S79_NAME="${S79_NAME}_ckpt${CKPT_SPAWN}"
  S79_TAG="${S79_TAG},\"ckpt${CKPT_SPAWN}\""
fi

# Plan S10.2: TABLE_DZ (mm) raises the table's top face from the baseline's height (22: the robot cell's
# table, 0.837 m on baseline B). In the run name and the tags only when set.
if [ -n "${TABLE_DZ:-}" ]; then
  export TABLE_DZ
  S79_NAME="${S79_NAME}_tabledz${TABLE_DZ}"
  S79_TAG="${S79_TAG},\"tabledz${TABLE_DZ}\""
fi

# Plan S7-10 (`pick_place/arm_table.py`): ARM_TABLE (none | watch | AC | AH | AT | AH10 -- arm 1-6 against
# the table: a sensor only, a contact penalty, + a clearance penalty, or + a termination on a hard
# contact). Unset is every run before S7-10; in the run name and the tags only when set.
[ -n "${ARM_TABLE:-}" ] && export ARM_TABLE
if [ -n "${ARM_TABLE:-}" ]; then
  S78_NAME="${S78_NAME}_arm${ARM_TABLE}"
  S78_TAG="${S78_TAG},\"arm${ARM_TABLE}\""
fi

# Plan S11 (two arms; `pick_place/kangaroo/env_cfg.py`, `pick_place/bimanual.py`): ARM_HOME (default | mirror: the
# right arm's home), TRACKER (B | split: the 6 + 4 mm bias split, the base through its own tracker), HANDOFF_ROUTE
# (none | up-out: the hand-off controller in the action term, for the eval), ARMS (left | both: JOINT, one policy
# for both arms), HANDOFF_TRAIN (none | left) and START (scratch | mixed), JOINT's. TARGET=T11 / T11W is TARGET's
# (above). Exported when set; in the run name and the tags only when set.
S11_NAME=""
S11_TAG=""
for _pair in ARM_HOME:home TRACKER:trk HANDOFF_ROUTE:route ARMS:arms HANDOFF_TRAIN:hot START:start; do
  _var=${_pair%%:*}
  _short=${_pair##*:}
  _val="${!_var:-}"
  if [ -n "$_val" ]; then
    export "${_var?}"
    S11_NAME="${S11_NAME}_${_short}${_val}"
    S11_TAG="${S11_TAG},\"${_short}${_val}\""
  fi
done

# Plan S9.1 (`pick_place/kangaroo/env_cfg.py`): BASE_OBS (goal | pose: SC-Y, the stack base's yaw on the actor and
# the yaw grade against the base) and CONTACT (box | ccd, decision 0004: unset is box-box through mujoco_warp's
# box_box collider; ccd gives back the GJK/EPA path every run before S9.1 trained on). Exported when set; in the
# run name and the tags only when set.
S91_NAME=""
S91_TAG=""
for _pair in BASE_OBS:base CONTACT:contact; do
  _var=${_pair%%:*}
  _short=${_pair##*:}
  _val="${!_var:-}"
  if [ -n "$_val" ]; then
    export "${_var?}"
    S91_NAME="${S91_NAME}_${_short}${_val}"
    S91_TAG="${S91_TAG},\"${_short}${_val}\""
  fi
done
# Plan §13.1 (`pick_place/hold.py`, `pick_place/insert_held.py`): S13 (L-H | L-HP | R-I | R-I-W), the two-arm
# insert's one-arm skills. Exported when set; in the run name and the tags only when set.
if [ -n "${S13:-}" ]; then
  export S13
  S91_NAME="${S91_NAME}_s13${S13}"
  S91_TAG="${S91_TAG},\"s13${S13}\""
fi

WANDB_PROJECT="${WANDB_PROJECT:-oppsie-pick-place-v0}"
EXPERIMENT="${EXPERIMENT:-kangaroo_pick_place}"

# Extra flags for a diagnostic run, e.g. --enable-nan-guard True. Word-split on purpose;
# keep them simple. Whatever goes here is echoed into the log, so a run that was not the
# plain configuration says so in its own provenance.
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Resume (plan S7, "5 k plus a resume if still rising"): RESUME=<path of a model_<iter>.pt>
# continues that run -- policy, optimiser, learning rate, iteration count and the reward
# curriculum's clock -- for MAX_ITERATIONS *more* iterations, in a new run directory and a new
# wandb run tagged "resumed". The checkpoint may belong to another checkout (the one the run
# started from); `pick_place_resume.sh` links its run directory into this one's log root.
# Resume on the code the run started on, or say in its record what changed.
RESUME="${RESUME:-}"

cd "$REPO"
actual_commit="$(git rev-parse HEAD)"
if [ "$actual_commit" != "$EXPECTED_COMMIT" ]; then
  echo "Refusing to run: expected commit $EXPECTED_COMMIT, found $actual_commit" >&2
  exit 1
fi

# Validate the names before the job takes a GPU for the ~4 min warp compile. The authority
# is the registry itself -- this imports the same module `train` will, so there is one list
# of valid names and it cannot drift from this script.
if ! uv run --extra cu128 python -c \
     "from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import kangaroo_pick_place_env_cfg as f, kangaroo_pick_place_rl_cfg as g; f(); g()" \
     >/dev/null 2>"$REPO/.reward_check.$$"; then
  echo "Refusing to run: REWARD=$REWARD DR=$DR PPO=$PPO SLEW=$SLEW BASELINE=$BASELINE" \
    "SPAWN=$SPAWN TOOL_SPEED=$TOOL_SPEED GAINS=$GAINS EFFORT=$EFFORT LAG=$LAG OCC=$OCC" \
    "OBJ_OBS=$OBJ_OBS ACTION=$ACTION CMD_OBS=${CMD_OBS:-(default)} TARGET=$TARGET" \
    "EPISODE_S=${EPISODE_S:-(default)} SHAPE=$SHAPE GOAL=$GOAL GATE=$GATE CRITIC_OBS=$CRITIC_OBS ARM_TABLE=${ARM_TABLE:-(default)} BOX_OBS=$BOX_OBS TABLE_DZ=${TABLE_DZ:-(default)}" \
    "S11:${S11_NAME:-(none)} S9.1:${S91_NAME:-(none)} did not validate." >&2
  tail -3 "$REPO/.reward_check.$$" >&2
  rm -f "$REPO/.reward_check.$$"
  exit 1
fi
rm -f "$REPO/.reward_check.$$"

RESUME_ARGS=()
RESUME_TAG=""
if [ -n "$RESUME" ]; then
  # shellcheck source=pick_place_resume.sh
  source "$REPO/scripts/pick_place_resume.sh"
  RESUME_TAG=',"resumed"'
fi

echo "=== provenance ==="
echo "commit      $actual_commit"
echo "task        $TASK"
echo "seed        $SEED"
echo "num_envs    $NUM_ENVS"
echo "iterations  $MAX_ITERATIONS"
echo "reward/DR   $REWARD / $DR"
echo "ppo         $PPO"
echo "slew        $SLEW"
echo "baseline    $BASELINE   spawn $SPAWN   tool speed $TOOL_SPEED"
echo "gains       $GAINS   effort $EFFORT   lag $LAG   occ $OCC   obj_obs $OBJ_OBS   occ_flag ${OCC_FLAG:-(default)}"
echo "action      $ACTION   cmd_obs ${CMD_OBS:-(default)}   target $TARGET   episode_s ${EPISODE_S:-(default)}   arm_table ${ARM_TABLE:-(default)}"
echo "shape       $SHAPE   goal $GOAL   gate $GATE   critic_obs $CRITIC_OBS   box_obs $BOX_OBS   box_feet ${BOX_FEET:-(default)}   ckpt_spawn ${CKPT_SPAWN:-(none)}   table_dz ${TABLE_DZ:-(none)}"
echo "s11         arm_home ${ARM_HOME:-(default)}   tracker ${TRACKER:-(default)}   handoff_route ${HANDOFF_ROUTE:-(default)}   arms ${ARMS:-(default)}   handoff_train ${HANDOFF_TRAIN:-(default)}   start ${START:-(default)}"
echo "s9.1        base_obs ${BASE_OBS:-(default)}   contact ${CONTACT:-(default: box)}   s13 ${S13:-(none)}"
echo "extra args  ${EXTRA_ARGS:-(none)}"
echo "resume      ${RESUME:-(none)}"
echo "node        $(hostname)   gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '?')"
uv pip list 2>/dev/null | grep -E '^(mjlab|mujoco|mujoco-warp|warp-lang|torch) ' || true
# Whether mujoco_warp_io.patch is applied is part of a result's provenance: the patch
# wraps check_toolkit_driver() in a try/except, so an unpatched tree calls it bare.
if grep -qE '^[[:space:]]+warp_util[.]check_toolkit_driver[(][)]' \
     .venv/lib/python*/site-packages/mujoco_warp/_src/io.py 2>/dev/null; then
  echo "patch       mujoco_warp_io.patch NOT applied"
else
  echo "patch       mujoco_warp_io.patch applied, or upstream moved -- check"
fi
# Decision 0005 (plan §12): mujoco_warp 3.11's box_box takes an edge-edge axis for a box resting on a box by
# float32 rounding and returns a contact up to 0.5 m deep (it popped 2 of 8 pins at the first reset). Every insert
# GOAL refuses to train without mujoco_warp_boxbox.patch (scripts/apply_mujoco_warp_boxbox_patch.sh .venv).
if grep -q BOX_BOX_EDGE_TOL .venv/lib/python*/site-packages/mujoco_warp/_src/collision_primitive_core.py 2>/dev/null; then
  echo "patch       mujoco_warp_boxbox.patch applied (decision 0005)"
else
  echo "patch       mujoco_warp_boxbox.patch NOT applied"
  case "$GOAL" in
    insert*)
      echo "Refusing to run: GOAL=$GOAL needs mujoco_warp_boxbox.patch (decision 0005)." >&2
      exit 1
      ;;
  esac
  # Plan §13.1's codes carry the socket block (on the table, in the hand, welded at H) and the pin.
  if [ -n "${S13:-}" ]; then
    echo "Refusing to run: S13=$S13 needs mujoco_warp_boxbox.patch (decision 0005)." >&2
    exit 1
  fi
fi
echo "=================="

# shellcheck disable=SC2086  # EXTRA_ARGS is meant to word-split.
exec uv run --extra cu128 train "$TASK" \
  $EXTRA_ARGS \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  --env.scene.num-envs "$NUM_ENVS" \
  --env.seed "$SEED" \
  --agent.seed "$SEED" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.logger wandb \
  --agent.wandb-project "$WANDB_PROJECT" \
  --agent.experiment-name "$EXPERIMENT" \
  --agent.run-name "pp_${RUN_TAG}_${REWARD}_${DR}_${PPO}_slew${SLEW}_${BASELINE}_${SPAWN}_tool${TOOL_SPEED}_${GAINS}_eff${EFFORT}${LAG_NAME}${OCC_NAME}${S78_NAME}${S8_NAME}${S9_NAME}${S79_NAME}${S11_NAME}${S91_NAME}_n${NUM_ENVS}_seed${SEED}" \
  --agent.wandb-tags "(\"ias\",\"pick-place-v0\",\"$RUN_TAG\",\"$REWARD\",\"$DR\",\"$PPO\",\"slew$SLEW\",\"base$BASELINE\",\"spawn$SPAWN\",\"tool$TOOL_SPEED\",\"$GAINS\",\"eff$EFFORT\"$LAG_TAG$OCC_TAG$S78_TAG$S8_TAG$S9_TAG$S79_TAG$S11_TAG$S91_TAG$RESUME_TAG)"
