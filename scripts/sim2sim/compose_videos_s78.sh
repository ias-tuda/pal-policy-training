#!/bin/bash
# Compose the RViz replay panels of an S7-8 sim2sim run (render_session.sh, 1280x720 each, real time unless
# SPEED_LABEL says otherwise) into:
#
#   <name>_mujoco_vs_pal_ep<K>.mp4   episode K: MuJoCo (training simulator) | PAL physics simulator
#   <name>_pal_<N>_episodes.mp4      PAL, all N episodes in a grid
#   <name>_mujoco_<N>_episodes.mp4   the MuJoCo reference, the same N episodes
#
#   compose_videos_s78.sh <panel_dir> <out_dir> <name> "<title>" <N> <cols> "<PAL placed>" "<MuJoCo placed>" [hero ep]
#
# Panels are <panel_dir>/pal_ep<K>.mp4 and mj_ep<K>.mp4, K = 0..N-1 (the panel order of the episodes spec).
# Needs an ffmpeg with drawtext and xstack.
set -eu
IN=$1
OUT=$2
NAME=$3
TITLE=$4
N=$5
COLS=$6
PAL_PLACED=$7
MJ_PLACED=$8
HERO=${9:-0}
mkdir -p "$OUT"
FONT=${FONT:-$(ls /System/Library/Fonts/Supplemental/Arial.ttf \
  /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf 2> /dev/null | head -1)}
FONT_BOLD=${FONT_BOLD:-$(ls "/System/Library/Fonts/Supplemental/Arial Bold.ttf" \
  /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf 2> /dev/null | head -1)}
SPEED_LABEL=${SPEED_LABEL:-real time}
PAL_NOTE=${PAL_NOTE:-PAL: arm 5-7 damping 1.5 + 64 ms reference lag (see the report)}
TXT=$(mktemp -d)
trap 'rm -rf "$TXT"' EXIT
ENC=(-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -r 25 -movflags +faststart)

# Header band above a video: a bold title line and a legend line, from text files (no escaping).
header() {
  echo "pad=iw:ih+90:0:90:color=white,drawtext=fontfile='$FONT_BOLD':textfile='$1':x=24:y=14:\
fontsize=30:fontcolor=0x141414,drawtext=fontfile='$FONT':textfile='$2':x=24:y=56:fontsize=22:\
fontcolor=0x3c3c3c"
}

# Grid of panels <prefix>_ep<K>.mp4, K = 0..N-1, COLS wide, tiles scaled to fit 1920 px.
grid() { # $1 prefix  $2 output  $3 title file  $4 legend file
  local inputs=() scales="" labels="" layout="" k r c W=$((1920 / COLS)) H=$((1080 / COLS))
  H=$((W * 9 / 16))
  for ((k = 0; k < N; k++)); do
    inputs+=(-i "$IN/$1_ep${k}.mp4")
    scales+="[$k:v]scale=${W}:${H}[s$k];"
    labels+="[s$k]"
    r=$((k / COLS))
    c=$((k % COLS))
    layout+="$((c * W))_$((r * H))|"
  done
  ffmpeg -y -loglevel error "${inputs[@]}" \
    -filter_complex "${scales}${labels}xstack=inputs=$N:layout=${layout%|}:fill=white,\
$(header "$3" "$4")[v]" -map "[v]" "${ENC[@]}" "$2"
}

printf '%s\n' "$TITLE: PAL physics simulator, $N episodes, $SPEED_LABEL" > "$TXT/t1"
printf '%s\n' "placed $PAL_PLACED (MuJoCo $MJ_PLACED)  ·  red: virtual cube, yellow while held  ·  green: goal  ·  policy log replayed in RViz" > "$TXT/l1"
grid pal "$OUT/${NAME}_pal_${N}_episodes.mp4" "$TXT/t1" "$TXT/l1"

printf '%s\n' "$TITLE: MuJoCo (training simulator), the same $N episodes, $SPEED_LABEL" > "$TXT/t2"
printf '%s\n' "the reference rollout the PAL runs are compared with (same policy, same episode specs)  ·  red: cube, yellow while held  ·  green: goal" > "$TXT/l2"
grid mj "$OUT/${NAME}_mujoco_${N}_episodes.mp4" "$TXT/t2" "$TXT/l2"

printf '%s\n' "$TITLE, episode $HERO: MuJoCo (left, training) and the PAL physics simulator (right), $SPEED_LABEL" > "$TXT/t3"
printf '%s\n' "red cube, yellow while held (PAL: virtual)  ·  green: goal  ·  $PAL_NOTE" > "$TXT/l3"
ffmpeg -y -loglevel error -i "$IN/mj_ep${HERO}.mp4" -i "$IN/pal_ep${HERO}.mp4" \
  -filter_complex "[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]xstack=inputs=2:layout=0_0|w0_0:fill=white,\
$(header "$TXT/t3" "$TXT/l3")[v]" -map "[v]" "${ENC[@]}" "$OUT/${NAME}_mujoco_vs_pal_ep${HERO}.mp4"
ls -la "$OUT"
