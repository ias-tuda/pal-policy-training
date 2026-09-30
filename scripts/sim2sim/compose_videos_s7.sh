#!/bin/bash
# Compose the RViz replay panels of an S7 (RCTP) sim2sim run (render_session.sh, 1280x720 each,
# panel names as in the jobs of the RCTP s3 test) into the videos:
#
#   rctp_s3_ep0_mujoco_vs_pal.mp4   episode 0: MuJoCo | PAL table only / PAL table + static
#                                   stand-in cube; 3 s real time, then 2 s at 5x slower
#   rctp_s3_pal_T_9_episodes.mp4    PAL, table only, episodes 0-8, 3x3, 5x slower
#   rctp_s3_mujoco_9_episodes.mp4   the MuJoCo reference, episodes 0-8, 3x3, 5x slower
#   rctp_s3_pal_T_ep0.mp4           PAL, table only, episode 0 alone; real time, then 5x slower
#
#   compose_videos_s7.sh <panel_dir> <out_dir> "<PAL placed>" "<MuJoCo placed>"
#
# Needs an ffmpeg with drawtext and xstack.
set -eu
IN=$1
OUT=$2
PAL_PLACED=${3:-"9 of 9"}
MJ_PLACED=${4:-"9 of 9"}
mkdir -p "$OUT"
FONT=${FONT:-$(ls /System/Library/Fonts/Supplemental/Arial.ttf \
  /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf 2> /dev/null | head -1)}
FONT_BOLD=${FONT_BOLD:-$(ls "/System/Library/Fonts/Supplemental/Arial Bold.ttf" \
  /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf 2> /dev/null | head -1)}
TXT=$(mktemp -d)
trap 'rm -rf "$TXT"' EXIT
ENC=(-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -r 25 -movflags +faststart)

# Header band above a grid: a bold title line and a legend line, from text files (no escaping).
header() {
  echo "pad=iw:ih+90:0:90:color=white,drawtext=fontfile='$FONT_BOLD':textfile='$1':x=24:y=14:\
fontsize=30:fontcolor=0x141414,drawtext=fontfile='$FONT':textfile='$2':x=24:y=56:fontsize=22:\
fontcolor=0x3c3c3c"
}

printf '%s\n' "RCTP seed 3 (S7-5, 5k iterations), episode 0: MuJoCo (training) and the PAL physics simulator" > "$TXT/t1"
printf '%s\n' "red cube, yellow while held (PAL: virtual)  ·  grey box: static stand-in, collision only  ·  green: goal  ·  3 s real time, then 0-2 s at 5× slower" > "$TXT/l1"
printf '%s\n' "PAL, all 9 episodes:" > "$TXT/n0"
{
  printf '%s\n' "table only: placed $PAL_PLACED (MuJoCo $MJ_PLACED), within 1.0 cm of the goal"
  printf '%s\n' "table + static stand-in: 3 of 9 (the box stays in the hand's way)"
  printf '%s\n' "wrist damping 1.5 + 64 ms reference lag (see the deployment notes)"
} > "$TXT/n1"
ffmpeg -y -loglevel error \
  -i "$IN/ep0_mujoco_rt.mp4" -i "$IN/mj_ep0_slow.mp4" -i "$IN/ep0_t_rt.mp4" -i "$IN/t_ep0_slow.mp4" \
  -i "$IN/ep0_tc_rt.mp4" -i "$IN/tc_ep0_slow.mp4" \
  -filter_complex "[0:v][1:v]concat=n=2[a];[2:v][3:v]concat=n=2[b];[4:v][5:v]concat=n=2[c];\
[a]scale=960:540[a2];[b]scale=960:540[b2];[c]scale=960:540[c2];\
[a2][b2][c2]xstack=inputs=3:layout=0_0|w0_0|w0/2_h0:fill=white,\
$(header "$TXT/t1" "$TXT/l1"),\
drawtext=fontfile='$FONT_BOLD':textfile='$TXT/n0':x=40:y=690:fontsize=26:fontcolor=0x141414,\
drawtext=fontfile='$FONT':textfile='$TXT/n1':x=40:y=740:fontsize=22:fontcolor=0x3c3c3c:line_spacing=12[v]" \
  -map "[v]" "${ENC[@]}" "$OUT/rctp_s3_ep0_mujoco_vs_pal.mp4"

# 3x3 grid of the slow panels <prefix>_epN_slow.mp4
grid() { # $1 prefix  $2 output  $3 title file  $4 legend file
  local inputs=() scales="" ep
  for ep in 0 1 2 3 4 5 6 7 8; do
    inputs+=(-i "$IN/$1_ep${ep}_slow.mp4")
    scales+="[$ep:v]scale=640:360[s$ep];"
  done
  ffmpeg -y -loglevel error "${inputs[@]}" \
    -filter_complex "${scales}[s0][s1][s2][s3][s4][s5][s6][s7][s8]xstack=inputs=9:\
layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0|0_h0+h3|w0_h0+h3|w0+w1_h0+h3,\
$(header "$3" "$4")[v]" -map "[v]" "${ENC[@]}" "$2"
}

printf '%s\n' "PAL physics simulator, table only: RCTP seed 3 (S7-5, 5k), episodes 0-8, first 2 s at 5× slower" > "$TXT/t2"
printf '%s\n' "placed $PAL_PLACED (MuJoCo $MJ_PLACED)  ·  red: virtual cube, yellow while held  ·  green: goal  ·  policy log replayed in RViz" > "$TXT/l2"
grid t "$OUT/rctp_s3_pal_T_9_episodes.mp4" "$TXT/t2" "$TXT/l2"

printf '%s\n' "MuJoCo (training simulator): RCTP seed 3 (S7-5, 5k), episodes 0-8, first 2 s at 5× slower" > "$TXT/t3"
printf '%s\n' "the reference rollout the PAL runs are compared with (same policy, same 9 episode specs)  ·  red: cube, yellow while held  ·  green: goal" > "$TXT/l3"
grid mj "$OUT/rctp_s3_mujoco_9_episodes.mp4" "$TXT/t3" "$TXT/l3"

ffmpeg -y -loglevel error -i "$IN/ep0_t_rt.mp4" -i "$IN/t_ep0_slow.mp4" \
  -filter_complex "[0:v][1:v]concat=n=2[v]" -map "[v]" "${ENC[@]}" "$OUT/rctp_s3_pal_T_ep0.mp4"
ls -la "$OUT"
