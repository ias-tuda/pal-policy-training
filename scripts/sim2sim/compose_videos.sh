#!/bin/bash
# Compose the PAL-simulator episode renders (pal-policy-deployment render_session.sh /
# render_episode.py, 1280x720 panels, jobs as in the report) into the videos for the report:
#
#   ep0_mujoco_vs_pal.mp4   episode 0, 2x2: MuJoCo | PAL table + stand-in / PAL table | PAL none;
#                           real time, then 10x slower
#   pal_tc_9_episodes.mp4   PAL, table + stand-in, episodes 0-8, 3x3, 10x slower
#   pal_tc_ep0.mp4          PAL, table + stand-in, episode 0, full size; real time, then 10x slower
#
#   compose_videos.sh <panel_dir> <out_dir>
#
# Needs an ffmpeg with drawtext and xstack.
set -eu
IN=$1
OUT=$2
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

cat > "$TXT/t1" << 'EOF'
S5.2 full fix (R-C-strict + M1 + M2 + M3), seed 1, 20k iterations: episode 0 in MuJoCo and in the PAL physics simulator
EOF
cat > "$TXT/l1" << 'EOF'
red cube, yellow while held: MuJoCo cube / PAL virtual cube   ·   grey box: PAL static stand-in (collision only)   ·   green: goal   ·   real time, then 10× slower
EOF
ffmpeg -y -loglevel error \
  -i "$IN/ep0_mujoco_rt.mp4" -i "$IN/ep0_mujoco_slow.mp4" -i "$IN/ep0_tc_rt.mp4" -i "$IN/ep0_tc_slow.mp4" \
  -i "$IN/ep0_t_rt.mp4" -i "$IN/ep0_t_slow.mp4" -i "$IN/ep0_f_rt.mp4" -i "$IN/ep0_f_slow.mp4" \
  -filter_complex "[0:v][1:v]concat=n=2[a];[2:v][3:v]concat=n=2[b];[4:v][5:v]concat=n=2[c];\
[6:v][7:v]concat=n=2[d];[a]scale=960:540[a2];[b]scale=960:540[b2];[c]scale=960:540[c2];\
[d]scale=960:540[d2];[a2][b2][c2][d2]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,\
$(header "$TXT/t1" "$TXT/l1")[v]" -map "[v]" "${ENC[@]}" "$OUT/ep0_mujoco_vs_pal.mp4"

cat > "$TXT/t2" << 'EOF'
PAL physics simulator, table + static stand-in cube: S5.2 full fix, seed 1 (20k), episodes 0-8, 10× slower
EOF
cat > "$TXT/l2" << 'EOF'
placed 8 of 9 (MuJoCo 9 of 9)   ·   red: virtual cube, yellow while held   ·   grey: static stand-in (collision only)   ·   green: goal
EOF
inputs=()
scales=""
for ep in 0 1 2 3 4 5 6 7 8; do
  f="$IN/tc_ep${ep}_slow.mp4"
  [ "$ep" = 0 ] && f="$IN/ep0_tc_slow.mp4"
  inputs+=(-i "$f")
  scales+="[$ep:v]scale=640:360[s$ep];"
done
ffmpeg -y -loglevel error "${inputs[@]}" \
  -filter_complex "${scales}[s0][s1][s2][s3][s4][s5][s6][s7][s8]xstack=inputs=9:\
layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0|0_h0+h3|w0_h0+h3|w0+w1_h0+h3,\
$(header "$TXT/t2" "$TXT/l2")[v]" -map "[v]" "${ENC[@]}" "$OUT/pal_tc_9_episodes.mp4"

ffmpeg -y -loglevel error -i "$IN/ep0_tc_rt.mp4" -i "$IN/ep0_tc_slow.mp4" \
  -filter_complex "[0:v][1:v]concat=n=2[v]" -map "[v]" "${ENC[@]}" "$OUT/pal_tc_ep0.mp4"
ls -la "$OUT"
