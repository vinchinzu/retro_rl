#!/bin/bash
# Render all SMB any% segments to individual MP4s, then concat into one video.
set -e
cd /home/v/01_projects/11_games/speedrun/retro_rl

OUTDIR=super_mario_bros/optimizer/videos
mkdir -p "$OUTDIR"

LEVELS=(smb_1_1 smb_1_2 smb_4_1 smb_4_2 smb_8_1 smb_8_2 smb_8_3 smb_8_4_1 smb_8_4_2 smb_8_4_3 smb_8_4_4 smb_8_4_5)

# Render each level
for lvl in "${LEVELS[@]}"; do
    actions="super_mario_bros/optimizer/runs/${lvl}/recording_000.json"
    out="${OUTDIR}/${lvl}.mp4"
    if [ -f "$actions" ]; then
        echo "=== Rendering $lvl ==="
        uv run python -m platformer_common.record_video --actions "$actions" --level "$lvl" --output "$out" --scale 3
    else
        echo "=== SKIP $lvl (no recording) ==="
    fi
done

# Create concat list for ffmpeg
CONCAT="${OUTDIR}/concat.txt"
> "$CONCAT"
for lvl in "${LEVELS[@]}"; do
    f="${OUTDIR}/${lvl}.mp4"
    if [ -f "$f" ]; then
        echo "file '$(realpath "$f")'" >> "$CONCAT"
    fi
done

# Concat into single video
FINAL="${OUTDIR}/smb_any_percent_full.mp4"
echo "=== Concatenating ==="
ffmpeg -y -f concat -safe 0 -i "$CONCAT" -c copy "$FINAL" 2>/dev/null
echo "Done! $FINAL"
