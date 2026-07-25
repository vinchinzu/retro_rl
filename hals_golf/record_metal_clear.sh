#!/bin/bash
# Record a full Amateur metal stroke-play clear with sound, then verify
# codecs + the printed scorecard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_PATH="${1:-$SCRIPT_DIR/recordings/metal_stroke_clear.ogv}"
LOG_PATH="${2:-$SCRIPT_DIR/recordings/metal_stroke_clear.log}"
MAX_TOTAL="${METAL_CLEAR_MAX_TOTAL:-90}"

if [[ "$OUTPUT_PATH" != /* ]]; then
    OUTPUT_PATH="$SCRIPT_DIR/$OUTPUT_PATH"
fi
if [[ "$LOG_PATH" != /* ]]; then
    LOG_PATH="$SCRIPT_DIR/$LOG_PATH"
fi

case "${OUTPUT_PATH,,}" in
    *.ogv|*.ogg) ;;
    *)
        echo "Output must end in .ogv or .ogg for VLC-compatible Theora/Vorbis encoding."
        exit 2
        ;;
esac

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffmpeg/ffprobe not found on PATH; install ffmpeg before recording."
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$LOG_PATH")"

set +e
HEADLESS=1 PYTHONUNBUFFERED=1 "$SCRIPT_DIR/run_bot.sh" clear \
    --mode stroke \
    --club-set metal \
    --state Title \
    --max-frames 250000 \
    --tee-state-prefix MetalTee \
    --video "$OUTPUT_PATH" \
    --video-scale 2 \
    --video-fps 60 \
    --post-complete-frames 1800 \
    2>&1 | tee "$LOG_PATH"
CLEAR_RC=${PIPESTATUS[0]}
set -e

if [[ "$CLEAR_RC" -ne 0 ]]; then
    echo "Metal clear failed (exit=$CLEAR_RC). See $LOG_PATH"
    exit "$CLEAR_RC"
fi

COMPLETE_LINE="$(
    grep -E '^\[CLEAR\] complete ' "$LOG_PATH" | tail -1 || true
)"
if [[ -z "$COMPLETE_LINE" ]]; then
    echo "Recording verification failed: no [CLEAR] complete line in $LOG_PATH"
    exit 1
fi

SCORECARD="$(
    sed -n 's/.*scorecard=\[\([^]]*\)\].*/\1/p' <<<"$COMPLETE_LINE" | tail -1
)"
TOTAL="$(
    sed -n 's/.* total=\([0-9][0-9]*\).*/\1/p' <<<"$COMPLETE_LINE" | tail -1
)"
TO_PAR="$(
    sed -n 's/.* to_par=\([-0-9][0-9]*\).*/\1/p' <<<"$COMPLETE_LINE" | tail -1
)"
OVER_PAR="$(
    sed -n 's/.* over_par=\(\[[^]]*\]\).*/\1/p' <<<"$COMPLETE_LINE" | tail -1
)"

if [[ -z "$SCORECARD" || -z "$TOTAL" ]]; then
    echo "Recording verification failed: could not parse scorecard/total from:"
    echo "  $COMPLETE_LINE"
    exit 1
fi

HOLE_COUNT="$(
    awk -F',' '{print NF}' <<<"$SCORECARD" | tr -d ' '
)"
if [[ "$HOLE_COUNT" -ne 18 ]]; then
    echo "Recording verification failed: expected 18 holes, got $HOLE_COUNT ($SCORECARD)"
    exit 1
fi

SUM="$(
    awk -F',' '{
        s=0
        for (i=1; i<=NF; i++) {
            gsub(/[[:space:]]/, "", $i)
            s+=$i
        }
        print s
    }' <<<"$SCORECARD"
)"
if [[ "$SUM" -ne "$TOTAL" ]]; then
    echo "Recording verification failed: scorecard sum=$SUM != total=$TOTAL"
    exit 1
fi
if [[ "$TOTAL" -gt "$MAX_TOTAL" ]]; then
    echo "Recording verification failed: total=$TOTAL exceeds max $MAX_TOTAL"
    exit 1
fi

VIDEO_CODEC="$(
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name -of default=nw=1:nk=1 "$OUTPUT_PATH"
)"
AUDIO_CODEC="$(
    ffprobe -v error -select_streams a:0 \
        -show_entries stream=codec_name -of default=nw=1:nk=1 "$OUTPUT_PATH"
)"
AUDIO_PEAK="$(
    ffmpeg -hide_banner -ss 60 -t 5 -i "$OUTPUT_PATH" -map 0:a:0 \
        -af volumedetect -f null - 2>&1 \
        | sed -n 's/.*max_volume: //p' \
        | tail -1
)"

if [[ "$VIDEO_CODEC" != "theora" || "$AUDIO_CODEC" != "vorbis" ]]; then
    echo "Recording verification failed: video=$VIDEO_CODEC audio=$AUDIO_CODEC"
    exit 1
fi
if [[ -z "$AUDIO_PEAK" || "$AUDIO_PEAK" == "-inf dB" ]]; then
    echo "Recording verification failed: audio stream is silent."
    exit 1
fi

echo "[SCORE] $COMPLETE_LINE"
echo "[SCORE] total=$TOTAL to_par=${TO_PAR:-unknown} over_par=${OVER_PAR:-[]} holes=$HOLE_COUNT"
echo "[VIDEO] verified VLC codecs: video=$VIDEO_CODEC audio=$AUDIO_CODEC"
echo "[VIDEO] verified non-silent audio: peak=$AUDIO_PEAK"
echo "[VIDEO] play with: vlc $OUTPUT_PATH"
echo "[LOG] $LOG_PATH"
