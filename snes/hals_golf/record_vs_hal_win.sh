#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_PATH="${1:-$SCRIPT_DIR/recordings/vs_hal_win_with_sound.ogv}"

if [[ "$OUTPUT_PATH" != /* ]]; then
    OUTPUT_PATH="$SCRIPT_DIR/$OUTPUT_PATH"
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

HEADLESS=1 PYTHONUNBUFFERED=1 "$SCRIPT_DIR/run_bot.sh" clear \
    --mode vs-hal \
    --club-set metal \
    --state Title \
    --video "$OUTPUT_PATH" \
    --video-scale 2 \
    --video-fps 60 \
    --post-complete-frames 1800

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

echo "[VIDEO] verified VLC codecs: video=$VIDEO_CODEC audio=$AUDIO_CODEC"
echo "[VIDEO] verified non-silent audio: peak=$AUDIO_PEAK"
echo "[VIDEO] play with: vlc $OUTPUT_PATH"
