#!/usr/bin/env bash
# Full end-to-end run: power-on -> Y3 Summer D30 credits, recorded, then a final scorecard.
#
#   snes/harvest/scripts/e2e_run.sh <tag>               # with video (default)
#   VIDEO=0 snes/harvest/scripts/e2e_run.sh <tag>       # no video (faster, lean emulator step)
#   VIDEO_SCALE=2 snes/harvest/scripts/e2e_run.sh <tag>
#   FOREGROUND=1 VIDEO=0 snes/harvest/scripts/e2e_run.sh <tag>  # supervised process/session
#   HARVEST_STRATEGY=score snes/harvest/scripts/e2e_run.sh <tag>   # what the run optimizes (default cash)
#
# Output, all under recordings/e2e/<tag>/:
#   report.json   run_to_day2 report (ending, ledger, journal, audit)
#   run.log       the [RUN] progress log
#   video_partNNN.mp4 (from --video video.mp4)   one-hour parts, playable while the run is still going
#   FINAL.md      final scorecard (written when the run ends, CLEAN or not)
# Day pins are saved as PR_e2e<tag>_* so any stall reproduces from the morning before it.
#
# The tree is copied first (as in y23_sweep.sh), so editing the repo mid-run cannot change it.
# A power-on run is deterministic for unchanged code; Clean intervention, no loads, no writes.

set -euo pipefail

[[ $# -eq 1 ]] || { sed -n '2,5p' "$0" >&2; exit 2; }
TAG="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAME="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN="$(cd "$GAME/../.." && pwd)"
PY="$GAME/.venv/bin/python"
[[ -x "$PY" ]] || PY="$MAIN/.venv/bin/python"
FROZEN="${TMPDIR:-/tmp}/e2e_$TAG"
OUT="$GAME/recordings/e2e/$TAG"

rm -rf "$FROZEN"
mkdir -p "$FROZEN/snes/harvest" "$OUT"
cp -r "$MAIN/retro_harness" "$FROZEN/"
for d in harvest maps HM-Decomp state_presets docs; do cp -r "$GAME/$d" "$FROZEN/snes/harvest/"; done
cp "$GAME/pyproject.toml" "$FROZEN/snes/harvest/"
for d in roms tasks custom_integrations saves; do ln -s "$GAME/$d" "$FROZEN/snes/harvest/$d"; done
ln -s "$MAIN/roms" "$FROZEN/roms"
find "$FROZEN" -name __pycache__ -prune -exec rm -rf {} +
# Keep the exact code this run played (a few tens of MB): /tmp is wiped by a reboot, and a 999 whose
# tree is gone cannot be replayed with video. Unpack into an empty dir and run with the same env + seed.
tar czf "$OUT/frozen_tree.tgz" -C "$FROZEN" --exclude='snes/harvest/roms' --exclude='snes/harvest/tasks' \
  --exclude='snes/harvest/custom_integrations' --exclude='snes/harvest/saves' --exclude='roms' . 2>/dev/null || true
env | grep -E '^(HARVEST_|VIDEO|JITTER)' | sort > "$OUT/env.txt" 2>/dev/null || true
echo "${HARVEST_STRATEGY:-cash}" > "$OUT/strategy.txt"
git -C "$MAIN" rev-parse HEAD > "$OUT/git_head.txt" 2>/dev/null || true
git -C "$MAIN" status --short snes/harvest retro_harness > "$OUT/git_dirty.txt" 2>/dev/null || true

video_args=()
if [[ "${VIDEO:-1}" != "0" ]]; then
  video_args=(--video "$OUT/video.mp4" --video-scale "${VIDEO_SCALE:-3}")
fi

cd "$FROZEN/snes/harvest"
run_full_campaign() {
  env PYTHONPATH="$FROZEN:$FROZEN/snes:$FROZEN/snes/harvest" HEADLESS=1 QT_QPA_PLATFORM=offscreen \
    "$PY" -m harvest.scripts.run_to_day2 --power-on --until-ending \
      --save-day-pins --pin-prefix "PR_e2e${TAG}_" --progress-every 50000 \
      --save-end-state "LN_e2e${TAG}_end" \
      "${video_args[@]}" --out "$OUT/report.json" > "$OUT/run.log" 2>&1 || true
  # Score whatever the run left behind (a failed run still writes a report).
  # The scorecard reads a JSON report only, so it runs from the live tree (newest formatter).
  if [[ -f "$OUT/report.json" ]]; then
    ( cd "$GAME" && "$PY" -m harvest.tools.final_score "$OUT/report.json" --out "$OUT/FINAL.md" ) \
      > "$OUT/final_score.log" 2>&1 || true
  fi
}
if [[ "${FOREGROUND:-0}" == "1" ]]; then
  run_full_campaign
else
  run_full_campaign > /dev/null 2>&1 &
  disown
  echo "started e2e $TAG; log $OUT/run.log, report $OUT/report.json, scorecard $OUT/FINAL.md"
fi
