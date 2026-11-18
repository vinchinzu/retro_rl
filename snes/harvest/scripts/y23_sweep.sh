#!/usr/bin/env bash
# Repeatability sweep for Years 2-3: N seeded, perturbed runs to the ending.
#
#   snes/harvest/scripts/y23_sweep.sh <pin> <tag> <seed>...
#   JITTER=90 FAULTS='*:fail' RATE=0.01 snes/harvest/scripts/y23_sweep.sh PR_y1b_Y2_Spring_D1 a 1 2 3
#   .venv/bin/python -m harvest.tools.sweep_report recordings/y23_sweep/<tag>_s*.json
#   UNTIL='--until-season 2 --until-day 1' JITTER=0 HARVEST_SADDLE=0 ...   # bounded A/B leg
#
# A power-on run is deterministic, so one clean finish proves one timeline. Each
# seed shifts it: HARVEST_PHASE_JITTER idles 0..JITTER neutral frames before every
# phase (sprite, animal and ROM RNG timing move; no RAM write), and FAULTS/RATE
# (HARVEST_FAULT_PHASES/RATE) force a share of phases to fail so recovery runs.
# The tree is copied first, so editing the repo mid-sweep cannot change a run.
# Reports and logs land in recordings/y23_sweep/<tag>_s<seed>.{json,log}.

set -euo pipefail

[[ $# -ge 3 ]] || { sed -n '2,5p' "$0" >&2; exit 2; }
PIN="$1"; TAG="$2"; shift 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAME="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN="$(cd "$GAME/../.." && pwd)"
PY="$GAME/.venv/bin/python"
[[ -x "$PY" ]] || PY="$MAIN/.venv/bin/python"
FROZEN="${TMPDIR:-/tmp}/y23_sweep_$TAG"
OUT="$GAME/recordings/y23_sweep"

rm -rf "$FROZEN"
mkdir -p "$FROZEN/snes/harvest" "$OUT"
cp -r "$MAIN/retro_harness" "$FROZEN/"
for d in harvest maps HM-Decomp state_presets docs; do cp -r "$GAME/$d" "$FROZEN/snes/harvest/"; done
cp "$GAME/pyproject.toml" "$FROZEN/snes/harvest/"
for d in roms tasks custom_integrations saves; do ln -s "$GAME/$d" "$FROZEN/snes/harvest/$d"; done
ln -s "$MAIN/roms" "$FROZEN/roms"
find "$FROZEN" -name __pycache__ -prune -exec rm -rf {} +

cd "$FROZEN/snes/harvest"
for seed in "$@"; do
  name="${TAG}_s${seed}"
  env PYTHONPATH="$FROZEN:$FROZEN/snes:$FROZEN/snes/harvest" HEADLESS=1 QT_QPA_PLATFORM=offscreen \
    HARVEST_PHASE_JITTER="${JITTER:-90}" HARVEST_FAULT_SEED="$seed" \
    HARVEST_FAULT_PHASES="${FAULTS:-}" HARVEST_FAULT_RATE="${RATE:-0.01}" \
    setsid nohup "$PY" -m harvest.scripts.run_to_day2 --state "$PIN" ${UNTIL:---until-ending} \
      --save-day-pins --pin-prefix "PR_${name}_" --progress-every 50000 \
      --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1 &
  echo "started $name (pid $!), log $OUT/$name.log"
done
