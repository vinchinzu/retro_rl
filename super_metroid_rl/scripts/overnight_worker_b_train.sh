#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

mkdir -p logs models

MAIN_LOG="logs/overnight_worker_b_${RUN_ID}.out"
STATUS_LOG="logs/overnight_worker_b_${RUN_ID}_status.out"

# Priority training plan (Feb 2026 overnight refinement).
#
# Status assessment:
#   parlor_descent    - COLLAPSED (ep_len=1, rew=-478 at 5M). MUST restart fresh.
#   morph_ball_return - Only 50k steps, maxing ep_len. Needs heavy training.
#   parlor_to_flyway  - Only 50k steps, low reward. Needs training.
#   flyway_to_torizo  - Only 50k steps, very low reward. Needs training.
#   elevator_return   - Working well (ep_len~200, completing). Light tune.
#   pit_room_return   - 5M steps, still learning (rew~1840). More training.
#   climb_return      - 5M steps, still learning (rew~2600). More training.
#   climb_descent     - Trained, decent. Light tune.
#   pit_room_descent  - Trained. Light tune.
#   elevator_descent  - Trained. Light tune.
#
# Segments marked FRESH will NOT load existing (collapsed) models.
SEGMENT_PLAN=(
  "parlor_descent:500000:FRESH"
  "morph_ball_return:500000:RESUME"
  "parlor_to_flyway:300000:RESUME"
  "flyway_to_torizo:300000:RESUME"
  "pit_room_return:300000:RESUME"
  "climb_return:300000:RESUME"
  "elevator_return:200000:RESUME"
  "climb_descent:200000:RESUME"
  "pit_room_descent:200000:RESUME"
  "elevator_descent:200000:RESUME"
)

{
  echo "[$(date -Iseconds)] Worker B overnight training start"
  echo "Run ID: $RUN_ID"
  echo "Python: $PYTHON_BIN"
  echo "Device: $DEVICE"
  echo "Segment plan:"
  for plan in "${SEGMENT_PLAN[@]}"; do
    echo "  $plan"
  done
} | tee "$MAIN_LOG" > "$STATUS_LOG"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[$(date -Iseconds)] ERROR: python binary not executable: $PYTHON_BIN" | tee -a "$MAIN_LOG" "$STATUS_LOG"
  exit 1
fi

# Verify ROM exists
integration_rom="$ROOT_DIR/custom_integrations/SuperMetroid-Snes/rom.sfc"
if [[ ! -f "$integration_rom" ]]; then
  fallback_rom="$ROOT_DIR/roms/rom.sfc"
  if [[ -f "$fallback_rom" ]]; then
    ln -sfn ../../roms/rom.sfc "$integration_rom"
    echo "[$(date -Iseconds)] Linked ROM: $integration_rom" | tee -a "$MAIN_LOG" "$STATUS_LOG"
  else
    echo "[$(date -Iseconds)] ERROR: ROM missing at $integration_rom and $fallback_rom" | tee -a "$MAIN_LOG" "$STATUS_LOG"
    exit 1
  fi
fi

# Smoke test: verify env creation works before committing to long training
echo "[$(date -Iseconds)] Smoke test: creating env..." | tee -a "$MAIN_LOG" "$STATUS_LOG"
if ! "$PYTHON_BIN" -c "
import stable_retro as retro
retro.data.Integrations.add_custom_path('$ROOT_DIR/custom_integrations')
env = retro.make(game='SuperMetroid-Snes', state='ZebesStart',
                 use_restricted_actions=retro.Actions.ALL,
                 inttype=retro.data.Integrations.ALL)
env.close()
print('OK')
" >> "$MAIN_LOG" 2>&1; then
  echo "[$(date -Iseconds)] ERROR: smoke test failed - env creation broken" | tee -a "$MAIN_LOG" "$STATUS_LOG"
  exit 1
fi
echo "[$(date -Iseconds)] Smoke test passed" | tee -a "$MAIN_LOG" "$STATUS_LOG"

# Back up collapsed parlor_descent model before fresh restart
if [[ -f "models/segment_parlor_descent.zip" ]]; then
  backup="models/segment_parlor_descent_collapsed_backup.zip"
  if [[ ! -f "$backup" ]]; then
    cp "models/segment_parlor_descent.zip" "$backup"
    echo "[$(date -Iseconds)] Backed up collapsed parlor_descent model" | tee -a "$MAIN_LOG" "$STATUS_LOG"
  fi
fi

# Training loop
failures=()
successes=()

for plan in "${SEGMENT_PLAN[@]}"; do
  IFS=: read -r segment steps mode <<< "$plan"
  seg_log="logs/overnight_worker_b_${RUN_ID}_${segment}.out"

  cmd=("$PYTHON_BIN" train_curriculum.py train --segment "$segment" --steps "$steps" --device "$DEVICE")

  if [[ "$mode" == "FRESH" ]]; then
    cmd+=(--fresh)
  elif [[ "$mode" == "RESUME" ]]; then
    load_path="models/segment_${segment}.zip"
    if [[ -f "$load_path" ]]; then
      cmd+=(--load "$load_path")
    fi
  fi

  {
    echo ""
    echo "[$(date -Iseconds)] START $segment steps=$steps mode=$mode"
    printf '[%s] CMD %s\n' "$(date -Iseconds)" "${cmd[*]}"
  } | tee -a "$MAIN_LOG" "$STATUS_LOG"

  if PYTHONUNBUFFERED=1 "${cmd[@]}" >> "$seg_log" 2>&1; then
    snapshot="models/segment_${segment}_worker_b_${RUN_ID}_${steps}steps.zip"
    if [[ -f "models/segment_${segment}.zip" ]]; then
      cp -f "models/segment_${segment}.zip" "$snapshot"
    fi
    successes+=("$segment")
    echo "[$(date -Iseconds)] DONE $segment snapshot=$snapshot" | tee -a "$MAIN_LOG" "$STATUS_LOG"
  else
    failures+=("$segment")
    echo "[$(date -Iseconds)] FAIL $segment (see $seg_log)" | tee -a "$MAIN_LOG" "$STATUS_LOG"
  fi
done

# Summary
{
  echo ""
  echo "[$(date -Iseconds)] ======== TRAINING SUMMARY ========"
  echo "Successes (${#successes[@]}): ${successes[*]:-none}"
  echo "Failures  (${#failures[@]}): ${failures[*]:-none}"
  echo "================================================"
} | tee -a "$MAIN_LOG" "$STATUS_LOG"

if (( ${#failures[@]} > 0 )); then
  exit 2
fi
echo "[$(date -Iseconds)] Completed successfully" | tee -a "$MAIN_LOG" "$STATUS_LOG"
