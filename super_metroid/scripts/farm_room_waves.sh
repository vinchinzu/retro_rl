#!/usr/bin/env bash
# Multi-round dual-track room farm: N OpenCode Luna agents (max thinking),
# review + clear between rounds. Relaxes continuous tip — segments only.
#
# Usage (from repo root):
#   ./super_metroid/scripts/farm_room_waves.sh
#   ./super_metroid/scripts/farm_room_waves.sh --rounds 10 --parallel 8
#   ./super_metroid/scripts/farm_room_waves.sh --rounds 1 --parallel 8 --dry-run
#   ./super_metroid/scripts/farm_room_waves.sh --rounds 3 --deadline-hours 2
#
# Between rounds:
#   1. Wait for all session EXIT lines
#   2. Parse residuals → farm rollup
#   3. Double-check no spine/continuous edits (git path guard)
#   4. Clear session history by design (fresh opencode run each card)
#   5. Generate next batch of disjoint SM-ROOM-SEG cards
#   6. Dispatch again
#
# Model: openrouter/openai/gpt-5.6-luna --variant max (via dispatch_opencode.sh)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck source=farm_rollup.sh
source "$ROOT/super_metroid/scripts/farm_rollup.sh"

ROUNDS=10
PARALLEL=8
DEADLINE_HOURS=0
DRY_RUN=0
VARIANT="${SM_OPENCODE_VARIANT:-max}"
FARM_LOG_DIR="$ROOT/super_metroid/docs/tasks/logs/farm"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
FARM_ID="farm_${STAMP}"
ROLLUP="$FARM_LOG_DIR/${FARM_ID}_rollup.md"
PID_DIR="$FARM_LOG_DIR/${FARM_ID}_pids"
START_EPOCH="$(date +%s)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rounds) shift; ROUNDS="${1:-10}" ;;
    --parallel) shift; PARALLEL="${1:-8}" ;;
    --deadline-hours) shift; DEADLINE_HOURS="${1:-0}" ;;
    --variant) shift; VARIANT="${1:-max}" ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h)
      sed -n '2,24p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$FARM_LOG_DIR" "$PID_DIR"

export SM_OPENCODE_VARIANT="$VARIANT"
export SM_OPENCODE_MODEL_LUNA="${SM_OPENCODE_MODEL_LUNA:-openrouter/openai/gpt-5.6-luna}"
export SM_OPENCODE_TIMEOUT_MINUTES="${SM_OPENCODE_TIMEOUT_MINUTES:-20}"

# Baseline dirty paths so path-guard only reports *new* spine dirt this farm.
BASELINE_DIRTY="$FARM_LOG_DIR/${FARM_ID}_baseline_dirty.txt"
git -C "$ROOT" status --porcelain 2>/dev/null \
  | awk '{print $NF}' \
  | rg 'super_metroid/(routes/continuous\.py|docs/STATUS\.md|routes/kpdr/|progression\.py)' \
  | sort -u >"$BASELINE_DIRTY" || true

echo "# Room farm $FARM_ID" | tee "$ROLLUP"
{
  echo
  echo "- Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Rounds: $ROUNDS"
  echo "- Parallel: $PARALLEL"
  echo "- Model: \$SM_OPENCODE_MODEL_LUNA=$SM_OPENCODE_MODEL_LUNA"
  echo "- Variant: $VARIANT (max thinking)"
  echo "- Mode: dual-track room segments only (continuous tip relaxed/parked)"
  echo "- Deadline hours: ${DEADLINE_HOURS:-0} (0 = rounds only)"
  echo
} | tee -a "$ROLLUP"

deadline_hit() {
  if [[ "${DEADLINE_HOURS}" == "0" || -z "${DEADLINE_HOURS}" ]]; then
    return 1
  fi
  local now elapsed max
  now="$(date +%s)"
  elapsed=$((now - START_EPOCH))
  max=$((DEADLINE_HOURS * 3600))
  [[ "$elapsed" -ge "$max" ]]
}

generate_batch() {
  local n="$1"
  local gen_flags=(--count "$n" --print-ids)
  if [[ "$DRY_RUN" -eq 1 ]]; then
    gen_flags+=(--dry-run)
  fi
  # stderr may include progress; keep ids on stdout only via awk.
  uv run python super_metroid/scripts/generate_room_segment_cards.py \
    "${gen_flags[@]}" 2>"$FARM_LOG_DIR/${FARM_ID}_gen_err.txt" \
    | tee -a "$FARM_LOG_DIR/${FARM_ID}_gen.txt" \
    | awk '/^SM-ROOM-SEG-/{print}'
}

# Extract latest log path for a card id from dispatch echo lines / filesystem.
latest_log_for() {
  local id="$1"
  ls -1t "$ROOT/super_metroid/docs/tasks/logs/${id}_"*.log 2>/dev/null | head -1 || true
}

wait_batch() {
  local -a pids=("$@")
  local pid
  if [[ ${#pids[@]} -eq 0 ]]; then
    return 0
  fi
  echo "Waiting on ${#pids[@]} pids: ${pids[*]}"
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || true
    fi
  done
}

review_batch() {
  local round="$1"
  shift
  local -a ids=("$@")
  local id log exit_line result next spine_hits
  local green=0 red=0 partial=0 blocked=0 unknown=0

  {
    echo
    echo "## Round $round review"
    echo
    echo "| Card | EXIT | Result | Next | Log |"
    echo "|------|-----:|--------|------|-----|"
  } | tee -a "$ROLLUP"

  for id in "${ids[@]}"; do
    log="$(latest_log_for "$id")"
    exit_line="?"
    result="UNKNOWN"
    next="?"
    if [[ -n "$log" && -f "$log" ]]; then
      exit_line="$(grep -E '^EXIT:' "$log" | tail -1 | cut -d: -f2 || echo '?')"
      # GREEN requires a filed residual result or runner JSON success; prose
      # such as "GREEN?" in a worker message is never evidence.
      result="$(farm_card_result "$id" "$log" "$ROOT/super_metroid/docs/tasks")"
      next="$(
        rg -i 'Next card ID:\s*\S+' "$log" | tail -1 \
          | sed -E 's/.*Next card ID:\s*//I;s/[[:space:]].*//' \
          || true
      )"
      [[ -z "$next" ]] && next="—"
    else
      log="(missing)"
      exit_line="—"
    fi
    case "${result^^}" in
      GREEN*) green=$((green + 1)) ;;
      RED*) red=$((red + 1)) ;;
      PARTIAL*) partial=$((partial + 1)) ;;
      BLOCKED*) blocked=$((blocked + 1)) ;;
      *) unknown=$((unknown + 1)) ;;
    esac
    echo "| $id | $exit_line | $result | $next | \`${log#$ROOT/}\` |" | tee -a "$ROLLUP"
  done

  {
    echo
    echo "Round $round tallies: GREEN=$green RED=$red PARTIAL=$partial BLOCKED=$blocked UNKNOWN=$unknown"
  } | tee -a "$ROLLUP"

  # Double-check: no *new* spine / continuous / STATUS dirt beyond farm baseline.
  local now_dirty new_hits
  now_dirty="$(
    git -C "$ROOT" status --porcelain 2>/dev/null \
      | awk '{print $NF}' \
      | rg 'super_metroid/(routes/continuous\.py|docs/STATUS\.md|routes/kpdr/|progression\.py)' \
      | sort -u || true
  )"
  new_hits="$(comm -13 "$BASELINE_DIRTY" <(printf '%s\n' "$now_dirty" | sort -u) 2>/dev/null || true)"
  if [[ -n "$new_hits" ]]; then
    {
      echo
      echo "### ⚠ PATH GUARD — NEW spine/continuous paths dirtied this farm"
      echo '```'
      echo "$new_hits"
      echo '```'
    } | tee -a "$ROLLUP"
  else
    echo "Path guard: no new continuous/STATUS/kpdr/progression dirt this farm." | tee -a "$ROLLUP"
  fi

  # Refresh practice queue board for honesty between rounds (cheap, offline).
  if [[ "$DRY_RUN" -eq 0 ]]; then
    uv run python super_metroid/scripts/export/room_work_queue.py 2>/dev/null \
      | tee -a "$FARM_LOG_DIR/${FARM_ID}_queue_export_r${round}.txt" \
      || echo "(queue export skipped/failed — non-fatal)" | tee -a "$ROLLUP"
  fi
}

dispatch_batch() {
  local -a ids=("$@")
  if [[ ${#ids[@]} -eq 0 ]]; then
    echo "empty batch" >&2
    return 1
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would dispatch: ${ids[*]}"
    return 0
  fi
  # Capture pids from dispatch stdout ("pid N  log ...")
  local out
  out="$(
    SM_OPENCODE_VARIANT="$VARIANT" \
      ./super_metroid/scripts/dispatch_opencode.sh --luna --variant "$VARIANT" \
      "${ids[@]}" 2>&1 | tee -a "$FARM_LOG_DIR/${FARM_ID}_dispatch.txt"
  )"
  local -a pids=()
  while IFS= read -r line; do
    if [[ "$line" =~ ^pid[[:space:]]+([0-9]+) ]]; then
      pids+=("${BASH_REMATCH[1]}")
    fi
  done <<<"$out"
  # Persist pids for crash recovery
  printf '%s\n' "${pids[@]}" >"$PID_DIR/round_pids.txt"
  printf '%s\n' "${ids[@]}" >"$PID_DIR/round_ids.txt"
  echo "Launched pids: ${pids[*]:-none}"
  # Export for caller via global
  BATCH_PIDS=("${pids[@]}")
}

# ── main loop ──────────────────────────────────────────────────────────────
echo "Farm $FARM_ID starting (rounds=$ROUNDS parallel=$PARALLEL variant=$VARIANT)"

for ((round = 1; round <= ROUNDS; round++)); do
  if deadline_hit; then
    echo "Deadline hit before round $round — stopping." | tee -a "$ROLLUP"
    break
  fi

  echo
  echo "======== ROUND $round / $ROUNDS ========" | tee -a "$ROLLUP"

  mapfile -t IDS < <(generate_batch "$PARALLEL")
  if [[ ${#IDS[@]} -eq 0 ]]; then
    echo "No more open segment cards to generate — farm complete early." | tee -a "$ROLLUP"
    break
  fi
  echo "Round $round cards: ${IDS[*]}" | tee -a "$ROLLUP"

  BATCH_PIDS=()
  dispatch_batch "${IDS[@]}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    review_batch "$round" "${IDS[@]}"
    continue
  fi

  wait_batch "${BATCH_PIDS[@]}"
  review_batch "$round" "${IDS[@]}"

  # Clear history: nothing to do — each card is a fresh opencode session.
  # Drop pid bookkeeping for next round.
  : >"$PID_DIR/round_pids.txt"
  echo "Session history clear (fresh sessions next round)." | tee -a "$ROLLUP"

  if deadline_hit; then
    echo "Deadline hit after round $round — stopping." | tee -a "$ROLLUP"
    break
  fi
done

{
  echo
  echo "## Farm end"
  echo "- Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Elapsed s: $(( $(date +%s) - START_EPOCH ))"
  echo "- Rollup: \`${ROLLUP#$ROOT/}\`"
} | tee -a "$ROLLUP"

echo
echo "Done. Rollup → $ROLLUP"
