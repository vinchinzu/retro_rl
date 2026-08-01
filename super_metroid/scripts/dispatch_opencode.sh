#!/usr/bin/env bash
# Dispatch one or more Super Metroid atomic task cards to OpenCode.
#
# Usage (from repo root):
#   ./super_metroid/scripts/dispatch_opencode.sh SM-K4-03
#   ./super_metroid/scripts/dispatch_opencode.sh --model flash SM-K4-05
#   ./super_metroid/scripts/dispatch_opencode.sh SM-K4-03 SM-K4-04 SM-K4-05
#   ./super_metroid/scripts/dispatch_opencode.sh --foreground SM-K4-04
#   ./super_metroid/scripts/dispatch_opencode.sh --flash SM-ROLLUP-STATUS
#   ./super_metroid/scripts/dispatch_opencode.sh --variant max SM-ROOM-SEG-01
#
# Defaults: Luna, --auto, --variant max (max thinking), background
# (logs under docs/tasks/logs/, gitignored). Parallel args launch concurrent
# sessions; only use when cards touch disjoint files. Ownership / hot-module
# conflict detection runs before launch.
#
# Override provider routing without editing the script:
#   SM_OPENCODE_MODEL_LUNA=... SM_OPENCODE_MODEL_FLASH=... ./.../dispatch_opencode.sh ...
#   SM_OPENCODE_VARIANT=max|high|medium|minimal  (default: max for Luna)
# Auth is never stored here — use the local OpenCode/user config outside the repo.
#
# Multi-round room farm (8-wide, dual-track segments):
#   ./super_metroid/scripts/farm_room_waves.sh --rounds 10 --parallel 8
#
# Process: super_metroid/docs/tasks/PROCESS.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_LUNA="${SM_OPENCODE_MODEL_LUNA:-openrouter/openai/gpt-5.6-luna}"
MODEL_FLASH="${SM_OPENCODE_MODEL_FLASH:-openrouter/deepseek/deepseek-v4-flash}"
MODEL="$MODEL_LUNA"
# Provider reasoning effort. Default max for Luna; Flash usually ignores.
VARIANT="${SM_OPENCODE_VARIANT:-max}"
# Bound a worker so an exploratory session cannot pin an entire farm round.
# Zero disables the wrapper for manual/interactive dispatches.
TIMEOUT_MINUTES="${SM_OPENCODE_TIMEOUT_MINUTES:-0}"
FOREGROUND=0
SKIP_OWNERSHIP=0
TASKS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      shift
      case "${1:-}" in
        luna|Luna) MODEL="$MODEL_LUNA" ;;
        flash|Flash) MODEL="$MODEL_FLASH" ;;
        *) MODEL="$1" ;;
      esac
      ;;
    --luna) MODEL="$MODEL_LUNA" ;;
    --flash) MODEL="$MODEL_FLASH" ;;
    --variant)
      shift
      VARIANT="${1:-max}"
      ;;
    --no-variant) VARIANT="" ;;
    --foreground|-f) FOREGROUND=1 ;;
    --force-parallel) SKIP_OWNERSHIP=1 ;;
    --help|-h)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    -*)
      echo "Unknown flag: $1" >&2
      exit 2
      ;;
    *)
      TASKS+=("$1")
      ;;
  esac
  shift
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "Usage: $0 [--flash|--luna|--model NAME] [--variant max|high|medium|minimal] [--foreground] [--force-parallel] SM-K4-NN [...]" >&2
  exit 2
fi

LOG_DIR="$ROOT/super_metroid/docs/tasks/logs"
mkdir -p "$LOG_DIR"

# Hot modules that must never be parallel-edited (PROCESS §5).
HOT_MODULES=(
  "routes/kpdr/business_climb.py"
  "routes/kpdr/hijump_return.py"
  "routes/kpdr/varia_return.py"
  "routes/spore_spawn_controller.py"
  "routes/continuous.py"
  "docs/STATUS.md"
)

pick_model_for_task() {
  local id="$1"
  # Prefer Flash for docs/report/rollup cards when caller left default Luna.
  # Implement suffixes (*B, *-IMPL) and geometry/scaffold stay Luna.
  if [[ "$MODEL" == "$MODEL_LUNA" ]]; then
    case "$id" in
      SM-TIGHTEN-*B|SM-TIGHTEN-*-IMPL|SM-*-IMPL) echo "$MODEL_LUNA" ;;
      SM-K4-03|SM-K4-05|SM-K3-TRACK) echo "$MODEL_FLASH" ;;
      SM-TIGHTEN-0[0-9]|SM-TIGHTEN-0[0-9]-report) echo "$MODEL_FLASH" ;;
      SM-TIGHTEN-0[0-9][A-Z]*) echo "$MODEL_LUNA" ;;  # e.g. 01B if above missed
      SM-ROLLUP-*|SM-*-report|SM-*-REPORT) echo "$MODEL_FLASH" ;;
      SM-PRIM-*) echo "$MODEL_LUNA" ;;
      *) echo "$MODEL" ;;
    esac
  else
    echo "$MODEL"
  fi
}

# Extract package-relative paths from "Own files" section of a card.
# Prints one path per line (best-effort; empty if section missing).
extract_own_files() {
  local card="$1"
  # Between "## Own files" and next "## " heading; lines starting with "- `"
  # or "- path" that look like source paths.
  awk '
    BEGIN { in_own=0 }
    /^## Own files/ { in_own=1; next }
    /^## / { if (in_own) exit }
    in_own && /^- / {
      line=$0
      sub(/^- /, "", line)
      # strip backticks and trailing comments
      gsub(/`/, "", line)
      sub(/ .*$/, "", line)
      sub(/^\*\*/, "", line)
      sub(/\*\*.*$/, "", line)
      if (line ~ /\// || line ~ /\.(py|md|csv|json)$/) {
        # normalize super_metroid/ prefix away
        sub(/^super_metroid\//, "", line)
        print line
      }
    }
  ' "$card"
}

# Map a path to a hot-module key if it matches.
hot_key_for_path() {
  local p="$1"
  local h
  for h in "${HOT_MODULES[@]}"; do
    if [[ "$p" == "$h" || "$p" == *"$h" ]]; then
      echo "$h"
      return 0
    fi
  done
  # basename match for kpdr hot files
  case "$p" in
    *business_climb.py) echo "routes/kpdr/business_climb.py" ;;
    *hijump_return.py) echo "routes/kpdr/hijump_return.py" ;;
    *varia_return.py) echo "routes/kpdr/varia_return.py" ;;
    *spore_spawn_controller.py) echo "routes/spore_spawn_controller.py" ;;
    *continuous.py) echo "routes/continuous.py" ;;
    *STATUS.md) echo "docs/STATUS.md" ;;
    *) return 1 ;;
  esac
}

check_ownership_conflicts() {
  local -a ids=("$@")
  if [[ ${#ids[@]} -le 1 || "$SKIP_OWNERSHIP" -eq 1 ]]; then
    return 0
  fi

  # Collect: path -> task id that owns it; hot key -> task id
  declare -A path_owner=()
  declare -A hot_owner=()
  local id card path hk other conflict=0

  for id in "${ids[@]}"; do
    card="$ROOT/super_metroid/docs/tasks/${id}.md"
    if [[ ! -f "$card" ]]; then
      continue
    fi
    while IFS= read -r path; do
      [[ -z "$path" ]] && continue
      if [[ -n "${path_owner[$path]:-}" && "${path_owner[$path]}" != "$id" ]]; then
        echo "OWNERSHIP CONFLICT: $path owned by both ${path_owner[$path]} and $id" >&2
        conflict=1
      else
        path_owner[$path]="$id"
      fi
      if hk="$(hot_key_for_path "$path")"; then
        if [[ -n "${hot_owner[$hk]:-}" && "${hot_owner[$hk]}" != "$id" ]]; then
          echo "HOT-MODULE CONFLICT: $hk touched by both ${hot_owner[$hk]} and $id" >&2
          echo "  Serialize per docs/tasks/PROCESS.md §5 (or pass --force-parallel)." >&2
          conflict=1
        else
          hot_owner[$hk]="$id"
        fi
      fi
    done < <(extract_own_files "$card")

    # Also scan full card for hot module path mentions under Own/Do if Own empty
    if [[ -z "$(extract_own_files "$card")" ]]; then
      for hk in "${HOT_MODULES[@]}"; do
        if grep -qF "$hk" "$card" 2>/dev/null || grep -qF "${hk##*/}" "$card" 2>/dev/null; then
          # only flag if appears near edit intent (routes/ or docs/)
          if grep -E "(Own files|Do not|edit|touch)" -A20 "$card" 2>/dev/null | grep -qF "${hk##*/}"; then
            if [[ -n "${hot_owner[$hk]:-}" && "${hot_owner[$hk]}" != "$id" ]]; then
              echo "HOT-MODULE CONFLICT (heuristic): $hk in $id and ${hot_owner[$hk]}" >&2
              conflict=1
            else
              hot_owner[$hk]="$id"
            fi
          fi
        fi
      done
    fi
  done

  if [[ "$conflict" -ne 0 ]]; then
    echo "Aborting parallel dispatch. Run serially or use --force-parallel." >&2
    return 1
  fi
  return 0
}

run_one() {
  local id="$1"
  local card="$ROOT/super_metroid/docs/tasks/${id}.md"
  if [[ ! -f "$card" ]]; then
    echo "Missing card: $card" >&2
    return 1
  fi
  local m
  m="$(pick_model_for_task "$id")"
  # Flash cards: skip variant unless explicitly forced via SM_OPENCODE_VARIANT
  # and caller passed --variant; default still max but OpenRouter may ignore.
  local variant_args=()
  if [[ -n "$VARIANT" ]]; then
    variant_args=(--variant "$VARIANT")
  fi
  local timeout_args=()
  if [[ "$TIMEOUT_MINUTES" != "0" && -n "$TIMEOUT_MINUTES" ]]; then
    timeout_args=(timeout --foreground "${TIMEOUT_MINUTES}m")
  fi
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local log="$LOG_DIR/${id}_${stamp}.log"
  local prompt
  prompt="$(cat "$card")"
  # Prefix hygiene so executor stays scoped even if card is thin.
  # Working directory is super_metroid/ (--dir). Prefer package-relative paths
  # (tests/..., routes/..., docs/...). Repo-root paths super_metroid/... also
  # work for shell verify commands launched from monorepo root via `uv run`.
  prompt="You are the Super Metroid cheap executor. Follow the task card exactly and be thorough.

Process (read if geometry / spine / residual): docs/tasks/PROCESS.md
Source states: docs/SOURCE_STATES.md
Template residual schema: docs/TASK_TEMPLATE.md

Throughput rules (dual-track room farm mode OK):
- Own only the files listed under Own files / Do. Never edit continuous.py or STATUS.md unless the card says so.
- Never edit routes/kpdr/* spine controllers, routes/continuous.py, or progression.py unless the card explicitly lists them.
- One problem / one segment per session — no cross-room policy edits.
- Read ALL Read-first files before editing. Use multiple tools (grep, read, pytest) — do not stop after the first skim.
- Implement the full Do list; prefer complete acceptance over a minimal partial.
- Run every Verify command; paste outputs into the final message.
- If pure probe / emu is optional and fails, report residual state — never force-pass or STATUS-promote.
- No progression/capacity/boss-bit RAM forges for green claims.
- One knob only: do not change a second interacting constant in the same session.
- Pure-first: do not claim continuous green; pure cards need listed --source.
- Practice promote is dual-track only — not continuous evidence.

Paths:
- OpenCode cwd is the super_metroid/ package root.
- Read/Edit: package-relative (tests/..., routes/..., combat/..., docs/..., policies/...).
- Shell: either \`cd\` to monorepo root and use super_metroid/... paths, or use paths that work from cwd; if a path 404s, retry without the super_metroid/ prefix.

Super-clean residual (required in final message — all sections):
1. Result: GREEN | RED | BLOCKED | PARTIAL
2. Files changed (path list) + one-line purpose each
3. Verify command stdout/stderr paste (or explicit skip reason)
4. Acceptance checklist: each [ ] from the card marked pass/fail
5. Residual risks: what still blocks pure-green / continuous / STATUS / practice promote
6. Next action (required):
   - Next card ID: SM-XXXX | PLANNER-GATE | none
   - One change: single knob or decision (one sentence)
   - Source state: path or needs capture: SM-*-SRC
7. Explicit non-claims: did not STATUS-promote; did not forge progression RAM; dual-track only if practice
8. If probe failed: last room/pose/x/y/door_transition (or N/A for docs-only)

--- TASK CARD ---
${prompt}"

  echo "=== $id → $m variant=${VARIANT:-none} (log: $log) ==="
  if [[ "$FOREGROUND" -eq 1 ]]; then
    "${timeout_args[@]}" opencode run --dir super_metroid \
      -m "$m" "${variant_args[@]}" --auto \
      --title "$id" \
      "$prompt" 2>&1 | tee "$log"
  else
    (
      "${timeout_args[@]}" opencode run --dir super_metroid \
        -m "$m" "${variant_args[@]}" --auto \
        --title "$id" \
        "$prompt" >"$log" 2>&1
      echo "EXIT:$?" >>"$log"
    ) &
    echo "pid $!  log $log"
  fi
}

# Normalize task ids
NORMALIZED=()
for id in "${TASKS[@]}"; do
  id="${id%.md}"
  id="${id##*/}"
  NORMALIZED+=("$id")
done

check_ownership_conflicts "${NORMALIZED[@]}"

PIDS=()
for id in "${NORMALIZED[@]}"; do
  if [[ "$FOREGROUND" -eq 1 ]]; then
    run_one "$id"
  else
    run_one "$id"
    PIDS+=($!)
  fi
done

if [[ "$FOREGROUND" -eq 0 && ${#PIDS[@]} -gt 0 ]]; then
  echo "Launched ${#PIDS[@]} background session(s). Tail logs:"
  echo "  tail -f super_metroid/docs/tasks/logs/*.log"
  echo "Wait pids: ${PIDS[*]}"
fi
