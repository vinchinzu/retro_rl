#!/usr/bin/env bash
# Dispatch one or more Super Metroid atomic task cards to OpenCode.
#
# Usage (from repo root):
#   ./super_metroid/scripts/dispatch_opencode.sh SM-K4-03
#   ./super_metroid/scripts/dispatch_opencode.sh --model flash SM-K4-05
#   ./super_metroid/scripts/dispatch_opencode.sh SM-K4-03 SM-K4-04 SM-K4-05
#   ./super_metroid/scripts/dispatch_opencode.sh --foreground SM-K4-04
#
# Defaults: Luna, --auto, background (logs under docs/tasks/logs/, gitignored).
# Parallel args launch concurrent sessions; only use when cards touch disjoint files.
#
# Override provider routing without editing the script:
#   SM_OPENCODE_MODEL_LUNA=... SM_OPENCODE_MODEL_FLASH=... ./.../dispatch_opencode.sh ...
# Auth is never stored here — use the local OpenCode/user config outside the repo.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_LUNA="${SM_OPENCODE_MODEL_LUNA:-openrouter/openai/gpt-5.6-luna}"
MODEL_FLASH="${SM_OPENCODE_MODEL_FLASH:-openrouter/deepseek/deepseek-v4-flash}"
MODEL="$MODEL_LUNA"
FOREGROUND=0
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
    --foreground|-f) FOREGROUND=1 ;;
    --help|-h)
      sed -n '2,14p' "$0"
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
  echo "Usage: $0 [--flash|--luna|--model NAME] [--foreground] SM-K4-NN [...]" >&2
  exit 2
fi

LOG_DIR="$ROOT/super_metroid/docs/tasks/logs"
mkdir -p "$LOG_DIR"

pick_model_for_task() {
  local id="$1"
  # Prefer Flash for docs/report-only cards when caller left default Luna.
  # Implement suffixes (*B, *-IMPL) and geometry/scaffold stay Luna.
  if [[ "$MODEL" == "$MODEL_LUNA" ]]; then
    case "$id" in
      SM-TIGHTEN-*B|SM-TIGHTEN-*-IMPL|SM-*-IMPL) echo "$MODEL_LUNA" ;;
      SM-K4-03|SM-K4-05|SM-K3-TRACK) echo "$MODEL_FLASH" ;;
      SM-TIGHTEN-0[0-9]|SM-TIGHTEN-0[0-9]-report) echo "$MODEL_FLASH" ;;
      SM-TIGHTEN-0[0-9][A-Z]*) echo "$MODEL_LUNA" ;;  # e.g. 01B if above missed
      *) echo "$MODEL" ;;
    esac
  else
    echo "$MODEL"
  fi
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

Throughput rules:
- Own only the files listed under Own files / Do. Never edit continuous.py or STATUS.md unless the card says so.
- Read ALL Read-first files before editing. Use multiple tools (grep, read, pytest) — do not stop after the first skim.
- Implement the full Do list; prefer complete acceptance over a minimal partial.
- Run every Verify command; paste outputs into the final message.
- If pure probe / emu is optional and fails, report residual state — never force-pass or STATUS-promote.
- No progression/capacity/boss-bit RAM forges for green claims.

Paths:
- OpenCode cwd is the super_metroid/ package root.
- Read/Edit: package-relative (tests/..., routes/..., combat/..., docs/...).
- Shell: either \`cd\` to monorepo root and use super_metroid/... paths, or use paths that work from cwd; if a path 404s, retry without the super_metroid/ prefix.

Super-clean residual (required in final message — all bullets):
1. Files changed (path list) + one-line purpose each
2. Verify command stdout/stderr paste (or explicit skip reason)
3. Acceptance checklist: each [ ] from the card marked pass/fail
4. Residual risks: what still blocks pure-green / continuous / STATUS
5. Planner next: single recommended next card or decision (one sentence)
6. Explicit non-claims: did not STATUS-promote; did not forge progression RAM
7. If probe failed: last room/pose/x/y (or N/A for docs-only)

--- TASK CARD ---
${prompt}"

  echo "=== $id → $m (log: $log) ==="
  if [[ "$FOREGROUND" -eq 1 ]]; then
    opencode run --dir super_metroid \
      -m "$m" --auto \
      --title "$id" \
      "$prompt" 2>&1 | tee "$log"
  else
    (
      opencode run --dir super_metroid \
        -m "$m" --auto \
        --title "$id" \
        "$prompt" >"$log" 2>&1
      echo "EXIT:$?" >>"$log"
    ) &
    echo "pid $!  log $log"
  fi
}

PIDS=()
for id in "${TASKS[@]}"; do
  # Normalize SM-K4-03.md → SM-K4-03
  id="${id%.md}"
  id="${id##*/}"
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
