#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="super_mario_bros/optimizer/logs/hybrid_bg_${TS}"
mkdir -p "$LOG_DIR"

REGISTRY="super_mario_bros/optimizer/model_registry.json"

# Heavy defaults for weak-segment improvement. Override via env if needed.
GA_GENERATIONS="${GA_GENERATIONS:-16}"
GA_POPULATION="${GA_POPULATION:-40}"
HILL_ITERS="${HILL_ITERS:-2500}"
HILL_RAW_ITERS="${HILL_RAW_ITERS:-2500}"
MAX_CANDIDATES="${MAX_CANDIDATES:-20}"

quote_args() {
  local out=""
  local arg
  for arg in "$@"; do
    out+=" $(printf '%q' "$arg")"
  done
  echo "${out# }"
}

launch_detached() {
  local name="$1"
  local log_file="$2"
  local pid_file="$3"
  shift 3

  local cmd_quoted
  cmd_quoted="$(quote_args "$@")"

  # Use setsid so process survives shell/tool teardown.
  setsid -f bash -lc "cd $(printf '%q' "$ROOT_DIR") && echo \\$BASHPID > $(printf '%q' "$pid_file") && exec env PYTHONUNBUFFERED=1 $cmd_quoted > $(printf '%q' "$log_file") 2>&1"

  sleep 0.2
  local pid=""
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
  fi

  printf "%s\t%s\t%s\n" "$name" "$pid" "$log_file" | tee -a "$LOG_DIR/pids.tsv" >/dev/null
  echo "started $name pid=$pid log=$log_file"
}

COMMON_ARGS=(
  uv run python -m super_mario_bros.hybrid_pipeline run
  --route smb_any_percent
  --selection-context chained
  --force-eval
  --max-candidates "$MAX_CANDIDATES"
  --weak-top-k 1
  --ga-generations "$GA_GENERATIONS"
  --ga-population "$GA_POPULATION"
  --hill-iterations "$HILL_ITERS"
  --hill-raw-iterations "$HILL_RAW_ITERS"
  --registry "$REGISTRY"
)

launch_segment() {
  local seg="$1"
  local log_file="$LOG_DIR/${seg}.log"
  local pid_file="$LOG_DIR/${seg}.pid"
  local report_file="super_mario_bros/optimizer/hybrid_report_${seg}_${TS}.json"

  launch_detached "$seg" "$log_file" "$pid_file" \
    "${COMMON_ARGS[@]}" \
    --segments "$seg" \
    --report "$report_file"
}

ANALYZE_LOG="$LOG_DIR/analyze.log"
ANALYZE_PID_FILE="$LOG_DIR/analyze.pid"
ANALYZE_REPORT="super_mario_bros/optimizer/hybrid_report_analyze_${TS}.json"

launch_detached "analyze" "$ANALYZE_LOG" "$ANALYZE_PID_FILE" \
  uv run python -m super_mario_bros.hybrid_pipeline analyze \
  --route smb_any_percent \
  --selection-context chained \
  --force-eval \
  --max-candidates "$MAX_CANDIDATES" \
  --report "$ANALYZE_REPORT" \
  --registry "$REGISTRY"

# Weak segments from current profile.
launch_segment "smb_8_1"
launch_segment "smb_8_2"
launch_segment "smb_8_4"

echo
echo "launched background jobs; pid table: $LOG_DIR/pids.tsv"
echo "tail -f $LOG_DIR/*.log"
