#!/usr/bin/env bash
# MK1 experiment runner — autoresearch-inspired eval + logging stub.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MK_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS="$SCRIPT_DIR/results.tsv"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$RESULTS" ]]; then
  printf 'experiment_id\tdate\thypothesis\tmodel\tfull_clear_rate\tper_stage\tnotes\tdecision\n' \
    > "$RESULTS"
fi

usage() {
  cat <<'EOF'
Usage:
  run_experiment.sh baseline
  run_experiment.sh eval <model_path> <experiment_id> [notes]

Examples:
  ./experiments/run_experiment.sh baseline
  ./experiments/run_experiment.sh eval mk1_fresh_ppo_final.zip E001
EOF
  exit 1
}

run_tournament() {
  local general="$1"
  local log="$2"
  cd "$MK_DIR"
  uv run python speedrun_multimodel.py \
    --general "$general" \
    --attempts 20 \
    --tournament 100 \
    2>&1 | tee "$log"
}

parse_clear_rate() {
  local log="$1"
  grep -E '^Full clears:' "$log" | tail -1 | sed -n 's/.*(\([0-9.]*\)%).*/\1/p'
}

append_result() {
  local exp_id="$1"
  local hypothesis="$2"
  local model="$3"
  local clear_rate="$4"
  local notes="$5"
  local decision="${6:-pending}"
  local date
  date="$(date -Iseconds)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$exp_id" "$date" "$hypothesis" "$model" "$clear_rate" "see_log" "$notes" "$decision" \
    >> "$RESULTS"
}

cmd="${1:-}"
case "$cmd" in
  baseline)
    # speedrun general for M2-M7; Fight/Goro/Shang use STAGE_MODELS overrides
    general="mk1_speedrun_ppo_final.zip"
    log="$LOG_DIR/baseline_$(date +%Y%m%d_%H%M%S).log"
    echo "Running baseline tournament (N=100) with $general"
    run_tournament "$general" "$log"
    rate="$(parse_clear_rate "$log" || echo "NA")"
    append_result "baseline" "multimodel baseline" "$general" "$rate" "log=$log"
    echo "Baseline full_clear_rate: ${rate}%"
    echo "Logged to $RESULTS"
    ;;
  eval)
    [[ $# -ge 3 ]] || usage
    model="$2"
    exp_id="$3"
    notes="${4:-}"
    log="$LOG_DIR/${exp_id}_$(date +%Y%m%d_%H%M%S).log"
    echo "Evaluating $model for $exp_id"
    run_tournament "$model" "$log"
    rate="$(parse_clear_rate "$log" || echo "NA")"
    append_result "$exp_id" "$notes" "$model" "$rate" "log=$log"
    echo "full_clear_rate: ${rate}%"
    echo "Logged to $RESULTS"
    ;;
  *)
    usage
    ;;
esac
