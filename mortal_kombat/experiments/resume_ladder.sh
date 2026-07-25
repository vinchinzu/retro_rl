#!/usr/bin/env bash
# Resume E003 ladder training from the latest mk1_ladder_ppo checkpoint.
# Uses 2 parallel envs by default to reduce RAM/GPU pressure after a crash.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$MK_DIR"

LATEST="$(basename "$(ls -t models/mk1_ladder_ppo_*_steps.zip 2>/dev/null | head -1 || true)")"
if [[ -z "$LATEST" ]]; then
  echo "No mk1_ladder_ppo_*_steps.zip checkpoint found."
  exit 1
fi

STEPS="${1:-6000000}"
N_ENVS="${N_ENVS:-2}"
LOG="experiments/logs/ladder_resume_$(date +%Y%m%d_%H%M%S).log"
mkdir -p experiments/logs

echo "Resuming from: $LATEST"
echo "Additional steps: $STEPS | n-envs: $N_ENVS"
echo "Log: $LOG"

uv run python train_speedrun.py \
  --curriculum ladder \
  --no-randomize \
  --lr 5e-5 \
  --prefix mk1_ladder_ppo \
  --load "$LATEST" \
  --steps "$STEPS" \
  --n-envs "$N_ENVS" \
  2>&1 | tee "$LOG"
