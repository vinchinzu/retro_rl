# Worker C Final Report: Torizo Integration Evaluation

**Date**: 2026-02-18
**Scope**: ZebesStart → flyway_to_torizo (12 segments)

---

## 1. Test Results

### Per-Segment Isolated Eval (stochastic policy, 5 trials each)

| # | Segment | Pass Rate | Status | Avg Steps | Training |
|---|---------|-----------|--------|-----------|----------|
| 1 | `landing_site` | 5/5 (100%) | OK | 843 | ~50k |
| 2 | `parlor_descent` | 0/5 (0%) | FAIL | 4000 (max) | 200k |
| 3 | `climb_descent` | 5/5 (100%) | OK | 716 | ~50k |
| 4 | `pit_room_descent` | 3/5 (60%) | OK | 1841 | ~50k |
| 5 | `elevator_descent` | 0/5 (0%) | FAIL | 3000 (max) | ~50k |
| 6 | `morph_ball_collect` | 0/5 (0%) | FAIL | 3000 (max) | ~20k |
| 7 | `morph_ball_return` | 0/5 (0%) | FAIL | 4000 (max) | ~50k |
| 8 | `elevator_return` | 5/5 (100%) | OK | 195 | ~50k |
| 9 | `pit_room_return` | 0/5 (0%) | FAIL | 3000 (max) | 5M |
| 10 | `climb_return` | 0/5 (0%) | FAIL | 4000 (max) | 5M |
| 11 | `parlor_to_flyway` | 0/5 (0%) | FAIL | 3000 (max) | ~50k |
| 12 | `flyway_to_torizo` | 0/5 (0%) | FAIL | 1563 | ~50k |

### Chained Route Eval (12 episodes, stochastic)

- **Success rate**: 0/12 (0%)
- **Best progress**: 1/12 segments (landing_site only)
- **Bottleneck**: `parlor_descent` blocks all 12 episodes (100% failure)
- Every episode completes landing_site then exhausts its 4000-step budget stuck in Parlor

---

## 2. Issues Found

### Critical: Deterministic Policy Deadlock
The original eval used `deterministic=True` which causes the landing_site model to converge to a single repeated action (action 10 = jump-left) at position x=37, y=1163. The agent reaches the left wall but never enters the door. Stochastic policy fixes this for landing_site but 8/12 segments still fail.

### Critical: 8/12 Segments Non-Functional
Only 4 segments can complete their transitions even in isolation. The route is fundamentally blocked.

### Suspicious: 5M-Step Models Still Fail
`pit_room_return` and `climb_return` each have 5M training steps but score 0/5 in isolated eval. Possible causes:
- **Env mismatch**: Training used `ActionHoldRepeat` with fixed hold=4 via `hold_sampler()`, but eval uses `SeededActionHoldRepeat` with random hold 2-4. The different frame skip changes the observation timing the model was trained on.
- **Reward wrapper absent**: Training wraps with `SegmentReward` (which terminates on success), but eval has no reward wrapper — the model may have learned to rely on reward-shaped exploration behavior that doesn't transfer.
- **Observation difference**: Training env creates env via `make_segment_env()` (includes `TimeLimit`), eval creates bare env. TimeLimit truncation may have helped exploration during training.

### Worker B Training Failures
The overnight Worker B training run (2026-02-17 20:52) failed for all 7 segments it attempted — all hit `FileNotFoundError: No romfiles found for game: SuperMetroid-Snes`. The ROM file exists and resolves correctly when running from the project directory, suggesting Worker B's training script ran from a different CWD.

---

## 3. Actionable Next Steps

### Priority 1: Fix Eval/Training Env Parity
The eval env must match the training env wrapper stack. Key differences to resolve:
- Use same action hold behavior (fixed hold=4 or match training's `hold_sampler`)
- Consider adding the `SegmentReward` wrapper during eval (even though we only check room transitions, the wrapper's termination logic may affect frame timing)

### Priority 2: Retrain `parlor_descent` (Route Blocker)
This is the single segment blocking all route progress. It's the hardest room (vertical descent with platforming).

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl/super_metroid_rl
.venv/bin/python train_curriculum.py train --segment parlor_descent --steps 800000 --device cuda
```

### Priority 3: Retrain All Failing Segments
Full batch (copy-paste ready):

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl/super_metroid_rl
for plan in \
  "parlor_descent:800000" \
  "elevator_descent:500000" \
  "morph_ball_collect:500000" \
  "morph_ball_return:500000" \
  "pit_room_return:500000" \
  "climb_return:500000" \
  "parlor_to_flyway:500000" \
  "flyway_to_torizo:500000"; do
  segment="${plan%%:*}"
  steps="${plan##*:}"
  echo "[$(date -Iseconds)] Training $segment for $steps steps..."
  .venv/bin/python train_curriculum.py train \
    --segment "$segment" --steps "$steps" --device cuda \
    2>&1 | tee "logs/retrain_${segment}.out"
done
```

### Priority 4: Investigate 5M Model Failures
Before retraining pit_room_return and climb_return again, verify:
1. Load the model in the **training** env (via `make_segment_env`) and check if it completes
2. If yes → eval env mismatch is the root cause, fix the eval wrapper stack
3. If no → models are degraded, retrain from scratch

```bash
# Quick diagnostic
.venv/bin/python -c "
from train_curriculum import *
import torch
for seg in ['pit_room_return', 'climb_return']:
    segment = ROUTE_SEGMENTS[seg]
    env = make_segment_env(segment)
    model = PPO.load(f'models/segment_{seg}.zip', device='cuda')
    obs, info = env.reset()
    for step in range(segment.max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc: break
    print(f'{seg}: room=0x{info.get(\"room_id\",0):04X} steps={step}')
    env.close()
"
```

### Priority 5: Fix Worker B ROM Resolution
Ensure overnight training scripts use absolute paths or `cd` to project root before running. The `overnight_worker_b_train.sh` script should set:
```bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
```

---

## 4. Artifacts Produced

| File | Description |
|------|-------------|
| `scripts/eval_torizo_integration.py` | Integration eval (isolated + chained, stochastic default) |
| `scripts/morning_worker_c_check.sh` | One-command morning validation |
| `logs/overnight_worker_c_eval.json` | Full metrics JSON (gitignored) |
| `logs/overnight_worker_c_summary.md` | Human-readable summary (gitignored) |
| `docs/worker_c_final_plan.md` | This document |

---

## 5. Morning Checklist

```bash
# 1. Quick eval (5 min)
bash scripts/morning_worker_c_check.sh

# 2. Read summary
cat logs/overnight_worker_c_summary.md

# 3. If retraining happened, re-eval
bash scripts/morning_worker_c_check.sh && cat logs/overnight_worker_c_summary.md
```
