# Worker A Final Status & Plan

**Written**: 2026-02-18 ~03:45 CST
**Role**: Planning + state audit for overnight Torizo training

---

## What Was Completed

1. **Full project audit**: Verified ROM, all 12 states, all 12 models, training pipeline, eval scripts
2. **Smoke test**: Ran 128-step training pass to confirm end-to-end pipeline works
3. **Runbook**: Updated `docs/overnight_torizo_plan.md` with verified preconditions, exact commands, success criteria, morning checklist
4. **Committed**: `89813dd plan: overnight torizo training runbook`

---

## Current State (as of wrap-up)

### Worker B Training — COMPLETE (all 10 segments done)

Worker B finished all 10 segments successfully (6h38m total):

| Segment | Steps | Duration | Finished |
|---------|-------|----------|----------|
| `parlor_descent` | 500k (FRESH) | ~73 min | 22:19 |
| `morph_ball_return` | 500k | ~63 min | 23:22 |
| `parlor_to_flyway` | 300k | ~39 min | 00:02 |
| `flyway_to_torizo` | 300k | ~39 min | 00:41 |
| `pit_room_return` | 300k | ~39 min | 01:20 |
| `climb_return` | 300k | ~39 min | 01:59 |
| `elevator_return` | 200k | ~26 min | 02:25 |
| `climb_descent` | 200k | ~26 min | 02:52 |
| `pit_room_descent` | 200k | ~26 min | 03:18 |
| `elevator_descent` | 200k | ~26 min | 03:44 |

**Not trained this run**: `landing_site` (already OK at 100% isolated), `morph_ball_collect` (not in Worker B plan).

### Worker C Eval — Results Available

Eval ran with both isolated and chained modes. Key findings:

**Isolated segment pass rates:**
- 100%: `landing_site`, `climb_descent`, `elevator_return`
- 60%: `pit_room_descent`
- 0%: `parlor_descent`, `elevator_descent`, `morph_ball_collect`, `morph_ball_return`, `pit_room_return`, `climb_return`, `parlor_to_flyway`, `flyway_to_torizo`

**Chained route**: 0/12 episodes succeeded. All 12 failed at `parlor_descent`.

**Important caveat**: Worker C eval ran at ~03:39 UTC (21:39 CST) — this was BEFORE Worker B finished training most segments. The eval used the OLD models. **A fresh eval is needed to measure post-training improvement.**

### Model State After Training

All models in `models/` now have significantly more training:
- `parlor_descent`: 500k steps (fresh start, was 200k)
- `morph_ball_return`: 550k total
- `parlor_to_flyway`, `flyway_to_torizo`: 350k total each
- `pit_room_return`, `climb_return`: 5.3M total each
- `elevator_return`: 250k total
- `climb_descent`, `pit_room_descent`, `elevator_descent`: 250k total each
- `landing_site`: ~50k (unchanged, was already 100% pass rate)
- `morph_ball_collect`: ~20k (unchanged, NOT retrained)

---

## Critical Next Steps (Morning)

### 1. Re-run eval with updated models (FIRST PRIORITY)

The Worker C eval used pre-training models. Must re-evaluate:

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl/super_metroid_rl
bash scripts/morning_worker_c_check.sh && cat logs/overnight_worker_c_summary.md
```

### 2. Likely bottlenecks to address

Based on the pre-training eval, these segments need attention if still failing:

| Segment | Issue | Suggested Fix |
|---------|-------|---------------|
| `parlor_descent` | Hardest room, 0% before retraining | Check if 500k fresh helped. If still <50%, needs 1M+ or demo data |
| `morph_ball_collect` | 0% isolated, NOT retrained overnight | Train 300-500k steps immediately |
| `elevator_descent` | 0% isolated, got 200k overnight | May need more; check post-eval |
| `morph_ball_return` | 0% isolated, got 500k overnight | Check post-eval |

### 3. If `parlor_descent` still fails

This is the single-point-of-failure for the entire chained route. Options:
- Train 1M+ more steps with increased `max_steps` (4000 → 6000)
- Add demo trajectories to `boss_data/nav_demos.npz`
- Add intermediate waypoint rewards for the Parlor → Climb path
- Consider the Parlor room topology — the path goes down-left, may need stronger left+down reward shaping

### 4. Train `morph_ball_collect` (skipped overnight)

```bash
.venv/bin/python train_curriculum.py train \
  --segment morph_ball_collect --steps 500000 --device cuda
```

### 5. Full retrain batch if many segments still failing

```bash
for plan in \
  "parlor_descent:1000000" \
  "morph_ball_collect:500000" \
  "elevator_descent:300000" \
  "morph_ball_return:300000"; do
  segment="${plan%%:*}"; steps="${plan##*:}"
  .venv/bin/python train_curriculum.py train \
    --segment "$segment" --steps "$steps" --device cuda
done
```

---

## Files Changed This Session

| File | Action |
|------|--------|
| `docs/overnight_torizo_plan.md` | Updated with full audit, verified commands, morning checklist |
| `docs/worker_a_final_plan.md` | This file — final status and next steps |
