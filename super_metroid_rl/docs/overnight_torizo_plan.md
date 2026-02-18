# Overnight Torizo Training Runbook

**Updated**: 2026-02-17 21:00 CST (Worker A audit)
**Goal**: Maximize probability of full ZebesStart -> Bomb Torizo route completion by morning
**GPU**: RTX 3060 12GB (CUDA available, ~12GB VRAM)
**Estimated wall-clock**: ~8-10 hours for full training plan

---

## 1. Current Audit Summary

### Verified Preconditions

| Check | Status | Notes |
|-------|--------|-------|
| ROM | OK | Symlink `custom_integrations/SuperMetroid-Snes/rom.sfc -> ../../roms/rom.sfc` resolves (3MB) |
| States | 12/12 OK | All segment start states present in `custom_integrations/SuperMetroid-Snes/` |
| Models | 12/12 exist | All in `models/segment_*.zip`, dated Jan 20-21 |
| data.json | OK | RAM addresses for room_id, samus_x/y, health, items, etc. |
| train_curriculum.py | OK | Smoke-tested: env creation + 128-step training pass |
| eval_torizo_integration.py | OK | 407 lines, deterministic chained evaluation |
| overnight_worker_b_train.sh | OK | Sequential training with ROM check, logging, snapshots |
| morning_worker_c_check.sh | OK | Quick 4-episode eval wrapper |

### Route (12 segments)

| # | Segment | Direction | Start State | Target Room | Trained Steps | Status |
|---|---------|-----------|-------------|-------------|---------------|--------|
| 1 | `landing_site` | left | ZebesStart | Parlor (0x92FD) | ~50k | Weak |
| 2 | `parlor_descent` | down | Parlor [from Landing Site] | Climb (0x96BA) | 200k | Moderate (HARD room) |
| 3 | `climb_descent` | down | Climb [from Parlor] | Pit Room (0x975C) | ~50k | Weak |
| 4 | `pit_room_descent` | down | Pit Room [from Climb] | Elevator (0x97B5) | ~50k | Weak |
| 5 | `elevator_descent` | down | Elevator [from Pit Room] | Morph Ball (0x9E9F) | ~50k | Weak |
| 6 | `morph_ball_collect` | collect | Morph Ball [from Elevator] | item bit 0x1 | ~20k | Weak |
| 7 | `morph_ball_return` | up | Morph Ball [from Constr. Zone] | Elevator (0x97B5) | ~50k | Weak |
| 8 | `elevator_return` | up | Elevator [from Morph Ball] | Pit Room (0x975C) | ~50k | Weak |
| 9 | `pit_room_return` | up | Pit Room [from Elevator] | Climb (0x96BA) | 5M | Strong |
| 10 | `climb_return` | up | Climb [from Pit Room] | Parlor (0x92FD) | 5M | Strong |
| 11 | `parlor_to_flyway` | right | Parlor [from Climb] | Flyway (0x9879) | ~50k | Weak |
| 12 | `flyway_to_torizo` | right | Flyway [from Parlor] | Torizo (0x9804) | ~50k | Weak |

### State File Verification (all 12 present)

```
ZebesStart.state                                    93338 bytes
Parlor and Alcatraz [from Landing Site].state        96345 bytes
Climb [from Parlor and Alcatraz].state               98283 bytes
Pit Room [from Climb].state                          99074 bytes
Blue Brinstar Elevator Room [from Pit Room].state    96731 bytes
Morph Ball Room [from Blue Brinstar Elevator Room].state  103423 bytes
Morph Ball Room [from Construction Zone].state      103361 bytes
Blue Brinstar Elevator Room [from Morph Ball Room].state   95235 bytes
Pit Room [from Blue Brinstar Elevator Room].state    96510 bytes
Climb [from Pit Room].state                          96502 bytes
Parlor and Alcatraz [from Climb].state               99191 bytes
Flyway [from Parlor and Alcatraz].state              96084 bytes
```

### Key Bottlenecks

- **8 of 12 segments** have only baseline training (20-50k steps) -- far too little
- `parlor_descent` is the hardest room architecturally (vertical descent with platforming)
- Return-phase segments (`morph_ball_return`, `elevator_return`) require climbing UP, which is harder
- `pit_room_return` and `climb_return` already have 5M steps -- these are the strongest, skip them
- `parlor_to_flyway` and `flyway_to_torizo` are connectors that need reliable transitions

### Missing Items

**None.** All states, models, ROM, and scripts verified. No creation steps needed.

---

## 2. Overnight Training Plan

### Strategy: Priority-weighted step budget

Focus training time on weakest segments with highest difficulty. Total budget: ~5.7M steps across 10 segments (skip `pit_room_return` and `climb_return` which already have 5M).

### Phase 1: Baseline Eval (15 min)

Run integration eval to identify actual failure points before training.

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl/super_metroid_rl
.venv/bin/python scripts/eval_torizo_integration.py \
  --headless \
  --episodes 8 \
  --max-steps 18000 \
  --seed-base 42 \
  --output-json logs/overnight_pre_eval.json \
  --output-summary logs/overnight_pre_eval_summary.md
```

### Phase 2: Sequential Training (~8 hours)

Train segments in priority order. Higher-failure and higher-difficulty segments get more steps.

| Priority | Segment | Steps | Est. Time | Rationale |
|----------|---------|-------|-----------|-----------|
| 1 | `parlor_descent` | 800000 | ~80 min | Hardest room -- vertical platforming, HARD tag |
| 2 | `morph_ball_return` | 800000 | ~80 min | Climbing UP through Morph Ball Room |
| 3 | `elevator_return` | 600000 | ~60 min | Vertical climbing in elevator room |
| 4 | `climb_descent` | 400000 | ~40 min | Straightforward but undertrained |
| 5 | `elevator_descent` | 400000 | ~40 min | Straightforward but undertrained |
| 6 | `morph_ball_collect` | 300000 | ~30 min | Item collection needs precision |
| 7 | `landing_site` | 300000 | ~30 min | Simple leftward movement |
| 8 | `pit_room_descent` | 300000 | ~30 min | Simple descent |
| 9 | `parlor_to_flyway` | 400000 | ~40 min | Rightward connector, critical path |
| 10 | `flyway_to_torizo` | 400000 | ~40 min | Final connector to boss room |

**Total: ~5.7M steps, ~8 hours**

### Phase 3: Post-Training Eval (15 min)

```bash
.venv/bin/python scripts/eval_torizo_integration.py \
  --headless \
  --episodes 12 \
  --max-steps 18000 \
  --seed-base 1729 \
  --output-json logs/overnight_post_eval.json \
  --output-summary logs/overnight_post_eval_summary.md
```

---

## 3. Execution Commands

### Option A: Use existing overnight script (recommended)

The script in `scripts/overnight_worker_b_train.sh` handles sequential training with logging, ROM symlinking, and snapshot backups.

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl/super_metroid_rl
PYTHON_BIN=.venv/bin/python DEVICE=cuda \
  nohup bash scripts/overnight_worker_b_train.sh "$(date +%Y%m%d_%H%M%S)" \
  > logs/overnight_full.out 2>&1 &
```

**Note**: The script's `SEGMENT_PLAN` array determines training order. Check that it matches the priority table above. Current script skips `pit_room_return` and `climb_return` (already 5M steps).

### Option B: Full 3-phase pipeline (pre-eval + train + post-eval)

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl/super_metroid_rl
PYTHON_BIN=.venv/bin/python

# Phase 1: Pre-eval
$PYTHON_BIN scripts/eval_torizo_integration.py \
  --headless --episodes 8 --max-steps 18000 --seed-base 42 \
  --output-json logs/overnight_pre_eval.json \
  --output-summary logs/overnight_pre_eval_summary.md

# Phase 2: Training (segments in priority order)
for plan in \
  "parlor_descent:800000" \
  "morph_ball_return:800000" \
  "elevator_return:600000" \
  "climb_descent:400000" \
  "elevator_descent:400000" \
  "morph_ball_collect:300000" \
  "landing_site:300000" \
  "pit_room_descent:300000" \
  "parlor_to_flyway:400000" \
  "flyway_to_torizo:400000"; do
  segment="${plan%%:*}"
  steps="${plan##*:}"
  echo "[$(date -Iseconds)] Training $segment for $steps steps..."
  $PYTHON_BIN train_curriculum.py train \
    --segment "$segment" --steps "$steps" --device cuda \
    --load "models/segment_${segment}.zip" \
    2>&1 | tee "logs/overnight_train_${segment}.out"
done

# Phase 3: Post-eval
$PYTHON_BIN scripts/eval_torizo_integration.py \
  --headless --episodes 12 --max-steps 18000 --seed-base 1729 \
  --output-json logs/overnight_post_eval.json \
  --output-summary logs/overnight_post_eval_summary.md
```

### Option C: Individual segment training

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl/super_metroid_rl
.venv/bin/python train_curriculum.py train \
  --segment SEGMENT_NAME --steps STEP_COUNT --device cuda \
  --load models/segment_SEGMENT_NAME.zip
```

### Useful utilities

```bash
# List all segments and their training status
.venv/bin/python train_curriculum.py list-segments

# Watch trained agent (visual, needs display)
.venv/bin/python train_curriculum.py run --render --start-state ZebesStart

# Monitor GPU during training
watch -n 5 nvidia-smi

# Tail training log
tail -f logs/overnight_train_parlor_descent.out
```

---

## 4. Success Criteria

### Per-Segment

- Segment model prints `[SUCCESS]` within `max_steps` window in >50% of training episodes
- Training reward trend is positive (check tensorboard or monitor CSV)
- No NaN/Inf in reward output

### Full Route Integration

- **Minimum**: >=1/12 episodes complete the full route (ZebesStart -> Torizo room)
- **Good**: >=3/12 episodes complete the full route
- **Excellent**: >=6/12 episodes complete the full route
- All 12 segments individually reachable (per-segment completion rate >60%)

### Failure Signals

- Any segment with 0% completion rate in eval = needs targeted 500k+ retraining
- `parlor_descent` remains the most likely bottleneck (vertical platforming)
- Return-phase climbing segments may need demo-augmented training
- If a segment trains 800k steps but still fails >80%, the reward shaping or max_steps budget may need tuning

---

## 5. Morning Validation Checklist

### Quick one-liner morning check:

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl/super_metroid_rl && \
  bash scripts/morning_worker_c_check.sh && \
  cat logs/overnight_worker_c_summary.md
```

### Step-by-step:

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl/super_metroid_rl

# 1. Check overnight training finished (look for "Completed successfully")
tail -20 logs/overnight_worker_b_*.out 2>/dev/null | tail -25

# 2. Verify models updated recently (should show today's date)
ls -lt models/segment_*.zip | head -14

# 3. Run quick integration eval (4 episodes, ~5 min)
bash scripts/morning_worker_c_check.sh

# 4. Read the eval summary
cat logs/overnight_worker_c_summary.md

# 5. Check for failure clusters
grep -A 20 "Failure Clusters" logs/overnight_worker_c_summary.md

# 6. Check per-segment completion rates
grep -A 15 "Transition Completion" logs/overnight_worker_c_summary.md

# 7. Visual spot-check (optional, needs display)
.venv/bin/python train_curriculum.py run --render --start-state ZebesStart
```

### If route doesn't complete:

1. Identify the failing segment:
   ```bash
   python3 -c "
   import json
   data = json.load(open('logs/overnight_worker_c_eval.json'))
   fails = data['summary']['transition_failure_clusters']
   for seg, count in sorted(fails.items(), key=lambda x: -x[1]):
       print(f'  {seg}: {count} failures')
   "
   ```

2. Retrain the worst segment with more steps:
   ```bash
   .venv/bin/python train_curriculum.py train \
     --segment FAILING_SEGMENT --steps 500000 --device cuda
   ```

3. Re-run eval to check improvement:
   ```bash
   bash scripts/morning_worker_c_check.sh && \
     cat logs/overnight_worker_c_summary.md
   ```

4. If `parlor_descent` is still the bottleneck after 1M+ total steps, consider:
   - Increasing `max_steps` from 4000 to 6000 in ROUTE_SEGMENTS
   - Adding demo data to `boss_data/nav_demos.npz`
   - Adding intermediate waypoint rewards for that room

---

## 6. Architecture Notes

### Segment Model Architecture
- **CNN**: 3-layer ConvNet (32->64->64 filters) + 512-unit FC
- **Policy**: PPO with pi=[256,128], vf=[256,128]
- **Input**: 4-frame RGB stack, downsampled 2x (12 channels, 112x128)
- **Actions**: 26 discrete actions (SNES button combos)
- **Action repeat**: Random hold 2-4 frames per action
- **PPO params**: lr=3e-4, ent_coef=0.02->0.005 (annealed), n_steps=1024, batch=256, epochs=4

### Reward Structure (SegmentReward)
- **Success**: +2000 for reaching target room + speed bonus (0.5 * frames_saved)
- **Direction progress**: +0.3 per pixel in correct direction
- **Wrong direction**: -0.4 per pixel backward
- **Door transition**: +50 for any door transition
- **Damage**: -2x HP lost
- **Death**: -300
- **Time penalty**: -0.2 per step
- **Action intent**: +/-0.08 for pressing correct directional buttons
- **parlor_descent special**: +0.1 bonus for leftward movement (helps find the down-left path)

### Integration Eval (eval_torizo_integration.py)
- Deterministic policy (no exploration)
- Seeded action-hold RNG for reproducibility
- Chains all 12 segment models sequentially
- Reports per-segment completion rates and failure clusters
- Outputs JSON metrics + Markdown summary
- Auto-generates retrain commands for failing segments

### Key File Paths
```
train_curriculum.py                    Main training + route runner
scripts/eval_torizo_integration.py     Integration evaluation
scripts/overnight_worker_b_train.sh    Overnight training orchestrator
scripts/morning_worker_c_check.sh      Morning eval one-liner
custom_integrations/SuperMetroid-Snes/ State files + data.json + ROM
models/segment_*.zip                   Trained segment models
logs/overnight_*.out                   Training/eval logs
world_map.json                         Room name -> hex ID mapping
```
