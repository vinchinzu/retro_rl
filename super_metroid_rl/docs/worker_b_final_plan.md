# Worker B Final Status & Plan

## Overnight Run: `overnight_feb17v2` (completed 2026-02-18 03:44 CST)

**All 10 segments trained successfully. Zero failures.**

### Bug Fixed This Session
- `SegmentReward.reset()` returned `room_id=0` because retro env gives empty info on reset.
  This caused `parlor_descent` to immediately terminate (ep_len=1) and collapse after 5M steps.
  Fixed with a no-op step in reset to populate info. Also added `--fresh` flag to `train_curriculum.py`.

---

## Segment Status After Training

### Completing segments (reaching target room)

| Segment | Total Steps | ep_rew | ep_len | Successes | Verdict |
|---------|-----------|--------|--------|-----------|---------|
| elevator_return | ~250k | **+6300** | 119 | **1239** | **STRONG** - completing fast |
| climb_descent | ~250k | **+3150** | 961 | **262** | **STRONG** - reliable |
| elevator_descent | ~250k | **+4100** | 960 | **114** | **STRONG** - reliable |
| pit_room_descent | ~250k | -445 | 1990 | 3 | Marginal - rare completions |

### Non-completing segments (never reached target room)

| Segment | Total Steps | ep_rew | ep_len | Successes | Verdict |
|---------|-----------|--------|--------|-----------|---------|
| parlor_descent | 500k (fresh) | -584 | 4000 | 0 | **WEAK** - exploring but stuck |
| morph_ball_return | ~550k | -1080 | 4000 | 0 | **WEAK** - no progress |
| parlor_to_flyway | ~350k | -471 | 2890 | 0 | **WEAK** - no completions |
| flyway_to_torizo | ~350k | -557 | 1830 | 0 | **WEAK** - no completions |
| pit_room_return | ~5.3M | -798 | 3000 | 0 | **STUCK** - 5M+ steps, no solution |
| climb_return | ~5.3M | -2160 | 4000 | 0 | **STUCK** - 5M+ steps, no solution |

### Not retrained (already good from before)

| Segment | Status |
|---------|--------|
| landing_site | Trained (50k), works |
| morph_ball_collect | Trained (20k), works |

---

## Checkpoint Locations

All live models: `models/segment_{name}.zip`
Overnight snapshots: `models/segment_{name}_worker_b_overnight_feb17v2_{steps}steps.zip`
Collapsed parlor_descent backup: `models/segment_parlor_descent_collapsed_backup.zip`

---

## Key Finding

**Pure RL with directional rewards cannot solve the hard navigation segments.**
After 500k-5M steps, 6 of 12 segments have ZERO completions. The agent explores
within rooms but never discovers the correct door transitions for:
- Upward returns (morph_ball_return, pit_room_return, climb_return)
- Complex navigation (parlor_descent, parlor_to_flyway, flyway_to_torizo)

The 4 segments that DO complete are simpler transitions where the directional
reward naturally guides the agent to the exit.

---

## Concrete Next Steps

### 1. Demo-guided training for stuck segments (HIGH PRIORITY)
The reward shaping alone isn't enough. These segments need human demonstrations:
- Record playthroughs with `play_session.py` or similar for each stuck segment
- Save as `boss_data/nav_demos.npz` or per-segment demo files
- The `_demo_match_reward()` infrastructure already exists in SegmentReward
- Even 1-2 demos per segment should dramatically help exploration

### 2. Waypoint rewards instead of pure directional (HIGH PRIORITY)
Current rewards only incentivize X/Y movement in one direction. Rooms have doors
at specific locations. Add intermediate waypoint rewards:
- For `parlor_descent`: reward approaching the door at bottom-left
- For return segments: reward approaching the upward door coordinates
- Can extract door positions from the states or world_map.json

### 3. Room transition chain rewards
Currently the agent gets +2000 for reaching the EXACT target room but -500 for
entering any other room. For multi-room segments, consider:
- Positive reward for entering any room that's "on the path"
- Remove the harsh -500 penalty for `parlor_descent` (it prevents exploration)

### 4. Increase max_steps for stuck segments
`parlor_descent` and return segments may need more than 4000 steps (with
ActionHoldRepeat, that's only ~10k-20k raw frames). Consider 8000-10000.

### 5. Fine-tune completing segments for speed
`elevator_return`, `climb_descent`, `elevator_descent` all complete reliably.
The speed bonus already exists but could be tuned higher to optimize frame count.

### 6. Integration test
Once more segments complete, run `python train_curriculum.py run --render` to
test the full ZebesStart→Torizo route end-to-end with all segment models chained.

---

## Files Changed This Session
- `train_curriculum.py`: Fixed reset() room_id bug, added --fresh flag
- `scripts/overnight_worker_b_train.sh`: Updated with FRESH mode, smoke test, priority plan
- `docs/overnight_worker_b_summary.md`: Pre-training status assessment
- `docs/worker_b_final_plan.md`: This file
