# Overnight Worker B Training Summary (Feb 17, 2026)

## Run ID: `overnight_feb17v2`

### Critical Bug Fixed
- **SegmentReward reset() returned room_id=0**: retro env returns empty `info` dict
  on `reset()`, so `start_room` was always 0. This caused `parlor_descent` (which has
  a "wrong room" termination check) to immediately terminate every episode (ep_len=1).
- **Fix**: Added no-op step in `reset()` to populate info before reading room_id.
- **--fresh flag added**: `train_curriculum.py train --fresh` skips auto-loading
  existing checkpoints, needed because parlor_descent's 5M model was collapsed.

### Segment Status Before Training

| Segment | Steps Trained | ep_rew_mean | ep_len_mean | Status |
|---------|-------------|-------------|-------------|--------|
| landing_site | 50k | good | ~200 | OK |
| parlor_descent | 5M (collapsed) | -478 | 1 | BROKEN - model collapsed |
| climb_descent | 50k | good | ~1000 | OK |
| pit_room_descent | 50k | decent | ~2000 | OK |
| elevator_descent | 50k | decent | ~300 | OK |
| morph_ball_collect | 20k | decent | ~4000 | OK |
| morph_ball_return | 50k | 2170 | 4000 | Needs more training |
| elevator_return | 50k | 6100 | ~200 | Strong (completing fast) |
| pit_room_return | 5M | 1840 | 3000 | Active learning |
| climb_return | 5M | 2600 | 4000 | Active learning |
| parlor_to_flyway | 50k | 560 | 3000 | Needs more training |
| flyway_to_torizo | 50k | 140 | 2000 | Needs more training |

### Training Plan (overnight_feb17v2)

| Priority | Segment | Steps | Mode | Est. Time |
|----------|---------|-------|------|-----------|
| 1 | parlor_descent | 500k | FRESH | ~75 min |
| 2 | morph_ball_return | 500k | RESUME | ~75 min |
| 3 | parlor_to_flyway | 300k | RESUME | ~45 min |
| 4 | flyway_to_torizo | 300k | RESUME | ~45 min |
| 5 | pit_room_return | 300k | RESUME | ~45 min |
| 6 | climb_return | 300k | RESUME | ~45 min |
| 7 | elevator_return | 200k | RESUME | ~30 min |
| 8 | climb_descent | 200k | RESUME | ~30 min |
| 9 | pit_room_descent | 200k | RESUME | ~30 min |
| 10 | elevator_descent | 200k | RESUME | ~30 min |

**Total estimated**: ~7.5 hours

### Key Observations
- `parlor_descent` was completely collapsed at 5M steps (kl=0, clip=0, entropy=-1.1).
  The model had converged to a degenerate policy. Root cause: incorrect start_room=0
  caused immediate termination, so the model learned to do nothing.
- `elevator_return` is the strongest segment - completing in ~200 frames with high reward.
- Return segments (morph_ball_return, pit_room_return, climb_return) are harder than
  descent - they require upward platforming.
- `parlor_to_flyway` and `flyway_to_torizo` had minimal training (50k each) and need
  significant improvement to complete the route.

### Files Changed
- `train_curriculum.py`: Fixed reset() room_id bug, added --fresh flag
- `scripts/overnight_worker_b_train.sh`: Updated training plan, added FRESH mode support,
  smoke test, improved logging
