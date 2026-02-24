# DKC Winky's Walkway Optimization Runs

## Current Best

**`speed_v4_fresh/ga_gen0050_best.json`** - GA with 10-action DKC_SPEED_ACTIONS, threshold=4600
- VERIFIED real level completion (level_id ends at 0x2E = next level)
- **1803 frames (30.05s)**, progress=4942, fitness=98198
- Hill climbing in progress for further frame savings
- **Watch:** `uv run python -m donkey_kong_country.optimizer watch --actions donkey_kong_country/optimizer/runs/speed_v4_fresh/ga_gen0050_best.json`

## Run History

### Phase 1: Original 14-action table (DEFAULT_PLATFORMER_ACTIONS)

Files in `runs/` root:
- `*_extracted.json` - BK2 recording extractions (human gameplay seeds)
- `best_human.json` - Best human recording
- `sprint_jump_seed*.json` - Synthetic seeds (run+jump patterns)
- `ga_gen0050_best.json` - GA best with old 14-action table, completion_min_progress=4000
  - 2452 frames (40.9s), but counted bonus room as completion
- `hillclimb_v1/` - Hill climb from old GA best
  - Improved to 2260 frames (37.7s), but same bonus room false positive

### Phase 2: DKC_SPEED_ACTIONS (10-action table with Y always held)

Action table change: removed walk-only actions (RIGHT, LEFT, RIGHT+B, LEFT+B)
that give 0 camera progress. Added RIGHT/LEFT without Y for cartwheel re-taps.

- `speed_seed_fresh.json` - Synthetic speed seed (run+jump+Y-release pattern)
- `speed_seed_converted.json` - Old GA best converted to speed action indices

#### speed_v3/ (OBSOLETE - bonus room false positive)
- GA with 10-action table but completion_min_progress=4000 (too low)
- "Completed" at 1662 frames but actually entered bonus room 2 (level_id=0x51)
- Second bonus room triggers level_id change at progress ~4001, just above old 4000 threshold

#### speed_v4_fresh/ (CURRENT - verified real completion)
- GA with 10-action table and completion_min_progress=4600
- Threshold 4600 blocks ALL bonus rooms (real exit is at progress ~4673)
- Gen 10: first real completion at 1845 frames (30.75s)
- Gen 30: improved to 1803 frames (30.05s) - verified ends at level_id=0x2E
- Hill climbing from gen 50 checkpoint for further optimization

## Key Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| completion_min_progress | 4600.0 | Blocks bonus rooms 1 (0x4F at ~2600 progress) and 2 (0x51 at ~4000 progress). Real exit at ~4673. |
| death_signals | lives_drop only | camera_reset triggers on both death AND level completion |
| max_stall_frames | 360 | Camera stops ~289 frames before level exit |
| action_table | DKC_SPEED_ACTIONS (10 actions) | Walking without Y = 0 px/frame. All movement holds Y (run). |

## Bonus Room False Positive Timeline

1. Old threshold 4000: blocked bonus room 1 (progress ~2600) but NOT bonus room 2 (progress ~4000)
2. New threshold 4600: blocks BOTH bonus rooms. Real level exit at progress ~4673.
3. DKC has multiple bonus barrel locations per level - threshold must exceed ALL of them.
