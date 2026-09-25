# DKC Winky's Walkway optimization runs

Local optimizer log. Not a STATUS result. STATUS still has no verified
autonomous level clear.

## Recorded file

`speed_v4_fresh/ga_gen0050_best.json`

- Action table `DKC_SPEED_ACTIONS` (10 actions), `completion_min_progress=4600`.
- Log records 1803 frames (30.05s), progress 4942, fitness 98198.
- Log records `level_id` ending at `0x2E`.
- Watch: `uv run python -m donkey_kong_country.optimizer watch --actions donkey_kong_country/optimizer/runs/speed_v4_fresh/ga_gen0050_best.json`

## Earlier logs

### 14-action table

Files in `runs/` root include BK2 extractions, `best_human.json`, synthetic
sprint-jump seeds, and `ga_gen0050_best.json` (2452 frames, progress threshold
4000). That threshold counted a bonus room as completion. `hillclimb_v1/`
reached 2260 frames with the same false positive.

### 10-action table

`DKC_SPEED_ACTIONS` drops walk-only actions that give 0 camera progress and
keeps a Y release for cartwheel re-taps. `speed_seed_fresh.json` and
`speed_seed_converted.json` are seeds for that table.

`speed_v3/` used threshold 4000. The log shows a finish at 1662 frames that
entered bonus room 2 (`level_id=0x51`) near progress 4001.

`speed_v4_fresh/` raises the threshold to 4600. The log's gen 10 finish is
1845 frames. Gen 30 and the gen 50 file above record 1803 frames and
`level_id=0x2E`.

## Parameters in that log

| Parameter | Value | Reason recorded in the log |
|-----------|-------|----------------------------|
| completion_min_progress | 4600.0 | Bonus room 1 is near progress 2600 (`0x4F`). Bonus room 2 is near 4000 (`0x51`). The log puts the real exit near 4673. |
| death_signals | lives_drop only | `camera_reset` fires on death and on level completion. |
| max_stall_frames | 360 | Camera stops about 289 frames before the exit. |
| action_table | DKC_SPEED_ACTIONS | Walking without Y is 0 px/frame. Movement in this table holds Y. |

Threshold 4000 blocked bonus room 1 and not bonus room 2. Threshold 4600 is
the log's attempt to sit above both. This file does not promote either run
to the STATUS gate.
