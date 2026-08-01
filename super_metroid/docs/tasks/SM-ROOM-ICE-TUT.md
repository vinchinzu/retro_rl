# TASK SM-ROOM-ICE-TUT: Practice track — Ice Beam Tutorial (K4-relevant)

## Recipe step
room practice (dual-track; K4 prep — not continuous)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local: `room_a865_from_a815_to_a8b9`
- optional note

## Context
- Ice Beam Tutorial `0xA865` sits on future K4 Ice branch graph.
- Practice green builds muscle for later pure; **does not** promote Ice
  continuous or graph verification.

## Read first
- ROOM_WORK_QUEUE, run_problem.py
- `tests/test_k4_speed_branches.py` (Ice path edge ids — read only)

## Do
1. Isolated teleport/run for the problem.
2. Promote only if green isolate.
3. Residual: link to future `SM-K4-ICE-PURE` after Business reverse.

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_a865_from_a815_to_a8b9
uv run python super_metroid/scripts/room/run_problem.py run room_a865_from_a815_to_a8b9
```
