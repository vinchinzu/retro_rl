# TASK SM-ROOM-EASY-01-R1: Boulder Room door-entry residual

## Recipe step
room practice residual (dual-track — never continuous)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_a1ad_from_9f64_to_a1d8.json` (and generated policy
  siblings for this problem id only)
- optional note `docs/tasks/SM-ROOM-EASY-01-R1-note.md`

## Context
- SM-ROOM-EASY-01 PARTIAL: teleport ok; run stalls
  pin `room=0xA1AD pose=138 x=85 y=187 door_transition=0`
- **One change:** door-entry geometry only — do not rewrite full traversal.

## Read first
- policy JSON for this problem
- `docs/tasks/SM-ROOM-EASY-01-note.md`
- `scripts/room/run_problem.py`

## Do
1. One-knob fix from left-side pin toward exit door.
2. Run isolate; promote only if green.
3. Residual next easy problem if still red.

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_a1ad_from_9f64_to_a1d8
```
