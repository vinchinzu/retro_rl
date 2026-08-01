# TASK SM-ROOM-EASY-03-R2: Grapple Tutorial 2 — land_shoot approach residual

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_abd2_from_ab64_to_ac00.json` (this problem only)
- optional note: `docs/tasks/SM-ROOM-EASY-03-R2-note.md`

## Context
- R1 **RED**: extended `open_exit_door` 20→32f; **same pin**.
- Pin: `room=0xABD2 pose=138 x=21 y=395 door_transition=0`
- R1 next: retune preceding **`land_shoot`** left-exit approach only
  (do **not** retouch open_exit_door).
- Practice only.

## Do
1. **One knob:** change only `land_shoot` approach timing / buttons for the
   left exit.
2. Isolate green or residual with pin + next card.
3. No continuous / STATUS / Metal / other rooms.

## Acceptance
- [ ] Isolate green **or** residual with pin + next card
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_abd2_from_ab64_to_ac00
```
