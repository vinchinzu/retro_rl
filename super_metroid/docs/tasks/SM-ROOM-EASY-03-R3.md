# TASK SM-ROOM-EASY-03-R3: Grapple Tutorial 2 — left-door open/entry only

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_abd2_from_ab64_to_ac00.json` only
- optional note: `docs/tasks/SM-ROOM-EASY-03-R3-note.md`

## Context
- EASY-03-R2 **RED**: `land_shoot` → LEFT+X 60f; **same pin**
  `room=0xABD2 pose=138 x=21 y=395 door_transition=0`
- R2 next: retune **left-door opening/entry boundary** only; keep preceding
  `land_shoot` sequence fixed.
- Two residual knobs on approach failed — this card is door-open/entry only.
- Practice only.

## Do
1. **One knob:** change only `open_exit_door` (and/or immediate post-open
   entry push into `0xAC00`) — frames and/or buttons. Do **not** retouch
   `land_shoot`.
2. Isolate green or residual with pin + next card.
3. If still same pin after this class, residual next = `PLANNER-GATE` park
   (door-system rethink) — do not invent R4 spam.

## Acceptance
- [ ] Isolate green **or** residual with pin + next card
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_abd2_from_ab64_to_ac00
```
