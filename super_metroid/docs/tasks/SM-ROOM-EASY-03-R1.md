# TASK SM-ROOM-EASY-03-R1: Grapple Tutorial 2 — left-exit residual

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_abd2_from_ab64_to_ac00.json` (this problem only)
- optional note: `docs/tasks/SM-ROOM-EASY-03-R1-note.md`

## Context
- SM-ROOM-EASY-03 **RED**: reaches lower-left of `0xABD2` but no exit door.
- Pin: `room=0xABD2 pose=138 x=21 y=395 door_transition=0`
- Policy remains `generated_unverified`. Practice only.

## Do
1. **One knob** on the left-exit approach / door trigger (not a full rewrite).
2. Isolate green or residual with pin.
3. No continuous / STATUS. Do not also retune Metal in this card.

## Acceptance
- [ ] Isolate run green **or** residual with pin + next card
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_abd2_from_ab64_to_ac00
```
