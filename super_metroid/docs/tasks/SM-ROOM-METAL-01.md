# TASK SM-ROOM-METAL-01: Metal Pirates — one combat-clear knob

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local policy for `room_b62b_from_b482_to_b5d5` only
- optional note: `docs/tasks/SM-ROOM-METAL-01-note.md`

## Context
- SM-ROOM-METAL **RED**: reaches far side but does not clear both Metal
  Pirates / exit door.
- Pin: `room=0xB62B pose=137 x=731 y=171 door_transition=0`
- Source: `custom_integrations/SuperMetroid-Snes/room_b62b_from_b482.state`
- Practice only — not continuous / KPDR spine.

## Read first
- prior note/log residual for SM-ROOM-METAL
- existing policy under `policies/room_clears/` for this problem id

## Do
1. **One knob:** replace the fixed rightward shoot-run slice with one Metal
   Pirate combat/clear tactic, then retest the existing exit sequence.
2. Promote only if isolate green; else residual with pin.
3. No continuous / STATUS / progression edits.

## Acceptance
- [ ] Isolate run green **or** residual with pin + next card
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_b62b_from_b482_to_b5d5
uv run python super_metroid/scripts/room/run_problem.py run room_b62b_from_b482_to_b5d5
```
