# TASK SM-ROOM-ICE-TUT-R2: Ice Tutorial — clear pose-138 + left door

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local: `room_a865_from_a815_to_a8b9` policy only
- optional note: `docs/tasks/SM-ROOM-ICE-TUT-R2-note.md`

## Context
- SM-ROOM-ICE-TUT-R1 **PARTIAL**: ends in `0xA865` without left exit.
- Pin: `room=0xA865 pose=138 x=277 y=139 door_transition=0`
- Pose 138 is knockback/spin pin — need one traversal/control change to clear
  it before the left-door approach.
- Practice only. **Not** continuous Ice path; do not open SM-K4-ICE-PURE.

## Do
1. **One knob:** replace one traversal span so Samus leaves pose-138 and can
   trigger the left door to the expected exit room.
2. Promote only if isolate green; else residual with pin.
3. No continuous / STATUS / progression.

## Acceptance
- [ ] Isolate run green **or** residual with pin + next card
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_a865_from_a815_to_a8b9
```
