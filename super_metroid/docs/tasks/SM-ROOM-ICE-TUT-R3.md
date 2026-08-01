# TASK SM-ROOM-ICE-TUT-R3: Ice Tutorial — jumpx7 pose-138 residual

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local policy only: `room_a865_from_a815_to_a8b9`
- optional note: `docs/tasks/SM-ROOM-ICE-TUT-R3-note.md`

## Context
- R2 **PARTIAL**: same pin after landx7 substitution.
- Pin: `room=0xA865 pose=138 x=277 y=139 door_transition=0`
- R2 next action: replace the single **`jumpx7`** traversal span so pose-138
  is not reintroduced before the left-door approach.
- Practice only. Not continuous Ice / not SM-K4-ICE-PURE.

## Do
1. **One knob:** change only the `jumpx7` span (buttons and/or frames) to leave
   pose-138 and enable left-door exit.
2. Isolate green or residual with pin + next card.
3. No continuous / STATUS / progression / other rooms.

## Acceptance
- [ ] Isolate green **or** residual with pin + next card
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_a865_from_a815_to_a8b9
```
