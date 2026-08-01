# TASK SM-ROOM-EASY-01-R2: Boulder Room — door-shot / transition residual

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_a1ad_from_9f64_to_a1d8.json` (this problem only)
- optional note: `docs/tasks/SM-ROOM-EASY-01-R2-note.md`

## Context
- SM-ROOM-EASY-01-R1 **PARTIAL**: shortened `door_open_wait` 55→4; still stuck
  in Boulder `0xA1AD`.
- Pin: `room=0xA1AD pose=138 x=85 y=187 door_transition=0` (totalFrames=710)
- Expected exit: `0xA1D8`. Need **different** one-knob than door_open_wait
  (already tried).
- Practice only — not continuous.

## Do
1. **One knob** (not door_open_wait again): door-shot timing, approach x-band,
   or face/shot before open — pick one.
2. Isolate green or residual with pin.
3. No continuous / STATUS.

## Acceptance
- [ ] Isolate run green **or** residual with pin + next card
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_a1ad_from_9f64_to_a1d8
```
