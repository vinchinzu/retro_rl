# TASK SM-ROOM-EASY-02B: Crab Hole — wrong-exit residual

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local only: `room_d21c_from_d3b6_to_d08a` policy / scaffold
- optional note: `docs/tasks/SM-ROOM-EASY-02B-note.md`

## Context
- SM-ROOM-EASY-02 **RED**: policy left Crab Hole into wrong room.
- Failure pin: `room=0xCF80 pose=82 x=984 y=118 door_transition=1`
  (expected exit `0xD08A`).
- Boulder (EASY-01) parked after two PARTIAL door pins — queue switched here.
- Practice only; not continuous Maridia.

## Do
1. **One knob** on the exit choice / final approach so the policy targets
   `0xD08A` instead of `0xCF80` (e.g. last door-facing span or approach
   direction — pick one).
2. Isolate green or residual with pin + next open easy.
3. No continuous / STATUS / spine files.

## Acceptance
- [ ] Isolate green **or** residual with pin + next card
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_d21c_from_d3b6_to_d08a
```
