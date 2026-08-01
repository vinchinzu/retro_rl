# TASK SM-ROOM-EASY-02C: Crab Hole — path-select top-left exit (not span extend)

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_d21c_from_d3b6_to_d08a.json` only
- optional note: `docs/tasks/SM-ROOM-EASY-02C-note.md`

## Context
- EASY-02B **RED**: extended `top_left` 40→80f; still exits wrong room
  `0xCF80` (top-right), expected `0xD08A`.
- Failure pin: end room `0xCF80` pose=82 x=984 y=118 door_transition=1
- Frame-length alone does not select the top-left doorway. Need a **path
  class** change on the final approach only (e.g. UP/left bias, stop-right
  before top, or replace `top_left` span buttons — **one** named span).
- Practice only; not continuous Maridia.

## Do
1. **One knob:** rewrite **only** the final exit approach span(s) so the
   policy cannot finish at the top-right door. Prefer UP-then-LEFT or
   explicit anti-right bias; do not only lengthen frames.
2. Isolate green or residual with pin + next card.
3. No continuous / STATUS / other rooms / spine.

## Acceptance
- [ ] Isolate green **or** residual with pin + next card
- [ ] Dual-track non-claim
- [ ] If still red, residual must say whether end room is still `0xCF80`

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_d21c_from_d3b6_to_d08a
```
