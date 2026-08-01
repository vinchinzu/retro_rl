# Residual — SM-ROOM-EASY-01-R2

## Result
PARTIAL

## Files changed
- `policies/room_clears/room_a1ad_from_9f64_to_a1d8.json` — changed only the final door-entry button hold from `LEFT+A+B` to `LEFT+X`.
- `docs/tasks/SM-ROOM-EASY-01-R2-note.md` — records the isolated residual and dual-track non-claim.

## Verify paste
- `uv run python super_metroid/scripts/room/run_problem.py run room_a1ad_from_9f64_to_a1d8` — exit 1; JSON `success=false`, failure `policy ended in 0xA1AD; expected 0xA1D8`.
- Final state: `room=0xA1AD`, `pose=138`, `x=85`, `y=187`, `door_transition=0`, `totalFrames=710`.

## Acceptance
- [ ] Isolated run green — failed; the final `LEFT+X` door-shot hold still remained in `0xA1AD`.
- [x] Residual recorded with final pin and next card.
- [x] Dual-track non-claim — this practice attempt is not continuous evidence.

## Residual risks
- The left Boulder door still does not trigger a `0xA1D8` transition after the final door-shot button change.
- The policy remains `generated_unverified`.
- No continuous, STATUS, progression, capacity, door-state, event, or boss-bit claim was made.

## Next action (required)
- **Next card ID:** SM-ROOM-EASY-02
- **One change:** Switch practice to the queued Crab Hole problem and leave the Boulder traversal unchanged.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_d21c_from_d3b6.state`

## Non-claims
- Did not STATUS-promote.
- Did not forge progression RAM.
- Not continuous evidence; this is isolated room practice only.

## Probe pin
room=0xA1AD pose=138 x=85 y=187 door_transition=0
