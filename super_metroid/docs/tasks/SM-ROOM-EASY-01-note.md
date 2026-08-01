# Residual — SM-ROOM-EASY-01-R1

## Result
PARTIAL

## Files changed
- `policies/room_clears/room_a1ad_from_9f64_to_a1d8.json` — shortened only the existing `door_open_wait` span from 55 to 4 frames.
- `docs/tasks/SM-ROOM-EASY-01-note.md` — records the R1 isolated practice residual and dual-track non-claim.

## Verify paste
- `uv run python scripts/room/run_problem.py teleport room_a1ad_from_9f64_to_a1d8` — exit 0; ordinary `0xA1AD`, `pose=2`, `x=448`, `y=121`, `door_transition=0`.
- `uv run python scripts/room/run_problem.py run room_a1ad_from_9f64_to_a1d8` — exit 1; JSON `success=false`, failure `policy ended in 0xA1AD; expected 0xA1D8`.
- R1 final state: `room=0xA1AD`, `pose=138`, `x=85`, `y=187`, `door_transition=0`, `totalFrames=710`.

## Acceptance
- [ ] One-knob door-entry fix green — failed; the shortened door-open wait still remained in `0xA1AD`.
- [ ] Isolated run green / promotion — failed; policy remains `generated_unverified`.
- [x] Honest residual — recorded with final probe pin and next easy problem.
- [x] Dual-track non-claim — this practice attempt is not continuous evidence.

## Residual risks
- The policy reaches the left side but does not trigger the `0xA1D8` door transition.
- R1 ended at `room=0xA1AD`, `pose=138`, `x=85`, `y=187`, `door_transition=0` after 710 frames.
- The room geometry/door-shot interaction still needs a deliberate single-knob investigation; policy remains `generated_unverified`.
- The next easy queue problem is rank 53, Crab Hole `room_d21c_from_d3b6_to_d08a`; its task card is `SM-ROOM-EASY-02`.
- No continuous, STATUS, progression, capacity, door-state, event, or boss-bit claim was made.

## Next action (required)
- **Next card ID:** SM-ROOM-EASY-02
- **One change:** Switch practice to the next easy Crab Hole problem; leave the Boulder Room traversal unchanged.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_d21c_from_d3b6.state`

## Non-claims
- Did not STATUS-promote.
- Did not forge progression RAM.
- Not continuous evidence; this is isolated room practice only.

## Probe pin
room=0xA1AD pose=138 x=85 y=187 door_transition=0
