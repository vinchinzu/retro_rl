## Residual — SM-ROOM-EASY-03-R2

### Result
RED

### Files changed
- `policies/room_clears/room_abd2_from_ab64_to_ac00.json` — changed only the `land_shoot` left-exit approach to `LEFT+X` for 60 frames; retained `open_exit_door` at 32 frames.
- `docs/tasks/SM-ROOM-EASY-03-R2-note.md` — recorded the residual and probe pin.

### Verify paste
`uv run python scripts/room/run_problem.py run room_abd2_from_ab64_to_ac00` — exit code 0; report returned `success: false`, `failure: "policy ended in 0xABD2; expected 0xAC00"`, `crossingFrame: null`, and `settledFrame: null`.

### Acceptance
- [x] Isolated residual with pin and next card.
- [x] Dual-track non-claim.

### Residual risks
- The left door still does not enter after the `land_shoot` approach and unchanged 32-frame door-open hold.
- This is practice-only evidence and does not establish continuous readiness or STATUS promotion.

### Next action (required)
- **Next card ID:** SM-ROOM-EASY-03-R3
- **One change:** Instrument or retune the left-door opening/entry boundary while keeping the preceding `land_shoot` sequence fixed.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_abd2_from_ab64.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence; this remains a dual-track room-practice result.

### Probe pin
`room=0xABD2 pose=138 x=21 y=395 door_transition=0`
