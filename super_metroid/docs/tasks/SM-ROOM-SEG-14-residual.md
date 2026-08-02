# Residual — SM-ROOM-SEG-14

### Result
RED

### Files changed
- `policies/room_clears/room_9969_from_99bd_to_9938.json` — replaced the scaffold's morphing traversal loop with the room-local 13 × 60-frame `LEFT+A+B+X` traversal pattern; policy remains unverified.
- `docs/tasks/SM-ROOM-SEG-14-residual.md` — recorded the isolated RED run, pin, and next one-knob action.

### Verify paste
- `uv run python scripts/room/run_problem.py teleport room_9969_from_99bd_to_9938` — exit 0; landed in `0x9969`, `ordinary_gameplay`, `x=960`, `y=121`, `pose=2`, `door_transition=0`.
- `uv run python scripts/room/run_problem.py run room_9969_from_99bd_to_9938` — exit 1; `success=false`, `failure=policy ended in 0x9969; expected 0x9938`, `totalFrames=1003`, final `0x9969`, `x=277`, `y=158`, `pose=138`, `door_transition=0`, `progression_writes=0`, `capacity_writes=0`, `deaths=0`.

### Acceptance
- [x] Isolated run produced an honest RED residual with a mandatory pin.
- [x] Only card-owned files were touched.
- [x] Dual-track non-claim is explicit below; this is not continuous evidence.
- [x] Next card ID and one change are filled.

### Residual risks
- The practice fixture starts with `selected_item=1` (missiles), while the continuous reference normalizes to beam (`selected_item=0`) before the Lower Mushrooms traversal.
- The adjusted movement pins at `x=549`, `y=171`, `pose=138` during the third traversal slice; the final door spans cannot cross from that state and end at `x=277`, `y=158`.
- The policy is not `verified_development_state` and must not be promoted until a green isolated replay.
- This room practice result does not establish a natural continuous predecessor or any continuous tip integrity.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-14-R1
- **One change:** Normalize the selected weapon to beam with one `SELECT` pulse plus its settle hold before the existing traversal loop, without changing movement timings.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_9969_from_99bd.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, event, boss-bit, or door-state RAM.
- Not continuous evidence; this is dual-track room practice only.
- Did not promote the practice policy.

### Probe pin (isolated practice)
room=0x9969 pose=138 x=277 y=158 door_transition=0
frames=1003 dwell=N/A last_pin=room=0x9969 pose=138 x=277 y=158 door_transition=0
