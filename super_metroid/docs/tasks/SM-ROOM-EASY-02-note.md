## Residual - SM-ROOM-EASY-02

### Result
RED

### Files changed
- `docs/tasks/SM-ROOM-EASY-02-note.md` - Records the isolated Crab Hole result.

### Verify paste
- `uv run python super_metroid/scripts/room/run_problem.py scaffold room_d21c_from_d3b6_to_d08a`
  - `existing_skipped`; the problem-local scaffold already existed.
- `uv run python super_metroid/scripts/room/run_problem.py teleport room_d21c_from_d3b6_to_d08a`
  - Exit 0; fixture loaded in `0xD21C`, ordinary gameplay, `x=192`, `y=377`.
- `uv run python super_metroid/scripts/room/run_problem.py run room_d21c_from_d3b6_to_d08a`
  - Exit 1; policy ended in `0xCF80`, expected `0xD08A`.

### Acceptance
- [ ] Green isolate or residual - residual recorded; isolate is red.
- [x] Dual-track non-claim - development-only room practice; no continuous claim.

### Residual risks
- The generated policy reaches the wrong room transition and remains
  `generated_unverified`.
- No promotion was attempted because the isolated run was not green.

### Next action (required)
- **Next card ID:** SM-ROOM-EASY-03
- **One change:** Start the next open easy practice problem, Ice Beam Tutorial Room.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_a865_from_a815.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression or capacity RAM.
- Not continuous evidence.

### Probe pin
- Start: room=`0xD21C`, pose=`2`, x=`192`, y=`377`, door_transition=`0`.
- Failure: room=`0xCF80`, pose=`82`, x=`984`, y=`118`, door_transition=`1`.
