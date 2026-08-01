## Residual - SM-ROOM-EASY-03-R1

### Result
RED

### Change tested
- Extended the existing `open_exit_door` LEFT+X hold from 20 to 32 frames.
- No other policy span was changed.

### Verify
`uv run python scripts/room/run_problem.py run room_abd2_from_ab64_to_ac00`

Result: policy ended in `0xABD2`; expected `0xAC00`.

### Probe pin
`room=0xABD2 pose=138 x=21 y=395 door_transition=0`

### Next action
- **Next card ID:** SM-ROOM-EASY-03-R2
- **One change:** Retune the preceding `land_shoot` left-exit approach timing only.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_abd2_from_ab64.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Practice-only result; not continuous evidence.
