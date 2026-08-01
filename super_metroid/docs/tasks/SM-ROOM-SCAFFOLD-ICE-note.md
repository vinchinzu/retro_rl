## Residual - SM-ROOM-SCAFFOLD-ICE

### Result
RED

### Files changed
- `policies/room_clears/room_a890_from_a8b9_to_a8b9.json` - Generated the doorway-natural Ice Beam room starter policy; it remains `generated_unverified`.
- `docs/tasks/SM-ROOM-SCAFFOLD-ICE-note.md` - Records the isolated collect-objective residual.

### Verify paste
- `uv run python scripts/room/run_problem.py scaffold room_a890_from_a8b9_to_a8b9`
  - Exit 0; generated the policy with `sameDoorReturn=true` and frame budget `into=74`, `approach=138`, `enter=110`.
- `uv run python scripts/room/run_problem.py teleport room_a890_from_a8b9_to_a8b9`
  - Exit 0; loaded ordinary gameplay in room `0xA890` at `x=64`, `y=121`, `pose=1`, `door_transition=0`.
- `uv run python scripts/room/run_problem.py run room_a890_from_a8b9_to_a8b9`
  - Exit 0 as a probe/report command; report result was `success: false`.
  - Reached target room `0xA8B9` at frame `851`, but failed `collected_beams did not change`.
  - `progression_writes=0`; `capacity_writes=0`; `deaths=0`.

### Acceptance
- [x] Scaffold created for the problem.
- [x] Teleport fixture loaded and matched expected room `0xA890`.
- [ ] Isolated run green; the policy crossed to `0xA8B9` but did not collect Ice Beam.
- [x] Policy was not promoted because the isolated run was red.

### Residual risks
- The generated `RIGHT+B` deeper traversal does not reach or trigger the Ice Beam PLM at objective block `[12, 7]`.
- The policy remains `generated_unverified`; this is room practice only and not continuous Ice collection evidence.

### Next action (required)
- **Next card ID:** SM-ROOM-ICE-R1
- **One change:** Replace only the `deeper_into_room` traversal span so it reaches and collects the Ice Beam before the return sequence.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_a890_from_a8b9.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin
- Start: room=`0xA890`, pose=`1`, x=`64`, y=`121`, door_transition=`0`.
- Failure: room=`0xA8B9`, pose=`10`, x=`472`, y=`395`, door_transition=`0`.
