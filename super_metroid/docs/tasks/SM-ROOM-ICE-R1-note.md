## Residual - SM-ROOM-ICE-R1

### Result
RED

### Files changed
- `policies/room_clears/room_a890_from_a8b9_to_a8b9.json` - Added `A` to the existing `RIGHT+B` item/pedestal approach span; duration and return sequence were unchanged.
- `docs/tasks/SM-ROOM-ICE-R1-note.md` - Records the failed isolated verification and probe pin.

### Verify paste
- `uv run python super_metroid/scripts/room/run_problem.py run room_a890_from_a8b9_to_a8b9`
  - Exit 0 as a probe/report command; report result was `success: false`.
  - Failure: `room objective incomplete: collected_beams did not change`.
  - Crossed to `0xA8B9` at frame `728`; settled at frame `851`.
  - `progression_writes=0`; `capacity_writes=0`; `deaths=0`.

### Acceptance
- [x] One item-touch/pedestal approach knob was changed.
- [ ] Isolated run green; adding jump to the approach did not collect Ice.
- [x] Policy was not promoted because the isolated run was red.
- [x] No continuous or STATUS files were changed.

### Residual risks
- The `RIGHT+A+B` span still does not reach or trigger the Ice Beam PLM at objective block `[12, 7]`.
- The policy remains `generated_unverified`; this is room practice only and not continuous Ice collection evidence.
- One-knob discipline prevents another geometry change in this card.

### Next action (required)
- **Next card ID:** SM-ROOM-ICE-R2
- **One change:** Replace the approach span with one explicitly staged pedestal-touch sequence, using a newly selected single movement/timing knob.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_a890_from_a8b9.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin
- Start: room=`0xA890`, pose=`1`, x=`64`, y=`121`, door_transition=`0`.
- Failure: room=`0xA8B9`, pose=`10`, x=`472`, y=`395`, door_transition=`0`.
