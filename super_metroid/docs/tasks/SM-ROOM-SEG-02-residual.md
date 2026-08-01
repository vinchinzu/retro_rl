## Residual - SM-ROOM-SEG-02

### Result
RED

### Files changed
- `policies/room_clears/room_aade_from_aa82_to_aa82.json` - Generated the doorway-natural scaffold and changed only the collection span to `LEFT+A`; it remains `generated_unverified`.
- `custom_integrations/SuperMetroid-Snes/room_aade_from_aa82.state` - Bootstrapped entry fixture for `0xAADE` from the `0xAA82` door.
- `custom_integrations/SuperMetroid-Snes/room_aade_from_aa82.provenance.json` - Records the doorway bootstrap contract and door pointer `0x943E`.
- `docs/tasks/SM-ROOM-SEG-02-residual.md` - Records the isolated RED result and next one-knob action.

### Verify paste
- `uv run python super_metroid/scripts/room/run_problem.py teleport room_aade_from_aa82_to_aa82`
  - Exit 0.
  - `room_id_hex=0xAADE`, `game_state=8`, `phase=ordinary_gameplay`, `door_transition=0`, `samus_x=192`, `samus_y=121`, `pose=2`.
- `uv run python super_metroid/scripts/room/run_problem.py run room_aade_from_aa82_to_aa82`
  - Exit 1 because the report was unsuccessful.
  - Failure: `policy ended in 0xAADE; expected 0xAA82`.
  - Final pin: `room=0xAADE`, `pose=137`, `x=139`, `y=139`, `door_transition=0`.
  - Objective progress did occur: `max_power_bombs=5` from `0`; assist report had `progression_writes=0` and `capacity_writes=0`.
- `uv run python super_metroid/scripts/room/run_problem.py run room_aade_from_aa82_to_aa82 --promote`
  - Skipped because the isolated run was RED; no promotion was attempted.

### Acceptance
- [x] Honest residual with a failure pin filed; isolated GREEN + promote was not achieved.
- [x] Only this problem's policy, entry fixture/provenance, and residual were changed by this task.
- [x] Dual-track only; this is not continuous evidence.
- [x] Next card ID and one change are filled below.

### Residual risks
- The `LEFT+A` collection span reliably raises `max_power_bombs` to `5`, but the idle `item_fanfare_wait` leaves the item information overlay active, so the existing exit spans receive no usable control.
- A bounded probe showed that `idle(360)` followed by `X` for 40 frames releases the overlay and lets the unchanged exit sequence cross to `0xAA82` at frame 790; that second knob was not applied in this card.
- The policy remains `generated_unverified`; practice promotion, continuous readiness, and STATUS promotion remain blocked.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-02-R1
- **One change:** Add one `X` dismissal span after the existing 360-frame item fanfare wait and before the unchanged exit approach.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_aade_from_aa82.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- This is isolated dual-track room practice only, not continuous evidence.

### Probe pin
room=0xAADE pose=137 x=139 y=139 door_transition=0; final failed run had `max_power_bombs=5` and did not cross the exit door.
