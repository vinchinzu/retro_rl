## Residual — SM-ROOM-SEG-10

### Result
RED

### Files changed
- `policies/room_clears/room_cda8_from_caf6_to_caf6.json` — generated the doorway-natural room policy scaffold; remains `generated_unverified`.
- `custom_integrations/SuperMetroid-Snes/room_cda8_from_caf6.state` — bootstrapped the room entry fixture from the controllable natural post-Spore source.
- `custom_integrations/SuperMetroid-Snes/room_cda8_from_caf6.provenance.json` — recorded the doorway pointer, source room, spawn, and development-only provenance.
- `docs/tasks/SM-ROOM-SEG-10-residual.md` — recorded the isolated verification failure and next action.

### Verify paste
- `uv run python super_metroid/scripts/room/run_problem.py teleport room_cda8_from_caf6_to_caf6` — exit code `0`
  - `room_id_hex: "0xCDA8"`
  - `phase: "ordinary_gameplay"`
  - `samus_x: 192`, `samus_y: 121`, `pose: 2`
  - `max_super_missiles: 0`
- `uv run python super_metroid/scripts/room/run_problem.py run room_cda8_from_caf6_to_caf6` — exit code `1`
  - `success: false`
  - `failure: "policy ended in 0xCDA8; expected 0xCAF6"`
  - `totalFrames: 798`
  - final `room_id: 0xCDA8`, `samus_x: 107`, `samus_y: 187`, `pose: 137`, `door_transition: 0`
  - `objectiveVerification.status: "not_reached"`
  - `progression_writes: 0`, `capacity_writes: 0`

The room reference data also marks the visible Super Missile as unavailable
until `f_DefeatedPhantoon` or `h_allItemsSpawned`. The natural post-Spore
fixture has neither a usable post-Phantoon source nor Super Missile capacity;
late full-loadout anchors tested for this fixture were input-frozen after the
door warp and were not retained.

### Acceptance
- [x] Isolated run is an honest residual with a probe pin; no promote was attempted after the RED run.
- [x] Only this problem's policy, entry fixture, provenance, and residual were changed by this task. The runner's ignored room report is a generated verification artifact.
- [x] Dual-track only: this is not continuous or STATUS evidence.
- [x] Next card ID and one change are filled below.

### Residual risks
- The room has no GREEN isolated clear and its policy is not promoted.
- A controllable post-Phantoon doorway source is still missing; the available late-game anchors freeze Samus input after this door warp.
- The current scaffold also does not solve the powered-off room traversal, so geometry should not be tuned against the wrong progression state.
- This practice result does not affect the continuous spine, STATUS, or queue promotion.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-10-R1
- **One change:** Capture one controllable doorway-natural source state after Phantoon that exposes the Super Missile without forging progression or capacity RAM.
- **Source state:** needs capture: `SM-ROOM-SEG-10-SRC`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Not continuous evidence; this is dual-track room practice only.

### Probe pin
room=0xCDA8 pose=137 x=107 y=187 door_transition=0
