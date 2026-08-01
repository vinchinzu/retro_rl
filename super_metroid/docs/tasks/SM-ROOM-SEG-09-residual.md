## Residual — SM-ROOM-SEG-09

### Result
GREEN

### Files changed
- `policies/room_clears/room_a9e5_from_aa41_to_aa41.json` — translated the bounded Hi-Jump collection and return choreography into a raw-button room policy; promoted to `verified_development_state` after a SHA-gated green run.
- `custom_integrations/SuperMetroid-Snes/room_a9e5_from_aa41.state` — doorway-natural `0xA9E5` entry fixture bootstrapped through the `0xAA41` door.
- `custom_integrations/SuperMetroid-Snes/room_a9e5_from_aa41.provenance.json` — records the fixture entry contract and `developmentOnly` status.
- `docs/tasks/SM-ROOM-SEG-09-residual.md` — records the isolated green result and non-claims.

### Verify paste
```text
uv run python scripts/room/run_problem.py teleport room_a9e5_from_aa41_to_aa41
exit 0
room=0xA9E5 game_state=8 phase=ordinary_gameplay x=192 y=121 door_transition=0

uv run python scripts/room/run_problem.py run room_a9e5_from_aa41_to_aa41 --promote
exit 0
success=true crossingFrame=1559 settledFrame=1684 totalFrames=1684
finalRoom=0xAA41 finalPose=9 finalX=39 finalY=395 door_transition=0
objective=collect_and_return status=passed hi_jump=true
progression_writes=0 capacity_writes=0 deaths=0 promoted=true

uv run python scripts/room/run_problem.py run room_a9e5_from_aa41_to_aa41
exit 0
success=true crossingFrame=1559 settledFrame=1684 totalFrames=1684
finalRoom=0xAA41 finalPose=9 finalX=39 finalY=395 door_transition=0
objective=collect_and_return status=passed policyStatus=verified_development_state
progression_writes=0 capacity_writes=0 deaths=0

uv run pytest tests/test_room_graph.py -q
exit 0
11 passed in 0.15s
```

### Acceptance
- [x] Isolated run **GREEN + promote** — Hi-Jump collected (`0x100` gain) and the same-door return settled in `0xAA41` ordinary gameplay.
- [x] Only own-files touched — target policy, this problem's fixture/provenance, and this residual; unrelated pre-existing worktree changes were left untouched.
- [x] Dual-track non-claim — this is isolated room practice only and is not continuous evidence.
- [x] Next card ID + one change filled — no follow-up knob is required for this green practice result.

### Residual risks
- The promoted status is `verified_development_state` for the doorway fixture only; it does not promote the room edge or continuous spine.
- The fixture uses the bootstrap boot state and remains development-only; re-bootstrap may re-roll room RNG while preserving the door boundary.
- Planner owns queue refresh, route composition, and any continuous natural-entry judgment.

### Next action (required)
- **Next card ID:** none
- **One change:** none; retain the promoted policy unless a later fixture/RNG stress run exposes a bounded residual.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_a9e5_from_aa41.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Not continuous evidence; this is dual-track isolated practice only.

### Probe pin (if pure/geometry)
room=0xAA41 pose=9 x=39 y=395 door_transition=0
