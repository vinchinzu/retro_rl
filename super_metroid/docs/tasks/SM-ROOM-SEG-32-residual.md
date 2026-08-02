## Residual - SM-ROOM-SEG-32

### Result
GREEN

### Files changed
- `policies/room_clears/room_afce_from_b026_to_a923.json` - Added the verified jump/run cadence and practice promotion metadata.
- `docs/tasks/SM-ROOM-SEG-32-residual.md` - Recorded the isolated practice result and non-claims.

The existing entry fixture was already present and was not modified.

### Verify paste
Commands were run from the repository root with exit code 0.

```text
uv run python super_metroid/scripts/room/run_problem.py teleport room_afce_from_b026_to_a923
{"problemId": "room_afce_from_b026_to_a923", "state": {"room_id_hex": "0xAFCE", "game_state": 8, "phase": "ordinary_gameplay", "samus_x": 960, "samus_y": 121, "door_transition": 0}}

uv run python super_metroid/scripts/room/run_problem.py run room_afce_from_b026_to_a923
{"problemId": "room_afce_from_b026_to_a923", "success": true, "targetRoomIdHex": "0xA923", "crossingFrame": 476, "settledFrame": 662, "totalFrames": 662, "progression_writes": 0, "capacity_writes": 0, "deaths": 0}

uv run python super_metroid/scripts/room/run_problem.py run room_afce_from_b026_to_a923 --promote
{"problemId": "room_afce_from_b026_to_a923", "success": true, "promoted": true, "policy": {"status": "verified_development_state"}}
```

### Acceptance
- [x] Isolated run GREEN + promote.
- [x] Only card-owned policy/fixture scope was used; the existing fixture was unchanged.
- [x] Dual-track non-claim recorded below; this is not continuous evidence.
- [x] Next card ID and one-change field filled below.

### Residual risks
- The result is an isolated doorway-natural practice green only.
- It does not establish natural predecessor entry, continuous route readiness, STATUS, or full-run integrity.

### Next action (required)
- **Next card ID:** none
- **One change:** none; no further knob is required for this practice problem.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_afce_from_b026.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Practice promotion is dual-track only and is not continuous evidence.

### Probe pin (if pure/geometry) - metrics
room=0xAFCE pose=2 x=960 y=121 door_transition=0
frames=662 dwell=186 last_pin=room=0xA923 pose=26 x=3288 y=642 door_transition=1
