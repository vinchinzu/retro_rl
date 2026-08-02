## Residual — SM-ROOM-SEG-34

### Result
GREEN

### Files changed
- `policies/room_clears/room_b139_from_af72_to_afce.json` — replaced scaffold LEFT+A+B cadence with left-ledge → mid-platform → right-gap fall plus pulsed DOWN+X door open; practice-promoted to `verified_development_state`.
- `docs/tasks/SM-ROOM-SEG-34-residual.md` — PROCESS residual for this isolated practice green.

Existing entry fixture was already present and was not modified:
`custom_integrations/SuperMetroid-Snes/room_b139_from_af72.state`.

### Verify paste
Commands were run from the repository root.

```text
uv run python super_metroid/scripts/room/run_problem.py teleport room_b139_from_af72_to_afce
exit=0
{"problemId": "room_b139_from_af72_to_afce", "state": {"room_id_hex": "0xB139", "game_state": 8, "phase": "ordinary_gameplay", "samus_x": 192, "samus_y": 121, "pose": 2, "door_transition": 0}}
statePath=custom_integrations/SuperMetroid-Snes/room_b139_from_af72.state
stateSha256=d3d0708f5a1d5047e1dbc56fd2c03e20d27b88b5fa2b53babdc9974982b3cbc7

# Pre-edit scaffold baseline (RED — not promoted):
uv run python super_metroid/scripts/room/run_problem.py run room_b139_from_af72_to_afce
exit=1
{"problemId": "room_b139_from_af72_to_afce", "success": false, "failure": "policy ended in 0xB139; expected 0xAFCE", "totalFrames": 443, "finalState": {"room_id_hex": "0xB139", "samus_x": 53, "samus_y": 276, "pose": 65, "door_transition": 0}, "assist": {"progression_writes": 0, "capacity_writes": 0, "deaths": 0}}

# Post one-knob policy edit:
uv run python super_metroid/scripts/room/run_problem.py run room_b139_from_af72_to_afce
exit=0
{"problemId": "room_b139_from_af72_to_afce", "success": true, "startRoomIdHex": "0xB139", "targetRoomIdHex": "0xAFCE", "crossingFrame": 422, "settledFrame": 537, "totalFrames": 537, "finalState": {"room_id_hex": "0xAFCE", "samus_x": 870, "samus_y": 80, "pose": 49, "door_transition": 0}, "assist": {"progression_writes": 0, "capacity_writes": 0, "deaths": 0}, "objectiveVerification": {"objective": "traverse_to_exit", "status": "passed"}}

uv run python super_metroid/scripts/room/run_problem.py run room_b139_from_af72_to_afce --promote
exit=0
{"problemId": "room_b139_from_af72_to_afce", "success": true, "promoted": true, "policy": {"status": "verified_development_state"}, "crossingFrame": 422, "settledFrame": 537, "totalFrames": 537, "assist": {"progression_writes": 0, "capacity_writes": 0, "deaths": 0}}

uv run pytest super_metroid/tests/test_room_graph.py -q -k "expand or scaffold or policy"
exit=0
5 passed, 6 deselected
```

### Acceptance
- [x] Isolated run GREEN + promote
- [x] Only card-owned policy/fixture/residual scope; existing fixture unchanged
- [x] Dual-track non-claim recorded below; this is not continuous evidence
- [x] Next card ID and one-change field filled below

### Residual risks
- Isolated doorway-natural practice green only.
- Does not establish natural predecessor entry, continuous route readiness, STATUS, or full-run integrity.
- Scaffold failure mode was stuck on the left mid ledge (x≈53, y≈276, pose 65); the fix is geometry-specific to this shaft and should not be treated as a general spine primitive without a second consumer.

### Next action (required)
- **Next card ID:** none
- **One change:** none; no further knob is required for this practice problem.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_b139_from_af72.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Practice promotion is dual-track only and is not continuous evidence.
- Did not edit STATUS, QUEUE, PROCESS, continuous routes, kpdr, progression, catalog, or sm_rev.
- Did not modify the entry fixture or any other room policy.

### Probe pin (if pure/geometry) — mandatory metrics
room=0xB139 pose=2 x=192 y=121 door_transition=0
frames=537 dwell=115 last_pin=room=0xAFCE pose=49 x=870 y=80 door_transition=0
cross=422 settle=537 progression_writes=0 capacity_writes=0 deaths=0
