## Residual — SM-ROOM-SEG-33

### Result
GREEN

### Files changed
- `policies/room_clears/room_affb_from_ae74_to_b026.json` — prior same-ticket work iterated the isolated jump/run cadence and practice-promoted the policy to `verified_development_state` (this closeout session did not re-edit the policy and did not re-run `--promote`).
- `docs/tasks/SM-ROOM-SEG-33-residual.md` — documentation-only residual with independent re-verify evidence.

The existing entry fixture was already present and was not modified:
`custom_integrations/SuperMetroid-Snes/room_affb_from_ae74.state`.

### Verify paste
Commands were run from the repository root. This closeout re-ran teleport + isolated run only (no `--promote`). Both exited 0.

```text
uv run python super_metroid/scripts/room/run_problem.py teleport room_affb_from_ae74_to_b026
exit=0
{"problemId": "room_affb_from_ae74_to_b026", "state": {"room_id_hex": "0xAFFB", "game_state": 8, "phase": "ordinary_gameplay", "samus_x": 704, "samus_y": 121, "pose": 2, "door_transition": 0}}
statePath=custom_integrations/SuperMetroid-Snes/room_affb_from_ae74.state
stateSha256=c9dbbda7526cb426323794c34eccd9b0e005a583680bd78bce7a7aadb3b5837b

uv run python super_metroid/scripts/room/run_problem.py run room_affb_from_ae74_to_b026
exit=0
{"problemId": "room_affb_from_ae74_to_b026", "success": true, "startRoomIdHex": "0xAFFB", "targetRoomIdHex": "0xB026", "crossingFrame": 678, "settledFrame": 796, "totalFrames": 796, "finalState": {"room_id_hex": "0xB026", "samus_x": 216, "samus_y": 118, "pose": 82, "door_transition": 0}, "assist": {"progression_writes": 0, "capacity_writes": 0, "deaths": 0}, "policy": {"status": "verified_development_state"}, "objectiveVerification": {"objective": "traverse_to_exit", "status": "passed"}}
```

`--promote` was intentionally not re-run: the policy was already practice-promoted to `verified_development_state` by the prior same-ticket execution. Isolated re-verify confirms GREEN without changing any production path.

### Acceptance
- [x] Isolated run GREEN + promote (policy already practice-promoted; this closeout re-verified GREEN without re-promote).
- [x] Only card-owned policy/fixture scope was used for the ticket; this closeout wrote residual only and did not touch the fixture or re-edit the policy.
- [x] Dual-track non-claim recorded below; this is not continuous evidence.
- [x] Next card ID and one-change field filled below.

### Residual risks
- The result is an isolated doorway-natural practice green only.
- It does not establish natural predecessor entry, continuous route readiness, STATUS, or full-run integrity.
- Practice promote remains dual-track only and must not be treated as continuous evidence.

### Next action (required)
- **Next card ID:** none
- **One change:** none; no further knob is required for this practice problem.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_affb_from_ae74.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Practice promotion is dual-track only and is not continuous evidence.
- Did not re-run `--promote` or modify the policy in this closeout session.
- Did not edit STATUS, QUEUE, PROCESS, continuous routes, kpdr, progression, catalog, or sm_rev.

### Probe pin (if pure/geometry) — mandatory metrics
room=0xAFFB pose=2 x=704 y=121 door_transition=0
frames=796 dwell=118 last_pin=room=0xB026 pose=82 x=216 y=118 door_transition=0
