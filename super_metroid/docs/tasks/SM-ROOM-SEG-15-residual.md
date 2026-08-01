## Residual — SM-ROOM-SEG-15

### Result
RED

### Files changed
- custom_integrations/SuperMetroid-Snes/room_affb_from_ae74.state — doorway-natural entry fixture for `0xAFFB` from `0xAE74`.
- custom_integrations/SuperMetroid-Snes/room_affb_from_ae74.provenance.json — bootstrap provenance and entry contract for this fixture.
- policies/room_clears/room_affb_from_ae74_to_b026.json — generated room policy with bounded traversal and door-input attempts; remains unverified.
- docs/tasks/SM-ROOM-SEG-15-residual.md — isolated-run residual and next action.

### Verify paste
```text
uv run python super_metroid/scripts/room/run_problem.py teleport room_affb_from_ae74_to_b026
exit code: 0
stdout (relevant):
{"problemId":"room_affb_from_ae74_to_b026","statePath":"custom_integrations/SuperMetroid-Snes/room_affb_from_ae74.state","state":{"frame":1,"game_state":8,"phase":"ordinary_gameplay","room_id_hex":"0xAFFB","samus_x":704,"samus_y":121,"pose":2,"door_transition":0}}
stderr: empty

uv run python super_metroid/scripts/room/run_problem.py run room_affb_from_ae74_to_b026
exit code: 1
stdout (relevant):
{"success":false,"failure":"policy ended in 0xAFFB; expected 0xB026","crossingFrame":null,"settledFrame":null,"totalFrames":565,"finalState":{"room_id_hex":"0xAFFB","samus_x":85,"samus_y":187,"pose":138,"door_transition":0},"assist":{"progression_writes":0,"capacity_writes":0},"policy":{"status":"generated_unverified"}}
stderr: empty
```

### Acceptance
- [x] Isolated run produced an honest residual with a pin; no promote was attempted because the run was RED.
- [x] Only this problem's fixture, policy, and optional residual were changed by this task; generated report output is ignored.
- [x] Residual states the dual-track non-claim.
- [x] Next card ID and one change are filled below.

### Residual risks
- The policy does not reach the target room `0xB026`; no practice promotion is available.
- The coarse traversal reaches the left wall at floor height, but the `y=7` exit door does not transition; explicit elevated-door geometry remains unresolved.
- This is not pure-green, continuous evidence, STATUS evidence, or a natural-entry full-run result.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-15-R1
- **One change:** Replace only the coarse exit approach with a bounded waypoint sequence that reaches the `y=7` left door from the pinned floor position.
- **Source state:** custom_integrations/SuperMetroid-Snes/room_affb_from_ae74.state

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Not continuous evidence; this is dual-track room practice only.

### Probe pin
room=0xAFFB pose=138 x=85 y=187 door_transition=0
