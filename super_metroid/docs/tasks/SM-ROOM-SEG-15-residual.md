## Residual -- SM-ROOM-SEG-15

### Result
RED

### Files changed
- `docs/tasks/SM-ROOM-SEG-15-residual.md` -- records the failed isolated replay, exact pin, and planner gate.

### Verify paste
```text
$ uv run python scripts/room/run_problem.py teleport room_cf80_from_d21c_to_cefb
exit=0
{"problemId":"room_cf80_from_d21c_to_cefb","statePath":"custom_integrations/SuperMetroid-Snes/room_cf80_from_d21c.state","state":{"frame":1,"room_id_hex":"0xCF80","phase":"ordinary_gameplay","door_transition":0,"samus_x":448,"samus_y":121,"pose":2}}

$ uv run python scripts/room/run_problem.py run room_cf80_from_d21c_to_cefb
exit=1
{"success":false,"failure":"policy ended in 0xCF80; expected 0xCEFB","crossingFrame":null,"settledFrame":null,"totalFrames":443,"finalState":{"room_id_hex":"0xCF80","samus_x":373,"samus_y":139,"pose":138,"door_transition":0}}
stderr: none
```

The bootstrap and scaffold steps were skipped because the problem-specific
doorway fixture and policy already existed on disk. No promote run was issued
because the isolated run was not green. Bounded diagnostics also tried flat
run, spin-jump, crouch/morph, bomb, and shot variants; all retained the same
barrier pin.

### Acceptance
- [x] Isolated run GREEN + promote **or** honest residual with pin: RED run is documented honestly; no promotion claimed.
- [x] Only own-files touched: this residual is the only tracked file added by this session; the runner's room report is an expected ignored artifact.
- [x] Dual-track non-claim is explicit below.
- [x] Next card ID and one change are filled below.

### Residual risks
- The policy remains `generated_unverified`; practice promotion is blocked by the missing exit transition.
- The doorway fixture begins with Morph/Bombs only (`collected_items=0x1004`), while the room's Green Gate blocks this reverse route; adding capability or capacity by RAM write would violate the task contract.
- The reference room data lists no ordinary minimal-inventory `node 3 -> node 1` strategy; direct listed routes are Grapple Teleport variants. The route may need a capability-valid source or planner re-scope before another timing edit.
- This result is isolated room practice only and provides no continuous, STATUS, or natural-entry evidence.

### Next action (required)
- **Next card ID:** PLANNER-GATE
- **One change:** Decide whether to re-scope this reverse problem to a legitimate capability-valid source/entry or leave it unresolved before changing any policy timing span.
- **Source state:** needs capture: `SM-ROOM-SEG-15-SRC`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- This is not continuous evidence; it is dual-track room practice only.

### Probe pin (if pure/geometry)
room=0xCF80 pose=138 x=373 y=139 door_transition=0
frames=443 dwell=N/A last_pin=0xCF80/pose138/x373/y139/door_transition0
