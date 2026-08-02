## Residual -- SM-ROOM-SEG-21

### Result
BLOCKED

### Files changed
- `policies/room_clears/room_d8c5_from_d6d0_to_d69a.json` -- changed the coarse exit approach to `LEFT`, matching the doorway contract; policy remains unverified.
- `docs/tasks/SM-ROOM-SEG-21-residual.md` -- records the blocked isolated replay, source gap, and exact pin.

### Verify paste
```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_d8c5_from_d6d0_to_d69a
exit=0
{"problemId":"room_d8c5_from_d6d0_to_d69a","statePath":"custom_integrations/SuperMetroid-Snes/room_d8c5_from_d6d0.state","state":{"frame":1,"game_state":8,"phase":"ordinary_gameplay","room_id_hex":"0xD8C5","door_transition":0,"samus_x":448,"samus_y":120,"pose":2,"equipped_items":4100,"collected_items":4100,"bombs":true,"varia":false,"hi_jump":false}}

$ uv run python super_metroid/scripts/room/run_problem.py run room_d8c5_from_d6d0_to_d69a
exit=1
{"success":false,"failure":"policy ended in 0xD8C5; expected 0xD69A","crossingFrame":null,"settledFrame":null,"totalFrames":381,"finalState":{"room_id_hex":"0xD8C5","samus_x":448,"samus_y":120,"pose":40,"door_transition":0},"assist":{"progression_writes":0,"capacity_writes":0,"deaths":0},"policy":{"status":"generated_unverified"}}
stderr: none
```

The fixture and policy were already present, so bootstrap and scaffold were
skipped. A bounded frame probe also held the fixture at `x=448`, `y=120`, with
zero velocity and `door_transition=0` for 20 `LEFT` frames; the input state is
not controllable enough to evaluate room geometry. The fixture has only Morph
and Bombs (`collected_items=0x1004`), not Gravity (`0x0020`), and the source
catalog requires a controllable natural Gravity + Bombs source with Spring Ball
still clear. The `--promote` run was not issued because the isolated run was
RED.

### Acceptance
- [x] Isolated run GREEN + promote **or** honest residual with pin: blocked RED run is documented honestly; no promotion claimed.
- [x] Only own-files touched by this session: the Shaktool policy and this residual; the existing Shaktool fixture was not edited.
- [x] Dual-track non-claim is explicit below.
- [x] Next card ID and one change are filled below.

### Residual risks
- The policy remains `generated_unverified`; practice promotion is blocked by the missing exit transition.
- The existing doorway fixture is input-frozen during replay and lacks Gravity, so further timing edits would not establish a valid Shaktool traversal.
- Adding Gravity, capacity, progression, door, event, or boss state by RAM write would violate the task contract.
- This result is isolated room practice only and provides no continuous, STATUS, or natural-entry evidence.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-21-R1
- **One change:** Capture and install only a controllable natural D6D0-to-D8C5 doorway source with Gravity + Bombs and Spring Ball clear, then rerun the current policy unchanged.
- **Source state:** needs capture: `SM-ROOM-SEG-21-SRC`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Not continuous evidence; this is dual-track room practice only.

### Probe pin (if pure/geometry) -- mandatory metrics
room=0xD8C5 pose=40 x=448 y=120 door_transition=0
frames=381 dwell=381 last_pin=room=0xD8C5/pose40/x448/y120/door_transition0
