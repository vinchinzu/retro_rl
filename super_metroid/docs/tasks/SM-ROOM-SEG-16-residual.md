## Residual - SM-ROOM-SEG-16

### Result
BLOCKED

The existing doorway-natural fixture and generated policy were present, so
bootstrap and scaffold were skipped. The isolated policy run was RED at the
Crab Gate green gate. The fixture has no Gravity (`collected_items=0x1004`, so
the `0x0020` Gravity bit is clear), while the source catalog requires a
controllable natural Gravity+Super source for this underwater branch. A
bounded check of the documented suitless gate timing variants also did not
cross the gate.

### Files changed
- `docs/tasks/SM-ROOM-SEG-16-residual.md` - records the blocked practice run and source gap.

### Verify paste

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_d08a_from_d21c_to_cfc9
{
  "problemId": "room_d08a_from_d21c_to_cfc9",
  "statePath": "custom_integrations/SuperMetroid-Snes/room_d08a_from_d21c.state",
  "state": {
    "frame": 1,
    "game_state": 8,
    "room_id_hex": "0xD08A",
    "door_transition": 0,
    "samus_x": 448,
    "samus_y": 121,
    "pose": 2,
    "collected_items": 4100,
    "super_missiles": 5,
    "varia": false,
    "hi_jump": false
  }
}
exit 0

$ uv run python super_metroid/scripts/room/run_problem.py run room_d08a_from_d21c_to_cfc9
{
  "success": false,
  "failure": "policy ended in 0xD08A; expected 0xCFC9",
  "totalFrames": 443,
  "crossingFrame": null,
  "finalState": {
    "room_id_hex": "0xD08A",
    "samus_x": 245,
    "samus_y": 171,
    "pose": 138,
    "door_transition": 0
  },
  "assist": {
    "progression_writes": 0,
    "capacity_writes": 0,
    "deaths": 0
  },
  "policy": {
    "status": "generated_unverified"
  }
}
exit 1
```

The `--promote` command was not run because the isolated run was RED. The
fixture and policy remain unchanged.

### Acceptance
- [x] Isolated run ended in an honest residual with the required pin; green promote was not possible.
- [x] Only the task-owned residual file was changed; the Crab Tunnel policy and fixture were not edited.
- [x] This is dual-track room practice only and is not continuous evidence.
- [x] Next card ID and one source-capture change are filled below.

### Residual risks
- The policy remains `generated_unverified` and is not practice-promoted.
- The current fixture cannot support the intended underwater green-gate branch without a valid controllable Gravity source.
- No pure-green, continuous, STATUS, or full-run claim is made.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-16-SRC
- **One change:** Capture and install only a controllable natural Crab Tunnel doorway source with Gravity (`collected_items & 0x0020`) and at least one Super, then re-run this problem unchanged.
- **Source state:** needs capture: SM-ROOM-SEG-16-SRC

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence; this is a blocked dual-track practice attempt.

### Probe pin (if pure/geometry) - mandatory metrics
room=0xD08A pose=138 x=245 y=171 door_transition=0
frames=443 dwell=443 last_pin=room=0xD08A/pose=138/x=245/y=171
