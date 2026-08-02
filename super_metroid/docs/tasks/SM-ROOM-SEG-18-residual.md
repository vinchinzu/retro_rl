## Residual - SM-ROOM-SEG-18

### Result
BLOCKED

The doorway-natural fixture and generated policy already existed, so bootstrap
and scaffold were skipped. The isolated run reached and settled in the target
room `0xD8C5`, but the Spring Ball objective was incomplete:
`collected_items` remained `0x1004` and the fixture has no Gravity capability.
The policy remains `generated_unverified`; no promotion was attempted.

### Files changed
- `docs/tasks/SM-ROOM-SEG-18-residual.md` - records the blocked isolated practice run and source gap.

### Verify paste

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_d6d0_from_d8c5_to_d8c5
{
  "problemId": "room_d6d0_from_d8c5_to_d8c5",
  "statePath": "custom_integrations/SuperMetroid-Snes/room_d6d0_from_d8c5.state",
  "state": {
    "game_state": 8,
    "room_id_hex": "0xD6D0",
    "door_transition": 0,
    "samus_x": 64,
    "samus_y": 121,
    "pose": 1,
    "collected_items": 4100,
    "morph_ball": true,
    "bombs": true,
    "varia": false
  }
}
exit 0

$ uv run python super_metroid/scripts/room/run_problem.py run room_d6d0_from_d8c5_to_d8c5
{
  "success": false,
  "failure": "room objective incomplete: collected_items did not change",
  "startRoomIdHex": "0xD6D0",
  "targetRoomIdHex": "0xD8C5",
  "crossingFrame": 786,
  "settledFrame": 938,
  "totalFrames": 938,
  "finalState": {
    "room_id_hex": "0xD8C5",
    "samus_x": 984,
    "samus_y": 139,
    "pose": 10,
    "door_transition": 0,
    "collected_items": 4100,
    "varia": false
  },
  "assist": {
    "progression_writes": 0,
    "capacity_writes": 0,
    "deaths": 0
  },
  "policy": {
    "status": "generated_unverified"
  },
  "developmentOnly": true
}
exit 1

$ uv run pytest super_metroid/tests/test_room_graph.py -q
...........                                                              [100%]
11 passed in 0.33s
exit 0
```

The bootstrap command was skipped because the problem state already existed.
The scaffold command was skipped because the problem policy already existed.

### Acceptance
- [x] Isolated run produced an honest residual with the required pin; green promote was not possible.
- [x] Only task-owned intent was changed; the existing Spring Ball policy and doorway fixture were not edited.
- [x] This is dual-track room practice only and is not continuous evidence.
- [x] Next card ID and one source-capture change are filled below.

### Residual risks
- The policy remains `generated_unverified` and is not practice-promoted.
- The current doorway fixture lacks the natural Gravity capability needed to test this underwater Spring Ball traversal.
- No pure-green, continuous, STATUS, or full-run claim is made.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-18-R1
- **One change:** Capture and install only a doorway-natural source with Gravity and Bombs while Spring Ball is uncollected, then rerun this policy unchanged.
- **Source state:** needs capture: SM-ROOM-SEG-18-SRC

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence; this is a blocked dual-track practice attempt.

### Probe pin (if pure/geometry)
room=0xD8C5 pose=10 x=984 y=139 door_transition=0
frames=938 dwell=938 last_pin=room=0xD8C5/pose=10/x=984/y=139/door_transition=0
