## Residual - SM-ROOM-SEG-19

### Result
BLOCKED

The doorway-natural fixture and generated scaffold policy were already present,
so bootstrap and scaffold were skipped. The required isolated run was RED: the
policy reached the lower-right wall of `0xD78F` but never crossed to the
`0xD72A` exit. The fixture has only `collected_items=0x1004` (Morph+B); it has
no Varia or Hi-Jump movement capability. A bounded same-problem jump/bomb
timing check did not establish an ascent from this early Maridia source. A
valid late-Maridia natural source is required before changing room geometry.

### Files changed
- `docs/tasks/SM-ROOM-SEG-19-residual.md` - records the blocked practice run, source gap, and required next capture.

The problem policy and doorway fixture were reused unchanged. The room runner
also wrote its expected ignored report at
`recordings/room_clears/room_d78f_from_da60_to_d72a.json`.

### Verify paste
Repository-relative paths are used below.

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_d78f_from_da60_to_d72a
exit 0
{
  "problemId": "room_d78f_from_da60_to_d72a",
  "statePath": "custom_integrations/SuperMetroid-Snes/room_d78f_from_da60.state",
  "stateSha256": "249108d4e27098c88ecfea4a7252428549f911b8650109ff1deac0e6ccf68805",
  "state": {
    "frame": 1,
    "game_state": 8,
    "phase": "ordinary_gameplay",
    "room_id_hex": "0xD78F",
    "samus_x": 64,
    "samus_y": 633,
    "pose": 1,
    "collected_items": 4100,
    "morph_ball": true,
    "bombs": true,
    "varia": false,
    "hi_jump": false
  }
}

$ uv run python super_metroid/scripts/room/run_problem.py run room_d78f_from_da60_to_d72a
exit 1
stderr: none
{
  "problemId": "room_d78f_from_da60_to_d72a",
  "success": false,
  "failure": "policy ended in 0xD78F; expected 0xD72A",
  "startRoomIdHex": "0xD78F",
  "targetRoomIdHex": "0xD72A",
  "crossingFrame": null,
  "settledFrame": null,
  "totalFrames": 181,
  "finalState": {
    "room_id_hex": "0xD78F",
    "samus_x": 219,
    "samus_y": 651,
    "pose": 137,
    "door_transition": 0
  },
  "objectiveVerification": {
    "objective": "collect_items_and_exit",
    "status": "not_reached"
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
```

The promote command was not run because the isolated run was not green.

### Acceptance
- [x] Isolated run ended in an honest residual with the required pin; green promote was not possible.
- [x] Only own-files were touched; the existing policy and fixture were unchanged and this residual is the only durable file added by this session.
- [x] Dual-track non-claim is explicit below; this is practice-only and is not continuous evidence.
- [x] Next card ID and one change are filled below.

### Residual risks
- The policy remains `generated_unverified` and is not practice-promoted.
- The current early-game fixture lacks the late-Maridia movement capability needed by the tested Precious Room ascent; the objective and exit were not reached.
- Geometry edits from this source would risk encoding a false suitless route. The required source must be a real predecessor entry, not a full-loadout state or RAM forge.
- Queue refresh, continuous composition, STATUS promotion, and natural-entry claims remain out of scope.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-19-SRC
- **One change:** Capture a controllable natural `0xDA60 -> 0xD78F` entry with the required late-Maridia movement loadout, then rerun this policy unchanged without forging progression state.
- **Source state:** needs capture: SM-ROOM-SEG-19-SRC

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss-bit RAM.
- Did not practice-promote the policy.
- Not continuous evidence; this is dual-track room practice only.

### Probe pin (if pure/geometry)
room=0xD78F pose=137 x=219 y=651 door_transition=0
frames=181 dwell=N/A last_pin=room=0xD78F/pose=137/x=219/y=651
