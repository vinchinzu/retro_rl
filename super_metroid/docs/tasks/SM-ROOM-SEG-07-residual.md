# Residual — SM-ROOM-SEG-07

## Result
RED

## Files changed
- `policies/room_clears/room_cda8_from_caf6_to_caf6.json` — changed only the `deeper_into_room` direction from `LEFT+B` to `RIGHT+B`; policy remains `generated_unverified`.
- `docs/tasks/SM-ROOM-SEG-07-residual.md` — recorded the isolated RED run and next one-knob action.

## Verify paste

Relevant stdout is reproduced with repository-relative paths.

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_cda8_from_caf6_to_caf6
exit=0
{
  "problemId": "room_cda8_from_caf6_to_caf6",
  "statePath": "custom_integrations/SuperMetroid-Snes/room_cda8_from_caf6.state",
  "stateSha256": "11f4d3d63910402aa9eb6314c91741e63ee5aa026bea72f20757c117a5c07189",
  "state": {
    "frame": 1,
    "game_state": 8,
    "phase": "ordinary_gameplay",
    "room_id_hex": "0xCDA8",
    "door_transition": 0,
    "samus_x": 192,
    "samus_y": 121,
    "pose": 2
  }
}

$ uv run python super_metroid/scripts/room/run_problem.py run room_cda8_from_caf6_to_caf6
exit=1
{
  "success": false,
  "failure": "room objective incomplete: max_super_missiles did not increase by 5",
  "startRoomIdHex": "0xCDA8",
  "targetRoomIdHex": "0xCAF6",
  "crossingFrame": 739,
  "settledFrame": 926,
  "totalFrames": 926,
  "finalState": {
    "room_id_hex": "0xCAF6",
    "door_transition": 4,
    "samus_x": 1063,
    "samus_y": 1675,
    "pose": 9
  },
  "objectiveVerification": {
    "objective": "collect_and_return",
    "status": "failed"
  },
  "assist": {
    "progression_writes": 0,
    "capacity_writes": 0,
    "deaths": 0
  },
  "policy": {
    "path": "policies/room_clears/room_cda8_from_caf6_to_caf6.json",
    "status": "generated_unverified"
  },
  "developmentOnly": true
}
```

Promotion was skipped because the isolated run was RED. Bootstrap and scaffold
were skipped because the existing doorway fixture and policy were present.

## Acceptance

- [x] Isolated run has an honest RED residual with a pin; no green or promote claim was made.
- [x] This session changed only the problem policy and this problem's residual; unrelated concurrent edits in `room_93aa_from_91f8_to_91f8.json`, `room_9c07_from_9bc8_to_9bc8.json`, and `room_ce40_from_c98e_to_93fe.json` were not touched.
- [x] Dual-track non-claim is explicit below.
- [x] Next card ID and one change are filled below.

## Residual risks

- The Super Missile PLM at block `(2,7)` was not collected, so this room is not practice-promoted.
- The changed `RIGHT+B` item approach reaches the return room but bypasses the collection objective.
- This result does not establish pure-green, continuous integrity, queue promotion, or STATUS readiness.

## Next action (required)

- **Next card ID:** SM-ROOM-SEG-07-R1
- **One change:** Tune only the `deeper_into_room` span to a bounded `LEFT+B` approach long enough to reach the Super Missile at block `(2,7)`; leave all return, door-open, and enter spans unchanged.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_cda8_from_caf6.state`

## Non-claims

- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- This is dual-track practice only and is not continuous evidence.
- Did not practice-promote the policy.

## Probe pin (room-practice failure)

room=0xCAF6 pose=9 x=1063 y=1675 door_transition=4
frames=926 dwell=N/A last_pin=room=0xCAF6 pose=9 x=1063 y=1675 door_transition=4
