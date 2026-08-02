# Residual — SM-ROOM-SEG-20

### Result
BLOCKED (required isolated run RED)

### Files changed
- `docs/tasks/SM-ROOM-SEG-20-residual.md` — records the failed practice run, last pin, capability blocker, and next action.

The existing problem fixture and generated policy were reused unchanged. No
other problem policy or fixture was edited.

The final scoped worktree check also observed unrelated changes in
`routes/continuous.py`, `routes/kpdr/k4_norfair.py`, and
`routes/kpdr/hops.py`; this card did not make or modify those changes.

### Verify paste

`uv run python super_metroid/scripts/room/run_problem.py teleport room_d7e4_from_d913_to_d95e`

Exit code: 0. Relevant stdout:

```text
{
  "problemId": "room_d7e4_from_d913_to_d95e",
  "statePath": "custom_integrations/SuperMetroid-Snes/room_d7e4_from_d913.state",
  "stateSha256": "bc3310a00f122eb67e6045ff8b9cbc6a7223da2c8ff2fe572b45b2cb0bffd187",
  "state": {
    "game_state": 8,
    "phase": "ordinary_gameplay",
    "room_id_hex": "0xD7E4",
    "door_transition": 0,
    "samus_x": 1736,
    "samus_y": 139,
    "pose": 2,
    "collected_items": 4100,
    "max_health": 199,
    "morph_ball": true,
    "bombs": true,
    "varia": false,
    "hi_jump": false
  }
}
```

`uv run python super_metroid/scripts/room/run_problem.py run room_d7e4_from_d913_to_d95e`

Exit code: 1. Stderr: none. Relevant stdout:

```text
{
  "problemId": "room_d7e4_from_d913_to_d95e",
  "success": false,
  "failure": "policy ended in 0xD7E4; expected 0xD95E",
  "crossingFrame": null,
  "settledFrame": null,
  "totalFrames": 443,
  "finalState": {
    "room_id_hex": "0xD7E4",
    "game_state": 8,
    "door_transition": 0,
    "samus_x": 1352,
    "samus_y": 203,
    "pose": 10
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
    "path": "policies/room_clears/room_d7e4_from_d913_to_d95e.json",
    "status": "generated_unverified"
  },
  "developmentOnly": true
}
```

Additional bounded movement probes from the same fixture reached the lower
sand ledge around `x=1349`, `y=203-217` and did not reach the upper morph
passage. The fixture inventory is Morph+B bombs (`collected_items=0x1004`),
without Gravity or Speed Booster. The room reference requires Gravity or a
validated suitless mid-air-morph route for the item path; development full-
loadout anchors are not a valid substitute.

### Acceptance
- [x] Isolated run was executed; its RED result is recorded honestly instead of being promoted.
- [x] Only own-files were touched by this session; the residual is the only repository file added.
- [x] Dual-track non-claim is explicit below; this is practice-only and not continuous evidence.
- [x] Next card ID and one change are filled below.

### Residual risks
- The policy remains `generated_unverified`; no practice promotion is justified.
- The current fixture cannot cross the lower sand obstacle with the available loadout, so the Energy Tank objective is not reached.
- A controllable natural-entry source retaining the required Maridia capability is missing; development full-loadout states must not be used to fabricate a green.
- Continuous integrity, `STATUS.md`, progression, and route claims are unaffected and were not tested by this card.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-20-R1
- **One change:** Capture a controllable natural-entry `D913 -> D7E4` fixture with Gravity present before changing policy geometry.
- **Source state:** needs capture: `SM-ROOM-SEG-20-SRC`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Did not claim continuous evidence; this is dual-track room practice only.
- Did not promote the unverified policy.

### Probe pin (if pure/geometry)
room=0xD7E4 pose=10 x=1352 y=203 door_transition=0
frames=443 dwell=N/A last_pin=room=0xD7E4 pose=10 x=1352 y=203 door_transition=0
