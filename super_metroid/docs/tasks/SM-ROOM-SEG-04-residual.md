# Residual - SM-ROOM-SEG-04

## Result

GREEN

## Files changed

- `policies/room_clears/room_93aa_from_91f8_to_91f8.json` - Replaced the coarse horizontal scaffold with a bounded jump/release traversal, PB pickup roll, mirrored return traversal, and promoted the policy after the green isolated run.
- `docs/tasks/SM-ROOM-SEG-04-residual.md` - Recorded isolated practice evidence, acceptance, and dual-track non-claims.

The existing doorway-natural fixture `custom_integrations/SuperMetroid-Snes/room_93aa_from_91f8.state` and its provenance file were present and unchanged.

## Verify paste

Repository-relative stdout summaries from the required commands:

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_93aa_from_91f8_to_91f8
exit=0
{
  "problemId": "room_93aa_from_91f8_to_91f8",
  "statePath": "custom_integrations/SuperMetroid-Snes/room_93aa_from_91f8.state",
  "state": {
    "frame": 1,
    "game_state": 8,
    "phase": "ordinary_gameplay",
    "room_id_hex": "0x93AA",
    "door_transition": 0,
    "samus_x": 64,
    "samus_y": 121,
    "pose": 1
  }
}

$ uv run python super_metroid/scripts/room/run_problem.py run room_93aa_from_91f8_to_91f8
exit=0
{
  "success": true,
  "crossingFrame": 1566,
  "settledFrame": 1764,
  "finalState": {
    "room_id_hex": "0x91F8",
    "samus_x": 2264,
    "samus_y": 395,
    "pose": 10,
    "health": 199,
    "power_bombs": 5,
    "max_power_bombs": 5
  },
  "objectiveVerification": {
    "objective": "collect_and_return",
    "status": "passed"
  },
  "assist": {
    "progression_writes": 0,
    "capacity_writes": 0,
    "deaths": 0
  },
  "policy.status": "generated_unverified"
}

$ uv run python super_metroid/scripts/room/run_problem.py run room_93aa_from_91f8_to_91f8 --promote
exit=0
{
  "success": true,
  "crossingFrame": 1566,
  "settledFrame": 1764,
  "objectiveVerification.status": "passed",
  "assist.progression_writes": 0,
  "assist.capacity_writes": 0,
  "assist.deaths": 0,
  "policy.status": "verified_development_state",
  "promoted": true
}
```

## Acceptance

- [x] Isolated run GREEN + promote.
- [x] Only the named problem policy and this problem residual were changed by this session; the existing entry fixture was unchanged.
- [x] Dual-track non-claim is explicit below.
- [x] Next card ID and one change are filled below.

## Residual risks

- This is an isolated doorway-natural practice result only.
- It is not pure-green continuous evidence, a natural predecessor-chain result, or a full-run integrity result.
- The planner still owns queue refresh, continuous composition, and STATUS promotion.
- No geometry residual remains for this problem after the promoted run.

## Next action (required)

- **Next card ID:** none
- **One change:** No further one-knob change is required for this room; planner may refresh the practice queue without changing this policy.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_93aa_from_91f8.state`

## Non-claims

- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Dual-track practice only; this is not continuous evidence.
- Did not claim a continuous tip or natural predecessor-chain clear.

## Probe pin

room=0x91F8 pose=10 x=2264 y=395 door_transition=0
frames=1764 dwell=N/A last_pin=room=0x91F8 pose=10 x=2264 y=395 door_transition=0
