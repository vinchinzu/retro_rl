# Residual - SM-ROOM-SEG-06

## Result
GREEN

## Files changed
- `policies/room_clears/room_ad1b_from_acf0_to_acf0.json` - Replaced the coarse scaffold with the Speed Booster platform, item, return, and door-entry sequence; promoted to `verified_development_state` after the green run.
- `docs/tasks/SM-ROOM-SEG-06-residual.md` - Recorded isolated practice evidence, acceptance, and non-claims.

The existing entry fixture `custom_integrations/SuperMetroid-Snes/room_ad1b_from_acf0.state` was present and unchanged.

## Verify paste

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_ad1b_from_acf0_to_acf0
exit 0
{
  "problemId": "room_ad1b_from_acf0_to_acf0",
  "statePath": "custom_integrations/SuperMetroid-Snes/room_ad1b_from_acf0.state",
  "state": {
    "game_state": 8,
    "phase": "ordinary_gameplay",
    "room_id_hex": "0xAD1B",
    "door_transition": 0,
    "samus_x": 64,
    "samus_y": 121,
    "pose": 1
  }
}

$ uv run python super_metroid/scripts/room/run_problem.py run room_ad1b_from_acf0_to_acf0
exit 0
{
  "success": true,
  "crossingFrame": 879,
  "settledFrame": 1011,
  "totalFrames": 1011,
  "finalState": {
    "room_id_hex": "0xACF0",
    "collected_items": 12292,
    "equipped_items": 12292,
    "phase": "ordinary_gameplay"
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
  "developmentOnly": true,
  "policy.status": "generated_unverified"
}

$ uv run python super_metroid/scripts/room/run_problem.py run room_ad1b_from_acf0_to_acf0 --promote
exit 0
{
  "success": true,
  "crossingFrame": 879,
  "settledFrame": 1011,
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
- [x] Only own-files touched.
- [x] Dual-track non-claim recorded below.
- [x] Next card ID and one change filled below.

## Residual risks
- This is an isolated doorway-natural practice result only.
- It is not pure-green continuous evidence, a natural predecessor-chain result, or a full-run integrity result.
- The generated practice report is development-only; the planner still owns queue refresh and any continuous composition.

## Next action
- **Next card ID:** none
- **One change:** No geometry residual remains for this problem; planner may refresh the practice queue without changing this policy.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_ad1b_from_acf0.state`

## Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Dual-track practice only; this is not continuous evidence.

## Probe pin
room=`0xACF0` pose=`10` x=`3032` y=`395` door_transition=`0` after settled exit; no failed probe.
