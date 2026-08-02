## Residual — SM-PATH-ROOM-W01a

### Result
GREEN

Isolated practice clear of Frog Speedway (`room_b106_from_af72_to_b167`) is
green and practice-promoted. Mid-room Boost Blocks require Speed Booster; the
doorway fixture grants Speed (+ Hi-Jump + Varia) on a controllable post-spore
boot after door-warp. Policy dashes `LEFT+B` through the blocks, opens the left
blue door, and settles in Frog Savestation `0xB167`.

### Files changed
- `policies/room_clears/room_b106_from_af72_to_b167.json` — speed-dash room policy; promoted `verified_development_state`
- `custom_integrations/SuperMetroid-Snes/room_b106_from_af72.state` — doorway-natural entry with Speed Booster grant
- `custom_integrations/SuperMetroid-Snes/room_b106_from_af72.provenance.json` — fixture provenance
- `recordings/room_clears/room_b106_from_af72_to_b167.json` — isolated run report (promote side-effect)
- `docs/tasks/SM-PATH-ROOM-W01a-residual.md` — this residual

### Verify paste

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_b106_from_af72_to_b167
{
  "problemId": "room_b106_from_af72_to_b167",
  "statePath": ".../custom_integrations/SuperMetroid-Snes/room_b106_from_af72.state",
  "state": {
    "game_state": 8,
    "room_id_hex": "0xB106",
    "door_transition": 0,
    "samus_x": 1992,
    "samus_y": 139,
    "pose": 2,
    "collected_items": 12549,
    "varia": true,
    "hi_jump": true
  }
}
exit 0

$ uv run python super_metroid/scripts/room/run_problem.py run room_b106_from_af72_to_b167
{
  "success": true,
  "failure": null,
  "startRoomIdHex": "0xB106",
  "targetRoomIdHex": "0xB167",
  "crossingFrame": 375,
  "settledFrame": 537,
  "totalFrames": 537,
  "finalState": {
    "room_id_hex": "0xB167",
    "samus_x": 216,
    "samus_y": 139,
    "pose": 10,
    "door_transition": 0,
    "game_state": 8,
    "phase": "ordinary_gameplay"
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
exit 0

$ uv run python super_metroid/scripts/room/run_problem.py run room_b106_from_af72_to_b167 --promote
{
  "success": true,
  "crossingFrame": 375,
  "settledFrame": 537,
  "totalFrames": 537,
  "finalState": { "room_id_hex": "0xB167", "phase": "ordinary_gameplay" },
  "policy": { "status": "verified_development_state" },
  "promoted": true
}
exit 0
```

### Acceptance
- [x] `run_problem.py run` green for this room (and promoted)
- [x] Residual with next card ID + one change
- [x] Dual-track non-claim

### Residual risks
- Fixture Speed Booster grant is development-only; continuous KPDR still lacks a
  natural Speed loadout at first Speedway visit (circular vs early route).
- Catalog entry `block [79,7]` disagrees with natural door-warp lip (`x=2048`);
  fixture uses warp lip + inset, not the stale catalog block.
- Practice promote ≠ continuous / STATUS / path-board continuous clearance.

### Next action (required)
- **Next card ID:** SM-PATH-ROOM-W01b
- **One change:** Path-room clear next open hop on the path board (e.g. Upper
  Norfair Farming `0xAF72` or Bubble Mountain approach) as dual-track practice only.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_b106_from_af72.state` (this problem only)

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM in continuous controllers
- Not continuous evidence (dual-track practice only; Speed grant is fixture-local)

### Probe pin (if pure/geometry) — mandatory metrics
room=0xB167 pose=10 x=216 y=139 door_transition=0
frames=537 dwell=settled@537 crossing=375 last_pin=room=0xB167/pose=10/x=216/y=139
