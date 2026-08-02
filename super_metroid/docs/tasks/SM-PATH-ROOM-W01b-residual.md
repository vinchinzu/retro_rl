## Residual — SM-PATH-ROOM-W01b

### Result
GREEN

Isolated practice clear of Bubble Mountain (`room_acb3_from_b07a_to_aedf`) is
green and practice-promoted. Top-right Bat Cave doorway entry → mid-room Power
Bomb floor clear → bottom blue door into Purple Shaft `0xAEDF`. Fixture boots
from a post-PB development state so the practice entry carries Power Bomb
capacity (required for the mid-room PB floor / obstacle A). Policy morphs on the
PB ledge, places one PB, drops the lower maze, and opens the bottom door with
pulsed `DOWN+X` shots (long hold morphs Samus instead of shooting).

### Files changed
- `policies/room_clears/room_acb3_from_b07a_to_aedf.json` — Bubble Mountain room policy; promoted `verified_development_state`
- `custom_integrations/SuperMetroid-Snes/room_acb3_from_b07a.state` — doorway-natural entry with PB capacity (post-PB boot)
- `custom_integrations/SuperMetroid-Snes/room_acb3_from_b07a.provenance.json` — fixture provenance + PB grant note
- `recordings/room_clears/room_acb3_from_b07a_to_aedf.json` — isolated run report (promote side-effect)
- `docs/tasks/SM-PATH-ROOM-W01b-residual.md` — this residual

### Verify paste

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_acb3_from_b07a_to_aedf
{
  "problemId": "room_acb3_from_b07a_to_aedf",
  "statePath": ".../custom_integrations/SuperMetroid-Snes/room_acb3_from_b07a.state",
  "state": {
    "game_state": 8,
    "room_id_hex": "0xACB3",
    "door_transition": 0,
    "samus_x": 448,
    "samus_y": 121,
    "pose": 2,
    "power_bombs": 5,
    "max_power_bombs": 5,
    "morph_ball": true,
    "bombs": true
  }
}
exit 0

$ uv run python super_metroid/scripts/room/run_problem.py run room_acb3_from_b07a_to_aedf
{
  "success": true,
  "failure": null,
  "startRoomIdHex": "0xACB3",
  "targetRoomIdHex": "0xAEDF",
  "crossingFrame": 1671,
  "settledFrame": 1816,
  "totalFrames": 1816,
  "finalState": {
    "room_id_hex": "0xAEDF",
    "samus_x": 127,
    "samus_y": 95,
    "pose": 49,
    "door_transition": 0,
    "game_state": 8,
    "phase": "ordinary_gameplay"
  },
  "objectiveVerification": {
    "objective": "collect_items_and_exit",
    "status": "passed"
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

$ uv run python super_metroid/scripts/room/run_problem.py run room_acb3_from_b07a_to_aedf --promote
{
  "success": true,
  "crossingFrame": 1671,
  "settledFrame": 1816,
  "totalFrames": 1816,
  "finalState": { "room_id_hex": "0xAEDF", "phase": "ordinary_gameplay" },
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
- Fixture Power Bomb capacity comes from a post-PB development boot
  (`dev_b1_red_tower_post_pb`); continuous KPDR still lacks natural PBs at first
  Bubble visit (Alpha PB is later on the route).
- Catalog objective is `collect_items_and_exit` (visible missile at block
  [20,60]); boot already has max missiles ≥ required pack, so collect check
  is skipped — exit-only practice path.
- Lower-maze navigation is coarse hop timing; enemy RNG / Sova position can
  still force re-bootstrap with different `boot_idle_frames`.
- Practice promote ≠ continuous / STATUS / path-board continuous clearance.

### Next action (required)
- **Next card ID:** SM-PATH-ROOM-W01c
- **One change:** Path-room clear Speed Booster Hall
  (`room_acf0_from_ad1b_to_b07a`) as dual-track practice only.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_acb3_from_b07a.state` (this problem only)

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM in continuous controllers
- Not continuous evidence (dual-track practice only; PB capacity is fixture-local)

### Probe pin (if pure/geometry) — mandatory metrics
room=0xAEDF pose=49 x=127 y=95 door_transition=0
frames=1816 dwell=settled@1816 crossing=1671 last_pin=room=0xAEDF/pose=49/x=127/y=95
