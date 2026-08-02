## Residual — SM-PATH-ROOM-W01c

### Result
GREEN

Isolated practice clear of Speed Booster Hall (`room_acf0_from_ad1b_to_b07a`)
is green and practice-promoted. Reverse trip from right-door (Speed Room)
entry uses Speed Booster dash over crumbles to the left blue door into Bat
Cave `0xB07A`. Doorway fixture grants Speed (+ Hi-Jump + Varia) on a
controllable post-spore boot after door-warp; catalog block `[111,23]` is
stale vs natural warp lip `x=3072` — fixture uses warp lip + inset.

### Files changed
- `policies/room_clears/room_acf0_from_ad1b_to_b07a.json` — speed-dash room policy; promoted `verified_development_state`
- `custom_integrations/SuperMetroid-Snes/room_acf0_from_ad1b.state` — doorway-natural entry with Speed Booster grant (warp-lip spawn)
- `custom_integrations/SuperMetroid-Snes/room_acf0_from_ad1b.provenance.json` — fixture provenance
- `recordings/room_clears/room_acf0_from_ad1b_to_b07a.json` — isolated run report (promote side-effect)
- `docs/tasks/SM-PATH-ROOM-W01c-residual.md` — this residual

### Verify paste

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_acf0_from_ad1b_to_b07a
{
  "problemId": "room_acf0_from_ad1b_to_b07a",
  "statePath": ".../custom_integrations/SuperMetroid-Snes/room_acf0_from_ad1b.state",
  "state": {
    "game_state": 8,
    "room_id_hex": "0xACF0",
    "door_transition": 0,
    "samus_x": 3016,
    "samus_y": 395,
    "pose": 2,
    "collected_items": 12549,
    "varia": true,
    "hi_jump": true
  }
}
exit 0

$ uv run python super_metroid/scripts/room/run_problem.py run room_acf0_from_ad1b_to_b07a
{
  "success": true,
  "failure": null,
  "startRoomIdHex": "0xACF0",
  "targetRoomIdHex": "0xB07A",
  "crossingFrame": 495,
  "settledFrame": 644,
  "totalFrames": 644,
  "finalState": {
    "room_id_hex": "0xB07A",
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

$ uv run python super_metroid/scripts/room/run_problem.py run room_acf0_from_ad1b_to_b07a --promote
{
  "success": true,
  "crossingFrame": 495,
  "settledFrame": 644,
  "totalFrames": 644,
  "finalState": { "room_id_hex": "0xB07A", "phase": "ordinary_gameplay" },
  "policy": { "status": "verified_development_state" },
  "promoted": true
}
exit 0
```

### Acceptance
- [x] Isolated run green for this room (and promoted)
- [x] Residual with next card ID + one change
- [x] Dual-track non-claim

### Residual risks
- Fixture Speed Booster grant is development-only; continuous KPDR still lacks
  natural Speed loadout at first Speed Hall reverse visit.
- Catalog entry block `[111,23]` disagrees with natural door-warp lip (`x=3072`);
  fixture uses warp lip + inset, not the stale catalog block.
- Hidden Missile PLM not collected in-run; boot already has `max_missiles=10` so
  objective harness skips ammo progress (pre-collected capacity rule).
- Practice promote ≠ continuous / STATUS / path-board continuous clearance.

### Next action (required)
- **Next card ID:** SM-PATH-ROOM-W01d
- **One change:** Path-room clear next open hop (Single Chamber
  `room_ad5e_from_b656_to_ae07`) as dual-track practice only.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_acf0_from_ad1b.state` (this problem only)

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM in continuous controllers
- Not continuous evidence (dual-track practice only; Speed grant is fixture-local)

### Probe pin (if pure/geometry) — mandatory metrics
room=0xB07A pose=10 x=216 y=139 door_transition=0
frames=644 dwell=settled@644 crossing=495 last_pin=room=0xB07A/pose=10/x=216/y=139
