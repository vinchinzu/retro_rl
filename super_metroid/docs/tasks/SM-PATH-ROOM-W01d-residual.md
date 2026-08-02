## Residual — SM-PATH-ROOM-W01d

### Result
GREEN

Isolated practice clear of Single Chamber bottom exit
(`room_ad5e_from_b656_to_ae07`) is green and practice-promoted. Fixture
door-warps Musketeers → Single Chamber then places Samus in the left-shaft
**bottom** free zone near exit door 2 (Spiky Platforms); policy opens the
right blue door and settles in `0xAE07`. Varia + Hi-Jump granted on
post-spore boot (heat survivability / jump assist). Dual-track practice only.

**Scope note (honest):** This is a **bottom-door segment**, not a full
far-right (node 5) → shaft → exit clear. Top-corridor bomb/speed pillars
(sm-json `5→6` Screw/PB/Bombs/BlueSuit) and multi-screen left-shaft descent
remain open for R1.

### Files changed
- `policies/room_clears/room_ad5e_from_b656_to_ae07.json` — bottom-door room policy; promoted `verified_development_state`
- `custom_integrations/SuperMetroid-Snes/room_ad5e_from_b656.state` — mid-room bottom free-zone entry (after door-warp)
- `custom_integrations/SuperMetroid-Snes/room_ad5e_from_b656.provenance.json` — fixture provenance + grant/placement notes
- `recordings/room_clears/room_ad5e_from_b656_to_ae07.json` — isolated run report (promote side-effect)
- `docs/tasks/SM-PATH-ROOM-W01d-residual.md` — this residual

### Verify paste

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_ad5e_from_b656_to_ae07
{
  "problemId": "room_ad5e_from_b656_to_ae07",
  "statePath": ".../custom_integrations/SuperMetroid-Snes/room_ad5e_from_b656.state",
  "state": {
    "game_state": 8,
    "room_id_hex": "0xAD5E",
    "door_transition": 0,
    "samus_x": 210,
    "samus_y": 883,
    "pose": 42,
    "collected_items": 4357,
    "varia": true,
    "hi_jump": true,
    "morph_ball": true,
    "bombs": true
  }
}
exit 0

$ uv run python super_metroid/scripts/room/run_problem.py run room_ad5e_from_b656_to_ae07
{
  "success": true,
  "failure": null,
  "startRoomIdHex": "0xAD5E",
  "targetRoomIdHex": "0xAE07",
  "crossingFrame": 228,
  "settledFrame": 350,
  "totalFrames": 350,
  "finalState": {
    "room_id_hex": "0xAE07",
    "samus_x": 39,
    "samus_y": 139,
    "pose": 9,
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

$ uv run python super_metroid/scripts/room/run_problem.py run room_ad5e_from_b656_to_ae07 --promote
{
  "success": true,
  "crossingFrame": 228,
  "settledFrame": 350,
  "totalFrames": 350,
  "finalState": { "room_id_hex": "0xAE07", "phase": "ordinary_gameplay" },
  "policy": { "status": "verified_development_state" },
  "promoted": true
}
exit 0
```

### Acceptance
- [x] Isolated run green for this room (and promoted) — bottom-door segment
- [x] Residual with next card ID + one change
- [x] Dual-track non-claim

### Residual risks
- Fixture is **not** doorway-natural at far-right lip (`x≈1536`); Samus is
  placed at left-shaft bottom free zone (`x≈210 y≈880`) after door-warp.
  Catalog entry block `[31,7]` is stale vs natural warp lip.
- Top corridor bomb/speed pillars (sm-json `5→6`) **not** cleared by policy;
  Morph+Bombs/PB/Screw/BlueSuit path from Musketeers lip remains open.
- Full left-shaft multi-screen descent from junction 6 → door 2 not encoded;
  partial drop probes stalled around mid-shaft ledges (`y≈520`).
- Fixture Varia + Hi-Jump grants are development-only; continuous KPDR lacks
  this loadout at first Single Chamber visit.
- Practice promote ≠ continuous / STATUS / path-board continuous clearance.

### Next action (required)
- **Next card ID:** SM-PATH-ROOM-W01d-R1
- **One change:** Same-room R1 — clear far-right Musketeers lip through
  top-corridor bomb/speed pillars and full left-shaft descent to door 2
  (or re-fixture doorway-natural at warp lip `x≈1480` and green the full
  path under dual-track practice only).
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_ad5e_from_b656.state` (this problem only)

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM in continuous controllers
- Not continuous evidence (dual-track practice only; Varia/Hi-Jump grants and
  mid-room placement are fixture-local)
- Did not clear top-corridor geometry from natural far-right entry

### Probe pin (if pure/geometry) — mandatory metrics
room=0xAE07 pose=9 x=39 y=139 door_transition=0
frames=350 dwell=settled@350 crossing=228 last_pin=room=0xAE07/pose=9/x=39/y=139
