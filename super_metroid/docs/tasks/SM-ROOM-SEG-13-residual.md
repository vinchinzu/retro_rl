## Residual — SM-ROOM-SEG-13

### Result
BLOCKED

### Files changed
- `docs/tasks/SM-ROOM-SEG-13-residual.md` — records the blocked dual-track Moat practice result and the required powered source capture.

No durable change was made to `policies/room_clears/room_95ff_from_93fe_to_948c.json` or the existing doorway fixture. The room runner generated its expected ignored report at `recordings/room_clears/room_95ff_from_93fe_to_948c.json`.

### Verify paste
```text
uv run python super_metroid/scripts/room/run_problem.py teleport room_95ff_from_93fe_to_948c
exit 0
{
  "problemId": "room_95ff_from_93fe_to_948c",
  "state": {
    "game_state": 8,
    "phase": "ordinary_gameplay",
    "room_id_hex": "0x95FF",
    "door_transition": 0,
    "samus_x": 448,
    "samus_y": 121,
    "pose": 2,
    "collected_items": 4100,
    "morph_ball": true,
    "bombs": true,
    "varia": false,
    "hi_jump": false
  }
}

uv run python super_metroid/scripts/room/run_problem.py run room_95ff_from_93fe_to_948c
exit 1
{
  "success": false,
  "failure": "policy ended in 0x95FF; expected 0x948C",
  "crossingFrame": null,
  "settledFrame": null,
  "totalFrames": 443,
  "finalState": {
    "room_id_hex": "0x95FF",
    "pose": 138,
    "samus_x": 245,
    "samus_y": 379,
    "door_transition": 0
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
```

The promote command was not run because the isolated run was not green.

### Acceptance
- [x] Isolated run **GREEN + promote** **or honest residual with pin** — honest residual path satisfied; the policy never crossed from `0x95FF` to `0x948C`, so promotion was correctly skipped.
- [x] Only own-files touched — no policy or fixture edit was made; this residual is the only durable file added by the session. The runner report is an expected ignored verification artifact.
- [x] Dual-track non-claim — this is isolated room practice only and is not continuous evidence.
- [x] Next card ID + one change filled.

### Residual risks
- The fixture is based on `natural_post_spore_spawn.state` and has `collected_items=0x1004`: morph ball and bombs are present, but Varia and Hi-Jump are absent; the source also predates the Speed Booster.
- The KPDR route board states that Moat traversal uses Speed + Hi-Jump (shinespark or platform jumps). The current early-game fixture therefore does not provide the intended movement source for this room.
- Movement/bomb timing changes cannot establish a valid Moat practice green from this source; the policy remains `generated_unverified`.
- Queue refresh, continuous composition, STATUS promotion, and natural-entry claims remain out of scope.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-13-R1
- **One change:** Capture a natural, controllable post-Speed/post-Hi-Jump Moat entry fixture from the real predecessor, then rerun the existing policy without forging progression or capacity state.
- **Source state:** needs capture: SM-ROOM-SEG-13-SRC

### Non-claims
- Did not STATUS-promote.
- Did not promote the practice policy.
- Did not forge progression, capacity, door, event, boss-bit, or item RAM.
- Did not claim continuous green; this is dual-track isolated practice only.

### Probe pin (if pure/geometry)
room=0x95FF pose=138 x=245 y=379 door_transition=0
frames=443 dwell=443 last_pin=room=0x95FF pose=138 x=245 y=379 door_transition=0
