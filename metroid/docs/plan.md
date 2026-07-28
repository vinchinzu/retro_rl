# Plan — Metroid (NES)

## Near term

1. **First missiles** — stop on `$687A` capacity > 0.
   - **Verified natural prefix:** use the low morph tunnel to return to the
     skree start, cross both long corridors, open three blue doors, and land
     on the third west-shaft platform at map (11,13), x≈106/y=225 with 14
     energy. `first_missiles.py` terminates this evidence point as `FRONTIER`.
   - **Current blocker:** clear the enemy-populated upper west shaft, cross
     the bridge, descend the east shaft, and collect the expansion.
   - Existing graph node `b_first_missiles` and route
     `metroid_first_missiles` stay planned until `$687A > 0` is observed.
2. **Bombs** — old lady sequence; equipment bit 0.
3. **Natural-entry chain** — morph → missiles → bombs without state loads.

## Medium term

- Expand Brinstar map graph with verified door edges (promote `planned` →
  `continuous` only with emulator evidence).
- Promote more Super Metroid progression helpers into `adventure_common` once
  both graphs share milestone/stop-predicate patterns.
- Long Beam / Ice Beam / Varia as optional side routes.

## Probe notes (2026-07-28)

- B shoots (projectile visible); doors should open to beam hits.
- AfterMorph settle: idle ~180 frames until `game_mode == 3` (leave fanfare).
- Morph return uses the floor tunnel: roll to map (3,14), x≈76/y=200,
  unmorph for one frame, then RIGHT+A into the skree room.
- Door transitions must stop on the first `in_door == 0` target-map frame;
  fixed post-transition holds change enemy timing and cost energy.
- Both AfterMorph and power-on entry reach the same frontier pose without
  loading a state during the natural run.

## Notes

- Cart WRAM equipment is at `$6878` (read via `env.data.memory.extract`).
- Morph room is map cell **(1, 14)** west of start **(3, 14)**.
- First missiles stop: `is_missiles_obtained` / capacity `$687A > 0`.
