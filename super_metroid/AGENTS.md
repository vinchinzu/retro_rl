# Agent Instructions — Super Metroid

Super Metroid scripted full-clear project. Shared process:
[`snes_oneshot/docs/FULL_RUN_PROCESS.md`](../snes_oneshot/docs/FULL_RUN_PROCESS.md).

## Evaluation contract

- Target: one continuous power-on-to-ending run.
- Allowed assists: unlimited health (in-game energy) and unlimited ammo,
  exactly as defined in
  [`docs/ASSIST_CONTRACT.md`](docs/ASSIST_CONTRACT.md).
- Resource assists remove attrition only. They must not grant uncollected ammo
  types, capacity, equipment, items, movement abilities, door state, map
  state, boss/event flags, rooms, or completion.
- Record every assist write in the full-run manifest.
- Completion requires the natural endgame escape and ending/credits evidence;
  defeating the final boss alone is not a clear.

## Organization

- Keep RAM addresses, route logic, maps, states, logs, recordings, and policy
  in `super_metroid/`.
- Save states belong under `custom_integrations/<GameId>/`.
- Use clean states for fast development and natural-entry states for
  acceptance.
- Prefer room/door/inventory progress vectors over coordinate-only watchdogs.
- Keep the last successful full-run baseline; candidates use separate reports.

## Immediate goal

The continuous power-on → Spore Spawn baseline is established. Continue from
the settled Spore Super room through Spore Farming/Big Pink to the first Power
Bomb expansion. Use `maps/room_problems.json` to develop isolated clears, but
promote only natural-entry continuous transitions.

Room-development commands:

```bash
uv run python super_metroid/scripts/export_room_problems.py
uv run python super_metroid/scripts/run_room_problem.py ready --run
uv run python super_metroid/scripts/run_room_problem.py route 0x9B5B 0x9E11 \
  --capability morph_ball --capability bombs --capability missiles \
  --capability spore_spawn_defeated --capability super_missiles
```

The generated graph/catalog, development states, reports, and screenshots are
gitignored local artifacts. The compact policies under
`policies/room_clears/` are curated source.
