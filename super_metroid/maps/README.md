# Super Metroid map assets

- `start_to_morph_graph.json` is generated from the typed, accepted graph in
  `super_metroid.progression`.
- `start_to_bomb_torizo_graph.json` extends that accepted graph through both
  early Missile expansions, the return climb, Bombs, Bomb Torizo, and the
  natural boss-room exit.
- `start_to_spore_spawn_graph.json` extends the accepted typed graph through
  Terminator, Green Brinstar, Spore Spawn, and the natural post-boss exit.
- `post_torizo_to_spore_spawn_plan.json` is an ability-aware, pre-calculated
  route from the external `super_metroid_editor` navigation export. Its source
  hashes and its explicit missing-direction patch are embedded in the file.
  It is planning input, not continuous-run evidence.
- `full_room_graph.json` merges reference physical topology with editor
  geometry for all 262 editor rooms. It contains 300 physical connections,
  583 directed traversals, and a 23-anchor full-game research sequence.
- `room_problems.json` assigns one canonical development problem to every
  editor room, including entry/exit endpoints, capability gates, static
  collision waypoints, queues, and expected practice artifacts.
- `legacy/world_map.json` is the previous project's room-name/ID catalog.
- `legacy/full_game_route.json` is an unverified objective-level full-game
  research route.
- `legacy/*.png` are prior area composites and local room references.

Regenerate the accepted graph with:

```bash
uv run python super_metroid/scripts/export_progression_map.py
```

The default export is the latest accepted `start_to_spore_spawn_graph.json`.
Pass `--graph start_to_bomb_torizo` or `--graph start_to_morph` to regenerate
an earlier prefix artifact.

Re-import reference maps with:

```bash
uv run python super_metroid/scripts/import_legacy_assets.py
```

Reference maps do not become verified navigation edges merely by being
present. Promote connections only after a continuous run observes the door
transition and required capabilities.

Regenerate the editor-backed Spore Spawn plan with:

```bash
uv run python super_metroid/scripts/export_spore_spawn_plan.py
```

Override the editor export location with `--editor-nav` or the
`SUPER_METROID_EDITOR_NAV` environment variable. The planner normalizes editor
ability names, rejects unavailable capabilities, and keeps route patches
marked `planned`.

The accepted Spore run observed every planned suffix edge, but the planner
artifact deliberately remains `planned_not_continuous`. The report connects
the two evidence layers by hash: the editor data proposes a route; the typed
emulator transition timeline proves which doors the continuous run took.
The editor reference route's Early Supers commentary is advisory and stale for
this slice; the accepted path reaches Spore Spawn without collecting Supers.

Regenerate the full research graph and room catalog with:

```bash
uv run python super_metroid/scripts/export_room_problems.py
```

The full graph uses the sibling editor export for geometry and the sibling
`sm-json-data` checkout for complete physical connections. Both generated JSON
files are local ROM-derived planning artifacts. See
[the room problem catalog](../docs/ROOM_PROBLEM_CATALOG.md) for source
semantics, teleport commands, queue policy, and the post-Spore route.
