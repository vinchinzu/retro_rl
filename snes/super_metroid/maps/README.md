# Super Metroid map assets

- `morph_graph.json` is generated from the typed, accepted graph in
  `super_metroid.progression`.
- `bombs_graph.json` extends that accepted graph through both
  early Missile expansions, the return climb, Bombs, Bomb Torizo, and the
  natural boss-room exit.
- `spore_graph.json` extends the accepted typed graph through
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
uv run python snes/super_metroid/scripts/export/progression_map.py
```

The default export is the latest accepted `spore_graph.json`.
Pass `--graph bombs` or `--graph morph` to regenerate
an earlier prefix artifact.

Re-import reference maps with:

```bash
uv run python snes/super_metroid/scripts/import_legacy_assets.py
```

Reference maps do not become verified navigation edges merely by being
present. Promote connections only after a continuous run observes the door
transition and required capabilities.

## Full-size visual references

The following local, gitignored images provide whole-map context while
developing room controllers:

| File | Dimensions | SHA-256 | Source |
|------|------------|---------|--------|
| `reference/scripterswar_zebes_tiles_full.png` | 16384×16384 | `95454764fe32ab0dd45c65a9961edc8fe3b8db24e8371d70a85aa94977d4380b` | [ScriptersWar interactive map](https://scripterswar.com/SuperMetroid/map), reconstructed from its 32×32 full-resolution tile set |
| `reference/snesmaps_zebes_full.png` | 16896×14336 | `114479f4b8c3d5dc60170f1ab8c61fc99ccaff59b784af063d239fa4fd477f9d` | [SNESMaps Zebes map](https://www.snesmaps.com/maps/SuperMetroid/SuperMetroidMapZebes.html) |

These are visual planning references only. They do not establish controller or
continuous-route evidence.

## Interactive path map (pixel-aligned CoG)

**Area basemaps** (`maps/legacy/crateria.png`, …) — not the ScriptersWar full
montage. `mapX`/`mapY` are **per-area**; global raw map squares do not line up
with the full website image.

```text
area_x = (mapX - area_min_map_x) * 256 + samus_x
area_y = (mapY - area_min_map_y) * 256 + samus_y
```

Polylines are **same-room + short step only** (default max 48 px). No
straight lines across doors or the map. Continuous tip JSON is markers only.

```bash
uv run python -m super_metroid.map_viewer serve --open --export-defaults

# Dense human CoG (recommended demo)
uv run python -m super_metroid.map_viewer export-path \
  tasks/parlor_left_human.json --id parlor_human

# Dense TAS series
uv run python -m super_metroid.map_viewer export-path \
  recordings/tas_import/resync_zebes_rooms --stride 2 --id resync_zebes
```

Generated assets: `maps/viewer/` (area PNGs, geojson, paths). UI package:
`super_metroid/map_viewer/`.

Regenerate the editor-backed Spore Spawn plan with:

```bash
uv run python snes/super_metroid/scripts/export/spore_spawn_plan.py
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
uv run python snes/super_metroid/scripts/export/room_problems.py
```

The full graph uses the sibling editor export for geometry and the game-local,
gitignored [`vg-json-data/sm-json-data`](https://github.com/vg-json-data/sm-json-data)
clone at `refs/sm-json-data` for complete physical connections. The reference
clone is pinned locally at
`d49da689b2620aa1a4223ebf505d4b7791d88662`; update it deliberately when
regenerating topology. Both generated JSON files are local ROM-derived
planning artifacts. See
[the room problem catalog](../docs/research/ROOM_PROBLEM_CATALOG.md) for source
semantics, teleport commands, queue policy, and the post-Spore route.
