# Editor Layers

## Goal

The Harvest editor should expose useful map structure, not just a flat tile-color view.

The current editor keeps the exact ROM-rendered snapshot map as the foundation, then adds focused live overlays on top of it.

## Start

```bash
./kickoff.sh
./startup.sh --state Y1_After_Buy_Potato
```

`kickoff.sh` starts the editor in the background, prefers `latest.state`, falls
back to `current.state`, and autostarts the embedded emulator session.

Export the current map and exit:

```bash
uv run python -m harvest.tools.editor_app --state Y1_After_Buy_Potato --export-dir debug_alignment/editor_exports
uv run python -m harvest.tools.editor_app --state latest --autostart --autoplay
```

## Current Layers

### Doors / transitions

- Draws known cross-map exits from `harvest/maps/map_config.py`.
- Also highlights door-like threshold tiles currently seen as:
  - `0xA4`
  - `0xC0`
  - `0xC3`
  - `0xD6`
- This is intended to make shop entrances, path exits, and similar transition tiles visible at a glance.

### Collision / blocked tiles

- Shades tiles that are not in the map-specific walkable set.
- Uses the walkable rules from `harvest/maps/map_config.py`, not one global walkability rule for every map.
- This makes town/path/shop collision overlays more useful than the earlier farm-only assumptions.

### Sprite clamp bounds

- Draws the ROM scene object clamp rectangle when the current scene exposes one.
- This is derived from the map scene model, not guessed from screen position.

### Sprite delta

- Live-session only.
- Compares the current emulator frame to the current base render and highlights pixels that differ.
- Useful for spotting NPCs, animated props, and other non-base-map elements.

Important: this is still a pixel-delta overlay. Use the separate Game objects /
NPCs layer when you want decoded WRAM object positions instead of frame-diff
pixels.

### Live viewport overlay

- Draws the current emulator viewport directly over the map canvas.
- Useful for checking camera alignment and validating visible tiles against the captured base pixels.
- This is an overlay only. Once a snapshot has seeded a full ROM render, live
  camera frames do not replace the full-map base with sprites or dialogue boxes.

### Player marker

- Shows the current player position on the full map.

### Game objects / NPCs

- Draws decoded WRAM game-object positions from `harvest/core/npc_catalog.py`.
- Known animals, candidate NPCs, and raw game objects get distinct markers and labels.
- This is the editor-visible version of the same dynamic object table used by
  world snapshots and bot diagnostics.

### Route waypoints

- Draws selected named routes from `harvest/maps/map_config.py`.
- Waypoints are filtered to the active tilemap, so cross-map routes show only
  the relevant segment while you inspect Farm, Path, Town, interiors, and so on.

### Day Plan Preview

- The right-side preview dock builds the current auto day plan from RAM-backed
  planner facts.
- It lists planned phases, deferred work, notes, and the facts that produced the
  decision. Route-backed phases expose their `map_config` route names.

## Current Sources

- Base map render:
  - exact ROM-rendered snapshot pixels placed in world coordinates by `harvest/tools/editor_app.py`
  - farm maps build a state-specific twin from `debug_alignment/reference_ranch_map.png` plus the current save-state tile grid; unchanged baseline tiles stay reference-backed, while crops, tilled ground, harvested plots, grass, debris changes, and other save-state differences render from the current state
  - farm twin pre-calcs are cached under `debug_alignment/editor_twin_cache/` and a `*_latest.png` copy is refreshed slowly during live play
- Live emulator:
  - shared `retro_harness.editor` bridge/panel with 1x, 2x, 3x, and 6x speed levels
  - `[` slows down, `]` speeds up, `F5` hot-saves, `F1` loads the hot save, `F6` toggles RAM recording, `F8` toggles embedded autoplay
  - the Harvest panel exposes `start_session()`, `set_autoplay_enabled()`, `toggle_autoplay()`, and `autoplay_enabled()` for programmatic control
  - editor-side map/RAM sync is throttled and WRAM travels over the bridge as a binary payload, so the emulator viewport can keep its frame rate while the full-map canvas updates periodically
- Doors / known exits:
  - `harvest/maps/map_config.py`
- Clamp bounds:
  - ROM scene metadata
- Collision:
  - `harvest/maps/map_config.py` walkable tile sets
- Sprite delta:
  - live frame minus current captured base map
- Dynamic game-object/NPC positions:
  - `harvest/core/npc_catalog.py` / `WorldSnapshot.entities`
- Route waypoints:
  - `harvest/maps/map_config.py`
- Plan preview:
  - `harvest/planner/day_plan_decision.py`

## Limits

- Game-object overlays are coordinate-backed markers; decoded sprite art and
  facing-specific render rules are not yet validated.
- Farm visual art still uses a reference-backed base until the ROM renderer
  fully emulates the game's `$7E:2000` graphics-map construction. Current
  save-state differences are patched from the state renderer and cached.
- Door candidates are still partly heuristic. Known transition rectangles are stronger than the generic door-like tile markers.
- Collision is only as good as the local walkable sets in `harvest/maps/map_config.py`.
- Unseen regions are intentionally left unknown instead of being synthesized.

## Good Next Steps

- Decode sprite art/facing for game-object overlays instead of marker-only rendering.
- Promote more map transitions from “door-like tile” to validated entrance rectangles.
- Split collision into separate overlays:
  - terrain blocked
  - building blocked
  - transition tiles
