# Agent Instructions — super_metroid_rl

Super Metroid-specific code and artifacts live here. Do not spill SM-specific
states, notes, or debug files into the repo root.

## Organization

- Save states belong in `custom_integrations/SuperMetroid-Snes/`.
- Active docs and runbooks belong in `docs/`.
- Historical plans and one-off run logs belong in `docs/archive/` or
  `scripts/archive/`.
- Debug captures belong in `debug_screens/` or `debug_frames/`.
- Maps and nav reference exports belong in `maps/`.
- Recordings, demos, logs, and models stay under the matching local folders.

## Active Code

- `__main__.py`: CLI entrypoint
- `bronze_tools.py`: doctor and boot-probe helpers
- `navigation/`: room graph, waypoint, and trace-map tooling
- `state_manager.py`: manual state management
- `record_tasker.py`: demo recording workflow
- `platformer_common/levels/super_metroid.py`: published segment configs

`legacy/` is frozen reference material unless the task explicitly revives it.

## State Handling

- Published anchors and quick saves should be written to
  `custom_integrations/SuperMetroid-Snes/`.
- If a tool creates states, make the output path explicit and keep it local to
  this directory.
