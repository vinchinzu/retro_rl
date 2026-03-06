# Super Metroid RL - Architecture

## Current Active Code

Entry point: `__main__.py` routes between two modes:
- **`nav-*` subcommands** (nav-path, nav-room, nav-waypoints, nav-info) -- handled in-house via the `navigation/` package
- **`doctor` / `boot-probe`** -- handled in-house via `bronze_tools.py` for Bronze readiness and true-start probing
- **Everything else** (play, verify, hillclimb, watch, selftest, optimize, auto-state) -- delegated to `platformer_common.runner.main()` after registering SM levels via `platformer_common.levels.super_metroid`

Top-level modules:
- `bronze_tools.py` -- Bronze doctor + boot-probe helpers for states/maps/nav exports
- `state_manager.py` -- create, rename, manage save states with `{Phase}_{Room}_{Direction}_{Items}.state` naming
- `train_curriculum.py` -- per-segment PPO training (largely superseded by the record-then-hillclimb workflow)
- `record_tasker.py` -- record human demo runs, save `.bk2` + JSON action sequences

## Navigation Stack (`navigation/`)

Pure-Python graph and waypoint system built from exported SM room/door data (default: `/tmp/sm_export/`).

| Module | Purpose |
|--------|---------|
| `map_data.py` | Load room collision grids, door info, nav graph from SMEDIT JSON exports |
| `world_graph.py` | `WorldGraph` -- BFS inter-room pathfinding with ability gates |
| `room_navigator.py` | `RoomNavigator` -- intra-room screen-path and door lookup |
| `waypoint_gen.py` | Auto-generate waypoints for hill climbing optimizer segments |
| `route.py` | `SPEEDRUN_ROUTE` -- ordered list of `RouteStep` segments (Landing Site through Bomb Torizo) |
| `seed_synth.py` | Synthesize action seeds from waypoint hints for optimizer bootstrapping |
| `trace_renderer.py` | Overlay a trace JSON onto area map PNGs (used by `trace-map` CLI command) |
| `room_analyzer.py` | Room geometry analysis utilities |
| `tests/` | Navigation unit tests |

Data dependency: the active nav loader expects the SMEDIT export layout (`nav_graph.json` plus `rooms/*.json`), usually produced by `super_metroid_editor/cli`. `refs/sm-json-data/` is reference material, not a drop-in replacement for the active loader.

## Scripts & Tooling (`scripts/`)

Operational scripts, not imported as library code.

- `bk2_to_mp4.py` -- convert emulator movie files to video
- `chop_recording.py` -- trim/split recorded action sequences
- `eval_full_route.py`, `eval_segments.py`, `eval_torizo_integration.py` -- evaluate trained models
- `pick_best_segments.py`, `verify_segments.py` -- compare segment recordings
- `record_hierarchical_runs.py`, `record_segment_episodes.py`, `record_segment_full_runs.py` -- batch recording helpers
- `resume_record.py`, `reprocess_recovered.py` -- recording recovery
- `tournament.py` -- run tournament-style segment evaluation
- `watch_full_route.py` -- visual route playback
- `analysis/` -- `plot_progress.py`, `view_stats.py`
- `maintenance/` -- `import_sm_data.py`, `migrate_stats.py`, `normalize_map.py`, `rename_room.py`, `sync_state_names.py`
- `debug/`, `fix_seg04.py`, `identify_rooms.py` -- one-off investigation scripts

Historical orchestration from the February 2026 overnight run now lives under:

- `docs/archive/2026-02-overnight_torizo/`
- `scripts/archive/2026-02-overnight_torizo/`

## Legacy Code (`legacy/`)

Frozen early experiments. Not imported by active code. Kept for reference only.

Includes: `super_metroid_naive.py`, `run_bot.py`, `metroid_env.py`, `metroid_rewards.py`, `train_bc_nav.py`, `train_ceres.py`, `train_general.py`, `train.py`, `hierarchical_ppo.py`, `play_human.py`, `record_demo.py`, `replay_demo.py`, `extract_state.py`, `extract_demo.py`, `extract_next.py`, `find_boss_x.py`, `verify_boss_x.py`, `record_boss.py`, `eval_boss.py`, `metroid_tool.py`, `run_bot_debug.py`, and associated shell scripts.

## Editor (`super_metroid_editor/`)

Git submodule. Separate Java project for exporting SM room data (collision, doors, items, enemies) to JSON. Outputs feed `navigation/map_data.py`. Not Python; not part of the training pipeline.

## Artifact Directories

| Directory | Contents | Gitignored |
|-----------|----------|------------|
| `custom_integrations/SuperMetroid-Snes/` | Save states (`.state`), `data.json`, `scenario.json` | States: no; ROM: yes |
| `optimizer/runs/` | Hill climber output (best actions, traces) per segment | Yes |
| `models/` | Trained PPO model checkpoints (`.zip`) | Yes |
| `recordings/` | `.bk2` emulator movie files | Yes |
| `demos/` | Saved demo action sequences | Yes |
| `logs/` | Training logs | Yes |
| `maps/` | Area map PNGs for trace rendering | No |
| `debug_frames/`, `debug_screens/` | Debug visualization output | Yes |
| `boss_data/` | Boss fight training data (monitor CSVs, demo NPZ) | Partial |
| `refs/sm-json-data/` | Community SM data (submodule or copy) | Varies |
| `roms/` | ROM file | Yes |
| `tests/` | Env diagnostics and combat mechanics tests | No |

## Shared Library Dependencies

- **`platformer_common/`** -- generic side-scroller optimizer (evaluator, GA, hill climbing, play/watch/verify CLI). SM levels are registered in `platformer_common/levels/super_metroid.py`. All `play`, `hillclimb`, `watch`, `verify`, `selftest`, `optimize`, `auto-state`, `trace-map` commands come from here.
- **`retro_harness/`** -- controller input handling (`keyboard_action`, `controller_action`, `sanitize_action`), play session management, bot runner. Used by `state_manager.py`, `record_tasker.py`, and legacy code.

## What Is Current vs Legacy vs Shared-Candidate

| Status | Code |
|--------|------|
| **Current** | `__main__.py`, `navigation/`, level configs in `platformer_common/levels/super_metroid.py` |
| **Current (standalone tools)** | `state_manager.py`, `record_tasker.py`, `scripts/` |
| **Superseded but kept** | `train_curriculum.py` (PPO approach; record+hillclimb is now primary) |
| **Legacy (frozen)** | Everything in `legacy/` |
| **Shared-candidate** | `record_tasker.py` recording logic overlaps with `platformer_common` play command; `state_manager.py` overlaps with `platformer_common` auto-state |
