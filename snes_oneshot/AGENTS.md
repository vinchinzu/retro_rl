# Agent Instructions — snes_oneshot

Shared scripted helpers for the SNES completion ladder.

Program notes: [docs/EASIEST_SNES_GAMES.md](docs/EASIEST_SNES_GAMES.md)  
Full-run workflow: [docs/FULL_RUN_PROCESS.md](docs/FULL_RUN_PROCESS.md)

## Working Rule

Save-state development, segmented stage/scene scripts, and retries are fine.
Get reliable clears first; chain later. Continuous title-to-credits is a
later hardening step, not a gate.

## Module Map

| File | Purpose |
|------|---------|
| `game_state.py` | Normalized `GameState` / enemy / projectile dataclasses (`living_enemies`, `threat_enemies`, `nearest_threat`) |
| `actions.py` | Compatibility imports for `retro_harness.actions` |
| `cursor.py` | Point-and-click cursor pose/target/step helpers |
| `primitives.py` | Compatibility imports for `retro_harness.input_script` |
| `behavior.py` | Minimal behavior-tree nodes (Selector/Sequence/Condition) |
| `combat.py` | Beat-em-up helpers (`align_vertical`, `fight_nearest`, segment tree) |
| `segment_runner.py` | Headless segment stop heuristics, report/PNG helpers |
| `watchdog.py` | Stuck detection / recovery signals |
| `ram_diff.py` | Differential RAM snapshot helpers for discovery |
| `rom_setup.py` | Unzip shared ROM into a game `roms/` + integration symlink |
| `setup_all_roms.py` | CLI: `uv run python -m snes_oneshot.setup_all_roms` |

## Norms

- Prefer development save states and segment bots over fresh-reset full runs.
- Validate important segments from both clean checkpoints and natural
  predecessor entries before chaining a full run.
- Keep the last successful full-run baseline immutable; promote candidate
  reports only after integrity checks pass.
- Elevate reusable logic here; keep RAM maps and game policies in `<game>/`.
- Prefer typed enums/dataclasses over stringly state.
- Line length 88; type hints on public APIs; tests under `tests/`.
- Emulator I/O, named actions, and generic timed/menu inputs stay in
  `retro_harness/`; this package owns scripted-agent policy built on them.
