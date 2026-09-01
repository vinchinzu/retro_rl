# retro_rl verification map

This directory is the maintained source for verifying the shared user-facing behavior of retro_rl. Read the index before driving, then use the matching feature file as the recipe. Game-specific ROM hops, bosses, and headed play belong to `.grok/skills/sm-session`, `zelda-session`, and `harvest-session`.

## Baseline preconditions

- Working directory is the retro_rl repo root (`pyproject.toml` name `retro_rl`).
- `./setup.sh` has been run; commands go through `uv run` on Python 3.12.
- `export QT_QPA_PLATFORM=offscreen` before pytest or any import of `PySide6`.
- `uv run python .cursor/skills/verify-retro-rl/scripts/doctor.py` reports `ok=true`.
- Evidence root is `.cursor/skills/verify-retro-rl/artifacts/<run-id>/`.
- Never drive a Qt editor or headed emulator this run did not start.

## Driving conventions

- Start every recipe from the baseline unless its preconditions say otherwise.
- Treat every command as literal. Keep module names, flags, and `./play` aliases unchanged.
- Run user commands through `.cursor/skills/verify-retro-rl/scripts/cli --out <dir> -- <command>`.
- Restore `docs/GAME_MATRIX.md` after a matrix drive if the run should not leave a dirty tree. Keep the snapshot in artifacts.
- Do not remove proof artifacts during cleanup.

## Proof and skip reporting

- Capture the user action and the resulting stdout or file, not only the exit code.
- CLI proof includes command, stdout, stderr, and exit code.
- Mutation proof includes a second read of the written file (matrix) or a JSON record (planner).
- Record the feature ID and entry point with every artifact.
- Report an unreachable path with the attempted command and the unmet precondition (missing ROM, display, extras).
- Do not report a skipped entry point as verified through a different path.
- A red `tests/test_docs.py` on a fresh clone that lacks gitignored `**/recordings/` or `*human*` route data is the documented cloud/fresh-clone miss, not a silent skip of the core gate.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with verify-retro-rl` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [List registered editors](./editor-list.md) covers `editor_launcher --list`, the no-args usage exit, and the harvest registration line.
- [Dispatch play scripts](./play-dispatch.md) covers `./play --list`, help, and unknown-game rejection.
- [Run the core test gate](./core-gate.md) covers the ROM-free `uv run pytest` core tier.
- [Plan an adventure path](./adventure-plan.md) covers the public planner and `inventory_aware_path` on the morph-door fixture.
- [Regenerate the game matrix](./game-matrix.md) covers `docs/generate_game_matrix.py` writing `docs/GAME_MATRIX.md`.
