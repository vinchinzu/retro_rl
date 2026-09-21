# List registered editors

Listing editors shows every game that registered a Qt editor with the shared launcher, without opening a window.

## Sub-features

- `editor-list` prints the registered project table.
- `editor-list-usage` prints the same table and exits `2` when no project id is given.
- `editor-list-harvest` includes the Harvest Moon row (`harvest` / `Harvest Moon`).

## How to get to it (user POV)

- Run `uv run python -m retro_harness.editor_launcher --list` from the repo root (README quickstart).
- Run `uv run python -m retro_harness.editor_launcher` with no arguments (usage / list, exit 2).
- Discover editors while launching one: `uv run python -m retro_harness.editor_launcher harvest -- --state Y1_Spring_D1_Farm` (GUI; not this feature's proof).

## Driving it with verify-retro-rl

Preconditions:

- Doctor reports `ok=true` and `editors` contains `harvest`.
- This run is not attached to a running Harvest editor window.

- **List editors.** Run `.cursor/skills/verify-retro-rl/scripts/cli --out $RUN/editor-list -- uv run python -m retro_harness.editor_launcher --list`. `exit.txt` is `0`. `stdout.txt` starts with `Registered editors:` and contains a line `harvest       Harvest Moon`.
- **Usage with no project.** Run `.cursor/skills/verify-retro-rl/scripts/cli --out $RUN/editor-list-usage -- uv run python -m retro_harness.editor_launcher`. `exit.txt` is `2`. `stdout.txt` still contains `Registered editors:` and `harvest`.
- **Proof.** Keep both step directories. The list output names `harvest` and `Harvest Moon` and does not launch a Qt window (no new PID in `$RUN/pids`).

## Gotchas

- `--list` does not import the Harvest Qt app. A passing list is not proof that `editor_launcher harvest` opens the GUI.
- Launching `harvest` with `DISPLAY` set and `QT_QPA_PLATFORM` unset opens a window on the user's session. Do not do that for this feature.
- `get_editor_project("earthbound")` is a missing id; the launcher will fail that name. Do not treat it as a registered editor.
- Editor discovery reads `snes/*/editor_registration.py` and `nes/*/editor_registration.py`. Only `harvest` is registered today; assert presence of `harvest`, not an exact singleton count, unless the map is updated.
