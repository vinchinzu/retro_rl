---
name: verify-retro-rl
description: "Verify retro_rl's shared CLI and library: editor launcher, ./play dispatcher, core pytest gate, adventure planner, and game-matrix generator. Use when proving a repo-wide change, running doctor on the checkout, or checking that list/plan/pytest commands still work. Game-specific ROM routes stay with the game session skills."
---

# Verify retro_rl

Primary surface is the **CLI + Python library** at the monorepo root (`uv run python -m …`, `./play`, `uv run pytest`). Secondary is the Qt game editor (currently Harvest Moon via `editor_launcher`). Headed emulator play needs legally supplied ROMs and is owned by game session skills, not this one.

There is no long-lived server. Launch means the checkout can import and run short-lived commands. Each drive is its own process.

Repo root is the directory whose `pyproject.toml` contains `name = "retro_rl"`. Run every command from there.

## Launch

One-time:

```bash
./setup.sh
# optional extras; not required for this skill's mapped features
# uv sync --all-extras
```

Ready when `uv run python -c "import retro_harness; print('ok')"` prints `ok` on Python 3.12. System Python is often 3.14; always go through `uv run`.

For any command that **imports PySide6** — including bare `uv run pytest`, which collects `retro_harness/tests/test_editor.py` — set:

```bash
export QT_QPA_PLATFORM=offscreen
```

Without it, collection raises `ImportError: libEGL.so.1` / xcb errors. A `pipewire-0.3` warning from Qt multimedia is harmless.

Teardown: there is no daemon. If a drive recorded PIDs, run Cleanup. Evidence stays.

## Doctor

Run this first whenever anything looks off, and before the first drive of a run:

```bash
export QT_QPA_PLATFORM=offscreen
uv run python .cursor/skills/verify-retro-rl/scripts/doctor.py
```

Worth driving when stdout has `ok=true`, `python_ok=true` (3.12.x), `harness_ok=true`, and `editors` includes `harvest`. ROM lines (`rom_super_metroid=…`) are informational: missing ROMs do not fail doctor, and they do not unlock ROM-gated features.

Refuse to drive:

- a Qt editor or headed `./play` session this run did not start
- a second `docs/generate_game_matrix.py` while another is writing `docs/GAME_MATRIX.md`

Two CLI doctor/list/plan processes may run side by side.

## Drive

Read `features/README.md`, then the matching feature file. Run every command through the `cli` helper so stdout, stderr, and the exit code land in the run directory.

```bash
SKILL=.cursor/skills/verify-retro-rl
RUN="$SKILL/artifacts/<run-id>"
mkdir -p "$RUN"
"$SKILL/scripts/cli" --out "$RUN/<step>" -- <command>
```

Stable handles (do not substitute):

| Surface | Handle |
|---|---|
| Editor list | `uv run python -m retro_harness.editor_launcher --list` — stdout starts with `Registered editors:` and includes `harvest` |
| Editor missing project | `uv run python -m retro_harness.editor_launcher` with no args — same list, exit `2` |
| Play list | `./play --list` — lines for `smb`, `sm`, `alttp-rando`, `sm-rando` |
| Play help | `./play --help` or `./play` with no args — `Usage: ./play <game>` |
| Core gate | `QT_QPA_PLATFORM=offscreen uv run pytest` |
| Adventure plan | `uv run python .cursor/skills/verify-retro-rl/scripts/plan_demo.py` |
| Game matrix | `uv run python docs/generate_game_matrix.py` — prints `Wrote docs/GAME_MATRIX.md` |

Prefer those argv strings over clicking the Harvest Qt window or poking emulator RAM. Pytest is the existing harness for library behavior; the mapped features still require the user-facing command, not a test-only import of a private helper.

Unknown `./play <game>` prints `Unknown game:` to stderr and exits `1`.

## Evidence

Root: `.cursor/skills/verify-retro-rl/artifacts/<run-id>/`. Cleanup must not delete this tree.

Each step directory from `cli` contains `command.txt`, `stdout.txt`, `stderr.txt`, `exit.txt`, `cwd.txt`, `env.txt`. Copy or snapshot side-effect files into the same step directory (the matrix feature snapshots `docs/GAME_MATRIX.md`).

Proof standards:

- Drive the real user command (`-m retro_harness.editor_launcher`, `./play`, `uv run pytest`, `docs/generate_game_matrix.py`, `plan_demo.py` which calls public `plan` / `inventory_aware_path`). Do not treat a passing unit test of a private function as proof of a listed CLI entry point.
- Capture the command and the resulting stdout/file, not only the final exit code.
- Confirm side effects: matrix writes `docs/GAME_MATRIX.md`; planner JSON has `path_edge_ids`; pytest terminal shows passed/failed counts.
- Mocks stay where production already isolates them (pytest fakes inside `retro_harness/tests`). Do not stub `editor_launcher --list` or the matrix generator for a proof.
- `QT_QPA_PLATFORM=offscreen` is required for pytest, not a dry-run. Confirm it is set in `env.txt`. ROM-gated commands (`./play sm …`, harvest editor GUI, `@pytest.mark.rom`) skip emulator boot when ROMs are absent — observe `No romfiles found` / skip, and do not report those paths as verified.

Record the feature id and entry point next to the artifacts (a `feature.txt` in the run dir is enough).

## Cleanup

```bash
.cursor/skills/verify-retro-rl/scripts/cleanup --run-dir .cursor/skills/verify-retro-rl/artifacts/<run-id>
```

Kills only PIDs listed in that run's `pids` file (one numeric PID per line). Short-lived CLI drives leave no `pids` file; cleanup then prints `no pid file` and exits 0.

After a matrix drive, restore `docs/GAME_MATRIX.md` from the snapshot in the evidence dir if the working tree should stay clean. Keep the snapshot and diff in artifacts.

Never `pkill python`, `pkill pytest`, or kill by window title. Never delete `artifacts/`.

## Helpers

All paths are from repo root. `cli` and `cleanup` are executable bash; the `.py` helpers run under `uv run python`.

```bash
export QT_QPA_PLATFORM=offscreen
SKILL=.cursor/skills/verify-retro-rl

uv run python "$SKILL/scripts/doctor.py"
"$SKILL/scripts/cli" --out "$SKILL/artifacts/<run-id>/<step>" -- \
  uv run python -m retro_harness.editor_launcher --list
uv run python "$SKILL/scripts/plan_demo.py"
"$SKILL/scripts/cleanup" --run-dir "$SKILL/artifacts/<run-id>"
```

`cli` always exits 0 so a failing command still leaves evidence; read `exit.txt` for the child's status.

## Feature map

Maintained source: [features/README.md](features/README.md). A proof that drives one convenient entry point is incomplete when the map lists others.
