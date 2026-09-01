# Run the core test gate

The core gate is the always-on, ROM-free pytest invocation that CI and `docs/TEST_TIERS.md` call `uv run pytest`.

## Sub-features

- `core-pytest` runs the default testpaths (`tests`, `retro_harness/tests`, `retro_harness/platformer/tests`, `retro_harness/adventure/tests`) with markers `not ml and not rom`.
- `core-qt` collects editor tests only when `QT_QPA_PLATFORM=offscreen`.
- `core-docs` includes `tests/test_docs.py` (manifests, links, generated matrix).

## How to get to it (user POV)

- Run `uv run pytest` from the repo root (README / TEST_TIERS core tier).
- On this machine, prefix `QT_QPA_PLATFORM=offscreen` so collection can import PySide6.
- Narrow the same gate while debugging: `uv run pytest retro_harness/tests -q` or `uv run pytest tests/test_docs.py -q`.

## Driving it with verify-retro-rl

Preconditions:

- Doctor reports `ok=true`.
- `QT_QPA_PLATFORM=offscreen` is set in the environment.
- The claim being proved is the **core** tier, not all-game or ROM smoke.

- **Core gate.** Run `.cursor/skills/verify-retro-rl/scripts/cli --out $RUN/core-pytest -- env QT_QPA_PLATFORM=offscreen uv run pytest`. Wait for pytest to finish. `env.txt` / the command must include `QT_QPA_PLATFORM=offscreen`.
- **Read the summary.** `stdout.txt` (or `stderr.txt`, pytest may split) ends with a `passed` / `failed` / `skipped` summary. `exit.txt` is `0` only if the core tier is green.
- **Confirm the filter.** The session did not collect `snes/` or `nes/` game suites as the primary tree (those are the all-game tier). Default addopts deselect `ml` and `rom`.
- **Proof.** Keep the full pytest log. A green run is proof of the core tier only. If `tests/test_docs.py` fails because gitignored `**/recordings/` or `*human*` route data is missing, copy the missing-path assertion into the run dir and report that entry as the documented fresh-clone miss, not as a silent skip of the whole gate.

## Gotchas

- Bare `uv run pytest` without `QT_QPA_PLATFORM=offscreen` often dies during collection on `libEGL.so.1`. That is an environment miss, not a product regression.
- `RETRO_RL_TEST_TIER=game-no-rom uv run pytest snes nes -m "not rom and not ml"` is a different tier. Do not cite it as core.
- `RETRO_RL_RUN_ROM_SMOKE=1 uv run pytest -m rom_smoke` needs ROMs. Doctor `rom_*=false` makes it unreachable.
- Passing unit tests prove at most fake-tested maturity for shared subsystems (`docs/TEST_TIERS.md`). Do not promote a planner or emulator pool to real-ROM tested from this feature.
