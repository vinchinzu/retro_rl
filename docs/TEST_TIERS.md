# Tests

A test boots a legally supplied ROM (or a named save state of it) and
evaluates on the live emulator: load, play, assert RAM / leave / goal.
Without that, the pytest is worthless. Do not close a bead on it. Do
not STATUS from it.

New emulator tests carry `@pytest.mark.rom`. A short bounded
representative may also carry `@pytest.mark.rom_smoke`.

```bash
uv run pytest <path> -m rom
RETRO_RL_RUN_ROM_SMOKE=1 uv run pytest -m rom_smoke
```

Bare `uv run pytest` deselects `@pytest.mark.rom`. CI has no copyrighted
ROMs; that job is import/docs hygiene.

| Gate | Command | Claim |
|------|---------|-------|
| Core | `uv run pytest` | Shared harness, graph, platformer/offline, and docs tests collect without ML extras or ROMs |
| ML extra | `uv sync --frozen --extra ml` then `uv run pytest retro_harness/fighters/tests retro_harness/platformer/tests -m "not rom"` | Optional Gymnasium/Torch/Stable-Baselines imports collect |
| All-game no-ROM | `RETRO_RL_TEST_TIER=game-no-rom uv run pytest snes nes -m "not rom and not ml"` | Game-owned unit/offline tests collect; real integrations stay skipped |
| Real-ROM smoke | `RETRO_RL_RUN_ROM_SMOKE=1 uv run pytest -m rom_smoke` | Selected local smoke matrix boots and steps real integrations |

CI runs the first three. ROM smoke is local.

Game directories are excluded from bare `pytest` because many share test
module names. The all-game job discovers `snes/` and `nes/` with
importlib isolation.
