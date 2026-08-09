# Test and CI Tiers

The repository has four fail-closed test tiers. A green tier only makes the
claim named below; it does not silently promote a shared subsystem's maturity.

| Tier | Command | Claim |
|------|---------|-------|
| Core | `uv run pytest` | Shared harness, graph, platformer/offline, and docs tests pass without ML extras or ROMs |
| ML extra | `uv sync --frozen --extra ml` then `uv run pytest retro_harness/fighters/tests retro_harness/platformer/tests -m "not rom"` | Optional Gymnasium/Torch/Stable-Baselines imports and offline training components collect and pass |
| All-game no-ROM | `RETRO_RL_TEST_TIER=game-no-rom uv run pytest snes nes -m "not rom and not ml"` | Every collected game-owned unit/offline test passes; real integrations remain skipped or deselected |
| Real-ROM smoke | `RETRO_RL_RUN_ROM_SMOKE=1 uv run pytest -m rom_smoke` | The selected local smoke matrix boots and steps real integrations |

The CI workflow runs the first three tiers. The ROM smoke job is opt-in because
copyrighted ROMs and save states are never stored in CI. New tests that call a
real emulator must carry `@pytest.mark.rom`; a short, bounded representative
may additionally carry `@pytest.mark.rom_smoke`.

Game directories are excluded from bare `pytest` deliberately: many retain the
same test module names and collectively form a much larger gate. The all-game
job discovers both `snes/` and `nes/` explicitly with importlib isolation, so a
new game suite is included automatically rather than relying on a hand-written
slug list.

## Shared subsystem maturity

Passing unit tests proves at most **fake-tested** maturity. Planner, benchmark,
pool, contract, and solver work must report the highest evidenced rung:

1. scaffolded
2. fake-tested
3. real-ROM tested
4. first real-game consumer
5. second independent consumer
6. publication-ready

A bead for shared infrastructure must not be closed as “complete” beyond its
evidenced rung. A first consumer is required for a reusable capability claim;
a second independent consumer is required before the interface is treated as
stable shared infrastructure.
