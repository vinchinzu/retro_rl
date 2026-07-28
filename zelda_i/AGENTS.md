# Agent Instructions — zelda_i

Scripted NES completion agent for **The Legend of Zelda** (graph_navigation track; maturity M1).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified (M1) |
| Integration | `LegendOfZelda-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Legend of Zelda, The.zip` |
| Local ROM | `zelda_i/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python zelda_i/scripts/setup_rom.py
uv run python zelda_i/scripts/boot_probe.py
uv run pytest zelda_i/tests -q
```

## Next milestone

first cave visit / first overworld segment policy.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
