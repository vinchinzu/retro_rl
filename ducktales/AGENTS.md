# Agent Instructions — ducktales

Scripted NES completion agent for **DuckTales** (platforming track; maturity M0→M1).

## Identity

| Field | Value |
|-------|-------|
| Status | scaffolded / boot in progress |
| Integration | `DuckTales-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Duck Tales.zip` |
| Local ROM | `ducktales/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python ducktales/scripts/setup_rom.py
uv run python ducktales/scripts/boot_probe.py
uv run pytest ducktales/tests -q
```

## Next milestone

first stage segment clear.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
