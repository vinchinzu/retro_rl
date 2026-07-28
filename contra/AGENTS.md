# Agent Instructions — contra

Scripted NES completion agent for **Contra** (run_and_gun track; maturity M0→M1).

## Identity

| Field | Value |
|-------|-------|
| Status | scaffolded / boot in progress |
| Integration | `Contra-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Contra.zip` |
| Local ROM | `contra/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python contra/scripts/setup_rom.py
uv run python contra/scripts/boot_probe.py
uv run pytest contra/tests -q
```

## Next milestone

first Stage 1 segment clear.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
