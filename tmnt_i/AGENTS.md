# Agent Instructions — tmnt_i

Scripted NES completion agent for **Teenage Mutant Ninja Turtles** (linear_combat track; maturity M1).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified (M1) |
| Integration | `TeenageMutantNinjaTurtles-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Teenage Mutant Ninja Turtles.zip` |
| Local ROM | `tmnt_i/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python tmnt_i/scripts/setup_rom.py
uv run python tmnt_i/scripts/boot_probe.py
uv run pytest tmnt_i/tests -q
```

## Next milestone

first Area 1 building/segment clear.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
