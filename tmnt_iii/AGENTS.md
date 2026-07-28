# Agent Instructions — tmnt_iii

Scripted NES completion agent for **Teenage Mutant Ninja Turtles III: The Manhattan Project** (linear_combat track; maturity M1).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified (M1) |
| Integration | `TeenageMutantNinjaTurtlesIII-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Teenage Mutant Ninja Turtles III - The Manhattan Project.zip` |
| Local ROM | `tmnt_iii/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python tmnt_iii/scripts/setup_rom.py
uv run python tmnt_iii/scripts/boot_probe.py
uv run pytest tmnt_iii/tests -q
```

## Next milestone

first Stage 1 segment clear.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
