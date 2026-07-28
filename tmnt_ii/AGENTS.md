# Agent Instructions — tmnt_ii

Scripted NES completion agent for **Teenage Mutant Ninja Turtles II: The Arcade Game** (linear_combat track; maturity M1).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified (M1) |
| Integration | `TeenageMutantNinjaTurtlesII-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Teenage Mutant Ninja Turtles II - The Arcade Game.zip` |
| Local ROM | `tmnt_ii/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python tmnt_ii/scripts/setup_rom.py
uv run python tmnt_ii/scripts/boot_probe.py
uv run pytest tmnt_ii/tests -q
```

## Next milestone

first Stage 1 wave/segment clear.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
