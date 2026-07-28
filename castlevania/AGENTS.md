# Agent Instructions — castlevania

Scripted NES completion agent for **Castlevania** (platforming track; maturity M1).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified (M1) |
| Integration | `Castlevania-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Castlevania.zip` |
| Local ROM | `castlevania/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python castlevania/scripts/setup_rom.py
uv run python castlevania/scripts/boot_probe.py
uv run pytest castlevania/tests -q
```

## Next milestone

first Stage 1 segment clear.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
