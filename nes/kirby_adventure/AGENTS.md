# Agent Instructions — kirby_adventure

Scripted NES completion agent for **Kirby's Adventure** (platforming track; maturity M1).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified (M1) |
| Integration | `KirbysAdventure-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Kirby's Adventure.zip` |
| Local ROM | `kirby_adventure/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python kirby_adventure/scripts/setup_rom.py
uv run python kirby_adventure/scripts/boot_probe.py
uv run pytest kirby_adventure/tests -q
```

## Next milestone

first stage/segment clear from Vegetable Valley hub.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
