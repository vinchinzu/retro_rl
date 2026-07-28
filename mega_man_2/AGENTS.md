# Agent Instructions — mega_man_2

Scripted NES completion agent for **Mega Man 2** (platforming, stage_action track; maturity M0→M1).

## Identity

| Field | Value |
|-------|-------|
| Status | scaffolded / boot in progress |
| Integration | `MegaMan2-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Mega Man II.zip` |
| Local ROM | `mega_man_2/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python mega_man_2/scripts/setup_rom.py
uv run python mega_man_2/scripts/boot_probe.py
uv run pytest mega_man_2/tests -q
```

## Next milestone

first Robot Master stage segment clear.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
