# Agent Instructions — smb

Scripted NES completion agent for **Super Mario Bros.** (platforming track; maturity M0→M1).

## Identity

| Field | Value |
|-------|-------|
| Status | scaffolded / boot in progress |
| Integration | `SuperMarioBros-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Super Mario Bros..zip` |
| Local ROM | `smb/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python smb/scripts/setup_rom.py
uv run python smb/scripts/boot_probe.py
uv run pytest smb/tests -q
```

## Next milestone

first 1-1 segment clear (flagpole).

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
