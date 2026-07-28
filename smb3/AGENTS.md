# Agent Instructions — smb3

Scripted NES completion agent for **Super Mario Bros. 3** (platforming track; maturity M0→M1).

## Identity

| Field | Value |
|-------|-------|
| Status | scaffolded / boot in progress |
| Integration | `SuperMarioBros3-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Super Mario Bros. 3.zip` |
| Local ROM | `smb3/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python smb3/scripts/setup_rom.py
uv run python smb3/scripts/boot_probe.py
uv run pytest smb3/tests -q
```

## Next milestone

first World 1-1 segment clear.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
