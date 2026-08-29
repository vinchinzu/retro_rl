# Agent Instructions — zelda_ii

Scripted NES completion agent for **Zelda II: The Adventure of Link** (graph_navigation track; maturity M1).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified (M1) |
| Integration | `ZeldaII-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Zelda II - The Adventure of Link.zip` |
| Local ROM | `zelda_ii/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python zelda_ii/scripts/setup_rom.py
uv run python zelda_ii/scripts/boot_probe.py
uv run python zelda_ii/scripts/run_leave_palace.py --trials 3
uv run pytest zelda_ii/tests -q
```

## Next milestone

first overworld walk / first encounter side-scroll from `NorthPalaceExit`.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
