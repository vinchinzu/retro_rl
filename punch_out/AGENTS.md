# Agent Instructions — punch_out

Scripted NES completion agent for **Mike Tyson's Punch-Out!!** (opponent_fsm track; maturity M0→M1).

## Identity

| Field | Value |
|-------|-------|
| Status | scaffolded / boot in progress |
| Integration | `PunchOut-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Mike Tyson's Punch-Out!!.zip` |
| Local ROM | `punch_out/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python punch_out/scripts/setup_rom.py
uv run python punch_out/scripts/boot_probe.py
uv run pytest punch_out/tests -q
```

## Next milestone

first bout win (Glass Joe) from Match1.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
