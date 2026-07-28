# Agent Instructions — punch_out

Scripted NES completion agent for **Mike Tyson's Punch-Out!!** (fighting_game_policies track; maturity M2).

## Identity

| Field | Value |
|-------|-------|
| Status | instrumented + first KD (M2) |
| Integration | `PunchOut-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Mike Tyson's Punch-Out!!.zip` |
| Local ROM | `punch_out/roms/` (via `scripts/setup_rom.py`) |
| Checkpoints | `Level1` (ring), `Match1` (clock live) |

## Commands

```bash
uv run python punch_out/scripts/setup_rom.py
uv run python punch_out/scripts/boot_probe.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python punch_out/scripts/run_glass_joe.py --goal knockdown
uv run pytest punch_out/tests -q
```

## Next milestone

Glass Joe full bout win from Match1 (third KD / KO / decision).

## Traps

- Get-up requires **2-frame** A/B presses with release (`A,A,idle,B,B,idle`), not single-frame mash.
- Holding LEFT/RIGHT does not dodge; use short pulses. Best survival so far: 3-frame L / 3 idle / 3-frame R / 3 idle.
- Glass Joe KD1 is the Vive La France taunt (`opp_pattern_set == 150`), not random jab spam.
- Level1 is pre-bell (~840f to clock); prefer Match1 for bout work.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
