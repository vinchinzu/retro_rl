# Agent Instructions — punch_out

Scripted NES completion agent for **Mike Tyson's Punch-Out!!** (fighting_game_policies track; maturity M3).

## Identity

| Field | Value |
|-------|-------|
| Status | isolated Glass Joe bout win (M3) |
| Integration | `PunchOut-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Mike Tyson's Punch-Out!!.zip` |
| Local ROM | `punch_out/roms/` (via `scripts/setup_rom.py`) |
| Checkpoints | `Level1` (ring), `Match1` (clock live), `GlassJoe_Clear` |

## Commands

```bash
uv run python nes/punch_out/scripts/setup_rom.py
uv run python nes/punch_out/scripts/boot_probe.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python nes/punch_out/scripts/run_glass_joe.py --goal win --trials 3 --record
uv run pytest nes/punch_out/tests -q
```

## Next milestone

M4 natural-entry: Glass Joe win from power-on / Level1 (not only Match1).

## Traps

- Get-up requires **2-frame** A/B presses with release (`A,A,idle,B,B,idle`), not single-frame mash.
- Holding LEFT/RIGHT does not dodge; use short pulses **timed** after attack act change (~32f wait + 5f hold). Continuous L/R spam desyncs.
- Glass Joe KD is Vive La France only (`opp_pattern_set == 150`); do not widen taunt detection (wastes hearts).
- Level1 is pre-bell (~840f to clock); prefer Match1 for bout work until M4.
- No mid-run RAM writes or state loads (Clean Bronze).

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
