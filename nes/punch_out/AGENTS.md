# Punch-Out agent notes

NES Mike Tyson's Punch-Out!!. Gate: `docs/STATUS.md` (M3, isolated Glass Joe).
Future work: `docs/plan.md`. RAM: `docs/ram_map.md`.

## Commands

```bash
uv run python nes/punch_out/scripts/setup_rom.py
uv run python nes/punch_out/scripts/boot_probe.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python nes/punch_out/scripts/run_glass_joe.py --goal win --trials 3 --record
uv run pytest nes/punch_out/tests -q
```

## Traps

- Get-up needs 2-frame A/B presses with a release (`A,A,idle,B,B,idle`), not a single-frame mash.
- Holding LEFT or RIGHT does not dodge. After attack acts 4, 6, 7, 10, 13, 17, 20, or 23, wait about 32 frames, then pulse LEFT or RIGHT for 5 frames. Continuous L/R spam desyncs.
- Glass Joe knockdowns are Vive La France only (`opp_pattern_set == 150`). Do not widen taunt detection.
- `Level1` is pre-bell (about 840 frames to the clock). Use `Match1` for bout work until natural entry exists.
- Checkpoints: `Level1` (ring), `Match1` (clock live), `GlassJoe_Clear`.
- No mid-run RAM writes or state loads.

## Pointers

`docs/STATUS.md` · `docs/plan.md` · `docs/ram_map.md`
