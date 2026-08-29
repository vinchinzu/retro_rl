# Agent Instructions — metroid

NES Metroid (metroidvania; **M5** boot → Maru Mari; first missiles WIP).
Shared: `retro_harness.adventure`, `retro_harness.nes`. Docs: `docs/STATUS.md`,
`docs/plan.md`, `docs/SCREEN_TIMER.md`.

## Commands

```bash
uv run python metroid/scripts/setup_rom.py
uv run python metroid/scripts/boot_probe.py
uv run python metroid/scripts/run_morph_ball.py --natural-entry
uv run python metroid/scripts/run_first_missiles.py --natural-entry
uv run python metroid/scripts/run_first_missiles.py --natural-entry --screen-timing
uv run python metroid/scripts/probe_screen_timer.py self-check
uv run pytest metroid/tests retro_harness/adventure/tests -q
```

## Layout

`ram.py` · `brinstar.py` · `morph_ball.py` · `first_missiles.py` ·
`screen_timer.py` / `screen_timing_session.py` · `routes.py`.

## Traps

- Morph is **left** of start (map 3,14 → 1,14), not right.
- Equipment in cart WRAM `$6878` via `env.data.memory.extract` — **not**
  `get_ram()` (2 KiB system only). Missiles capacity `$687A`.
- Do not mash START after play begins (pauses).
- West corridor: LEFT+A climb spans; floor-only LEFT sticks at x≈4.
- After morph: low ball tunnel → unmorph ~(3,14) x≈76/y=200 before skree room.
- AfterMorph often in item fanfare (`game_mode == 9`); idle until mode 3.
- East doors elevated — stop on first controllable target-map frame.

## Next

Morph → first missiles (`$687A` > 0) natural-entry from leftover
(11,12) west-shaft hold through bridge + east shaft. Do not reopen
morph/doors/corridor or the (11,13)→(11,12) climb.
