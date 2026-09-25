# Agent instructions: great_waldo_search

SNES The Great Waldo Search. Shared cursor math: `retro_harness.cursor`.
Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/ram_map.md`.

## Commands

```bash
uv run python -m retro_harness.setup_all_roms great_waldo_search

SDL_VIDEODRIVER=dummy uv run python great_waldo_search/scripts/boot_probe.py
SDL_VIDEODRIVER=dummy uv run python great_waldo_search/scripts/ram_probe.py

SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python great_waldo_search/scripts/record_full_run.py --dry-run
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python great_waldo_search/scripts/record_full_run.py

uv run --frozen pytest great_waldo_search/tests retro_harness/tests/test_cursor.py -q
```

Per-scene scripts are `clear_scene1.py` through `clear_scene5.py`. Save states
live under `custom_integrations/GreatWaldoSearch-Snes/`.

## Traps

- Runs use `players=2`. P2-A is the seek. P1-A is the click.
- After the scroll click, Scenes 2, 4, and 5 need a long P2-A hold. Manual
  LEFT or RIGHT pan does not open the Waldo window.
- Scene 5 Waldo needs at least 200 frames of settle. Score RAM lies
  mid-animation.
- Cleared `.state` rebuild timings are not the continuous `pre_idle` path.
  Loading a state mutates layout RNG.
- Do not mash A on the five-scrolls screen. That leaves the ending.
