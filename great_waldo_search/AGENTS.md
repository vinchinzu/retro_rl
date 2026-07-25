# Agent Instructions — great_waldo_search

SNES The Great Waldo Search scripted-completion workspace (rank 1 / tier 0). Shared
helpers: `snes_oneshot/`. Program notes:
`snes_oneshot/docs/GAME_SELECTION_NOTES.md`.

## Norms

- Prefer development save states and scene-segment scripts over uninterrupted
  title-to-credits runs.
- Store `.state` files under `custom_integrations/GreatWaldoSearch-Snes/`.
- Keep RAM maps and cursor policies here; elevate only reusable helpers to
  `snes_oneshot/`.
- Headless probes: `SDL_VIDEODRIVER=dummy` (and audio dummy as needed).
- Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/ram_map.md`.

## Immediate goal

**Continuous power-on → five-scrolls ending done** — one emulator session,
no mid-run state loads. Artifact:
`recordings/great_waldo_search_full_credits.mp4`. Re-record with
`scripts/record_full_run.py`.

## Clear recipes

```bash
# Scene1: scroll (32,100) + RIGHT+Y×80 + Waldo (36,28)
SDL_VIDEODRIVER=dummy uv run python \
  great_waldo_search/scripts/clear_scene1.py

# Scene2 cave: scroll (224,100) + P2-A×500 + Waldo (32,120)
SDL_VIDEODRIVER=dummy uv run python \
  great_waldo_search/scripts/clear_scene2.py

# Scene3 monks: P2-A×300 + (160,100) then P2-A×200 + (198,100)
SDL_VIDEODRIVER=dummy uv run python \
  great_waldo_search/scripts/clear_scene3.py

# Scene4 giants: P2-A×500 + (34,100) then P2-A×500 + (196,140)
SDL_VIDEODRIVER=dummy uv run python \
  great_waldo_search/scripts/clear_scene4.py

# Scene5 Land of Waldos (final search): P2-A×300 + (32,100)
# then P2-A×500 + (180,60) → five-scrolls ending
SDL_VIDEODRIVER=dummy uv run python \
  great_waldo_search/scripts/clear_scene5.py
```

Scene2/4/5: do not replace post-scroll P2-A with manual LEFT/RIGHT pan alone.
Scene5 Waldo needs a longer settle (≥200f warm) than Scenes 1–4.

## Commands

```bash
uv run python -m snes_oneshot.setup_all_roms great_waldo_search

SDL_VIDEODRIVER=dummy uv run python great_waldo_search/scripts/boot_probe.py
SDL_VIDEODRIVER=dummy uv run python great_waldo_search/scripts/ram_probe.py

# Continuous power-on → five-scrolls ending (dry-run, then encode)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run python \
  great_waldo_search/scripts/record_full_run.py --dry-run
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run python \
  great_waldo_search/scripts/record_full_run.py

uv run --frozen pytest great_waldo_search/tests snes_oneshot/tests/test_cursor.py -q
```
