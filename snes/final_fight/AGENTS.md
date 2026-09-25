# Agent instructions: final_fight

SNES Final Fight. Shared helpers: `retro_harness.combat` and
`segment_runner`. Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/ram_map.md`.

## Commands

```bash
uv run python -m retro_harness.setup_all_roms final_fight
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/boot_probe.py
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/ram_probe.py
uv run python final_fight/scripts/stage3_advance.py
uv run python final_fight/scripts/stage3_area1_probe.py
uv run --frozen pytest final_fight/tests -q
```

Save states live under `custom_integrations/FinalFight-Snes/`.

## Traps

- Up increases player Y.
- `CLEAR_AREA` (`game_status` `0x08`) is a RAM write. It is not a Clean clear
  and it is not natural entry. Damnd and Sodom underflow leave `0x0CD2` at 0,
  so idling does not finish the round.
- `--heal-hp` and `--force-enemy-hp` are dev-only. They do not count as the
  Area 1 kill or as Boss 3.
- Area 1: continuous `LEFT+Y` deals no damage. Do not hold LEFT while
  punching (left gutter). Plant the HP 0 ghost before scrolling.
- Prefer resume states named in `docs/STATUS.md`. Do not clobber a preferred
  mid with a lower-HP rerun.
- Headless probes need `SDL_VIDEODRIVER=dummy`.
