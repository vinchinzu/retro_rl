# Agent instructions: Donkey Kong Country

Platformer harness with play, autosplit, and replay. Docs: `docs/STATUS.md`,
`docs/plan.md`. Root rules: [root AGENTS.md](../../AGENTS.md).

## Commands

```bash
./run_bot.sh play --autosplit
./run_bot.sh refresh-best
uv run python -m pytest tests/ -q -m rom
```

ROM: `roms/DonkeyKongCountry.sfc`, symlinked to
`custom_integrations/DonkeyKongCountry-Snes/rom.sfc`.

## Traps

- Wayland: `run_bot.sh` defaults `SDL_VIDEODRIVER=x11`.
- Level id is RAM `0x003E`. In-game timer is `0x0046` / `0x0048`.
- Save states live under `custom_integrations/DonkeyKongCountry-Snes/`.
- `optimizer/RUNS.md` is a local log. It is not the STATUS gate. No
  autonomous level clear is verified.

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[README.md](README.md)
