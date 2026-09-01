# Agent Instructions — Donkey Kong Country

Platformer harness with play/autosplit/replay. Docs: `docs/STATUS.md`,
`docs/plan.md`. Root rules: [../../AGENTS.md](../../AGENTS.md).

## Commands

```bash
# Play with autosplit
./run_bot.sh play --autosplit

# Refresh best times from split log
./run_bot.sh refresh-best

uv run python -m pytest tests/ -q -m rom
```

ROM: `roms/DonkeyKongCountry.sfc` → symlink into
`custom_integrations/DonkeyKongCountry-Snes/rom.sfc`.

## Traps

- Wayland: `run_bot.sh` defaults `SDL_VIDEODRIVER=x11`.
- Level ID is RAM `0x003E`; in-game timer `0x0046`/`0x0048`.
- Save states live under `custom_integrations/DonkeyKongCountry-Snes/`.
