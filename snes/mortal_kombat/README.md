# Mortal Kombat

Verified facts: [docs/STATUS.md](docs/STATUS.md). Future work:
[SPEEDRUN_PLAN.md](SPEEDRUN_PLAN.md). Commands and traps:
[AGENTS.md](AGENTS.md). RAM: [docs/ram_map.md](docs/ram_map.md).

This file is the manifest status note. It does not add a result beyond
`docs/STATUS.md`. A checkpoint zip is not an arcade clear.

```bash
uv run python snes/mortal_kombat/scripts/setup_rom.py
uv run python snes/mortal_kombat/scripts/boot_probe.py
uv run --extra ml python snes/mortal_kombat/scripts/eval_roster.py --attempts 5
uv run python snes/mortal_kombat/scripts/replay_natural_fight7.py --repeat 5
```
