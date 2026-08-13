# Ceres Ridley — probe commands

Public policy lives in `combat/ceres_ridley.py` and
https://wiki.supermetroid.run/Ridley#Ceres_Station.
Pin-bench table lives in `docs/plan.md`.

```bash
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py capture
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py dump --frames 400
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py strategy --policy wait
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py strategy --policy tail_tank
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py bench
```

Enter pin: `custom_integrations/SuperMetroid-Snes/scratch/ceres_ridley_enter.state`.
Bench JSON: `scratch/ceres_ridley_bench.json`.
