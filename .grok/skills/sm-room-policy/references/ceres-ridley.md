# Ceres Ridley — probe commands

Public policy lives in `combat/ceres_ridley.py` (`CeresRidleyStrategy`) and
https://wiki.supermetroid.run/Ridley#Ceres_Station.
Pin-bench table lives in `docs/plan.md` § Ceres Ridley fight.

```bash
# The Ridley fight runs inside the full-station play (hops report vs TAS).
PYTHONPATH=snes uv run python -m super_metroid.routes.kpdr.ceres.spine station

# Policy unit tests (no emulator): wait / tail_tank / fresh_fifth_jump.
uv run pytest snes/super_metroid/tests/test_ceres_ridley_combat.py -q
```

Product default is `fresh_fifth_jump` (no route flag).
Same-pin fight bench: `routes/kpdr/ceres/data/ceres_ridley_bench.json`.
