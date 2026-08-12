# Harvest Agent Notes

Package `harvest` (disk: `snes/harvest/`; nested import root). Repo-wide rules:
[root AGENTS.md](../../AGENTS.md).

## Commands

```bash
./run_bot.sh play --autoplay --state latest

# Live power-on with bot (window + [ ] speed): title → D1 handoff → multi-day
uv run python -m harvest.runtime.harvest_bot play --autoplay --power-on --end-of-spring
# --no-d1-handoff skips town talks/shed/sleep after power-on

# Boot / power-on (clean diary → Spring D1 07:00 town)
uv run python -m harvest.scripts.boot_probe --state Y1_Inside_House
HEADLESS=1 uv run python -m harvest.scripts.boot_probe --power-on \
  --out recordings/power_on_boot_probe.json

# D1 town recon (docs/town_day1_recon.md)
uv run python -m harvest.scripts.town_day1_recon checklist
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon auto \
  --state Y1_Spring_D1_AnnEve --out recordings/town_day1_rest_auto.json

# Multi-day soak (M3)
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --state Y1_Inside_House --end-of-spring \
  --out recordings/run_spring_month.json \
  --save-end-state Y1_Summer_D1_Morning

# Power-on continuous (rr-5in): D1 handoff auto + multi-day
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on --until-day 2 \
  --out recordings/power_on_d1_handoff_d2.json
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on --end-of-spring \
  --out recordings/power_on_spring_to_summer.json
# --no-d1-handoff disables auto town talks+shed after power-on

# Harvest + ship + post-5pm wallet credit (rr-53g)
HEADLESS=1 uv run python -m harvest.scripts.harvest_ship_money_probe \
  --state Y1_Day09_Harvest_Mode_Start \
  --out recordings/harvest_ship_5pm_money.json

# Gate A multi-day successor: harvest phases + money>$100 (rr-y8n)
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --state Y1_Day09_Harvest_Mode_Start --days 1 \
  --out recordings/run_spring_gate_a_day09.json

# Record task (F5) / tests
uv run python -m harvest.runtime.harvest_bot play --state latest --record <name> --no-day-plan
uv run python -m unittest tests.test_day_plan_sequences tests.test_task_progress -v

# Editor
./kickoff.sh
PYTHONPATH=.. uv run --project .. python -m retro_harness.editor_launcher harvest -- --state latest
```

## Layout

| Path | Role |
|------|------|
| `harvest/core/`, `maps/`, `planner/`, `runtime/`, `tasks/` | Package (import `harvest.*`) |
| `custom_integrations/HarvestMoon-Snes/` | Save states |
| `tasks/*.json` | Human task recordings (not the package) |
| `docs/STATUS.md`, `plan.md`, `PLANNING_STACK.md`, `ram_map.md` | Specs |

Register ROMs only via `harvest.runtime.retro_setup.register_harvest_integration`
(and `backup_mutable_start_state` before recording). Never hand-roll
`Integrations.add_custom_path` in new scripts.

## Traps

- Viewport BFS is ~16×14 tiles; hop targets ≤7 tiles or use `densify_waypoints`.
- Walkable tile IDs come from **recordings**, not static save-state dumps.
- Tasks must not import `day_plan` / orchestrator (circular); shared facts live
  in `ram_catalog` / `tile_catalog` / `map_config`.
- Prefer skill composition (`tasks/skills.py`) over new phase machines.
- Nested import: workspace is `snes/harvest/`; package is `snes/harvest/harvest/`.
  Root `conftest` / `repo.ensure_import_paths` put the workspace on `sys.path`.

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/MILESTONES.md](docs/MILESTONES.md) ·
[docs/plan.md](docs/plan.md) · [docs/CODE_QUALITY_REVIEW.md](docs/CODE_QUALITY_REVIEW.md) ·
[docs/PLANNING_STACK.md](docs/PLANNING_STACK.md) · [docs/town_day1_recon.md](docs/town_day1_recon.md)

## Structure rule (from 2026-08-10 review + structure pass)

Do **not** grow `crop_planter.py` / `home.py` / `cow_task.py` mono FSMs with
new thrash `if`s. Land residuals as extracted modules:

| Concern | Module(s) |
|---------|-----------|
| Pond charges / hop densify / thrash rules | `pond_policy`, `pond_charges`, `pond_hop`, `pond_thrash` (barrel `pond_corridor`) |
| Plot/water pure geometry | `crop_geometry` |
| Crop dual-FSM enums / work modes | `crop_fsm` (`CropState`, `PlotPhase`) |
| Crop hoe/plant arms | `crop_establish` |
| Crop water-step + residual recovery | `crop_water_ops` |
| Crop can-refill / pond access thrash | `crop_refill` |
| Crop multi-phase navigate / stuck | `crop_navigate` |
| House approach zones | `home_approach` |
| Return-home failure policy | `home_recover` |
| Pathfinding | `tasks/nav` (not `farm_clearer`) |
| Tile scan / tool helpers | `tasks/farm_ops` (not `farm_clearer`) |
| Inventory/shed/exit | `inventory_shed` / `inventory_exit` / `inventory_time` |
| Cow stands / care actions | `cow_geometry` / `cow_care` |
| Cow phase mixins / enum | `cow_fsm`, `cow_talk_ops`, `cow_brush_ops`, `cow_milk_ops`, `cow_feed_ops`, `cow_exit_ops` |
| Day-plan sequence tests | `tests/test_day_plan_{crop,home,coop,power_on,common}.py` (+ helpers) |

Prefer skill composition (`tasks/skills.py`) over new phase machines.
Production: `CoopChoresTask` feed_nav + ship_nav far approach use
`coop_nav_to_feed_bin_skill` / `coop_nav_to_shipping_bin_skill` (host navigate).
Gate board: `docs/MILESTONES.md`.
