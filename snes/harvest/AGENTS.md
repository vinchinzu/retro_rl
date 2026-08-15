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

# First mountain berry from Spring D2 house (reactive path segments)
HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \
  --state Y1_Inside_House --screenshot recordings/mountain_grape_stand.png
# Ground-grape pick + Don't eat (rr-14xx)
HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \
  --state Y1_Inside_House --pick --screenshot recordings/mountain_grape_kept.png
# Full Spring D2 pick/keep/return/bin route (rr-nn3x)
HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \
  --state Y1_Inside_House --ship --out recordings/mountain_grape_ship.json
# Hour-by-hour RAM clock + 12:00 lunch stand (frames vs BERRY_SHIP_BENCH)
HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \
  --state Y1_Inside_House --ship --until-lunch \
  --out recordings/mountain_segments_clock.json

# Classify a pick/talk from an existing tape or a live pin (do not re-record)
uv run python -m harvest.scripts.interact_scan tape mountain_grape_stand
uv run python -m harvest.scripts.interact_scan search grape
# Seed shop: nav shop_door + RAM buy (not a day tape)
HEADLESS=1 uv run python -m harvest.scripts.buy_seeds_probe \
  --state Y1_Inside_House --out recordings/buy_seeds_d2_probe.json
# West-pocket weeds/stones (inside y=31 fence; lift only, no plant tape)
HEADLESS=1 uv run python -m harvest.scripts.pocket_clear_probe \
  --state Y1_Inside_House --out recordings/pocket_clear_probe.json

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
- Travel BFS must not route onto WEED `0x03` (ROM-walkable, pins movement) or
  a tile the farmer is **pushing**. Clear from a neighbor stand. D2 sections
  are `rr-20w.2.*` — do not soak the whole 3–4h day as one ticket.
- Interact: scan an existing tape / UnlinkedText before recording. Face-walk
  is movement. Item box with held forage is Eat/Don't eat, not Gotz. See
  [docs/INTERACT.md](docs/INTERACT.md).
- Tasks must not import `day_plan` / orchestrator (circular); shared facts live
  in `ram_catalog` / `tile_catalog` / `map_config`.
- Prefer skill composition (`tasks/skills.py`) over new phase machines.
- Nested import: workspace is `snes/harvest/`; package is `snes/harvest/harvest/`.
  Root `conftest` / `repo.ensure_import_paths` put the workspace on `sys.path`.

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[docs/INTERACT.md](docs/INTERACT.md) ·
[docs/PLANNING_STACK.md](docs/PLANNING_STACK.md) · [docs/town_day1_recon.md](docs/town_day1_recon.md)

Skills (repo `.grok/skills/`): `harvest-interact` · `harvest-route` · `harvest-shop`

## Structure rule (1k LOC + no mono thrash)

Soft max **~1000 LOC / file** (repo Working Norms). Do **not** grow monofiles
with residual thrash `if`s — extract a module or data rule first.

| Concern | Module(s) |
|---------|-----------|
| MultNav | `multi_nav` (not `navigation.py`) |
| Pond / crop thrash | `pond_*`, `crop_{establish,water_ops,refill*,navigate,detect,act_verify,step}` |
| Home | `home_return`, `home_sleep`, `home_approach`, `home_recover` |
| Coop / cow | `coop_{layout,feed_ops,egg_ops}`, `cow_*` |
| Maps / routes | `map_config` facade + `map_types` / `farm_pond` / `map_routes` |
| Day plan | `day_plan_orchestrator`, `multi_day_planner`, `day_phase_{catalog,berry,chicken,cow}` |
| D1 / ROM / editor | `town_day1_*`, `rom_*` / `save_state_io` / `map_render`, `editor_*` |

Prefer skill composition (`tasks/skills.py`) over new phase machines.
Gate board: `docs/plan.md` · `bd ready -l harvest`.
