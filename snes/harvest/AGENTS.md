# Harvest Agent Notes

Package `harvest` (disk: `snes/harvest/`; nested import root). Repo-wide rules:
[root AGENTS.md](../../AGENTS.md). Session loop:
`.grok/skills/harvest-session/SKILL.md`. Tracker:
`bd ready -l harvest -l spine`.

## Immediate goal

`rr-20w.2.3` D2 whole-farm clear (P0) + living residual
[`docs/tasks/rr-20w.2.3-residual.md`](docs/tasks/rr-20w.2.3-residual.md).
Water-refill `rr-3ae8` is also on the spine filter — claim **one**. Do not
promote [STATUS.md](docs/STATUS.md) from a fixture or pin.

## Commands

```bash
bd ready -l harvest -l spine

HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on \
  --stop-after-d2-shipping --save-end-state Y1_D2_PostShipper_WorkStart \
  --out recordings/power_on_d2_spine_clear_final.json

HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \
  --state Y1_Inside_House --ship --out recordings/mountain_grape_ship.json

uv run python -m harvest.scripts.interact_scan tape mountain_grape_stand
uv run python -m harvest.scripts.interact_scan search grape

HEADLESS=1 uv run python -m harvest.scripts.buy_seeds_probe \
  --state Y1_Inside_House --out recordings/buy_seeds_d2_probe.json

HEADLESS=1 uv run python -m harvest.scripts.pocket_clear_probe \
  --state Y1_After_Buy_Potato --out recordings/pocket_clear_probe.json

uv run python -m harvest.scripts.d2_leftover_probe --headed --section fences \
  --state Y1_D2_After_Bushes
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --section stones --chunk sw --state Y1_D2_After_Stones \
  --out recordings/d2_leftover_stones_sw.json
```

`HEADLESS=1`; no MP4. Glance is `harvest.clock_glance`. Parked CLIs:
[docs/plan.md](docs/plan.md) § CLI catalog. Natural entry is power-on.
Do not start D2 from `Y1_D2_Morning_After_D1`.

## Layout

| Path | Role |
|------|------|
| `harvest/core/`, `maps/`, `planner/`, `runtime/`, `tasks/` | Package (`harvest.*`) |
| `custom_integrations/HarvestMoon-Snes/` | Save states |
| `docs/STATUS.md`, `plan.md`, `FARM_CLEAR_D2.md`, `INTERACT.md` | Specs |

Register ROMs only via `harvest.runtime.retro_setup.register_harvest_integration`.
Nested import: workspace is `snes/harvest/`; package is `harvest.*` (disk
`snes/harvest/harvest/`). Split a file **before 500 lines**; refuse a new
knob on a file **≥800**. Extract before 1k; module map in plan.md.

## Traps

- Viewport BFS is ~16×14 tiles; hop targets ≤7 tiles or `densify_waypoints`.
- WEED `0x03` is not travel-walkable. Never BFS onto debris/push. Clear from
  a neighbor stand. D2 sections are `rr-20w.2.*`.
- Interact: scan an existing tape / UnlinkedText before recording.
- 5pm farm ShippingScene: pulse A (press/release). Do not hold A.
- Do not start D2 from `Y1_D2_Morning_After_D1` — grape return-to-bin seals
  at the house fence (rr-oqri).

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[docs/FARM_CLEAR_D2.md](docs/FARM_CLEAR_D2.md) ·
[docs/INTERACT.md](docs/INTERACT.md)

Skills: `harvest-session` · `harvest-route` · `harvest-interact` · `harvest-shop`
