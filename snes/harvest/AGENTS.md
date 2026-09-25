# Harvest agent notes

Package `harvest` (disk: `snes/harvest/`; nested import root). Repo rules:
[root AGENTS.md](../../AGENTS.md). Session loop:
`.grok/skills/harvest-session/SKILL.md`. Tracker:
`bd ready -l harvest -l spine`. One living residual:
[docs/tasks/rr-20w-run16-defects.md](docs/tasks/rr-20w-run16-defects.md).

## Commands

```bash
bd ready -l harvest -l spine

# Farm-clear and pocket: live planner until D2FarmStatus.is_complete.
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on \
  --stop-after-d2-clear --save-end-state Y1_D2_PowerOn_FarmClear \
  --out recordings/power_on_d2_farm_clear.json

HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --state Y1_After_Buy_Potato --stop-after-d2-clear \
  --out recordings/d2_pocket_farm_clear.json

# Grape, then shop (BUY_SEEDS), stop after the D2 5pm ship.
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --state Y1_Inside_House --stop-after-d2-shipping \
  --out recordings/d2_grape_shop.json

# Grape only (named mountain_berry sequence; ship is on).
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --day-plan mountain_berry --state Y1_Inside_House \
  --out recordings/mountain_grape_ship.json

uv run python -m harvest.scripts.interact_scan tape mountain_grape_stand
uv run python -m harvest.scripts.interact_scan search grape
```

`HEADLESS=1`. No MP4. Glance is `harvest.clock_glance`. Natural entry is
power-on. Nested import: workspace `snes/harvest/`, package `harvest.*`.

## Traps

- Viewport BFS is about 16 by 14 tiles. Keep hop targets to 7 tiles, or call
  `densify_waypoints`.
- WEED `0x03` is not travel-walkable. Do not BFS onto debris or push tiles.
  Clear from a neighbor stand.
- Do not record a walk a search can already close. Scan an existing tape
  before a new interact recording.
- A shop menu, or a CrossMap return to the origin map, is not a completed
  buy. Require the shop tilemap and a wallet or stock change.
- 5pm farm ShippingScene: pulse A. Do not hold A.
- Do not start D2 from `Y1_D2_Morning_After_D1`. Grape return-to-bin
  seals at the house fence (rr-oqri).
- The clock does not advance indoors. A stuck `[RUN]` time means tilemap `0x15` or `0x26`, not a hang.
- Do not write a pin or fixture result into `docs/STATUS.md`.
  `Y1_D3_Morning` spring runs stay in the living residual.

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[docs/FARM_CLEAR_D2.md](docs/FARM_CLEAR_D2.md) ·
[docs/INTERACT.md](docs/INTERACT.md).
Skills: `harvest-session`, `harvest-route`, `harvest-interact`, `harvest-shop`.
