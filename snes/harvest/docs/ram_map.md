# RAM Map — Harvest Moon (SNES)

Authoritative field metadata: `harvest/core/ram_catalog.py` (`SCALAR_FIELDS`).
Save-state RAM is direct WRAM; live `env.get_ram()` may be offset by `+0x4000`
(see `LIVE_RAM_WRAM_OFFSET`).

## Calendar & weather

| Field | Addr | Kind | Notes |
|-------|------|------|-------|
| year | `0x11F18` | u8 | 0-based display year |
| season | `0x11F19` | u8 | 0 spring … 3 winter |
| weekday | `0x11F1A` | u8 | 0–6; Sunday special-cased in planner |
| day | `0x11F1B` | u8 | 1–30 |
| hour | `0x11F1C` | u8 | 0–23 |
| minute | `0x11F1D` | u8 | |
| weather / weather_tomorrow | `0x098C` | u8 | Event codes include festivals |
| weather flags | `0x0196` | u16 | Rain bits used by `is_rainy` |

## Player / scene

| Field | Addr | Kind | Notes |
|-------|------|------|-------|
| tilemap | `0x0022` | u8 | live_offset=False; farm `0x00–0x03`, house `0x15–0x17`, path `0x0C`, town `0x04`, sleep `0x0F` |
| player_x / player_y | `0x00D6` / `0x00D8` | u16 | Pixel coords |
| player_state | `0x00D2` | u8 | Carry / transition bits |
| player_action | `0x00D4` | u8 | Live: 0 idle/walk/run/**push** (no distinct push code), 3 jump/water, 4 carry, 9 dialogue |
| player_direction | `0x00DA` | u8 | Facing: 0 down, 1 up, 2 right, 3 left |
| input_lock | `0x019A` | u8 | 1 = free; dismiss when not 1 |
| stamina | `0x0918` | u8 | Current. Script object: `Stamina.from_ram(ram)` / `WorldSnapshot.player.stamina` |
| max_stamina | `0x0917` | u8 | Spa restores current to this (often 100–150) |
| exhaustion_level | `0x096C` | u8 | Decomp `!exaustion_level` |
| tool_hit_counter | `0x096D` | u8 | Hammer/axe 2×2 hits; breaks at 6 then STZ |
| dialog_text_id | `0x0183` | u16 | Dialogue / shop menus |
| dialog_menu_cursor | `0x018A` | u8 | |

## Inventory / farm

| Field | Addr | Kind | Notes |
|-------|------|------|-------|
| tool_selected | `0x0921` | u8 | |
| tool_backpack | `0x0923` | u8 | |
| held_item | `0x091D` | u8 | Debris / egg / chicken carry |
| potato_seeds etc. | seed fields in catalog | u8 | Seasonal planting |
| money | money lo/hi in catalog | | Display multiplier on money |

## Animals

Chicken/cow slot tables and feed counts live in `harvest/core/animal_status.py`
and matching catalog entries (`num_chickens`, `num_cows`, feed flags, incubator).

## Scene modes (derived)

`harvest/core/scene.py` classifies tilemap + locks into NORMAL, DIALOGUE, MENU,
MAP_TRANSITION, SLEEP_WAKE_TRANSITION, ENDING_CREDITS, etc. Prefer the
classifier over raw tilemap checks in new tasks.

## Discovery

```bash
uv run python -m harvest.runtime.harvest_bot ram-fields
uv run python -m harvest.runtime.harvest_bot world --state Y1_Inside_House --compact
```

Diff save states before/after an action; promote stable fields into
`ram_catalog.py` with source `decomp` / `retro` / `state`.
