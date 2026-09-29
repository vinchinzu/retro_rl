# RAM Map - Harvest Moon (SNES)

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

## Ending sequence (decomp + ROM-probed)

Trigger. The game ends on the morning of **Year 3, Summer D30** (RAM
`year=2 season=1 day=30`; `$7F1F18/19/1B`). `NightReset` (bank_82 `$8282AC`)
advances the calendar; on waking with that date the ROM queues the ending event
(bank_83 `$83DED3`: `year==2 && season==1 && day==0x1E` -> `CODE_83F4D8`, CC event
`0x2E`, sets FLAG5E bit 2, zeroes the stepper `$7F1F47`, tallies cows/farm
development). A cutscene at Y3 Summer D29 6:00 (`$83E340`, CC event `0x0A`/5)
precedes it. Nothing else on the calendar ends the game: Y2 and Y3 play normally.

Debug shortcut. bank_82 `.endgamecheat` (~L787, end of the sleep routine): joypad
`Joy1_Current & 0x2030 == 0x2030` (Select + L + R) at bedtime sets
`year=2 season=1 day=30` outright. ROM-probed: holding Select+L+R through the
normal sleep task jumps a Y1 D2 pin to Y3 S1 D30 and the ending plays.

Sequence (probed frame counts from the moment the bot's sleep task reports
"morning house ready", pin `LN_ending_Y3S29_eve`; the ROM drives all of it):

| frame | RAM signal | what is on screen |
|---|---|---|
| +0 | `year=2 season=1 day=30`, house tilemap 0x15, `ending_scene_index`=0 | wake in house |
| ~+680 | tilemap 0x3D, then 0x00 farm, `ending_scene_index`=0x14 | farm/dog scene |
| ~+2400 | tilemaps 0x4B..0x52, index 0x15..0x1B, 0x20 | black/eval fades |
| ~+5500 | tilemap 0x36 (mountain), index 0x25, `time_running`=1 | sunset -> dusk mountain |
| ~+6200..+11900 | `dialog_text_id` 1092..1105, 1164, 1108, 1155 (`input_lock` 6 while text prints) | evaluation screens: Money, each girl's love, Happiness rate, ranch rating |
| ~+11900 | `dialog_text_id` 1147, `input_lock` 1 | final line ("...you still need to be trained more" in the probe; wording is branch dependent) |
| terminal | index stays 0x25, tilemap 0x36, static screen | night mountain, empty; **never returns to the title** |

Signals (`harvest/planner/tasks/ending.py`, catalog fields in `ram_catalog.py`):

- `ending_scene_index` = `$7F1F47` (RAM `0x11F47`): the ending stepper. 0 in
  normal play; 0x14..0x25 during the sequence. Dispatch at bank_83 `$83EC11`;
  `0x20` runs `RanchMasteryCalculator`, `0x25` -> `CODE_83F4D7` (a bare `RTS`) =
  the terminal state. **Credits finished = index 0x25 + `input_lock==1` + the
  screen key (index, tilemap, text id, lock) unchanged for 900 frames.**
- `ending_aux_scene_index` = `$7F1F49`: per-scene sub-step, written 0..0x1D by the
  event-script block at `$83E76D` (stayed 0 in the probes).
- The in-game clock is meaningless during the ending (hour cycles 6 / 16-18).
- **Input is not required.** A run with no buttons after the sleep is frame for
  frame identical to one pulsing A (both probed to frame 30000). Do not press;
  just wait ~12k frames after waking on D30. Only the sleep needs the bot.

Bot support: `scene.scene_indicates_ending` (any non-zero stepper),
`EndingTask` (waits for the terminal static screen),
`run_to_day2 --until-ending` (hands the run to `EndingTask` once the date is
Y3 S1 D30 or the planner reports "ending reached"; out JSON `ending` block).

## Year handling

`year` is 0-based (`0x11F18`, `$7F1F18`; Year 1 = 0). `NightReset` increments it
when Winter D30 rolls to Spring D1. `read_world_date` stays `(season, day)`;
use `read_world_ymd` / `format_world_date` (`Y2S1D5`, 1-based year) when a
comparison can cross a year. `run_to_day2 --until-year N` (1-based) plus the
existing `--until-season/--until-day` name an inclusive morning target; without
`--until-year` the target is in the start year.

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
