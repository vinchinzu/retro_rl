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
| dialog_choice_result | `0x018F` | u8 | Choice row; a Down edge toggles a two-row menu |
| dialog flags | `0x019B` | u8 | `0x80` text ended / menu opening, `0x10` menu live; an A between them answers row 0 (peddler trace) |
| fade_current_brightness | `0x0025` | u8 | `ScreenFadein` steps it to 15; the pad ignores input until then (~70 frames after a coop exit lands) |

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

## Year-conditional events (non-festival, non-weather)

Scouted from the decomp (`bank_83` morning/date dispatcher `$83CD..$83E4CA`,
`bank_81`) with a ROM event-script decoder (event = `UNK_LoadCCDataLong`
group X / event Y, scripts in bank B3). All 30 `!year` reads were classified;
flag bits are from `HM-Decomp/TODO.txt` (`$7F1F64` upgrades, `$7F1F66` wife,
`$7F1F68`/`$7F1F6A` events). The decomp disassembly is partly misaligned, so
conditions are hand-checked and the "bot-reachable" rows were ROM-probed.
Probes are what-if pins (evening before, date edited in the save state), 2 days
with the normal planner on main `d213f635` (SceneGuard present).

| event (group/id) | trigger | what it does | needs input / blocks | bot-reachable | ROM result |
|---|---|---|---|---|---|
| Y2 Summer D1 6:00 (`$83E340`, 0A/4) | year=1 season=1 day=1, hour 6:00:00, wherever the farmer is (bed) | time-stopped text box 0x0479 | text auto-clears or SceneGuard pulses A | yes | day ends in bed; guard logged text 0x0479 "gave up" but planner continued (Summer D1 is also the festival day, calendar lane) |
| **Y3 Summer D29 6:00 (`$83E340`, 0A/5)** | year=2 season=1 day=29, 6:00:00 | time-stopped text box 0x047A (1146), then D30 = ending | one text box | yes, mandatory | passed: guard handled text 0x047A in the house, day slept, D30 reached (`ev_Y3S1D29`, 13997f) |
| Y3 Summer D30 (`$83DED3`) | year=2 season=1 day=30 | ending, see "Ending sequence" | none (wait) | yes | see above |
| Y2+ money event (`$83D0FF`, 13/0) | year>0, 6:00:00, `$0196&0x1A==0` (clear weather), money >= 3000 raw (30,000G), flag68 bit 0x0400 unset | visitor NPC on the farm road, text 333-335 (`$014D`): the peddler's money-tree seed. Row 0 "Buy it" adds -2000 raw = **20,000G** and sets flag68 0x0800; row 1 declines. Fires once (flag68 0x0400 set at load). The bot holds the pad from `$019B` 0x80 to 0x10, then picks row 1 (`peddler_decline_action`, pin `LN_y23_whatif_peddler_y2w1`); y23_final lost 20,000G on Y3 Spring D30 before that | dialogue 0x014D/0x014E pulsed by guard; the NPC sprite blocks the farm road (nav "yield to sprite", pixel-stuck ~10k frames) | only if money >= 3000G (bot reached 7060G in Y1) | fired Y2 Spring D5: -2000G, MOUNTAIN_BERRY timed out (12001f), return_home recovered, day ended in bed (26k frames vs ~14k) |
| Y2+ Fall/Winter visitor (`$83DCDF`, 21/0) | year>0, season>=2, day 2-7/13-17/25-28, clear weather, flag6A bit 0x0200 unset, weekday Mon-Fri | 2-op script chaining a nested CC scene (group 0x21, visitor-style) | none seen | maybe | probed Y2 Fall D3 (weekday Mon): no dialogue appeared, day ended in bed (`ev_Y2Fall`, 13648f); whether the event fired silently is unproven |
| Y2+ Saturday event (`$83D163`, 16/0) | (year>0 or Winter), Saturday 6:00:00, clear weather, house upgraded (flag64 & 0xC0), flag68 bit 0x4000 unset | yes/no question (texts 801-803), happiness +/-20, sets flag68 0x8000, flag6A 0x2 | a choice prompt | no: bot never upgrades the house | ruled out on paper |
| Post-collapse wake (`$83E407`, 0D/0, 0D/1) | wake after passing out (flag5C bit 7); in Y2+ (or Y1 Winter) 0D/0 once then 0D/1 | doctor scene, texts 1086/374 | text | only if the farmer collapses at 2:00 | not probed (bot sleeps before then) |
| Cow year event (`bank_81 $81E80E`, 00/7C) | year>0, flag66 bit 0x8000 unset, touching an adult cow | cow scene | scripted | no: bot owns no cows | ruled out |
| Marriage / kid / girl-heart events (`$83DABF..$83DD95`, groups 1-0x1D, 0x44/0x45) | flags 66/6A/6C (wife bits, hearts >= 200, kid ages) | wife/child cutscenes | text | no: never married, no kids, hearts low | ruled out |
| Sprinkler shop unlock (`bank_81 $81AFxx`) | year>=1 or Summer D20+ | shop stock only | none | n/a | no event |
| Year-1-only (`$83CFA4`, `$83DDB9`, bank_81 `$81F14E`, thunder/snowstorm in NightReset) | year==0 | tutorials, thunder, special weather | | n/a | absent after Y1 |

Probe status: the 2000G-event run was repeated with SceneGuard disabled locally
(`quiet_frames=10**9`, uncommitted) and was frame-identical (26192f), so the guard
did not change that day; the sprite blocking the road, not a dialogue, cost the
frames. `--until-ending` from a Y3 Summer D24 evening what-if pin ran the planner
through D25-D29 and the ending (77783f, `ev_D25`, "ending credits complete").

Risk ranking for a farm/ship/sleep/festival bot: (1) the Y2+ money event, the
only one with a physical blocker and a 2000G cost (needs 3000G banked);
(2) Y3 Summer D29 and Y2 Summer D1 text boxes (harmless, guard-handled);
(3) Y2+ Fall/Winter visitor; (4) post-collapse wake; everything else needs a
wife, kids, cows or a house upgrade.

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
