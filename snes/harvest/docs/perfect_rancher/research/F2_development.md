# F2 development coverage

Date: 2026-09-29. Bead PR-011. Research only. No route pin was written. No emulator was started. Census is the persistent farm array `$7EA4E6` (64×64) read from save-state WRAM, plus the USA ROM tables below.

ROM: `roms/Harvest Moon.smc`, sha1 prefix `a64a5634429a`. Reader: `HarvestMoonRom`. States under `snes/harvest/custom_integrations/HarvestMoon-Snes/`.

Command, from the repo root, `PYTHONPATH=snes/harvest`, `QT_QPA_PLATFORM=offscreen`, `uv run python` inline. Frame count: 0.

## Verdict

On `Y1_D2_PowerOn_FarmClear` the persistent map has **2,968** tiles below `0xA0`. Every one of them can be spring/summer grass `0x70`, which `CalculateFarmDevelopment` counts and which the farmer can walk. Uncovered tiles in that layout: **0**. Slack above 2,950 is 18. `projected_farm_development_percent(2968)` is 100, so the score term is 50.

`Y1_Fall_D1_Pin` and `Y1_Winter_D1_Pin` each have **2,974** tiles below `0xA0`. The same cover rule plants all 2,974. That is the plan's 2,974. It is the fall/winter mask, not the D2 clear mask.

Paths from the house stand reach the bin lip, the barn, the coop, the shed approach, and the west exit. Distances are in the layout draft.

A hurricane on the sleep into Summer D30 can still destroy counted grass after the last morning the farmer can replant. The geometry hits 2,950. That night is a separate durability constraint, already named in the plan's phase H.

## What the ROM counts

`CalculateFarmDevelopment` (`bank_82.asm` `$82AA0C`, lines 2603–2632) reads `!farm_map_array`. A tile counts when its id is below `0xA0` and the id is `0x05`, `0x06`, or `≥ 0x1D`. The counter is `!ranch_development` (`$7F1F56`). The mastery term is `min(100, tiles×10/295)` then `>> 1`, divisor `0x0127`. Python `is_farm_development_tile` matches this test.

Counted examples: fence `0x05`, broken fence `0x06`, grass seed `0x1D`, grass `0x70`–`0x7C`, crops `0x1E`–`0x6F`. Untilled `0x01`, dried `0x02`, weed `0x03`, stone `0x04`, tilled `0x07`/`0x08`, and every id `≥ 0xA0` do not count.

Daytime tool writes go through `CODE_81A688` → `SUB_82B060` into `!current_map_array` (`bank_82.asm` lines 3561–3600). `CopyCurrentMaptoFarmMap` (`$82A682`) runs on a farm map leave (`bank_82.asm` line 4919, `tilemap < 4`; also `bank_80.asm` line 3622). The night pass and the ending scan read the persistent array.

## Census

| State | Tiles `< 0xA0` | Already counted |
| --- | ---: | ---: |
| `Y1_D2_PowerOn_FarmClear` | 2968 | 8 |
| `Y1_D2_After_Stumps` | 2968 | 8 |
| `Y1_D3_Morning` | 2968 | 8 |
| `Y1_Summer_D1_Pin` | 2968 | 17 |
| `Y1_Fall_D1_Pin` | 2974 | 14 |
| `Y1_Winter_D1_Pin` | 2974 | 17 |
| `LN_calendar_Y1_Winter_D1` | 2974 | 5 |

`LN_oracle_whatif_farm_covered` is a RAM fill of every `< 0xA0` cell with `0x79`. It reports 2,974 and development byte 100. It is not a cleared-farm census and is not route evidence.

FarmClear composition of the 2,968: `0x01` × 1809, `0x02` × 1135, stump quad ids `0x09`–`0x0C` × 16, and `0x55` × 8. No `0x03`, `0x04`, `0x05`, or `0x06` on that pin. The 8 counted tiles are the `0x55` cells.

The six cells that are `0xA0` on FarmClear and `0x02` on the fall and winter pins:

`(36,45) (37,45) (38,45) (36,46) (37,46) (38,46)`

Fall and winter agree with each other (symmetric difference 0). Summer D1 agrees with FarmClear. `0xA0` fails the seed check (`CheckToolSuccess` rejects `≥ 0xA0`), so those six are outside the clear-pin pool. On the fall pin they are hoeable `0x02`.

## Which tiles can be grass, fence, or crop

Farm property table for maps 0–3 is `UNK_PointersTable` → `DATA16_82B3B4` (`bank_82.asm` lines 3771–3774). Tool bits are byte 1 of the 4-byte record (`UNK_CheckTileProperty` with A=`1`, then the mask in `CheckToolSuccess`):

| Bit | Tool |
| --- | --- |
| `0x01` | sickle |
| `0x02` | hoe |
| `0x04` | hammer |
| `0x08` | axe |
| `0x10` | water / sprinkler |
| `0x20` | seed (crop and grass) |

Seed bit `0x20` is set on `0x07` and `0x08` only, among ids below `0xA0`. Hoe bit `0x02` is set on `0x01`, `0x02`, `0x07`, and `0x08`. Id `0x00` fails the tool check (`CPX #$0000`).

`CODE_81A688` indexes `DATA8_81B363` (`$81B363`) by 6 bytes. `SUB_81A801` (`bank_81.asm` lines 3750–3781) stores byte 2 as height, byte 3 as width, and the last word as the value `SUB_82B060` writes. Measured records:

| Index | Who calls it | Write |
| --- | --- | --- |
| `$17` | hoe (`ToolUsedHoe` line 165); sickle on a farm tile that is not weed `0x03` and not grass `0x79` | 1×1 tile `0x07` |
| `$0E` | sickle on weed; hammer on farm stone `0x04` and on `0x06` | 1×1 tile `0x02` |
| `$0D` | axe / hammer 2×2 break after 6 hits (`$096D`) | 2×2 tile `0x02` |
| `$1E` | grass seed, season `< 2` | 1×1 tile `0x70`, and `INC !planted_grass` |
| `$ED` | grass seed, season `≥ 2` | 1×1 tile `0x1D`, no `planted_grass` increment |
| `$19` / `$1A` / `$1B` / `$1C` | corn / tomato / potato / turnip in season | 1×1 tiles `0x3A` / `0x26` / `0x54` / `0x66` |
| `$EB` | those crops out of season | 1×1 tile `0x1E` |
| `$52` | sickle on grass `0x79` | 1×1 tile `0x7A` |
| `$14` / `$15` | records exist | 1×1 tiles `0x05` / `0x06` |

Grass on this farm: hoe `0x01` or `0x02` to `0x07`, then the grass bag writes `0x70` in spring or summer. Weed, stone, `0x06`, stumps, and large rocks break to `0x02`, then the same hoe and seed. `0x70` is `≥ 0x1D` and below `0xA0`, so it counts.

`0x55` on the clear pin has tool byte `0x00`. Hoe, seed, sickle, hammer, and axe all fail. It already counts. `MonthlyFarmTilesCheck` turns `0x1D`–`0x6F` into `0x07` on Winter D1, and `0x07` takes the seed. The layout plants those eight after that winter.

Crops count, and the in-season plant writes a `≥ 0x1D` id. Many later growth ids are in `CROP_SOLID_TILES` (`tile_catalog.py`), so a crop path closes. Fall decrements odd crop ids. Winter D1 turns the whole `0x1D`–`0x6F` band into `0x07`. Crops are a counted cover with a season limit. The layout uses grass.

Fence: `0x05` and `0x06` count. `GetFenceFromStack` (`bank_81.asm` `$81D644`) calls `AddWood` with `#$FFFE` (−2 lumber) and, on success, stores item `0x57` and `player_action` 4. `DATA8_81A2AD[0x57]` is 0, so `SUB_818000` skips `CODE_81A688` for that item. The clear pin has no `0x05` to keep. The layout does not place new fences. Records `$14` and `$15` would write `0x05` and `0x06`; the lumber item's drop path does not call them.

## Walk

`FARM_WALKABLE` and `TRAVEL_SOLID_TILES` in `tile_catalog.py` are the 2026-09-28 ROM walk measurement (id written at `(19,29)`, UP held from `(19,30)`).

| Counted id | Walk |
| --- | --- |
| `0x05`, `0x06` | no (`TRAVEL_SOLID_TILES`) |
| `0x1D`, `0x70`–`0x7C`, `0x80`–`0x85` | yes |
| crops outside `CROP_SOLID_TILES` | yes |
| crops in `CROP_SOLID_TILES` | no |

`0x70` and the sickle result `0x7A` are both in the walkable grass span. A grass cover keeps the path. A fence cover would count and would block travel.

Structure ids the farmer can use as path, and that are not part of the 2,968: `0xA0`–`0xA3`, `0xA8`, `0xC5`. `FARM_NO_GO_TILES` still blocks the well, the shed body, the barn faces, and the ditch fringe even when the id is in that set. The bin lip `(9,28)` is in that fringe and is `0xA8`; F8 stood on it (`F8_shipping.md`, boot tile `(9,28)`). The layout allows that cell.

## Cost per tile

Grass, one cell that is already `0x07` or `0x08`:

- Pattern: `PreCheckToolSuccessSquare` (`bank_82.asm` `$8292D6`) steps `$096B` through nine squares. `DATA8_8292FA` offsets from the faced tile are `(0,0), (0,1), (1,1), (1,0), (1,−1), (0,−1), (−1,−1), (−1,0), (−1,1)`.
- Each square that passes the seed bit is one `CODE_81A688`. Spring/summer writes `0x70`. The ninth step clears `$096B`, decrements `!seeds_grass_N`, and calls `ChangeStamina` with `#$FF` (−1). The first eight steps do not charge stamina.
- A failed square still advances `$096B`. The bag is nine attempts. Hoe the 3×3 first so all nine writes land.
- Hoe, when the cell is `0x01` or `0x02`: one swing, effect `$17` writes `0x07`, `ChangeStamina` `#$FE` (−2) at the hoe return. Stamina refills on `NightReset`.

Price of a grass bag was not found as an `AddMoney` immediate. `SEED_BAG_PRICE["grass_seeds"]` in `buy_seeds.py` is 500. Potato 200G and corn/tomato 300G are the measurements named in that file (corn and tomato on Summer D13). The shed gives one grass bag for free: `INC !seeds_grass_N` at `bank_81.asm` `$8184D0`.

Crop bags use the same nine-step pattern and the same −1 stamina on the ninth step. In-season writes are the `$19`–`$1C` tiles above. Potato and turnip require season 0; corn and tomato require season 1; any other season writes `0x1E` (still counted, until Winter D1).

Fence in hand: 2 lumber, item `0x57`. Map id after the drop is not `0x05` on the path that was decoded.

Frames: grass seeds store animation id `#$0046` at `$0901` (`ToolAnimationGrassSeeds`, `bank_82_toolanimation_subrutines.asm` line 105). Hoe stores `#$0054 + direction` (line 18). Sickle stores `#$0050 + direction` (line 9). `CODE_81CFA0` points that id at `DATA8_81D210` and runs the sprite script until `$A7 == #$FFFF`. That script was not reduced to a frame count. `farm_clearer.py` `SWING_COOLDOWN_FRAMES = 20` is the bot's cooldown.

## Cut grass and `0x1D`

`ToolUsedSickle` returns immediately in winter (`season == 3`). Otherwise, on the farm:

- Weed `0x03` → effect `$0E` → `0x02` (below `0x1D`; hoe it back).
- Grass `0x79` only: `AddGrass(+1)`, decrement `$092E`, effect `$52` → **`0x7A`**. `0x7A ≥ 0x1D`, so the cut tile still counts, and `0x7A` is walkable.
- Any other farm tile the sickle accepts → effect `$17` → `0x07`, which does not count.

No other grass stage is the sickle's grass target. `0x70`–`0x78` and `0x7A`–`0x7C` have no sickle bit except `0x79` (property byte of `0x79` is `0x01`).

Nightly, while it is not winter (`NightlyFarmTilesCheck`, `bank_82.asm` `$82A811`):

- `0x1D` is skipped (stays `0x1D`).
- `0x79` is skipped (stays mature).
- `0x7C` is rewritten to `0x73`.
- Other grass `≥ 0x70` increments every night, with no water check.
- `0x7A` therefore becomes `0x7B`, then `0x7C`, then `0x73`, and grows toward `0x79`. All of those ids are `≥ 0x1D`.

So a successful cut of mature grass stays at or above `0x1D` through the nightly grass cycle. Hurricane and New Year damage are the writes that put grass back on `0x02`.

Cow feeding `Sub_82A9A0` finds one of `0x76`–`0x79` and writes `0x7A`. That tile still counts.

## Hurricanes, season changes, winter

`NightReset` (`$8282AC`) order on the sleep:

1. Calendar. If the season wraps past winter, `LoadsDamageProbabilities(4)` and `ClimateFarmDamageCheck` run **before** season is stored as 0. That is New Year night (Winter D30 → Spring D1).
2. `SetWeatherFlags` copies `weather_tomorrow` onto `$0196`.
3. Clock to 06:00, then `NightlyFarmTilesCheck`, then `ClimateFarmDamagePrep`.

`MonthlyFarmTilesCheck` runs only from the nightly pass when season is already 3 and day is 1. That is the sleep into Winter D1. It does not run on New Year night, because the season has already been stored as spring. Spring→Summer and Summer→Fall do not wipe the map.

Winter D1, persistent map, ids below `0xA0`:

- `≥ 0x70` → `0x7A` (still counted, still walkable).
- `0x1D`–`0x6F`, including fall/winter grass seed `0x1D` and every crop → `0x07` (not counted).
- Fences, untilled, weeds: unchanged.

Stable cover is grass planted in spring or summer (`0x70`), which winter stores as `0x7A`. Grass planted in fall or winter is `0x1D` and is cleared on the next Winter D1.

`ClimateFarmDamageCheck` hits a tile when `RNGReturn0toA(N)` returns 0. `DamageProbabilityTable` (`$828298`):

| Weather | Fence `0x05` → `0x06` | Grass `0x1D` and `≥ 0x70` → `0x02` | Tilled / crops → `0x02` |
| --- | --- | --- | --- |
| Rain | 1/96 | none | none |
| Snow | 1/64 | none | none |
| Hurricane | 1/8 | 1/16 | 1/4 |
| Flag `0x0200` row | 1/8 | none | 1/4 |
| New Year | none | 1/64 | 1/32 |

`0x06` still counts. `0x02` does not. Grown-grass destruction also decrements `planted_grass` when that counter is nonzero. `0x1D` destruction does not.

`WeatherTomorrow` (`$828CF9`): hurricane and thunder are summer (`Hurricane_Chance_Table` / `Thunder_Chance_Table` are `db $00,$1E,$00,$00`, so 30). The hurricane roll is skipped when the day is already 30. Turtle shell (`$7F1F6C` bit `0x1000`) doubles that 30 before the roll. Year-1 summer D29 forces thunder (weather 5) when the thunder flag is clear. Thunder sets `$0196` bit `0x0100`, and `ClimateFarmDamagePrep` returns without damaging on that bit.

The roll that matters for the ending: the night that enters Summer D29 runs `WeatherTomorrow` with day 29, so a hurricane can be stored for the next day. The sleep into D30 applies that weather in `SetWeatherFlags`, grows tiles, then `ClimateFarmDamagePrep`. D30 morning's `CalculateFarmDevelopment` sees the damaged map. The farmer does not get a D30 work morning before the tally. One hurricane on that sleep at 1/16 of ~2,968 grass tiles is on the order of 185 tiles, larger than the 18-tile slack on the clear pin and the 24-tile slack on the 2,974 mask. Rain and snow do not destroy grass. New Year (1/64) is Y3 Spring D1, with spring and summer left to replant.

## Access

Measured on the FarmClear map with every hoeable or breakable `< 0xA0` cell set to `0x70`, the eight `0x55` cells left as `0x55` (walkable and counted), `FARM_NO_GO_TILES` blocked, and the bin lip allowed. BFS from `(8,26)` reaches 3,875 of 3,876 walkable cells. The missed cell is `(26,28)`, id `0xA1`, boxed by the shed no-go. It is outside the 2,968.

Every `< 0xA0` cell that does not already count has a reached orthogonal neighbor on the **original** clear map (stumps included, as targets to face rather than to enter). Unfaced plant targets: 0. The four stump anchors `(34,42)`, `(42,44)`, `(36,51)`, `(42,58)` are all faceable.

Door and exit ids on this pin, and distances on the grass map, are in `F2_layout.md`.
