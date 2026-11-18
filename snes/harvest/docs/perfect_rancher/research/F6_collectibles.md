# F6 collectibles

Date: 2026-09-29. Bead PR-015. Dep PR-007 is done (`research/whatif_pins.md`, live load `fails=0`). Static decode of the USA ROM and `HM-Decomp`. No emulator. Frames: 0.

## Corrected inventory, 2026-11-03: ten sources, nine with ROM receipts

The "designed seven" conclusion below is **wrong**. Group 23's NEWCC at $B48E44 targets $B48E94 in its own bank, not the berry
script at $B38E94, and its positive choice writes tool 0x14 (gold axe), not a berry. The real set is ten sources (the web
walkthroughs fogu.com and almarsguides.com list the same ten); the ROM's persistent receipt flag and the green test for each:

| # | source | receipt flag | ROM receipt |
| ---: | --- | --- | --- |
| 1-2 | hoe, twice (1/64 a swing) | `$7F1F64` 0x0800, 0x1000 | `test_field_berry_is_dug_picked_and_eaten` |
| 3 | mountain stump (1/16 a six-chop tree) | `$7F1F64` 0x0400 | `test_stump_berry_is_acquired_and_eaten` |
| 4 | chicken statue, hammer (1/4 a swing) | `$7F1F64` 0x0200 | `test_statue_berry_is_hammered_picked_and_eaten` |
| 5 | cave, via the pocket stump after the first Winter snowstorm | `$7F1F6A` 0x0040 | `test_cave_berry_*` |
| 6 | fish: catch one, put it back in the pool | `$7F1F66` 0x2000 | `test_fish_is_caught_returned_and_its_berry_eaten` |
| 7 | wandering hawker, first Fall hen sale (the hen goes, no gold) | `$7F1F66` 0x4000 | `test_hawker_chicken_trade_berry_is_eaten` |
| 8 | Harvest Festival (Fall D12), a fish offered | `$7F1F6C` 0x8000 | `test_harvest_festival_fish_offering_berry_is_eaten` |
| 9 | Egg Festival (Fall D20) win: coin flip, 0x1000 turtle shell or 0x2000 arms the house berry (0x4000 eaten) | `$7F1F6C` | `test_egg_festival_win_*` (both faces) |
| 10 | snow flower seed (2,000G, Flower Festival peddler Spring D23) planted on the summit in Winter | purchase `$7F1F6E & 0x0800`; seed consumed on planting | `test_snow_flower_is_eaten_and_kept_through_the_return_and_night` |

The persistent `count == 10` ending check is unchanged. Notes that cost a fix each: the permanent bit of a *spawned* berry is set
at spawn, so an unpicked hoe or stump berry is gone for good (e2e `s4` lost three); the Egg Festival's house berry lies on the
table's edge and lifts only from its east side facing left; the hunt's barrel 7 sits in a pen whose gate tile is `0xA6`.
Details and the planner hookup: [plan.md](../../plan.md#power-berry-calendar-star-night-and-wife-drift-2026-11-03-handoff).

The rows below are the 2026-09-29 static decode and keep their original errors in the "seven" count and the goddess-axe row.

**Cave berry (row 5), measured 2026-11-02.** The `$29` spawn and the eat flag (`$7F1F6A & 0x40`) are real:
the native recipe takes 3,530 frames. But "first day `$29` can be entered" is not the earliest date. The
chamber is sealed from the cave's rooms; the way in is the special stump at mountain (10,38), whose hole
drops the farmer beside the berry (17 frames). That stump is in a west pocket behind a 2x2 boulder at tile
(23,39), which `ReplaceTilesMoutain` (`$81AC7F`) removes only once `$7F1F64` bit 0x0002 is set, by weather 4
(a Winter snowstorm; `WeatherTomorrow` forces it when the day reads 7 in Year 1 Winter). Earliest is the
first Year 1 Winter snowstorm, and no later than the Winter D8 morning.


## Verdict

Designed one-shot power berries are 7. The bead asks to confirm 10. `MASTER_PLAN` F6 says "All 10 power berries" and "Festival berries give 2–3 chances." The ROM does not. This file is the contradiction. Best-ending `$83F075` compares `power_berry_N` (`$7F1F36`) to `#$0A` at `$83F097` and branches away on any other value. Seven eaten berries leave max stamina at 170 and the stamina term at 35, short of 50.

New game stores `power_berry_N = 0` (`bank_83.asm` `$83` line 4566). Item id `#$08` is the berry: `Event_UNK3` (`$84B93C`) hardcodes `LDA.B #$08` / `STA.W !item_on_hand`. The eat completion at `$81CAAB` runs only when `$0984 == 8`.

No `3B 08` and no `3E 08` exist in script banks `$B3`–`$B6`. The only opcode `0x4A` on a real script body is group 0 at `$B38E94` (`2B 20 87 00`, then `4A`, then a jump back to the `4A`). That object grants item 8 when `[$CC]+$0C` is nonzero, A is pressed, and `item_on_hand` is empty. Its only script spawn is `09 02 94 8E` at `$B48E44`.

## Score bits

| Term | Test | Points | Writer |
|---|---|---|---|
| Clock | `$7F1F6E & $4000` | +22 | `$81F370` |
| Turtle shell | `$7F1F6C & $1000` | +21 | script SetFlag bit 12 at `$B4BFE0` |

`ReplaceTilesHouse1` (`$81ACCC`) draws turtle tile `$00FD` when the turtle bit is set, and clock tile `$00FE` when the clock bit is set. The turtle bit also shifts the hurricane chance up once before `RNGReturn0toA` (`bank_82.asm` line 1522).

`RNGReturn0toA` (`$8089F9`) returns 0 with probability 1/A when A divides 256. The berry rolls use `#$40` (1/64), `#$10` (1/16), and `#$04` (1/4). A nonzero result is the miss.

Night reset (`bank_82.asm` lines 500–509) zeros `$7F1F5A` through `$7F1F62`. Permanent berry flags live on `$7F1F64` and are not in that list. `$7F1F70` bit `$0004` (the clock arm) is not cleared there. The night pass at lines 409–411 clears only `$7F1F70` bit `$0002` (`AND #$FFFD`).

## Designed berries

Shed row 1 is `#$0F` at new game (`bank_83.asm` line 4646, comment "all the basic tools"). `ram.asm` names the low nibble sickle, hoe, hammer, axe. Shipping returns match: tool 2 hoe clears bit `$02`, tool 3 hammer clears bit `$04`, tool 4 axe clears bit `$08` (`$818671`, `$8186A8`, `$8186DF`). Hoe, axe, and hammer are owned on Y1 Spring D1. Gold hoe, gold hammer, and gold axe hit the same permanent flags as the basic tools. They are not extra berries.

Map-entry table `UNK_TableFunctionSelect` at `$83CAA9` indexes by `tilemap_to_load`. Confirmed words: `$0B → $83E6B2`, `$15/$16/$17 → $83DED3`, `$29 → $83E4CB`.

| # | Item | Trigger | RNG and chances | Flag | Earliest | Latest |
|---|---|---|---|---|---|---|
| 1 | Field berry 1 | Hoe or gold hoe, `bank_82_toolused_subrutines.asm` `.powerberry1check` line 249 | 1/64 per swing. A same-swing coin (1/16, script `$3A`, +1 money) or bag (1/16, script `$3B`, +5) returns before the berry roll and sets `$7F1F5C` bit `$0200`. Later swings that day skip the coin section and can still roll. One spawn per day | `$7F1F64` bit `$0800`. Daily `$7F1F5C` bit `$0100` is set on the successful roll. Both require `$7F1F60` bit `$0008` clear | Y1 Spring D1 | Y3 Summer D29, while the permanent bit and the daily bit are clear |
| 2 | Field berry 2 | Same roll, after bit `$0800` is set | Same 1/64. The second success sets bit `$1000` and spawns. A later 1/64 still sets the daily bit and then returns with no spawn | `$7F1F64` bit `$1000` | The day after berry 1, or the same season any later day | Y3 Summer D29 while bit `$1000` is clear |
| 3 | Mountain stump | Axe, six chops, tile property in `{$09,$0A,$0B,$0C}`, tilemap `>= 4`. Gold axe skips the six-chop counter and the `$7F1F60` bit `$0008` check. Both spawn at line 627 / line 1814 | 1/16. A miss leaves bit `$0400` clear | `$7F1F64` bit `$0400` | Y1 Spring D1, on a stump off the farm | Y3 Summer D29 while the bit is clear |
| 4 | Chicken statue | Hammer or gold hammer. Tilemap `$0C`–`$0F`, tile property `$00F8`. Loads script `Y=$0016` before the roll | 1/4. A miss returns without setting the daily bit, so the same day can retry. Success sets the permanent bit, spawns, then sets the daily bit | `$7F1F64` bit `$0200`. Daily `$7F1F5C` bit `$0080` | Y1 Spring D1, on those maps | Y3 Summer D29 while bit `$0200` is clear |
| 5 | Cave berry | Entry of tilemap `$29` at `$83E4CB`. Spawn at `($00E0,$0180)`, `Y=$001F`, if `$7F1F6A` bit `$0040` is clear. The function does not set that bit | No RNG. One spawn per entry while the bit is clear. Eating on tilemap `$29` sets the bit (`$81CAF9`). Eating on any other map leaves it clear, and the next `$29` entry spawns another | `$7F1F6A` bit `$0040`, set only by the eat | First day tilemap `$29` can be entered. The spawn has no calendar compare. `map_config.py` sends the mountain-spring west hole to `$29`. This pass did not decode an earthquake gate | Y3 Summer D29. The function also has no day compare on Y3 Summer D30 |
| 6 | Indoor berry | House entry `$83DED3` (tilemaps `$15`–`$17`) after the festival arm | No RNG on the spawn. One designed berry. See the festival row for the arm | Arm `$7F1F6C` bit `$2000`. Eat on `$15/$16/$17` sets `$7F1F6C` bit `$4000` (`$81CB08`), which suppresses the next spawn | First house entry after the arm. Y1 Fall D20 if the prize runs and you enter the house that day | Y3 Summer D29. On year 2, season 1, day `$1E` the same function jumps to the ending at `$83F4D8` before the spawn |
| 7 | Goddess-axe berry | Axe on tilemaps `$10`–`$13`, tile property `$00F4`, `$0196 & $001A` clear, `$7F1F6A` bit `$0020` clear. The tool sets the bit, then loads group `$17` / script 0 (`LDX #$0017`, line 727) | No RNG. The script spawns the berry object at `$B48E44` before the cursor. The flag is already set, so leaving the berry does not spawn another. Cursor 0 is text `$031F`, SetFlag `$7F1F6A` bit 3, happiness −30. Cursor 1 is text `$0320`, happiness +20 | `$7F1F6A` bit `$0020` | Y1 Spring D1 if that tile is reachable and the weather mask is clear. Otherwise the next clear day | Y3 Summer D29 while bit `$0020` is clear |

The spawn sites of `LDY #$001F` / `JSL SUB_8480F8` are exactly `$81EF47`, `$8295A7`, `$8297A6`, `$829898`, `$829FDA`, `$82A1A8`, `$82A279`, `$83DF40`, `$83E4E8`. The first is chicken death. The next six are the hoe, statue, stump, and the three gold tools. The last two are the indoor and cave spawns.

## Festival: turtle or the indoor arm

`$83D304` loads group `$28` script 2 when season is 2, day is `$14`, and hour is below `$0F`. No year check. That is Y1 Fall D20 and Y2 Fall D20, before 15:00. Y3 never reaches Fall. The preceding block at `$83D2E0` loads a different script when `$7F1F60` bit `$0400` is set. Its `AND #$FBFF` stores to direct-page `$60`, not back to `$7F1F60`.

Group `$28` script 3 is what tilemap `$0B` runs on every entry (`$83E6B2`, no date check). From `$B4C3AC` it sets `$7F1F60` bit 9, spawns the festival actors, and compares `$8009A2`. Values 3, 4, and 5 jump to `$B4C403`: text `$02EA`, opcode `0x41` adds +30 to happiness, then `09 09 98 BF` spawns the prize actor at `$B4BF98`. Any other value shows text `$038F` and does not spawn it. The ROM has no `STA $09A2`. The festival loader zeros `$09A2` at `$83D3A5`. This pass did not find the instruction that stores 3, 4, or 5, so the prize is "the `$0B` script while `$09A2` is 3, 4, or 5," which is the festival result path, not a daily roll.

Prize at `$B4BF98`, `RNG /2`:

- 0, and `$7F1F6C` bit 12 clear: SetFlag bit 12 at `$B4BFE0` (the turtle), opcode `0x41` adds +50 to happiness. No item 8.
- 1, and bit 13 clear: SetFlag bit 13 at `$B4BFF7` (the indoor-berry arm). No item yet.
- If the rolled bit is already set, that arm jumps to the other. Two wins, one on each Fall D20, cover both bits. One win gives exactly one of the two, at 1/2.

The blocks at `$B4BFA3` and `$B4BFBE` set bits 10 and 11, add +30 to `$7F1F10` (`stored_grass`), and show text `$02EB`. The labeled string at `DATA16_B9B0F4` is "fodder for one whole month." Those blocks are not targeted by the `/2`. They are not berries. There is no third festival berry.

Turtle earliest: Y1 Fall D20, before 15:00, if `$09A2` is 3, 4, or 5 and the `/2` selects bit 12. Latest: Y2 Fall D20 under the same conditions. After Y2 Fall D20 the bit stays clear for the rest of the game.

Indoor berry earliest and latest are in the table above. The arm is this festival result, not a separate berry lying on the festival map.

## Clock

The score bit is written in one place. House-completion handler `$81F2DA` (`$096E = $48`, set by morning `$83D060` when `$7F1F66` bit `$0100` is set). After the timer and the house text, `$81F354` checks `$7F1F64` bit `$0080`. If that bit is still clear and `$7F1F70` bit `$0004` is set, `$81F370` does `ORA #$4000` onto `$7F1F6E` and shows text `$025C`. The string at `DATA16_B8E4C8` is the souvenir clock. If bit `$0080` is already set, or the arm bit is clear, the handler loads group 9 scripts `$38` and `$39` and does not set the clock. The first completion sets house bit `$0040` (`bank_82.asm` line 561). The second sets bit `$0080` (line 554). The clock is the first-upgrade gift only.

The arm bit is also written in one place. Quote handler `$81F121` (`$096E = $47`). When `$0970` hits `$20`, if year is 0 and season is below 2, `$81F162` does `ORA #$0004` onto `$7F1F70`. That store happens before the wood check and before the yes/no cursor. A decline or a failed payment on that morning still leaves the arm set. Season 2 or 3, or any later year, skips the store. The bit is not cleared by the night zeros.

Quote morning is `$83CFF8`: `$7F1F66` bit `$0040` loads group 9 script `$2C` and clears the bit. Night pass lines 517–526 promote bit `$0020` to bit `$0040` on the night after `$0020` is set. New game sets `wood_need_for_upgrade` to 250, or to 500 if house bit `$0040` is already set (`bank_83.asm` lines 4863–4869). The quote charges `AddMoney` of `$FE0C` (−5,000G) when the need is `$00FA`, and requires 250 wood. The 500-wood path charges `$FC18` (−10,000G) and requires `$01F4` wood. Success zeros `development_rate` and sets `$7F1F66` bit `$0080`. Each following night increments the rate. The night it becomes 4 sets bit `$0100`, clears bit `$0080`, and sets the house upgrade bit. The gift is the next morning. Accept on morning D, gift on morning D+4.

Window the ROM does enforce:

- Latest arm morning: Y1 Summer D30. The check is year 0 and season below 2, with no day compare. Day `$1E` is a real morning of that season (the ending uses that date).
- Earliest gift, once a quote in that window is accepted: four mornings later. A Y1 Spring D1 accept gifts on Y1 Spring D5.
- Latest gift: Y3 Summer D29, with the arm already set and house bit `$0080` still clear. The Y3 Summer D30 house entry is the ending. The completion dispatcher itself is the morning chain at `$83D060`, which can still see bit `$0100` on a D30 wake-up. Schedule the gift by D29.

The request bit `$0020` has two script writers, both SetFlag index 5 on `$7F1F66`: `$B3AD90` (group 1 script 9) and `$B3CC50` (group 4 script 7). The only `LDX #$0001` / `LDY #$0009` load is `$83DE2F`, inside the tilemap `$2B` entry at `$83DD95`. Every path that can fall into that load has already set `$7F1F68` bit `$0008`, and `$83DE0A` skips the load when that bit is set. The dated path (year 0, spring, or summer before day `$1A`) loads group `$0B` script `$0A` instead and returns. No other `24 AD` spawn of script 9 turned up. This pass did not find a reached writer of bit `$0020`. The clock's score store and its arm window are decoded. The scene that starts the quote is not.

## Not berries

- Hawker. Group 17 script 0 holds the offer. The accept path at `$B48850` fades, spawns a placement script, shows text `$014A`, and Ends at `$B48861`. No opcode `0x4A` and no spawn of `$8E94`. `Text_Hawker_Berry1` offers a chicken trade. `Text_Hawker_Berry2` says "this cow is now mine."
- Fishing. `Event_UNKFishing` (`$84AA5D`) stores item `#$07`.
- Forage. `DATA8_82CFB4` through `$82D1C0` is 524 bytes and contains no `#$08`.
- Flower, harvest, and winter festival loaders (`$83D3B8`, `$83D3E2`, `$83D40E`) load other groups. The harvest entry does not spawn `$8E94`.
- The morning loader at `$83D23E` (6 chickens, weather, 6:00) loads group `$29` script 0. It is not an item-8 grant.
- Chicken death at `$81EF47` clears the coop slot, decrements `chicks_N`, spawns `Y=$001F`, happiness −20, and sets `$7F1F60` bit `$1000`. It does not check a one-shot before the spawn. Each death can raise `power_berry_N`. It is not one of the seven.

## Duplication, not used

`$81CAAB` increments `power_berry_N` with no cap, adds 10 happiness, and adds 10 to `max_stamina`, also with no cap. That block does not store `item_on_hand`. `Event_TransitiontoHouse` (`$84B51B`) and `Event_TransitiontoX` (`$84B54F`) both `ORA #$4000` onto `$0196`. `MASTER_PLAN` names the resulting edge "tired animation + screen transition." Eating the cave berry off tilemap `$29`, or the indoor berry off tilemaps `$15`–`$17`, leaves the suppress bit clear, so the next entry of that map spawns another. Either path can push `power_berry_N` off 10 and fail `$83F097`. Neither is on the route. This pass did not execute them.

## What 999 would need

Stamina maxes at 10 eaten berries, +10 each, from a base of 100. The seven one-shots above are the whole designed set. Chicken deaths, off-map respawns, and the transition edge are the only ways the eaten count reaches 10, and each of them can also pass 10. The clock and the turtle are separate +22 and +21 terms. Their plans are above. The clock plan still has no reached writer for the quote-request bit.

## Unfinished repair findings, 2026-09-30

The cabin request decode needs correction: `$83DDC2 BNE $83DE0A` jumps
into normal dispatch. With intro bit `$7F1F68.0008`, map 0x2B selects group
1/script 9 (rain: group 4/script 7). The native request/arm/gift receipt was supplied on 2026-10-15 by the
first-house continuation in [EVE_BAR.md](../../EVE_BAR.md). The door is
mountain tile (32,33), index 28, destination 0x2B.

Interior navigation also has a stride bug: `$82B13C` uses width 16 for
`$0181=1`, 32 for 2, and 64 otherwise; the existing generic tile reader/BFS
assumes 64. Cabin 0x2B and cave 0x29 require their native widths. Native cave
entry was measured at 436 frames, but acquisition to exactly ten and final
sleep were not completed. The unfinished interior approach remains in the
repair-home worktree; it is not a merged or route-ready capability.
