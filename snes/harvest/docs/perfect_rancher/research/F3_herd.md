# F3 herd and flock

## Repair evidence, 2026-09-30

The missing slot-0 stall conclusion below is superseded. The edited pin
cleared feed flags while retaining a filled trough tile. Native barn
exit/re-entry regenerates slot 0 as tile E2. The task now feeds actual
slot masks, including the pregnant 0x1000 mask, rather than a prefix.
Merged-main [F3_all12_feed_night_probe.json](F3_all12_feed_night_probe.json)
ran 7,507 frames: flags 0x0FFF, hay 500→488, and all twelve cows stayed
at 200 through a fed Spring D16→17 night. It starts from an edited research fixture
and audits zero live writes, mid-run loads and assists. It proves feed and
night survival, not acquisition or fifty days of care. A replacement calf
is no longer required for slot 0.


Date: 2026-09-29, stall-1 night 2026-09-30. Bead PR-012. Dep PR-007 is done (`research/whatif_pins.md`, oracle `fails=0`). USA decomp under `HM-Decomp`, plus two buy walks, two care days, and one fed-barn night on research pins `LN_mgr-f3_whatif_*`. Those pins are RAM what-ifs. They are not route evidence.

The bought day-0 byte is 0. Talk is +1 and brush is +3. A fed adult barn night adds 0. Slot 0 never reaches 200. A feedable slot reaches 200 on care day 50. The dates are under "Latest date".

## Audit qualification, 2026-09-30

The 50-care-day result is extrapolated from +4 care and one measured
zero-delta fed slot-1 night. No 50-day replay or born-slot-0 route was
measured. PR-048 is mandatory for native slot-0 replacement, inherited
happiness, naming, care and final-night survival. F3's research closure
is not an all-twelve-cow route proof.

## Slot

Cow slots: 12, base `$7EC1C6` (`GetsCowPointer` `$83C9A7`), stride `0x10`. `AddNewCow` (`$83C8DC`) returns A=1 when every slot has status bit 0.

| Offset | Use |
|---|---|
| +0 | Status. Exists `$01`, baby `$02`, child `$04`, adult `$08`, cranky `$10`, sick `$20`, pregnant `$40`. Birth ORs `$80` and clears `$40`. |
| +1 | Daily flags. Brushed `$01`, milked `$02`, talked `$04`. Night pass `AND #$F8` clears those three. Bit `$80` is a pending pregnancy. |
| +2 | Home map. New cows store `$27` (barn). |
| +3 | Age or sick or pregnancy counter. |
| +4 | Happiness, u8. |
| +5 | Annoyance. Night cranky test is `== $0A`. |
| +8 / +A | Position. |
| +C..+F | Name. |

`AddCowHappiness` (`bank_84.asm` `$84A5D4`) takes a signed amount in A and the cow pointer in `$72`. The sum is stored 0 when the 16-bit add is negative, and 255 when it is `>= 256`.

Score term 14 walks the 12 slots. Existence is status bit 0. Points are `min(25, byte >> 3)`. Byte `>= 200` is 25 points. `cow_N` `$7F1F0A` is a separate term, `3 × cow_N`, max 36.

## Buy value and birth value

`AddNewCow` clears the 16 slot bytes, stores map `$27`, pregnancy 0, happiness 0, annoyance 0, daily flags 0, then status `$05` (exists|child). A=0 keeps that happiness 0. A=1 (born) stores status `$03` (exists|baby), copies happiness from `$7F1F2B`, and calls `AddPlayerHappiness` with 70.

`SetCowNameBought` (`$80815F`) calls `AddNewCow` with A=0 during the name-screen fade, then copies the four name bytes. `SetCowNameBorn` (`$8081D2`) calls it with A=1 and clears `$7F1F64` bits `$0004` and `$0008`. The slot byte +4 is written in that fade, before a talk can land.

Shop payment is earlier and does not create the slot. `CODE_81E08A` (`$81E08A`) ORs `$7F1F5A` with `$0022`, INCs `cow_N`, and adds player happiness 20 (`#$0014`). `AddNewCow` does not INC `cow_N`. `AnimalPurchaseSession` treats the count and the wallet as success while the slot is still empty. A day-0 read has to be the first frame the slot exists, after `SetCowNameBought`.

Name kind `$099F`: 1 at `$81E777` (bought), 2 at `$81F044` (born).

Birth stages `$7F1F2B` on the birth night, from the mother's byte: 20, or 50 when the mother is `>= 96` and `< 192`, or 100 when the mother is `>= 192` (`CowFeedingandStatus` `$83BF` birth block, `bank_83.asm` around line 7176). The calf slot receives that byte at naming. This session did not see a birth or a name screen.

ROM store for a bought cow is 0. A replay of `tasks/buy_cow.json` read that byte on the first frame the slot existed. See "Day-0 slot".

## Talk, brush, milk, feed

`Event_UNKCowRelated` (`bank_84.asm`):

- A-button talk (`$849C86`): if +1 bit `$04` is clear, set it and `AddCowHappiness` +1. No adult check.
- Brush, tool `$0F` (`$849EAE`): if +1 bit `$01` is clear, set it and `AddCowHappiness` +3. No adult check.
- Milk, tool `$0E` (`$849DED`): requires status bit `$08`, +1 bit `$02` clear, and status bits `$70` clear. Sets the milked flag. Stores item `$15` when happiness `< $60`, `$16` when `< $C0`, `$17` otherwise. Does not call `AddCowHappiness`.
- Wrong-tool path `$849FB6`: INC byte +5 unless it is already `$0A`.
- Medicine (`$849FBC`), reached when `$7F1F5A` bit `$0080` is set and the cow is adult. `ToolUsedCowMedicine` (`$829B11`) sets that bit. The apply ANDs status with `#$DF` and stores 0 to byte +3. It does not add happiness back.

`ToolUsedMiraclePotion` (`$829B31`) sets `$7F1F5A` bit `$0100` and clears the selected tool. The cow apply is `$849E4E`, reached when that bit is set and the cow is adult. It refuses when `$7F1F64` bit `$0004` is already set or `cow_N == $0C`. On success it sets `$7F1F64` bit `$0004`, clears status bits `$30`, sets +1 bit `$80`, `AddCowHappiness` +10, stores pregnancy byte `$15` (21) and the same 21 into `$7F1F12`, and INCs `cow_N` before any calf slot exists.

Feed during the day writes `fed_cows_flags` `$0932`. It does not call `AddCowHappiness`. The night test is below.

## Night decay, outdoors, sickness, death

`CowFeedingandStatus` (`$83BC5A`). The first loop's `BEQ .isbaby` is the branch taken when status bit `$02` is clear. Babies jump to `.skipcow` and skip the unfed penalty. Children (bit `$04`) on a map `< 4` call `Sub_82A9A0` and also skip the penalty. The growth loop runs after this loop, so the night a child becomes an adult still skips the feed check.

Adults in the barn (map `== $27`): a pregnant cow tests `Cow_Feed_Flags` at index `$18` (the word `$1000`). Any other adult tests `Cow_Feed_Flags[slot]`. The table at `$82A571` is `$0001,$0002,$0004,$0008,...` so slot 0's mask is `$0001`. A clear mask takes happiness −8 (`#$FFF8`) and a sick roll with Y=4.

Adults outside the barn with map `>= 4` take the same −8. Map `< 4` calls `Sub_82A9A0`, then `$092E == 0` takes −8. Otherwise `$0196` selects the outdoors delta:

| `$0196` bit | Delta | Sick-roll A |
|---|---|---|
| `$0002` | −8 | 4 |
| `$0008` | −16 (`#$FFF0`) | 2 |
| `$0010` | −24 (`#$FFE8`) | 2 |
| `$0200` | −8 | 8 |
| none of those | 0 (`.cowfed`) | |

`SetWeatherFlags` (`$828C09`) writes `Climate_Flag_Table` `$828CED`: `$0004,$0002,$0008,$0010,$0200,$0100` for sunny-in-summer, rain, snow, hurricane, fair, and weather 5. Bits `$0004` and `$0100` fall through to the 0 delta when the grass test passed. Bit `$0400` is not in that table. The cranky path still tests it. That path was not seen live.

There is no positive feed delta and no positive outdoors delta. A fed barn adult with a matching flag bit and none of the four weather bits above changes happiness by 0 in this loop.

Sick roll: `RNGReturn0toA` (`$8089F9`) returns 0, and the cow is not pregnant, +1 bit `$80` is clear, and status bit `$20` is clear. Then status ORs `$20`, byte +3 stores 7, happiness −12 (`#$FFF4`), player happiness −30 (`#$FFE2`). The routine buckets `255/A`. This note does not reduce that to a fraction.

Death, second loop, sick adults: byte +3 decrements. The night that store hits 0, `$0196 & $001A == 0` and `weather_tomorrow < 6` clears the slot, DECs `cow_N`, and ORs `$7F1F6E` bit `$0040`. Death does not call `AddCowHappiness`. If the weather test fails, the byte stays 0 and a later night retries the check. Two removal scenes also zero a slot and DEC `cow_N` (`$81E9C3`, `$81EB59`) and call `AddPlayerHappiness` with `#$FFCE`.

Cranky, same first loop, after the feed branch: byte +5 `== $0A`, or `$7F1F5A` bit `$1000` with the cow outside the barn and `RNGReturn0toA` A=`$10` returning 0 (that arm also applies −8 first), or `$0196` bit `$0400` with the cow outside the barn. If not pregnant, +1 bit `$80` clear, and not sick: status ORs `$10`, byte +3 stores 3, byte +5 stores 0, happiness −30 (`#$FFE2`). The second loop decrements the timer and clears bit `$10` on the night it hits 0. Bit `$1000` of `$7F1F5A` is set by `ToolUsedHoe` (`$829460`) and `ToolUsedGoldHoe` (`$829E8E`) on their spring/summer RNG arm. That cranky arm was not seen live.

The second loop discards the happiness byte it loads, then clears +1 bits 0–2.

## Growth

Bought cow, status `$05`, timer 0. The child arm increments byte +3 and at 21 stores status adult (`ORA #$08`, `AND #$FB`) and ORs `$7F1F6E` bit `$1000`. The decomp label on that store says "healthy Child". The compare is `#21`. From 0 that is 21 nights, and the feed check starts the following night (night 22), because the feed loop already ran.

Born calf, status `$03`. The baby arm increments byte +3 and at 14 stores child (`ORA #$04`, `AND #$FD`). The decomp label says "healthy Adult". Then the child arm takes 21 nights. The slot does not exist until `SetCowNameBorn`, so those counters start at the first `CowFeedingandStatus` after naming. Naming itself was not run live, so the calendar day of that first night is not measured.

Every adult night ORs `$7F1F6E` bit `$1000`. The animal-shop shelf (`ReplaceTilesAnimalShop` `$81AFC0`) shows a miracle potion when that bit is set, `$7F1F64` bit `$0004` is clear, `cow_N != $0C`, `$7F1F68` bit 0 is set, and neither tool slot already holds item `$0A`.

## Pregnancy clock

The potion stores 21 in byte +3 and in `$7F1F12`, with +1 bit `$80` set. The same night's second loop sees bit `$80`, sets status bit `$40`, clears bit `$80`, and does not decrement. Each later night decrements both counters. The night `$7F1F12` decrements to 0 falls into the birth block, and birth also retries on later nights once the counter is already 0. Birth requires map `$27`. It sets `$7F1F64` bit `$0008`, marks the mother with status `$80` and clears `$40`, and stages `$7F1F2B`. It does not INC `cow_N` and does not create the calf slot.

`cow_N` therefore counts the fetus from the potion apply, and counts a shop cow from the payment, before either slot exists. The happiness sum only scores slots with the exists bit. Shop, potion, and `AddNewCow` all refuse at 12. Twelve existing slots is the barn. Twelve counted animals, including pregnancies, is the `cow_N` cap. A 12th counted cow cannot be impregnated.

## Chickens

`AddNewChicken` (`$83C807`) searches 13 slots (`CPX #13`), base `$C286`, stride 8. A=0 stores status `$43` (incubating egg) at `($00E8,$00B8)`. A=1 stores status `$09` from `Table_ChicketSpawnPos`. A=2 stores status `$09` at `($0238,$0318)` and writes the season into +1. The shop calls A=2 (`$81E1D9`). The header comment "A=0 hatched" does not match those stores.

`chicks_N == $0C` refuses a buy (`$81DF24`, text `$03B7`). Payment is `AddMoney` `#$FF9C` with high byte `$FF` (−100 raw, −1000G), then INC `chicks_N`, `$7F1F5A` bit `$0020`, player happiness +10. The same bit `$0020` is the one-animal-per-day gate for cows and chickens (`$81DEBE`, `$81DF15`).

The night chicken loops use `CPX #$000D` (13). An egg (status bit `$02`) increments byte +2. The night that byte reaches 3 hatches only when `chicks_N < $0C`: timer cleared, status `ORA #$04` and `AND #$BD` (chick), position `($00D8,$00B8)`, INC `chicks_N`. From timer 0 that hatch is the third night. When `chicks_N == $0C` the egg stays at timer 3 and does not increment the counter. A later night retries the hatch while the counter is full. Counted chickens do not exceed 12. The 13th physical slot can hold an egg, and that egg adds no `chicks_N` points. `animal_status.py` `CHICKEN_SLOT_COUNT` is 12. The ROM search is 13. The score term reads `chicks_N`.

A chick (bit `$04`, egg bit clear) increments byte +2 and at 7 becomes an adult (`ORA #$08`, `AND #$FB`, status `$09`). That is 7 nights. Chickens have no happiness byte.

`CODE_83C05E`: status bit 0 set and bits `$0E` clear clears the slot, DECs `chicks_N`, and adds player happiness `#$FFE2`.

## Shop gates and potion price

Cow buy (`$81DEBC` onward). `$0196 & $000A` nonzero shows text `$03D0` and stops (`$81DE48`). `cow_N == $0C` shows text `$03B7`. Grass test is `planted_grass >= 16 * (chicks_N + cow_N + 1)` (`$81DFDC`). Payment is `AddMoney` of `#$FE0C` with high byte `$FF` (−500 raw, −5000G). Failure text `$0304`. Success is the INC and the +20 above, text `$0308`.

The buy pin's live boot had money raw 800 (8000G displayed), planted grass 256, `cow_N` 0, weekday 2, hour 9, `$0196` `0x80E1` (`0x80E1 & 0x000A` is 0), `$7F1F5A` `0x8000` (bit `$20` clear). The shop script was never reached.

Miracle potion price was not bought live. Item id `$0A` is the potion (`ram.asm` tool list, and `CMP.B #$0A` at `$81AFC0`). The shop record index for an item `< $44` is `item − $1B + $13` (`$8189A6`). For `$0A` that is 2, and the `< $44` arm does not `XBA` before the 16-bit store, so a nonzero high byte would select a different record. Record 2 of `DATA8_81A1A5` (12-byte stride, `bank_81.asm` line 3094) is `3C 03 F7 02 11 02 F9 02 B8 E2 FF FF`. The price field at +9 is `E2 FF FF` = `0xFFFFE2` = −30 raw. Treat −300G as that table read, not a wallet measurement.

## Probes

Commands from `snes/harvest`. The script sets `HEADLESS=1`, `SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy`, `QT_QPA_PLATFORM=offscreen`. One emulator per process, `setsid nohup uv run python /tmp/f3_herd_probe.py buy` and the same with `care`. The script boots `make_harvest_env(pin)` and `HarnessRunner.reset()` with no state argument. An earlier launch called `reset(state=pin)` and raised `TypeError` before any ROM step (`/tmp/f3_buy_probe.log`). That crash is not one of the two walks.

Pins, written once, not route evidence:

- `LN_mgr-f3_whatif_buy0` from `Y1_After_Buy_Potato`
- `LN_mgr-f3_whatif_care1` from `buy_cow_end`, hour stored 16
- `LN_mgr-f3_whatif_care2` from `buy_cow_end`, hour stored 18, happiness byte stored 40

### Buy, two walks, no slot

Try 1, pid 2315811, `/tmp/f3_buy_probe2.log`. Boot frame 0: tilemap 0, xy (93, 448), hour 9 minute 4, weekday 2, day 2, season 0, year 0, money display 8000, raw 800, cows 0, planted grass 256, weather 32993, `$7F1F5A` 32768, cows `[]`. `farm_to_animal_shop_counter` left the farm at frame 29, relocalized onto tilemap `0x04` at (744, 424), then the hop toward (696, 890) push-faced tile (45, 26). Frame 674: expected tilemap `0x04`, got `0x0C`. `PROBE_BUY_FAIL nav`. No payment, no name screen, no slot.

Try 2, pid 2330091, `/tmp/f3_buy_probe3.log`. Same pin, same boot. The seed-shop face reached tilemap `0x04` and passed (608, 288). The south hops after that push-faced tiles (39, 22), (40, 22), (40, 23), (37, 34), (38, 34), then tiles along columns 37–41. Frame 8000: tilemap `0x04`, xy (670, 834), hour 18, cows `[]`, text `0x02F7`. The animal shop is closed from 17:00. The process was still oscillating (670, 830)/(670, 834) at frame 76000 with the same empty cow list and hour 18. It ended there with no `PROBE_BUY_FAIL` line and no JSON. Pid 2330091 is dead. No third walk.

### Care, daytime +4, night −8

Try 1, pid 2337931, `/tmp/f3_care_probe_try1.log`, pin `LN_mgr-f3_whatif_care1`. Injected start 40. Talk frame 1480: happiness 40→41. Brush frame 1702: 41→44. Chores success frame 2184, `talk=True brush=True milk=1 ship=1 fed=1`, fed flags `0x0008`. `GoToSleepTask` treated barn tilemap `0x27` as outdoors and idled at (201, 329), hour 16, through frame 47184. Happiness stayed 44. Day stayed 16. `PROBE_CARE_FAIL sleep`. The log's age 13 and map 45 came from a raw index that does not match `home_map_raw` / `pregnancy_raw`. Discard those two fields. The happiness frames stand.

Try 2, pid 2352582, `/tmp/f3_care_probe2.log`, JSON `/tmp/f3_care_probe.json`, pin `LN_mgr-f3_whatif_care2`. Boot frame 0: tilemap 39, xy (113, 277), hour 18 minute 0, weekday 2, day 16, money display 3690, raw 369, cows 1, grass 2296, hay 179, weather 192 (`0x00C0`), `$7F1F5A` 32800. Slot 0 status 9, flags 0, happiness 40, age 0, map 39. The 40 is the pin write.

| Frame | What changed |
|---|---|
| 961 | Flags `0x02` (milked). Happiness still 40. |
| 1316 | `fed_cows_flags` `0x0008`. Happiness still 40. |
| 1610 | Flags `0x0E`. Happiness 41. |
| 1832 | Flags `0x0F`. Happiness 44. |
| 2369 | Chores success. Happiness 44, map 39, fed `0x0008`. Wallet still 3690. |
| 3000 | House tilemap `0x15`, phase `nav_bed`, hour 18, happiness 44. |
| 3518 | Flags 0, happiness 36, fed flags 0. |
| 3565 | Morning. Day 17, weekday 3, hour 6, tilemap 21, happiness 36, status 9, age 0, map 39, hay 178, weather 0, `$7F1F5A` 0, wallet 3690. `PROBE_CARE_OK`. |

Daytime on this adult: talk +1, brush +3, milk +0, feed +0. The log line `Milk shipped` did not move the wallet. Hay fell 179→178, so one feed unit was consumed. Weather `0x00C0` includes none of `$0002/$0008/$0010/$0200`.

The morning delta is 44→36, which is the unfed −8, and the sick −12 did not also land (status stayed 9). Slot 0's night mask is `Cow_Feed_Flags` word 0, `$0001`. The chore left `$0932` at `0x0008`, which is word 3 of that table and `COW_FEED_SPOTS[0].flag` in `cow_geometry.py`. `$0001 & 0x0008` is 0, so the night pass takes `.cowunfed`. This run is a measured unfed adult night after talk and brush. It is not a measured fed-barn night. A third care emulator was not started.

## Day-0 slot

`research/F3_herd_day0.json`. Pin `LN_mgr-f3_whatif_day0boot` is a copy of the trashed `latest_backup_buy_cow_20260424_135648` state, loaded by name. From `snes/harvest`, `HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen` and `uv run python /tmp/f3_pr012.py day0` replayed `tasks/buy_cow.json` for 6260 frames. `make_harvest_env(pin)` then `env.reset()` with no state argument.

Boot frame 0: tilemap 21, xy (136, 120), hour 6, cows empty, hay 180, weather 0.

Frame 4803 is the first frame the slot exists. Status `$05`, flags 0, map `$27`, age 0, happiness 0, name `1c2830b3`, raw `0500270000000000a80016011c2830b3`. Clock is Y0 Spring D16, weekday 2, hour 9 minute 1. Tilemap 0, player (0, 0), weather `0xC0E1` (49377), `num_cows` 1, hay 180, fed flags 0. Bits `$0002/$0008/$0010/$0200` are clear. Bit `$4000` is the `SetCowNameBought` OR. The bought day-0 byte is 0.

## Care on the adult pin

`LN_mgr-f3_whatif_care3` is the what-if from `buy_cow_end`: status `$09`, daily flags cleared, happiness left at 1, hour 18, brush `$0F` in hand, milker in the pack, fed flags 0. Two boots. Neither reached 06:00.

`research/F3_herd_care3.json`. `CowChoresTask` from (113, 277). Frame 8000: tilemap 39, xy (141, 185), happiness 1, daily flags 0, fed flags 0, hay 150 (boot hay 179), talked false, brushed false, milked 0, fed 0. `PROBE_CARE_FAIL chores`, reason max steps exceeded.

`research/F3_herd_care4.json`. Log `/tmp/f3_pr012_care4.log`. Same pin, booted again, no pin write. Boot frame 0: cow (167, 278), player (113, 277), weather `0x00C0`, happiness 1.

| Frame | What changed |
|---|---|
| 762 | `buy_cow.json` frames 5416:5776 from (125, 359). Player (163, 280). Happiness 1→2. Flags `0x0C`. Cow (150, 278). |
| 858 | Brush, face left, tool `$0F`. Happiness 2→5. Flags `0x0D`. |
| 1170 | Frames 5776:6088. Player (113, 273). Hay 179→178. Held 0. Fed flags 0. Happiness 5. Flags `0x05`. |
| 1294–2062 | Four fodder pickups at (217, 185). Hay 177, then 176, 175, 174. Held `$1A` after each pickup. |
| 2193 | Four A presses at (113, 275). Each left fed flags 0 and held 0. Happiness 5. Flags `0x0D`. Weather `0x00C0`. `PROBE_CARE4_FAIL feed`. |

`ToolUsedCowFeed` (`$82A4C2`) ORs `Cow_Feed_Flags` when the tile in front has property `$E2`..`$EE`. These presses took the drop path. Slot 0's bit `$0001` stayed clear. Sleep did not run. Talk on this adult is +1 and brush is +3, the same daytime pair as the earlier 40→44 day. This care boot did not measure the fed-barn morning byte.

## Latest date

Two dates. Day-0 happiness is 0 (`research/F3_herd_day0.json`). Talk is +1 and brush is +3, and the night pass clears those daily flags, so a care day adds 4 before the night. Milk adds 0. Feed adds 0 until the night test. The 21 child nights skip that test. The adult barn night in the stall-1 section adds 0 when the slot bit is set.

Slot 0 never reaches 200. Its mask is `$0001`, property `$E2`, and no stall has that id, so every adult night is the unfed −8. A bought cow starts at 0. Twenty-one child days of talk and brush reach 84. The first adult afternoon can brush to 88, then the night subtracts 8, and each later day is +4−8. The peak is 88.

A feedable slot reaches 200 on care day 50. Slot 1's mask is `$0002`, stall (8, 15) property `$E3`. Twenty-one care days leave happiness 84 on the adult morning. Twenty-nine more brushes add 116, and 84+116=200 on the brush of care day 50. The measured zero night delta would leave it at 200; the live night
started at 1, not 200.

Care day 1 on the measured buy is Y1 Spring D16 (year raw 0, season 0, day 16). Spring D16 through D30 is 15 days, Summer is 30, and Fall D1 through D5 finishes day 50. The brush that stores 200 is Y1 Fall D5. Fall D5 at 06:00 is still 196. The first 06:00 at 200 is Y1 Fall D6. A later buy shifts both dates by the same number of days. Slots whose mask is one of the other stall flags take this same 0 night when that bit is set. Only slot 1 was fed live.

## Barn night attempt

`research/F3_herd_barn0.json`. Research pin `LN_mgr-f3_whatif_barn0` is a copy of `LN_mgr-f3_whatif_care3`. The only RAM write was slot 0 byte +2 = `$27`. Happiness stayed 1. Status stayed `$09`. This pin is not route evidence.

From `snes/harvest`, `HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen`, `setsid nohup uv run python /tmp/f3_barn0.py`. Log `/tmp/f3_barn0.log`. One emulator. Frame 611, `PROBE_BARN_FAIL feed`.

Boot frame 0: tilemap 39, xy (113, 277), direction 2, hour 18, day 16, weather flags `0x00C0`, fed flags 0, hay 179, happiness 1, status 9, map byte 39 (`$27`), cow (167, 278). `$0181` is 1, so `GetTileIndex` strides 16.

Frame 164: fodder held `$1A` at (217, 185). Hay 179→178.

Frame 281: A at (113, 277), direction 2, the `cow_feed_ops` timing (face right 4, A for 8, face released while A is down). `CalculateTileinFront` (A=1, X=6, Y=6) reads tile (8, 17). That byte is `$2F`. `ToolUsedCowFeed` at `$82A4C2` accepts `$E2`..`$EE` only. Held fodder cleared. Fed flags stayed 0. Happiness stayed 1. Frame 316 is the settled mark.

The stall bytes on this map, and the `Cow_Feed_Flags` word they OR, are:

| Tile | Id | Flag |
|---|---|---|
| (8, 5) | `$E7` | `$0020` |
| (8, 7) | `$E6` | `$0010` |
| (8, 9) | `$E5` | `$0008` |
| (8, 13) | `$E4` | `$0004` |
| (8, 15) | `$E3` | `$0002` |
| (5, 5) | `$ED` | `$0800` |
| (5, 7) | `$EC` | `$0400` |
| (5, 9) | `$EB` | `$0200` |
| (5, 13) | `$EA` | `$0100` |
| (5, 15) | `$E9` | `$0080` |
| (5, 17) | `$E8` | `$0040` |

`$E2` occurs once, at tile (2, 34), on the swatch row `$E2`..`$ED`. Slot 0's night mask is `Cow_Feed_Flags` word 0, `$0001`, which is tile id `$E2`. No stall has that id. The second press in this boot scanned a 64-wide grid, walked toward (318, 72), stuck at the fodder stand, and pressed there. That press is not a property match. Sleep did not run. Happiness before sleep and at 06:00 are unmeasured.

Day-0 happiness is 0. Talk is +1 and brush is +3 on the measured care days. The decoded child growth is 21 nights. This boot did not measure the fed-barn delta. Slot 1's night is the next section.

## Stall 1 fed night

`research/F3_herd_stall1.json`. Research pin `LN_mgr-f3_whatif_stall1` is a copy of `LN_mgr-f3_whatif_barn0`. The only RAM write copied the 16 slot-0 bytes into slot 1 and cleared slot 0. Happiness byte +4 was not given a new value; the copy kept 1. The file's slot 0 is 16 zero bytes. This pin is not route evidence.

From the repo root, `HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen`, `setsid nohup uv run python /tmp/f3_stall1.py`. Log `/tmp/f3_stall1.log`. One emulator. The run ended at frame 1404.

Boot frame 0: tilemap 39, xy (113, 277), direction 2, hour 18, Y1 Spring D16, weekday 2, weather `0x00C0`, hay 179, fed flags 0, `$0181` 1. Slot 1 status 9, flags 0, map 39, age 0, happiness 1, annoy 0, cow (167, 278). Slot 0 status 0 and happiness 0. Tile (8, 15) is `$E3`. Facing right from the boot pixel still reads (8, 17) as `$2F`. No A press was made at (113, 277).

Frame 115: fodder held `$1A` at (217, 185). Hay 179→178.

Frame 231: first A at (113, 244), one pixel above (113, 245). Face right 4, A for 8, face released while A is down. `CalculateTileinFront` (A=1, X=6, Y=6) reads tile (8, 15), id `$E3`. Fed flags stayed 0. Held fodder stayed `$1A`. Happiness stayed 1.

Frame 251: second A, same pixel, same faced property `$E3`. Fed flags `0x0002`, `fed_cows_n` 1, held 0. That is the one retry. No third press. The stall byte then read `$2F`. Happiness stayed 1. Weather still `0x00C0`.

`GoToSleepTask` left the barn. Frame 335 is farm tilemap 0 at (133, 372). Frame 857 is house tilemap 21. Frame 1362: day 17, hour still 18, house (49, 86), happiness 1, status 9, map 39, fed flags still `0x0002`, weather `0x00C0`. Frame 1404: hour 6, minute 0, day 17, weekday 3, happiness 1. Slot 0 status stayed 0 and its happiness stayed 0. The only slot-1 happiness in the log is 1, so the night delta is 0. The cow was already an adult; the 21 child nights were not re-run.
