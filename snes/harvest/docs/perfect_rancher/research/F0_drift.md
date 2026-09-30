# F0 end-of-game drift

Date: 2026-09-29. Bead PR-009. Deps PR-007 is already on disk in `whatif_pins.md` (`fails=0`).

Blocked on a plan contradiction. MASTER_PLAN F0 and `state_presets/ranch_master_999.json` say the final sleep adds 2 to Maria, and the preset stores 509 so that add would land on 511. The live ROM subtracts 2. Opcode `41 1F 1F 7F FE FF` at `$B5805A` is a signed add of −2, clamped to 0..999 (`Event_UNKCCchanges14`, `bank_84.asm` `$84B5A9`, compare `#$03E7`). On `latest_y3_ending_ranch_master_999` Maria moved 509 → 507 at frame 1132, still Y3 Summer D29, and stayed 507 through ending index `0x25`. Both 507 and 511 score `(hearts & 0x1FF) >> 4 = 31`, so this pin's `ranch_mastery` stayed 999. The sign is still wrong in the plan, and a wife at 496 would land at 494 and lose the point.

Probe JSON: `research/F0_drift_probe.json`. No new save state. No `LiveRamEditor`. No `--ram-set`.

## Pins and frames

Both runs are one process, `GoToSleepTask` then A pulses, `--trials 1`, `HEADLESS=1`, dummy video and audio, cwd `snes/harvest`. Natural entry for this bead is the named pin.

| Pin | Load | Maria −2 | Sleep done | Index `0x20` | Index `0x25` | RAM score at `0x20` / `0x25` | Oracle at `0x20` |
|---|---|---|---|---|---|---|---|
| `LN_ending_Y3S29_eve` | D29 19:00, tilemap `0x15`, Maria 0, dev byte 0, tiles 80 | none | 474 | 5609 | 5996 | 0 / 1 | 1 |
| `latest_y3_ending_ranch_master_999` | D29 17:04, tilemap `0x17`, Maria 509, dev byte 100, tiles 2965 | frame 1132, 509 → 507 | 1923 | 35487 | 35894 | 999 / 999 | 999 |

`GoToSleepTask` (`home_sleep.py`) will not take the bed before hour 18. The house clock is frozen, so from 17:04 the task walks outside, waits until hour 18 (frame 1069, tilemap 0), then re-enters the house (frame 1089, tilemap `0x17`). The −2 is 43 frames after that re-entry, before the day rolls (frame 1816) and before NightReset sets hour 6 (frame 1858). The eve pin is already hour 19 indoors, sleeps without that re-entry, and Maria stays 0. There is no +2 on either pin.

Index `0x20` is the frame before `RanchMasteryCalculator` stores the sum. The eve pin still has RAM score 0 there and oracle 1; the store lands at index `0x25` (RAM score 1). The 999 pin already held 999, and the inputs at `0x20` oracle to 999.

## Maria −2

House tilemaps `0x15`, `0x16`, and `0x17` are indexes 21–23 of `UNK_TableFunctionSelect` (`bank_83.asm` `$83CAC9`) and all three enter `CODE_83DED3`. On Y3 Summer D30 (`year == 2`, `season == 1`, `day == 0x1E`) that entry jumps straight to `CODE_83F4D8` and does not load the wife. On any earlier day it falls through to `CODE_83E144`.

`CODE_83E144` picks the first set wife bit in `$7F1F66` (Maria `0x0001`, then Ann, Nina, Ellen, Eve). Hearts `>= 200` (`#$00C8`) go to `CODE_83E210`. That loader skips the script when `$7F1F6E` bit `0x0002` is set. Otherwise it loads CC group `0x43`, event 0–4 for that wife (`LDX #$0043`, `UNK_LoadCCDataLong`). The 999 pin has `marriage_flags` `0x1E1F` (all five wife bits) and `incubator_flags` `0x5C0C` (bit `0x0002` clear), so Maria wins and the others are not loaded. Ann, Nina, Ellen, and Eve stayed 509.

Group `0x43` event 0 starts at `$B58020` (`DATA8_B58000` in `bank_B5.asm`). Live ROM bytes:

- `$B58020` `JumpifFlagSet` `$0196` bit index 1 → `$8036`
- `$B58027` `JumpifEquals` `season == 3` → `$8036`
- `$B5802E` `JumpifBetween` `hour` in `[0x0B, 0x0E]` inclusive → `$84D4` (set `$7F1F5E` bit 12, then disable). `JumpifBetween` (`$848DFB`) increments the high bound and uses `BCS`, so the written high byte is included.
- `$B58036` `JumpifEquals` `hour == 0x12` → `$805A`
- `$B5805A` `41 1F 1F 7F FE FF` adds −2 to `hearts_maria` (`$7F1F1F`)

Hour `0x12` is 18 (`ram.asm`: `!hour = $7F1F1C`, military time in hex). It is outside `[0x0B, 0x0E]`, so a script that starts at hour 18 always reaches the −2. The following bytes set `$7F1F5E` bit 12 and jump to `$8096`. They do not fall through to the later +2.

A `+2` does exist in the same event, at `$B584C4` (`41 1F 1F 7F 02 00`), then a jump to `$8096`. Nothing in `$B58020`–`$B584DA` has destination `$84C4`. It is the fall-through of the branch at `$B58087`, which jumps to `$84A0` only when `$7F1F5E` bit 11 is set, and `$B58087` sits on the path taken when hour is not 18. The hour-18 edge used by a legal sleep never executes it. Ann's copy of the −2 is `$B58514` (`41 21 1F 7F FE FF`) in event 1, reached by `hour == 0x12` → `$8514`. F9 already listed the five hour-18 sites.

D30 morning does not run this loader, because `CODE_83DED3` has already jumped to `CODE_83F4D8`. The probe shows no second heart write after the day roll.

## D29 6:00 event `0A/5`

`CODE_83E340` (`$83E340`) loads group `0x0A` event 5 when `year == 2`, `season == 1`, `day == 0x1D`, and the clock is 6:00:00. `Event0A` at `$B3E83B` points event 5 at `$B3E9AE`. The live script is `02 08 09 01 B6 E9 38 10` then the spawned actor at `$B3E9B6`: positions, text `0x047A` mode 1, and `37 11 10`. There is no opcode `0x41` and no wallet, heart, crop, happiness, or development store. An evening pin is already past that morning. The sleep rolls the day to 30, so `0A/5` does not run again. Neither probe changed a score input at D30 6:00.

## NightReset (`$8282AC`)

Writers that the score actually sees, from `bank_82.asm`:

- Weekday `+1` mod 7. Day `+1`. Day `0x1F` wraps the season; season 4 wraps the year and calls `ClimateFarmDamageCheck`. D29 → D30 does not wrap the season. Both probes: weekday 5 → 6, day 29 → 30.
- Clock becomes 6:00:00. This is after the −2, which already happened at hour 18.
- `AddMoney` of the 24-bit shipping pile (`$7F1F07`), then the pile is zeroed. Both pins had pile 0 at the bed, and `money_raw` stayed 30 and 997330. The 999 pin's pile later reads 5 during ending scene 4 (frame 4192). That write is inside the credits, the wallet does not change, and the calculator does not read the pile.
- Shipped counters `$7F1F4A`–`$7F1F50` have no store here. Corn, tomatoes, turnips, and potatoes stayed 0 on the eve pin and 511 on the 999 pin.
- Hearts have no store here. `WifePregnanacyandChilds` (`$828790`) reads them.
- `max_stamina` is copied into current stamina, then `ChangeStamina` of +20. The score reads `max_stamina`. Eve stayed 100. The 999 pin stayed 200 (current stamina 100 → 200). `power_berry_count` stayed 10. The display byte `power_berries` bounces during credits (10 → 4 and other values) and is not the stamina term.
- Fed cow and chicken counts and flags are zeroed after `CowFeedingandStatus`, so the night sees today's fed flags. The 999 pin fed all 12 (`fed_*_flags` 4095 → 0).
- `$7F1F5A`, `$7F1F5C`, `$7F1F5E`, `$7F1F60`, `$7F1F62`, `$7F1F74`–`$7F1F78` are zeroed. `$7F1F68` is `ORA #$0001`. On the 999 pin `$7F1F68` went `3583 → 5631` (`0x0DFF → 0x15FF`, bit `0x0800`) at frame 1874, the same frame as the pregnancy tick. That word is not a calculator input.
- `$7F1F64` keeps the house bits. `AND #$FFCF` clears bits `0x0010` and `0x0020` only. Upgrade flags stayed `0x1BC0` (house `0x40` and `0x80` still set).
- `$7F1F66` bits `0x001F` stay. The 999 pin stayed `0x1E1F`. A set bit `0x0020` is moved to `0x0040`. Bit `0x0080` is the upgrade timer: `development_rate` increments, and at 4 the matching house bit is set. `development_rate` on the 999 pin stayed 100, and the house bits did not change.
- `$7F1F6C` bit `0x1000` (turtle) is outside the masks this night clears. Family flags stayed `0x1060`.
- `$7F1F6E` child bits `0x0004` / `0x0008` and clock bit `0x4000` stayed. Incubator flags stayed `0x5C0C`.

`WifePregnanacyandChilds` clears `$7F1F6C` bits `0x0080` and `0x0100` only (`AND #$FC7F`). If any wife bit is set it increments `wife_pregnancy`. Birth still requires `wife_pregnancy >= 20`, house bit `$7F1F64 & 0x0080`, and hearts `>= 450` (child 1) or `>= 650` (child 2, and `kid1_age >= 90`). It sets `$7F1F6E` bit `0x0004` then `0x0008`. It does not write hearts. Each night a set child bit increments that child's age. Age landing on exactly 60 adds +50 happiness (child 1) or +100 (child 2) through `AddPlayerHappiness`. The 999 pin moved `wife_pregnancy` 0 → 1 and both kid ages 180 → 181. Happiness stayed 999. The eve pin is unmarried (`marriage_flags` 0) and happiness stayed 10.

`CowFeedingandStatus` (`$83BC5A`) leaves a fed adult in the barn on the `.cowfed` path with no happiness add. Unfed is `−8` (`#$FFF8`) plus a sickness roll. Weather outsides use `−16`, `−24`, or `−8`. A new sickness also adds `−12` cow happiness and `AddPlayerHappiness #$FFE2` (−30). A cranky byte (`slot +5 == 0x0A`) subtracts 30 cow happiness (`#$FFE2`) and does not call `AddPlayerHappiness`. The 999 herd stayed happiness 255, status 9, 25 points each, sum 300. Slot byte `+1` went 7 → 0 on all 12 cows (the daily brushed / milked / talked bits `0x01|0x02|0x04`). That byte is not the score. The eve pin has no cows. Happiness did not move on either pin, so the −30 sickness sink did not fire. It remains a conditional night sink when a cow gets sick.

`NightlyFarmTilesCheck` (`$82A811`) walks `!farm_map_array` (`$7EA4E6`). Grass `0x1D` takes the skip path. `CalculateFarmDevelopment` (`$82AA0C`) counts the same tiles the oracle counts: below `0xA0`, and fence `0x05` / `0x06` or `>= 0x1D`. Counted tiles were unchanged overnight: eve 80, 999 pin 2965. A season wrap or storm is a different night. This summer D29 → D30 pair did not move the count.

## `CODE_83F4D8` (`$83F4D8`)

Runs from `CODE_83DED3` when the loaded map function sees Y3 Summer D30. It loads ending group `0x2E` event 0, sets `$7F1F5E` bit `0x0002`, zeros the ending stepper `$7F1F47`, picks a cow name into the name buffer, then:

1. `CalculateFarmDevelopment` stores the raw tile count in `ranch_development` (`$7F1F56`). The probe watches that byte climb (eve through 80, 999 pin through 2965) across consecutive frames.
2. `Unk_Audio23` (`$838000`) multiplies A by `$7E`. Here A is `#$000A` and `$7E` is the tile count.
3. `DivisionUnsigned` (`$838082`) divides `$7E` by `$80`. `$80` is `#$0127` (295). The quotient returns in A. The remainder overwrites `$7E`.
4. The quotient is stored back to `ranch_development`, capped at `#$0064` (100).

So the byte the calculator later shifts is `min(100, tiles * 10 / 295)`, not the evening byte. Eve: stored byte 0, 80 tiles, `80 * 10 // 295 = 2`. The morning write is 2, and the development term becomes `2 >> 1 = 1`. Oracle total goes 0 → 1, and RAM score becomes 1 at index `0x25`. The 999 pin: 2965 tiles, `2965 * 10 // 295 = 100`, byte stays 100, development term stays 50.

The ending script itself (`group 0x2E` event 0, bank B4 prefix at `$B4C614`) sets hour 7 and branches on the wife bits. It has no opcode `0x41` in the prefix that runs before the scene index advances. Heart and crop values at `0x20` match the post-drift inputs above.

## Per-term drift

Measured on this one summer night, then applied by the writers above. "Evening" means the state before the hour-18 house re-entry. "At `0x20`" is what the calculator reads.

| Term | Evening → `0x20` on these pins | Rule for the ledger |
|---|---|---|
| Money | 30 → 30 and 997330 → 997330. Pile was 0 at the bed. | `money = money + shipping_pile`, then pile 0. Score `min(78, (money & 0xFFFF) >> 7)`. |
| Cows / chickens | 0 and 12, unchanged. | Count unchanged unless a cow actually dies on the sick path. These pins did not. |
| Cow happiness | 12 × 255, 25 points, sum 300. Daily flags in slot `+1` cleared. | Fed adult in the barn: +0. Otherwise the penalties in `CowFeedingandStatus`. Points `min(25, happiness >> 3)` per existing cow. |
| Stamina | `max_stamina` 100 and 200, `power_berry_count` 0 and 10. | No night writer. Score `(max_stamina - 100) >> 1`. |
| Girls | Maria 509 → 507. Ann, Nina, Ellen, Eve stayed 509. Eve pin stayed 0. | The loaded wife, and only that wife, `−2` when the hour-18 house entry runs group `0x43` with `$7F1F6E` bit `0x0002` clear and her hearts `>= 200`. Clamp 0..999. Other girls +0. Score `(hearts & 0x1FF) >> 4`. |
| Crops | 0 and 511, all four, unchanged. | No writer in NightReset, `0A/5`, or `CODE_83F4D8`. |
| Happiness | 10 → 10 and 999 → 999. | +0 on this path. `−30` if a cow sickness roll fires. `+50` / `+100` if a child's age lands on 60. Cap 999. Score `happiness >> 5` capped at 999. |
| House | `0x1BC0` unchanged. Eve 0. | Unchanged unless the upgrade timer (`$7F1F66` bit `0x0080`, `development_rate` hitting 4) sets `0x40` or `0x80` the same night. |
| Marriage | `0x1E1F` unchanged. | Bits `0x001F` stay. |
| Children / clock | `0x5C0C` unchanged. Ages 180 → 181, pregnancy 0 → 1. | Flag bits unchanged unless this night is the birth night (`wife_pregnancy >= 20` and the heart and house gates). Ages and the pregnancy counter are not terms. |
| Turtle | `0x1060` unchanged. | Bit `0x1000` survives this night. |
| Development | Eve byte 0 → 2 from 80 tiles. 999 byte 100 → 100 from 2965 tiles. | Replace the byte with `min(100, post-night_tiles * 10 // 295)`. Score is that byte `>> 1`. 2950 tiles is the first count that stores 100. |

## Ledger projection

For a D29 evening state `S`, after the last farm action and before bed:

```
wife = first set bit of S.marriage_flags in (Maria, Ann, Nina, Ellen, Eve)
hearts' = S.hearts
if the sleep re-enters house 0x15/0x16/0x17 while hour is 0x12
   and S.incubator_flags & 0x0002 == 0
   and wife hearts >= 200:
    hearts'[wife] = clamp(hearts'[wife] - 2, 0, 999)
money' = S.money_raw + S.shipping_money_raw
tiles' = counted development tiles after NightlyFarmTilesCheck
dev' = min(100, tiles' * 10 // 295)
```

Happiness and each cow's happiness byte change only through the conditional sinks above. On a fed, in-barn, already-born herd whose child ages are not 59, both stay put. House, marriage, child, clock, and turtle bits stay put unless that same night is a birth or a house-upgrade completion.

`D30 score = evaluate_ranch_mastery_score` of those inputs, which is the sum the calculator writes at index `0x25`, capped at 999. Read the inputs on the first frame whose ending index is `0x20`. The RAM score cell can still hold the pre-ending value on that frame. The eve pin is the witness: oracle 1 at `0x20`, RAM score 1 at `0x25`.

Wife window that still scores 31 after the −2: evening hearts 498–511, which land on 496–509. 496 and 511 both score 31 before the drift; 496 does not survive it (`494 >> 4 = 30`). The other four girls, and all four crops, keep the evening window 496–511, because this night does not write them. The 999 pin's Maria 509 → 507 stays inside the window and the total stays 999:

`78 + 36 + 36 + 50 + 31×5 + 31×4 + 31 + 32 + 32 + 32 + 43 + 50 + 300 = 999`.
