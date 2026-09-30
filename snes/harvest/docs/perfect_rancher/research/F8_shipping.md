# F8 shipped counts

Date: 2026-09-29. Bead PR-017. USA decomp under `HM-Decomp`, plus live drops from `Y1_After_Ship_Berry` on research pins `LN_mgr-ship_whatif_*`. Those pins are RAM what-ifs, not route evidence. The live results match MASTER_PLAN F8: the crop counter increments once per accepted item when `$7F1F5E` bit 0 is set.

## Addresses

`ram.asm`: `shipped_corn` `$7F1F4A`, `shipped_tomatoes` `$7F1F4C`, `shipped_turnips` `$7F1F4E`, `shipped_potatoes` `$7F1F50`. Shipping pile is the 24-bit raw value `shipping_moneyL/H` `$7F1F07` / `$7F1F09`. Wallet is `$7F1F04` / `$7F1F06`. Both money fields are stored in 10G units (`ram.asm`: "Money is actually x10").

`RanchMasteryCalculator` (`bank_83.asm` `$83EE19` onward) only reads each counter: `AND #$01FF`, four `LSR`, add into `ranch_mastery`. The flavor checks at `$83EDF6`, `$83EE19`, `$83EE3C`, and `$83EE5F` compare corn, tomato, turnip, and potato to `#$00C8` (200) and do not store them.

## Bit 0 around `$818140`

`SUB_81809A` (`bank_81.asm` `$81809A`) takes the item id in `$0984`, indexes `DATA8_8197C0`, JSRs the handler, then `AND #$FFFE` on `$7F1F5E` and RTS. The increment runs inside the JSR. The clear runs before the subroutine returns, so a later frame read sees bit 0 clear. Potato 3 below incremented with `flag_1f5e` 0 on both the before and after snap.

Jump table `DATA8_8197C0` at `$8197E0` (`32 81 4A 81 62 81 7A 81 92 81 96 81 9A 81 9E 81 …`) sends items 16–25 to:

| Item | Handler | Store |
|---|---|---|
| 16 `$10` corn | `$818132` | `INC` `shipped_corn` if bit 0 (`$818140`) |
| 17 `$11` tomato | `$81814A` | `INC` `shipped_tomatoes` |
| 18 `$12` potato | `$818162` | `INC` `shipped_potatoes` |
| 19 `$13` turnip | `$81817A` | `INC` `shipped_turnips` |
| 20–25 `$14`–`$19` | `$818192`–`$8181A6` | `JSR CODE_8188E5` / `RTS` |

Each crop `INC` adds 1. There is no cap and no stack-size add. The other six handlers return without writing the four counters. Egg is item 20. Milk S/M/L are items 21–23 (`$15`–`$17`).

`CODE_8188E5` (`$8188E5`) dispatches on `$0976`. State 0 (`CODE_819010`) stores 2 and returns. State 3 (`CODE_819379`) is the bin: tilemap `< 4`, or barn `$27`, or coop `$28`, and the tile 32px in front (`CalculateTileinFront` with A=2) has property `$00F2`, then `Dropedonsaleplace` (`$8193BF`).

`Dropedonsaleplace` reads the odd byte of `Items_Price_Table` (`$819FDE`, X = item×2+1). A zero price falls through to `Droped_on_special_place`. A nonzero price and `hour < 17` adds that byte into `shipping_money` and `ORA #$0001` onto `$7F1F5E` (source comment: "Things in the shipping bin"). Hour 17 and later skips the add and the bit. Either way the handler then `STZ item_on_hand`. The crop `INC` sees the bit in that same call, so the count is written at the drop.

The second `ORA #$0001` is `CODE_819195` (`$819195`). A drop with `player_action == 5` and `CBBBB` nonzero jumps there from `CODE_819049`. It sets the bit, and adds the same price byte into `shipping_money`, only when `$7F1F5C & #$0010` is already set, the price byte is nonzero, and `hour < 17`. It also `ORA #$0040` onto `$7F1F5C` and stores `$0976 = 8`. With bit `$0010` clear it still stores state 8 and skips the money and the count bit.

`Event_UNKPlayerPos` (`bank_84.asm` `$84AB27`) clears `$7F1F5C` bit `$0010` on entry (`AND #$FFEF` at `$84AB35`). It sets that bit again at `$84AC51` only on the branch where `horse_age` `$7F1F32 == #$15`. `HORSE_ADULT_AGE` in `harvest_state.py` is `0x15`. `HORSE_SADDLEBAG_SCENE_COMPLETE_FLAG` is upgrade-flag bit `0x0100`, a different word. This path has no item array and no night dump: the `INC` is the same handler call as the collision.

## Prices

Odd bytes of `Items_Price_Table` for items 16–23: 12, 10, 8, 6, 5, 15, 25, 35. One potato therefore adds 8 raw (80G) to the shipping pile. Milk S adds 15 raw (150G) on a path that sets bit 0.

Peddler records are `DATA8_81A094`, three bytes each (text lo, text hi, price), index item×3. Items 16–23 at `$81A0C4`: prices `$18,$14,$10,$0C,$0A,$14,$1E,$28` (24, 20, 16, 12, 10, 20, 30, 40 raw). `CODE_818F5A` (`$818F5A`) loads that price byte into `AddMoney` (`$83B1C9`) and `STZ item_on_hand`. It does not write `$7F1F5E`. The dialog opens only from `CODE_819049` when tilemap is `$04`–`$07`, `weekday == 0` (`BEQ` at `$8190EB`; `WEEKDAY_NAMES` maps 0 to Sunday), `hour < $11`, weather `$0196` bit `$0002` is clear, the front tile property is `$00F0`, and the text id is nonzero.

## Gameplay writers

`STA.L` on the four `shipped_*` symbols appears in three places: new-game zeros (`bank_83.asm` around line 4621), the SRAM load (`bank_83.asm` around line 5827), and the four `INC` handlers above. `NightReset` (`bank_82.asm` `$8282AC`) `AddMoney`s the shipping pile into the wallet, stores 0 to `shipping_money`, and stores 0 across `$7F1F5A`, `$7F1F5C`, `$7F1F5E`, `$7F1F60`, `$7F1F62`, `$7F1F74`–`$7F1F78`. It does not address `$7F1F4A`–`$7F1F50`. `ShippingScene` (`$828131`) runs when `hour == 17` on a farm map and opens the daily shipping text. It does not increment the counters.

## Live bin drops

Base pin `LN_mgr-ship_whatif_potato_bin`, minted from `Y1_After_Ship_Berry`. Boot (`/tmp/f8_ship_probe.log` frame 30): tilemap 0, weekday 2 (Tuesday), season 0, year 0, day 2, hour 10 minute 5, xy (146, 457), tile (9, 28), held `$12`, horse_age 0, `$7F1F5C` 0, `$7F1F5E` 0, counters 0, `ship_raw` 15, `money_raw` 30. Feet on the north lip of metatile `$F2`, facing down, is the bin. Product stands (11, 30) and (8, 32) are walled off; both walks returned `arrived=False` at frame 30.

Three separate down-facing drops on that lip, each a fresh boot, hour still 10:

| Drop | Log | Frames | Potatoes | `ship_raw` | Wallet raw |
|---|---|---|---|---|---|
| 1 | `/tmp/f8_ship_probe3.log` | 77 | 0→1 | 15→23 | 30 |
| 2 | same | 77 | 1→2 | 23→31 | 30 |
| 3 | `/tmp/f8_ship_probe4.log`, JSON `bin_potato_3` | 65 (snaps at frames 32 and 65) | 2→3 | 31→39 | 30 |

Corn, tomato, and turnip stayed 0. Each drop is +1 potato and +8 shipping raw. N drops add N, in the drop, with the wallet unchanged. A third attempt from y=461 (`f8_ship_probe3.log` frame 74) consumed the potato and added 0. A throw aimed at stand (8, 32) facing up (`f8_ship_probe.log` `[DROP-SEEK]` frame 103) consumed the potato from (146, 457) and added 0.

The same session then sat through the clock. `/tmp/f8_ship_probe4.log`: frame 6000 is 17:00 with potatoes 3, `ship_raw` 39, `money_raw` 30; frame 9000 is 18:04 with the same three numbers. JSON `sleep.end` is frame 17924, xy (136, 176), hour 18 minute 2, day 2, potatoes 3, `ship_raw` 39, `money_raw` 30. `GoToSleepTask` returned `TaskStatus.FAILURE` "could not find house (tilemap=0x00)". `NightReset` did not run. The 17:00 scene left the counter and the wallet where the drop had put them.

## Peddler, horse, milk

The live pins stay on tilemap 0, weekday 2, `horse_age` 0, `$7F1F5C` 0. The Sunday town gate and the adult-horse bit were not opened. On those pins the peddler dialog and `CODE_819195` cannot set bit 0. The peddler handler pays the wallet through `AddMoney` and leaves the four counters alone. The horse path, when `horse_age == $15` and bit `$0010` is set, uses the same per-item `INC` as the bin, at the collision, and puts the price in `shipping_money`.

Milk S (item `$15`, held 21) was thrown twice. JSON `bin_milk_s`: y=185, facing down, frames 29→62, held 21→0, all four counters unchanged, `ship_raw` 39, wallet 30. The Y coordinate on that pin had been stored as the low byte only, so the farmer was off the `$F2` block. JSON `bin_milk_s_at_lip` (pin `LN_mgr-ship_whatif_milk_lip`, 58 frames): (150, 456) facing up (`dir` 1), held 21→0, same zero deltas. Facing up throws away from the bin. Neither throw is a `$F2` sale. Items 21–23 have no `INC`. A real bin hit would still add milk S's price byte 15 to `shipping_money` inside `Dropedonsaleplace`, then return from `$818196` without touching the four counters. That on-tile milk drop was not measured.

## Pickup table note

`SUB_82AD0E` indexes `DATA8_82CFB4` as `tile×4 + season`. Tile `$60` is item `$12` (potato, the measured handler). Tiles `$6C`/`$6D` are item `$13` (the turnip `INC`). Tiles `$6E`/`$6F` are items `$44`/`$45`. In fall and winter (`season == 2 or 3`) the same routine jumps to `.CODE_82AEE5` on `$6E`/`$6F` before the table. `MATURE_CROP_TILES` in `tile_catalog.py` labels `$6E`/`$6F` as turnip. That harvest path was not run. The counter write stays the item-19 handler.

## Probe files

JSON: `snes/harvest/docs/perfect_rancher/research/F8_shipping_probe.json`. Pins under `snes/harvest/custom_integrations/HarvestMoon-Snes/`: `LN_mgr-ship_whatif_potato_bin`, `_p2`, `_p3`, `_ground`, `_milk`, `_milk_lip`. Commands from `snes/harvest` with `HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen`, `--trials 1`, `setsid nohup uv run python`. Logs: `/tmp/f8_ship_probe.log`, `/tmp/f8_ship_probe3.log`, `/tmp/f8_ship_probe4.log`, `/tmp/f8_ship_probe5.log`.
