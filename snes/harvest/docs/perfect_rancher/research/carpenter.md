# Carpenter house upgrades

Date: 2026-09-29. Bead PR-061. USA decomp under `HM-Decomp`, plus one paid order on research pin `LN_mgr-home_whatif_upgrade1`. That pin is a RAM what-if, not route evidence. The live order matches the upgrade-1 branch: text `0x134`, then `0x135`, −5000G, −250 wood, `$7F1F66` bit `0x0080` set, `$7F1F64` still 0.

Native arrival verified 2026-10-15: the power-on predecessor earned 252 wood
and 6,200G, ordered on Summer D28, and completed on Fall D2 after exactly
four sleeps. `recordings/house_marriage_resources.json` records the
5,000G/250-wood debit; `recordings/house_build_nights.json` records counters
0,1,2,3,4 and the house bit appearing only at 4. All segments have zero RAM
writes and zero mid-run loads. Checkpoint lineage and the subsequent Sunday
sale/wedding are in [EVE_BAR.md](../../EVE_BAR.md).

House score is term 8. `ranch_master.py` adds 16 for `$7F1F64 & 0x0040` and 16 for `& 0x0080`, max 32. `npc_catalog.py` labels those bits `house_upgraded_once` and `house_level_2`. The order bits live on `$7F1F66`, catalog field `marriage_flags` at file offset `0x11F66`.

## Addresses

`ram.asm`: `stored_wood` `$7F1F0C`, `wood_need_for_upgrade` `$7F1F0E`, `moneyL` `$7F1F04` ("Money is actually x10"), `development_rate` `$7F1F35` with the source comment "Also used too keep track of house upgrade days", `ranch_development` `$7F1F56`. The build counter is `$7F1F35`. It is not the credits percentage.

`house_size` in `ram_catalog.py` is `$0970`. The order and completion handlers `INC` it to `$20` and then `STZ` it. Mid-run values are a dialogue step, not remodel level.

`AddMoney` is `$83B1C9`. A negative 24-bit add that would cross zero returns A=1 and stores nothing. `AddWood` is `$83B224`. It adds a signed 16-bit delta, fails on a negative result, and caps `stored_wood` at 999. `GetFenceFromStack` (`bank_81.asm` `$81D644`) passes `#$FFFE` (−2). That is fence lumber. PR-062 owns the chop pipeline.

`FirstNightReset` (`$83ABF0`) stores 250 into `wood_need_for_upgrade`. If `$7F1F64 & 0x0040` is already set, it stores 500 instead.

## Sign dialog

Two copies of the same script, `bank_B3.asm` at `$B3AD65` and `$B3CC25`. Opcodes from `bank_84.asm`: `0x14` `Event_JumpifFlagSet` (`$848D44`), `0x1C` `Event_StartTextBox`, `0x1D` `Event_StartTextBoxCopy`, `0x20` `Event_UNKJumpif018F` (`$84919D`, compare one byte with `$018F`), `0x23` `Event_SetFlag` (`$849306`). `Events_Flags_Table` (`$8494D3`) maps bit index 5 to mask `0x0020`, 6 to `0x0040`, 7 to `0x0080`.

`$B3AD65`, and the same tests at `$B3CC25`:

| Test | Taken |
|---|---|
| `$7F1F64` bit 7 (`0x0080`, house max) | text `0x0133` |
| `$7F1F66` bit 5, 6, or 7 (`0x0020`, `0x0040`, or `0x0080`) | text `0x0139` |
| else | text `0x0130` |
| `$018F == 0` (Yes) | `SetFlag` `$7F1F66` bit 5, then text `0x0131` |
| `$018F == 1` (No) | text `0x0132` |

Pointer table `bank_83.asm` and `UnlinkedText.txt`:

| Id | Line |
|---|---|
| `0x128` | "We are carpenters. Please come to us when you want to enlarge your house." |
| `0x129` | Call when you have enough wood. "There's a sign by the house." |
| `0x12A` | Extension campaign. Enlarge by the end of summer for a free gift. |
| `0x12E` | Chop logs on the mountain. |
| `0x12F` | Materials also go to fences. |
| `0x130` | "Do you want to enlarge your house?" Yes / No |
| `0x131` | "OK. I'll be at the ranch tomorrow." |
| `0x132` | "Let me know whenever you want to do it." |
| `0x133` | "How do you like living in the Super Deluxe Great Log House?" |
| `0x139` | "Hee-hee, look forward to it." |

`calendar_schedule.py` records intro text `0x128` and that map `0x2B` is not a carpenter door. This probe did not walk the sign. The pin pre-set `$7F1F66` bit `0x0040`, so the next farm load started the paid order.

`NightReset` (`bank_82.asm` `$8282AC`, the block at lines 517–526) sees bit `0x0020`, clears it, and sets bit `0x0040`. That is the overnight conversion from "Yes on the sign" to "visit on the next farm load."

## Paid order

Farm maps are tilemap indices 0–3. `UNK_TableFunctionSelect` (`$83CAA9`) points those four slots at `UNK_FuncInTable` `$83CB69`. The carpenter check is `$83CFF8`, after the dog and intro gates at `$83CE5B`.

`$83CE5B` returns early if `$7F1F5A & 0x0002`, or if `$7F1F68` lacks bit `0x0001`, `0x0020`, or `0x0080`. Later 6:00:00 returns also sit in front of `$83CFF8`: festival when `weather_tomorrow >= 6`, `$7F1F70` bit `0x0002`, `$7F1F64` bit `0x0010`, and Y1 summer days under `0x14` (`$83CFA4`, which sets `$7F1F70` bit `0x0001`). Hour, minute, or second other than 6:00:00 jumps to `$83CE5B` from `$83CC53` and still reaches the carpenter check. Wife bits `0x0001` through `0x0010` on `$7F1F66` take that same jump. The check at `$83CFF8` does not read the hour.

`$83CFF8` if `$7F1F66 & 0x0040`: load CC `(9, 0, 0x2C)`, store player action `0x47` in `$096E`, zero `$096F` and `$0970`, set `game_state` bit `0x0040`, clear `player_action`, clear `$7F1F66` bit `0x0040`, RTS.

`PlayerActionPointerTable` is `$81D593`. The entry at `$81D621` is `EMPTY_00F121`, index `(0x81D621 − 0x81D593) / 2 = 0x47`. The handler bytes start at `$81F121`. The next entry, `$81D623`, is action `0x48` at `$81F2DA`. Action `0x49` at `$81F40E` opens text `0x300` and is the horse-saddlebag scene.

`$81F14C` runs once `$0970` hits `0x20`:

1. If `year == 0` and `season < 2` (Y1 spring or summer), `ORA #$0004` onto `$7F1F70`. This runs before the wood test and before Yes/No.
2. `STZ $0970`.
3. If `wood_need == 0x00FA` (250): `stored_wood < 250` goes to text `0x137`, else text `0x134`.
4. If `wood_need != 250`: `stored_wood < 0x01F4` (500) goes to text `0x137`, else text `0x13B`. The 500 compare is constant. The later wood subtract uses the live `wood_need` value.
5. Yes is `$018F == 0`. Upgrade 1 (`wood_need == 250`) subtracts `0xFFFE0C` (−500 raw, −5000G). Any other `wood_need` subtracts `0xFFFC18` (−1000 raw, −10000G). `AddMoney` A=1 opens text `0x304` and skips the purchase. Success calls `AddWood` with the negated `wood_need`, opens text `0x135`, stores 0 to `$7F1F35`, and `ORA #$0080` onto `$7F1F66`.
6. No opens text `0x136`.
7. After text `0x137` is dismissed, the handler loads CC `(9, 0, 0x2D)`.

| Id | Line | When |
|---|---|---|
| `0x134` | "It costs 5000G to enlarge your house." Yes / No | upgrade 1, wood at least 250 |
| `0x13B` | "It costs 10000G to enlarge your house." Yes / No | `wood_need != 250`, wood at least 500 |
| `0x135` | "Alright. We'll start off tomorrow." | payment accepted |
| `0x136` | "Oh... OK. Call me again any time." | No on the cost menu |
| `0x137` | Not enough materials. Check the sign. | wood short |
| `0x304` | "You don't have enough money." | `AddMoney` would go negative |

While bit `0x0080` is set and `$7F1F35 != 0`, the next farm load plays CC `(0, 0, 0x2E)` and returns. That is the work cutscene. The accept path leaves `$7F1F35` at 0, so the cutscene waits until a later morning.

## Build nights

`NightReset` increments the day first (`$8282AC`, day `INC` at line 364). Later, if `$7F1F66 & 0x0080`, it increments `$7F1F35` (`bank_82.asm` lines 530–562). When that byte becomes 4:

- `$7F1F66 |= 0x0100`
- `wood_need_for_upgrade = 500`
- `$7F1F66` bit `0x0080` clears
- if `$7F1F64 & 0x0040` already set, `$7F1F64 |= 0x0080`, else `$7F1F64 |= 0x0040`

The second completion ORs `0x0080` and leaves `0x0040` set, so both score bits stay on. The same night also `AND #$FFCF` on `$7F1F64` (line 407), which clears bits `0x0010` and `0x0020` only.

Four sleeps after the accept. The morning after the fourth sleep, farm load `$83D060` sees bit `0x0100`, clears it, loads CC `(9, 0, 0x36)` and `(10, 0, 0x37)`, and starts action `0x48`. This probe did not sleep. The 4-night length is the `CMP #$04` in `NightReset`, not a counted run.

Action `0x48` at `$81F302`: if `$7F1F64 & 0x0080` is clear, `AddPlayerHappiness(0x32)` and text `0x13D` ("Super Log House"). If the bit is set, happiness `+0x64` and text `0x140` ("Super Deluxe Great Log House"). NightReset has already written the new house bit, so upgrade 1's morning dialog takes the +50 line and upgrade 2's morning dialog takes the +100 line. Neither completion was run on this pin.

## Clock and the fall strawman

After the upgrade-1 completion text is dismissed, `$81F354` checks `$7F1F64 & 0x0080` again. While that bit is clear and `$7F1F70 & 0x0004` is set, it `ORA.W #$4000` onto `$7F1F6E` (`$81F370`) and opens text `0x25C` ("souvenir Clock", `DATA16_B8E4C8`). Term 11 is that bit, +22. The date is not read again here. The latch is the `ORA #$0004` at order time, and that `ORA` requires year 0 and season under 2.

`ORA.W #$4000` onto `$7F1F6E` appears at `$81F370` only. The other `ORA.W #$4000` sites write different words: `$7F1F6C` at `$81CB08`, `$7F1F66` at `$81EC44`, `$7F1F6A` at `$83CD1C`, `$7F1F68` at `$83D1B7`. Plenty of other instructions `STA.L $7F1F6E`. This pass did not open every one of them.

MASTER_PLAN term 11 still says the clock source is TBD. Section 4 is a strawman and puts house upgrade 1 in Y1 Fall. Fall is season 2, so an order there does not set `$7F1F70` bit `0x0004`, and the completion handler then skips the clock. The upgrade itself still runs. The house bits do not depend on the season. A Fall upgrade-1 leaves the clock term at 0 if `$81F370` is the grant. This file records that date gate. It does not edit the plan.

## Live order

Pin `LN_mgr-home_whatif_upgrade1.state`, minted from `Y1_D3_Morning` before boot. Edits on that new file only: `money_raw` 2000 (20000G), `stored_wood` 300, `wood_need` 250, `$7F1F66 |= 0x0040`, clock 9:00:00, `development_rate` 0, `$7F1F5A` bit `0x0002` cleared, `$7F1F64` left 0. No `LiveRamEditor` and no `--ram-set` during the run.

Command, cwd `snes/harvest`, one emulator, log `/tmp/pr061_carpenter.log`:

`HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen PYTHONPATH=<repo>:<repo>/snes:<repo>/nes:<repo>/snes/harvest setsid nohup snes/harvest/.venv/bin/python /tmp/pr061_carpenter_probe.py --trials 1`

A bare `uv run` under `setsid` did not import `harvest`. The venv binary plus that `PYTHONPATH` did. The run exited after the order.

JSON: `snes/harvest/docs/perfect_rancher/research/carpenter_probe.json`. 1009 frames. Ordered at frame 828.

| Frame | What changed |
|---|---|
| 1 | House tilemap `0x15` at (136, 120), hour 9, `money_raw` 2000, wood 300, `wood_need` 250, `$7F1F66` `0x0040`, `$7F1F64` 0, `$7F1F70` 0 |
| 391 | Farm tilemap 0, action 71 (`0x47`), `$7F1F66` cleared to 0. The visit consumed bit `0x0040`. |
| 527 | Outdoor exit settled, tilemap 0, y 344 |
| 619 | Text 308 (`0x134`), input lock 2, cursor 0. Money still 2000, wood still 300. `$7F1F70` is already 4. |
| 828 | Text 309 (`0x135`). `money_raw` 1500, wood 50, `$7F1F66` 128 (`0x0080`), `$7F1F64` 0, `$7F1F70` 4, `$7F1F35` 0, action 71 |
| 1009 | Lock 1, hour 9 minute 1, same wallet, wood, and flags. `wood_need` still 250. |

Deltas: `money_raw` −500, wood −250. Texts `0x136`, `0x137`, and `0x304` did not open. `$7F1F64` stayed 0 because the house bit is written on the fourth `NightReset`, not on the receipt.
