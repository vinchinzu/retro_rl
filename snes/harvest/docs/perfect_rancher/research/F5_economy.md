# F5 economy

Date: 2026-09-30. Bead PR-014. Deps PR-011, PR-012, PR-013 are on disk as `F2_development.md`, `F3_herd.md`, and `F4_family.md`. F3 has measured dates, so this pass did not stop on PR-012.

No new emulator. Frame count for this pass: 0. Prices below are the ROM shop table, the ship table, the carpenter handler, and the probes those files already recorded. `spring_opt.py` is unchanged. `build_rancher_ledger` in `harvest/planner/livestock_econ.py` reads `Calendar.shop_open` and `CROPS` from it.

Ledger: `research/F5_ledger.json`. 300 mornings, index 0 through 299.

## Audit: model is not a compatible campaign, 2026-09-30

PR-014 is reopened. The 620,000G ledger is retained as a historical paper
model: twelve bought cows, no births or miracle-potion spend. F7 needs
nine namings and F10 needs a born replacement for slot 0. The late wedding
also cannot supply child 2's age-60 happiness bonus before the ending.
Rebuild one dated herd/happiness/cash plan, including actual milk/egg
receipts, slot turnover, 750 lumber, care frames, losses and food repairs.
The model numbers below do not establish a feasible 999 campaign.

## Verdict

Y3 Summer D30 morning (index 299) is **620,000G**. The window is 99,840G–655,350G. 620,000G is inside it. The low 16 bits of the money word are 62,000. `62000 >> 7` is above 78, and 62,000 is below 65,536, so the money term stays at 78.

The path buys the F2 grass cover, the F3 herd and flock, and both F4 house upgrades in time for the Y2 Fall D9 wife bit. From Y3 Spring D16 it ships only enough milk and eggs to hold the wallet at 620,000G. Withheld animal gold is 180,040G. Crop ships are not cut. Lowest balance is 40G, on Y1 Spring D6 after seed buys. No spend crosses zero.

## Price table

Shop buys charge the 24-bit field at `DATA8_81A1A5` byte +9 (`CODE_8189BB`, then `AddMoney`). The index is `shelf_id - $44`. Displayed G is `-raw × 10`. A tool id below `$44` aliases a different row. The miracle-potion row is shelf `$53`, not tool id `$0A`.

| Item | G | Where the number comes from | Bought on this path |
| --- | ---: | --- | ---: |
| House upgrade 1 | 5,000 | `AddMoney` of `$FFFE0C` when `wood_need == 250` (`$81F14C`). Live wallet: `carpenter_probe.json`, pin `LN_mgr-home_whatif_upgrade1`, 1,009 frames, order at frame 828, raw 2000→1500 | Y1 Summer D1 |
| House upgrade 2 | 10,000 | Same handler, `wood_need != 250`, `AddMoney` of `$FFFC18`. ROM immediate. No wallet probe | Y1 Summer D12 |
| Brush | 800 | Shelf `$4C`, price `B0 FF FF`, ORs shed row2 bit `$40` | Y1 Summer D1 |
| Milker | 1,800 | Shelf `$4B`, price `4C FF FF`, ORs shed row2 bit `$20` | Y1 Summer D2 |
| Sprinkler | 2,000 | Shelf `$51`, price `38 FF FF`, ORs shed row3 bit `$10`. Unlock: year ≥ 1, or season ≥ 2, or summer day ≥ `$14` (`$81AF` sprinkler check) | Y1 Summer D20 |
| Grass seed | 500 | Shelf `$44`, price `CE FF FF`. Same 12-byte table as the measured corn, tomato, and potato rows. No new grass wallet probe | 328 bags |
| Cow fodder | 600 | Shelf `$55`, price `C4 FF FF`, `STA feed_cow_N = 12` | 0 |
| Chicken feed | 300 | Shelf `$54`, price `E2 FF FF`, `STA feed_chicks_N = 12` | 188 stacks, 56,400G |
| Miracle potion | 4,000 | Shelf `$53`, price `70 FE FF` (−400 raw), ORs shed row2 bit `$02` | 0 |
| Medicine | 1,000 | Shelf `$52`, price `9C FF FF`, ORs shed row2 bit `$01` | 0 |
| Tomato seed | 300 | Shelf `$46`, price `E2 FF FF`. `buy_seeds.py` names a Summer D13 measurement at this price | 8 bags, Y1 Summer D1 |
| Corn seed | 300 | Shelf `$45`, price `E2 FF FF`. Same Summer D13 note | 8 bags, Y1 Summer D1 |
| Potato seed | 200 | Shelf `$47`. `buy_seeds.py` names this measurement | 56 bags |
| Turnip seed | 200 | Shelf `$48`, same table, price `EC FF FF` | 56 bags |

`F3_herd.md` read tool id `$0A` as shop record 2 and called the potion −300G. Record 2 is the tomato shelf. The handler that ORs the potion bit is shelf `$53` at 4,000G. This path buys cows at 5,000G and does not buy the potion.

Cow payment is `AddMoney` of `$FE0C` with high byte `$FF` (−500 raw, 5,000G) on the animal-shop success path. Chicken payment is `#$FF9C` (−100 raw, 1,000G). Both are ROM immediates. The two F3 buy walks did not reach the counter: try 1 lost tilemap `0x04` at frame 674, try 2 was still on the closed shop at frame 76,000. Those walks were not repeated. Cow feed on this path is cut grass (`AddGrass` from the sickle). The 600G fodder stack is the priced alternative and is not spent. Medicine is priced and not spent: the plan's fed barn nights are the 0-delta night from `F3_herd_stall1.json`, not a sick roll.

Ship prices are the odd byte of `Items_Price_Table` (`$819FDE`), ×10. Item 16 corn 120, 17 tomato 100, 18 potato 80, 19 turnip 60, 20 egg 50, 21–23 milk S/M/L 150/250/350. `CROPS` in `spring_opt` matches the four seed and ship pairs. The builder raises if that match breaks.

New game stores raw 30 (`bank_83.asm` `LDA #30` into `moneyL`). Starting wallet is 300G.

## Milk size and eggs

Milk at `$849E17` stores item `$15` when happiness `< $60` (96), `$16` when `< $C0` (192), and `$17` otherwise. Ship prices 150 / 250 / 350. Happiness 0 still stores `$15`. The writer does not call `AddCowHappiness`. F3's care logs did not record `item_on_hand`, so the size bands are this compare, not a live item id.

F3's measured care day adds 4 before the night (talk +1, brush +3, milk +0, daytime feed +0). A fed adult barn night adds 0 when the stall bit matches (`F3_herd_stall1.json`, frame 1,404, happiness stayed 1). The 21 child nights skip the feed test. Care day 1 is the buy day. Care day 22 is the first adult morning, at 84, if every earlier day was talk and brush.

| Morning | Feedable stall | Slot 0 |
| --- | --- | --- |
| Care days 1–21 | no milk (child) | no milk |
| Care day 22–24 | 84, 88, 92 → S, 150G | 84, then −4 a morning → S |
| Care day 25–48 | 96 through 188 → M, 250G | still falling, S, 150G |
| Care day 49 on | 192, then 200 from care day 51's 06:00 → L, 350G | 0 after the drop, still S, 150G |

The brush that stores 200 is care day 50. That morning is still 196. Milk is taken before talk and brush, matching the F3 chore order, so the morning byte sets the size.

Slot 0's night mask is `$0001` (tile `$E2`). No stall has that id (`F3_herd.md`). It never reaches 200. The first bought cow is slot 0 and ships S milk only. Cows 2–12 use feedable stalls and ship L milk from care day 49.

Eggs: one floor egg per fed adult whose byte +1 is `$28` and whose status bit `$10` is clear (`CODE_83C1FF` ORs a bit into `$7F1F45`). Ship price 50G. A shop hen is `AddNewChicken` with A=2, which overwrites byte +1 with `season+4`. `CODE_818E1B` then stores `tilemap_to_load` into that byte. The plan drops her on the coop the buy day. An incubator egg is A=0: `AddNewChicken` has already stored `$28`, and that arm does not overwrite it. The egg counter hits 3 on the third night and hatches if `chicks_N < 12`. The chick counter hits 7 on the seventh night after that and the same night's lay loop can fire, because the feed loop still saw a chick. The ledger incubates one egg at a time until `chicks_N` is 12. `adult_count` on the ledger is 12. This egg timing is the night loops, not a live egg count.

Chicken feed is a stack of 12, and a buy stores 12 rather than adding. The ledger buys only when tonight's eaters exceed the stock. Sunday is closed (`Calendar.shop_open`). An unfed coop night sets status bit `$10` and drops the lay for the penalty timer. Cow fodder is not that stack.

## Path under F2, F3, and F4

Index: Y1 Spring D1 morning = 0. Y3 Summer D30 morning = 299. Shop days are `Calendar.shop_open` (Sunday closed, festival tuple empty).

F2. `Y1_D2_PowerOn_FarmClear` has 2,968 tiles below `0xA0`, 8 of them already `0x55`. The other 2,960 take spring/summer grass `0x70`. `NewGameSetup` stores `seeds_grass_N = 1`, one free bag, 9 tiles. Paid bags are 328 × 500G = 164,000G. Planted grass on the ledger is 2,961. Stable cover is the spring/summer write. A hurricane on the sleep into Summer D30 can destroy counted grass. That is a development loss. This ledger does not buy replacement bags after the last work morning.

F4. Upgrade 1 is paid on Y1 Summer D1 (index 30), which is year 0 and season 1, so the visit can latch `$7F1F70` bit `$0004` before the wood test (`carpenter.md`). Four nights later the first house bit is on. Upgrade 2 is paid on Y1 Summer D12 (index 41). The level-2 bit is on the morning of index 45 (Y1 Summer D16), before the Y2 Fall D9 wife bit (index 188). Wood is the carpenter's lumber check (250, then 500). It is not a gold line. The ledger pays the gold on those mornings and assumes the wood is already in `stored_wood`.

F3. One animal per open day, after `planted_grass >= 16 × (chicks_N + cow_N + 1)`. Hen on Y1 Summer D1 (1,000G), cows on Y1 Summer D13 through D28 (12 × 5,000G). Cow 1 is slot 0. Cows 2–12 are the feedable stalls. The last of those is at 200 on the Y1 Winter D18 morning (buy index 57, care-day-51 morning index 107), which is before D30. Brush and milker are in hand before the first cow. The hen is incubated up to 12. Chickens past the first cost no purchase gold.

Y2 Spring D5 (index 124) is the morning `ram_map.md` records for the `$83D0FF` visitor. The ROM gate is `ADC #$F448` / `ADC #$FF` / `BMI`: money raw ≥ 3,000, which is 30,000G. That morning's ledger balance before the debit is 200,730G, so the gate passes. The debit used here is the measured −2,000G from that ram_map row. The group `$13` script's own `AddMoney` was not read in this pass.

Score crops, one harvest generation, 504 shipped each (inside 496–511):

| Crop | Bags | Tiles | Ship |
| --- | ---: | ---: | ---: |
| Turnip | 56 × 200G | 504 | 504 × 60G = 30,240 |
| Potato | 56 × 200G | 504 | 504 × 80G = 40,320 |
| Tomato, planted Summer D1 | 8 × 300G | 72 × 7 harvests | 504 × 100G = 50,400 |
| Corn, planted Summer D1 | 8 × 300G | 72 × 7 harvests | 504 × 120G = 60,480 |

Seed spend 27,200G. Crop gold posted 181,440G. Tomato and corn each have 7 harvests when the 8 bags go in on Summer D1 (`days_to_first_harvest` 10, regrow 3, horizon 30).

| Gold | G |
| --- | ---: |
| Start | 300 |
| Crops shipped | 181,440 |
| Milk shipped | 713,190 |
| Eggs shipped | 55,270 |
| Seeds | −27,200 |
| Grass, 328 bags | −164,000 |
| 12 cows | −60,000 |
| Hen | −1,000 |
| Brush, milker, sprinkler | −4,600 |
| Houses | −15,000 |
| Chicken feed | −56,400 |
| Y2 visitor | −2,000 |
| D30 morning | 620,000 |

Animal gold withheld after the hold: 180,040G. From Y3 Spring D16 the next morning is brought back to 620,000G when feed spends 300G. Index 298 ships 300G. Index 299 adds that ship and spends nothing.

## Command

From the repo root. No ROM boot.

```bash
QT_QPA_PLATFORM=offscreen uv run pytest snes/harvest/tests/test_livestock_econ.py -q
```

6 passed in 1.23s. `CHICKEN` and `COW` are still missing `purchase_cost_g`, `products_per_day`, `feed_cost_g_per_day`, and `chore_frames_per_day`. `plan_purchase` still returns those names. The 299-day path is `build_rancher_ledger`.

```bash
QT_QPA_PLATFORM=offscreen uv run python -c \
  "from harvest.planner.livestock_econ import build_rancher_ledger; print(build_rancher_ledger().end_g)"
```

Prints `620000`. The JSON beside this note is that object, including all 300 mornings.
