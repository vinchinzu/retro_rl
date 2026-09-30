# F7 happiness

Date: 2026-09-29. Bead PR-016. Static decode of the USA ROM and `HM-Decomp`. No emulator frames. The delta histogram matches MASTER_PLAN section F7, so this file is the plan, not a contradiction.

## Audit: dated floor must be rebuilt, 2026-09-30

PR-016 and PR-073 are reopened. The weak-case sum below assumes both
child age-60 bonuses occur before the ending. Current milestones marry
at index 188 (Y2 Fall D9), put child 2 at 298, and end at 299. Its +100
age-60 bonus arrives too late. With seven berries, the stated weak floor
is 900 rather than 1000; with ten it is 930, before other unbudgeted
sources. Nine namings also conflict with F5 buying all twelve slots.
Date an earlier marriage or verified repeatable food/social repairs,
and coordinate birth space, potions, slot-0 replacement and the cash model.
The decoded source/sink table remains research evidence.

## Score

`RanchMasteryCalculator` scores `happiness >> 5`, max 31. That needs `$7F1F33 >= 992`. `AddPlayerHappiness` (`bank_83.asm` `$83B282`) adds the signed value in A. A negative sum is rejected and the old value stays. A sum `>= 999` stores 999. 992 through 999 all score 31. 991 scores 30.

At 999 the spare room is 7. One diary answer of -1 still scores 31. A sink of -8 or worse drops the term to 30 unless a later positive source lands. 992 has no spare room at all. The D29 evening target is 999.

New game stores 0 (`$83` line 4603, `LDA.W #0` / `STA.L !happiness`). Load game copies SRAM offset `$6C` back (`$83` line 5800).

Event opcode `0x41` (`Event_UNKCCchanges14`, `$84B5A9`) adds a signed word to a 24-bit pointer and clamps to 0..999. A negative sum stores 0. The same byte pattern on `$7F1F33` is `41 33 1F 7F`. The ROM has 99 of those. Other `33 1F 7F` hits are the code loads and stores above, plus compares. No opcode `0x32` / `0x47` / `0x48` writes this address.

Signed deltas on the 99 sites: +30 x38, +5 x28, +10 x17, +20 x4, +50 x3, -5 x1, -10 x2, -20 x3, -30 x1, -50 x1, -100 x1. That is the MASTER_PLAN scan line.

## Code callers of AddPlayerHappiness

Amounts are the `LDA` immediately before the `JSL`. `#$` is hex. A bare asar number is decimal, so the calf is 70 and the first child's day-60 bonus is 50.

| Where | Delta | When it fires | Repeat |
|---|---|---|---|
| `SetCowNameBorn` `$8081D2` passes A=1 into `AddNewCow` `$83C8DC` | +70 | Naming screen of a calf. `AddNewCow` fails if all 12 slots are full. | Per birth |
| `SetCowNameBought` `$80815F` passes A=0 | 0 | Bought cow. The slot is created, happiness is unchanged. | |
| Power berry `$81CAAB` | +10 | `$0984 == 8`. Increments `power_berry_N` and `max_stamina` by 10. | Once per berry eaten. Ten berries are +100. The best-ending check wants `power_berry_N == 10`, so the route eats ten. |
| House completion `$81F315` / `$81F325` | +50 or +100 | After the `$0970` timer hits `$20`. Bit `$7F1F64 & $0080` clear pays +50 and text `$013D` ("Look, isn't it an excellent house?"). Bit set pays +100 and text `$0140` ("It's finished! Look at this the Super..."). | Once per completion scene |
| `WifePregnanacyandChilds` `$828790` | +50, then +100 | The day `kid1_age` becomes 60 (`LDA.W #50`). The day `kid2_age` becomes `$003C` (60), +100 (`LDA.W #$0064`). Each compares equal, so later days skip it. Pregnancy itself starts only while house bit `$7F1F64 & $0080` is set. | Once each |
| First cow payment `$81E0C6` | +20 | Cursor `$018F == 0`, grass check passes, `AddMoney` of `$FFFE0C` succeeds (5,000G), then `cow_N` increments and `$7F1F5A` bits `$0022` are set. | The flag bits make the success path one scene, not every later purchase |
| First chicken `$81E1FA` | +10 | `AddNewChicken` with A=2, `chicks_N` increments, `$7F1F5A` bit `$0020` set. | Once |
| Incubator `$81` `Droped_on_special_place` | +5 | Tile property `$00F7`, `$7F1F6E` bit `$2000` clear. The success path sets that bit and calls `AddNewChicken` with A=0. | Again only after something clears the bit |
| `ToolUsedPaint` `$829BCB` | +10 four times | Tile properties `$00E2`, `$00E3`, `$00E4`, and the remaining branch. Each sets one `$7F1F66` bit: `$0400`, `$0800`, `$1000`, `$0200`. | Once per tile |
| Summer sickle `$82` `.frogcheck` and `$829E5D` | +2 | Season == 1, `$7F1F60` bit `$0008` clear, `$7F1F5A` bit `$2000` clear, `RNGReturn0toA` of `$10` returns 0. Success sets the `$2000` bit. The second path repeats the same compares. | Once, about 1/16 per try until it hits |
| Shop food `$81F68E` | +1 | Text `$024B` ("Good-day, today! Here are some famous..."). Cursor 0, `AddMoney` `$FFEC` with high byte `$FF` succeeds (200G). Text `$024C` on success, `$0213` if the wallet refuses, `$024D` if the cursor is not 0. Then `player_action` `$0019`. | No flag in this handler |
| Shop food `$81F7B6` | +3 | Same shape, text `$024A`, `AddMoney` `$FFE2` (300G). | No flag in this handler |
| Girl diary, five sites | -1 | Maria `$81DCBD` (`$7F1F1F`), Nina `$81DE0B`, Ellen `$81E338`, Ann `$81E3F3`, Eve `$81E58B`. Prompt text `$0023` is "Found the diary!". Cursor 0 shows the heart-scaled text (`$03A6` plus the heart tier) and subtracts 1. Any other cursor closes with no change. `StartTextBox` zeroes `$018F`, so the default line is the sink. | Each time that prompt is answered on cursor 0 |
| Cow sick, `CowFeedingandStatus` `$83BC5A` | -30 | Sick roll returns 0 (`RNGReturn0toA` with A=4, 2, or 8 depending on the weather branch). Cow not pregnant (status bit `$40`), flag byte bit `$80` clear, not already sick (bit `$20`). Sets sick, writes pregnancy byte 7, cow happiness -12, player -30 (`$FFE2`). | Each sick proc |
| Chicken night pass `$83C05E` | -30 | Chicken status bit 0 set and bits `$0E` clear. Status cleared, `chicks_N` decremented, player `$FFE2`. | Each such chicken |
| Cow removal `$81E9DB` | -50 | Clears the cow slot, decrements `cow_N`, `LDA.W #$FFCE`. A second `JSL` at `$81EB7B` is the same amount on the other branch of that scene. | The scene |
| Chicken removal `$81ED9E` and `$81EF53` | -20 | `LDA.W #$FFEC`, `chicks_N` decremented. | The scene |
| Action index 19 `$81C317` | +2 | Inside `DATA16_81C027`. Handler `$81C2AF` when `$0901 == $008B` loads group 0 event `$35` and adds 2. | Uncounted until that action is named |
| Clear morning `$83D2DA` | +2 | Weather `$0196 & $001A` is 0, the clock is 6:00:00, and `RNGReturn0toA` of 8 returns 0. Also loads group `$23` event 0. | No one-shot in this block. Left out of the floor |

The ROM has 30 `JSL AddPlayerHappiness` instructions. `SetCowNameBought` is not one of them. The two sickle paths are two of the 30 and share one flag. The definition at `$83B282` is not a caller.

## Opcode 0x41 on `$7F1F33`

Scripts live in the group table at `$B38000` (0x48 groups, 1288 scripts). Each site below is owned by the script whose start is the greatest start at or before the opcode. Full addresses are in `F7_happiness_probe.json`.

Group 43 events 0..4 hold 4 x +30 each (20 of the 38). `$83E210` loads exactly one of them, A=`$13`, picked by the `$7F1F66` wife bits `$0001/$0002/$0004/$0008/$0010`. In 43/00 the three opcodes at `$B58146`, `$B58153`, `$B58160` are the taken sides of `$7F1F6C` bits 7, 8 and 9 inside the item handoff (opcode `0x2D` at `$B5808E`, item pointer `$B5809C`, `old_item_on_hand` `$091E == 6` jumps to `$B58120`). The fall-through of those three bits adds +8 to Maria's hearts and shows "Wow, thanks a lot", with no player happiness. The fourth +30, `$B5842C`, is a later branch of the same script. The other four wife scripts repeat the shape. These are exclusive gift reactions for one wife, not 20 stackable payouts.

Group 44 events 0..4 hold 3 x +30 each (15 of the 38). Map entry `$83CB69` loads events 0, 2 and 3 for wife bits `$0001/$0004/$0008`, and skips the load when `$7F1F6E` bit `$0002` is set. `$83E6EE` and `$83E6FF` load events 1 and 4 for the other two wife bits. On 44/00 the three +30s at `$B598E8`, `$B598F5`, `$B59902` share one decision stack: weather bit 1 clear, hour outside 6..10 and outside 15..18, not winter, both kids outside age `$3C..$59`, opcode `0x2D` taken, `old_item_on_hand != 2`. Treat the three as pages of one scene (+90 upper bound), and only the married girl's script runs.

The other three +30s are 28/01, 28/03 and 29/00. 28/03's +30 at `$B4C407` requires `$8009A2 == 5`, text "The winner of this ye...", and sets `$7F1F60` bit 9. 29/00's +30 is the `$80091E == 20` side. Its other side is -5.

| Script | Deltas | Loader | Route note |
|---|---|---|---|
| 16/00 | +20 and -20 | `$83D1C9`, group `$16` event 0 | The Saturday event. See sinks. |
| 15/00 | +20 and -10 | `$83DE7A` | Choice. Texts around `$0329`. |
| 17/00 | +20 and -30 | `bank_82_toolused_subrutines.asm` line 729 | Cursor 0 is -30 at `$B48E69`. Cursor 1 is +20 at `$B48E80`. Default cursor 0 is the sink. |
| 18/00 | +20 and -20 | `$83D239` | `$80091E == 1` is +20, `== 2` is -20. |
| 21/00 | +50 and -10 | `$83DD39` | Both sit past the entry `END`. Choice, texts `$0030`..`$0032` (sick-cow lines). |
| 0D/01 | +10 | `$83E4BD` | Post-collapse wake. The only happiness opcode in range is +10 at `$B3F0BB`. |
| 0C/00 | +10 | `$83CEC4` | Reached on the first path. Sets weather bit 13. |
| 1E/00 | +50 | `$83E5DA` | Reached when season is not 0, 1, 2, or 3 on the opening compares, so the fall-through of four season checks. Read it as one specific season miss, not a free +50. |
| 20/00 | -100 | `$83E13F` | Cow funeral flag. See sinks. |
| 22/00 | -50 | `$83E1F1` | Low-heart girl at 6:00. See sinks. |
| 28/02 | +50 | `$83D3B3` | Inside the script, not on the entry fall-through. |
| 27/00 | +10 x4 | `$83D435` | Four +10s behind `$7F1F66` bit checks. Compare with the paint flags before counting both. |
| 2A/00, 2B/00, 2C/00 | +10 each | `bank_80` `$809EB5` A=18, `$809E77` A=19, `$809E39` A=17 | Reached with an empty condition list. |
| 45/00, 45/03 | +10 x2 each | `$83E2E5` / `$83E308`, and also `bank_81` `$81C0EC` / `$81C0FD` | After the wife dispatch: kid1 age `>= 60` and `$7F1F6E` bit `$0004` loads 45/00. Kid2 age `>= 60` and bit `$0008` loads 45/03. One of the two +10s was reached, gated on `old_item_on_hand == 6`. Separate from the code +50 / +100 at the same age. |
| 01/* and 07/* | the +5 pool (28 sites, a few scripts hold two or four) | `$83D4B2` through `$83DE4F`, A=0 | Flag checks on `$7F1F66` / `$7F1F6A` / `$7F1F6C`. The walker did not reach the opcodes from the entry. Count them as small one-shots, not a daily +5. |
| 05/00 | +5 x4, three reached | `$83DC19` | Same caution. |
| 26/00, 27/01, 27/02, 28/01 | +10, and 28/01 also +30 and -20 | `$83D409`, none, `$83D63A`, `$83E6AD` | 26/00 is season 2, day `$0C`, hour below `$0F`. 27/02 is season 3, day `$0A`, hour below `$11`. 27/01 has no `LDX #$0027` / `LDY #$0001` pair. 28/01 is season 0, day 1. Its three deltas are `JumpIfEqualsRNG` arms, and the entry walk does not reach them. |

## Ranked positive sources

1. Calf naming, +70, every `SetCowNameBorn`. This is the source that reaches 992. Barn cap is 12, and a full barn refuses the birth.
2. Ten power berries, +100. Required by the stamina term anyway.
3. The two house completion scenes, +50 and +100 when bit `$7F1F64 & $0080` is clear and then set. If both scenes see the bit the same way, the pair is +100 or +200. The plan below uses the smaller pair.
4. Two children reaching age 60, +50 and +100. Required by the children term anyway.
5. Four paint tiles, +40, one-shot flags.
6. Saturday cursor 0, +20, once.
7. First cow scene +20 and first chicken scene +10, once each.
8. Incubator +5 whenever bit `$7F1F6E & $2000` is clear.
9. Shop food, +1 for 200G and +3 for 300G, cursor 0, no flag in the handler. Useful as a small repair. The NPC's own once-a-day rule is not in this function, so the 992 math does not depend on it.
10. Wife scripts. One girl. Group 44 is at most one +90 scene. Group 43's +30s are separate `$7F1F6C` gift bits. Festivals and the contest line (28/03) are extra one-shots. They are margin on top of the births.
11. Summer sickle frog, +2 once, about 1/16, two code paths and one flag. The clear-morning +2 at `$83D2DA` is the same kind of leftover. Leave both out of the sum.

Bought cows add 0 from `AddNewCow`. The first-cow scene is the +20, and it is not a birth.

## Sinks

Any of these after the last +70, once the value is already 999, misses the term. A later birth repairs one of them. Two repairs are needed after a -100.

- Saturday, group 16/00, loaded at `$83D163`. Year nonzero or season == 3, hour 6:00:00, weekday `$7F1F1A == 6` (ram_map.md calls this Saturday), weather `$0196 & $001A == 0`, house `$7F1F64 & $00C0` nonzero, `$7F1F68` bit `$4000` clear. The loader sets that bit before the script, so the event happens once. Text `$0321` is "Excuse me but can I borrow your hammer?". Cursor 0 shows `$0322` ("Thank you...") and adds +20 at `$B48DEA`. Cursor 1 shows `$0323` ("What? I won't ask you any more!"), sets `$7F1F6A` bit 2, and adds -20 at `$B48DFC`. Press cursor 0.
- Girl diary cursor 0, -1, five girls. At 999 a single answer is harmless. Eight of them after the last plus are not.
- Group 17/00 cursor 0, -30 at `$B48E69`. Cursor 1 is +20.
- Group 18/00, -20 when `$80091E == 2`. The `== 1` side is +20.
- Group 15/00, -10 on the other branch from +20.
- Group 21/00, -10 next to a +50. The text block is the sick-cow lines `$0030`..`$0032`.
- Group 29/00, -5 on the hour 18..24 side. The other side is +30.
- Group 22/00, -50 at `$B4A00B`, unconditional once the script starts. Loader `$83E1F1`: a `$7F1F66` girl bit is set, that girl's hearts are below 200, the clock is 6:00:00, and `$7F1F6E` bits `$000C` and `$0001` are clear. The loader then sets `$7F1F6E` bits `$0003` and `$7F1F5E` bit `$1000`. Hold the flagged girl at 200 hearts or more before that morning. Maria, Ann, Nina, Ellen and Eve all have the compare.
- Group 20/00, -100 at `$B49EB8`. Loader `$83E0DB` runs when `$7F1F6E` bit `$0040` is set, clears it, copies the cow's name bytes, and starts the script. The night pass sets that bit when a cow dies and does not itself call `AddPlayerHappiness`. The -100 is the funeral script.
- Cow removal scene `$81E9DB` / `$81EB7B`, -50, and it decrements `cow_N`.
- Chicken removal `$81ED9E` / `$81EF53`, -20.
- Cow sick, -30. Feed, and do not leave cows out in the weather that feeds the sick roll.
- Chicken night pass, -30, when status bit 0 is set and bits `$0E` are clear.
- Opcode `0x41` stores 0 if the sum goes negative. A script -100 from a low total empties the bar. `AddPlayerHappiness` refuses instead of wrapping.

The collapse wake is script 0D/01, and its happiness opcode is +10. Passing out still burns the day. There is no collapse penalty in the caller list or the 0x41 set.

## Dated plan to 999 by Y3 Summer D29

Power-on starts at 0. The controllable window is Y1 Spring D1 through Y3 Summer D29.

The birth count is nine. That still clamps at 999 in the weak case: both house scenes pay +50 (100), Saturday is answered on cursor 1 (-20), and the rest of the code floor lands.

```
berries 100
house scenes, weak case 50+50 = 100
child age-60 bonuses 50+100 = 150
first cow scene 20
first chicken scene 10
paint tiles 40
Saturday cursor 1 = -20
subtotal 400
nine births x 70 = 630
total 1030, stored as 999
```

Eight births in that weak case land at 960, and `960 >> 5` is 30. The route therefore names nine calves, not eight. In the strong case (house +50 then +100, Saturday cursor 0) the same nine births are surplus past 999, which is what the clamp is for. A tenth birth is the repair if a -30 or -50 has already landed. A -100 needs two repairs.

Barn size is 12. Three purchased cows (the first of them is the +20 scene, the other purchases add 0) plus nine births fill the barn. Leave the ninth naming until after the last animal-risk day, and early enough that the calf can still satisfy the cow-happiness term. F3 owns that deadline. F7 only needs the +70 to land on or before Y3 Summer D29.

- Y1 Spring through Summer. Eat each berry as F6 finds it (+10, up to ten). Incubate while the hatch bit is clear (+5). Take the first chicken scene (+10) and the first cow scene (+20). Do not sell, starve, or leave animals out.
- Y1 Fall. House completion while `$7F1F64` bit `$0080` is clear: +50, text `$013D`.
- Y1 Winter. Second house completion. If the bit is now set, this one pays +100 and shows the "Super" text. Paint the four tiles (+40) once each flag is clear. The Saturday loader can run once a house bit in `$00C0` is set: cursor 0, +20. Cursor 1 spends the one-shot flag and subtracts 20.
- Y2 Spring. Marriage itself has no `AddPlayerHappiness` call. Wife scripts 43 and 44 become available for the one wife. Take the positive branch when a choice is on screen. Keep every flagged girl at 200 hearts or more before a 6:00 so group 22 does not fire its -50. Start the births. Each naming screen is +70.
- Y2 Summer through Fall. The day child 1's age becomes 60: +50. Keep naming calves. The group 45 script may add a further +10 on the `old_item_on_hand == 6` branch. That is extra, not a substitute for the code bonus.
- Y2 Winter through Y3 Spring. The day child 2's age becomes 60: +100. Finish the birth count if the ninth naming is not being held as a repair.
- Y3 Summer D1 through D29. The ledger should already read 999. Through D29: feed every cow, bring animals in, answer no removal prompt, skip diary cursor 0 unless a later +1 or +3 is still available the same day, and do not start group 17 on cursor 0. Shop food repairs a diary -1. It does not repair a -30. If a sink of -8 or worse has already happened, the ninth (or a tenth) naming is the repair, and it has to happen before D29 evening.
- Y3 Summer D29 evening. `$7F1F33` is 999. 992 also scores 31 and leaves nothing for a diary answer on the last night, so the route stops at the clamp.

No live probe was required. The signs, the one-shot flags, and the birth math are in the ROM. `F7_happiness_probe.json` is the address table.
