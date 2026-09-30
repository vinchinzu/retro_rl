# F4 family timeline

Date: 2026-09-29. Bead PR-013. USA decomp under `HM-Decomp`, ROM `/home/v/01_projects/11_games/speedrun/roms/Harvest Moon.smc` (headerless, 2,097,152 bytes). Live step is on `LN_mgr-f4_whatif_preg2`, a RAM what-if. That pin is not route evidence.

Pins: `custom_integrations/HarvestMoon-Snes/LN_mgr-f4_whatif_preg.state` and `LN_mgr-f4_whatif_preg2.state`. JSON: `F4_family_probe_try1.json`, `F4_family_probe_try2.json`.

## Wife bits `$7F1F66`

Low five bits are the wives (`npc_catalog.ROMANCE_MARRIAGE_BITS`, `Events_Flags_Table` at `$8494D3`):

| Girl | Index | Bit | Hearts | SetFlag | Proposal text |
|---|---|---|---|---|---|
| Maria | 0 | `$0001` | `$7F1F1F` | `$B3B5FF` | `$0173` "I'm happy...." |
| Ann | 1 | `$0002` | `$7F1F21` | `$B3B137` | `$0175` "...Are you sure? Me?" |
| Nina | 2 | `$0004` | `$7F1F23` | `$B3C321` | `$0177` "Yes! You!" |
| Ellen | 3 | `$0008` | `$7F1F25` | `$B3B2FB` | `$0179` "What? ......Yes." |
| Eve | 4 | `$0010` | `$7F1F27` | `$B3B817` | `$017B` "Titter...alright. I say OK." |

`RanchMasteryCalculator` (`bank_83.asm`, after the two house-upgrade adds) does `LDA.L $7F1F66` / `AND #$001F` and adds 32 when any wife bit is set. New game zeros the word (`bank_83.asm` around the `$7F1F66` `STA #$0000` with the heart zeros). SRAM load stores it back (`STA.L $7F1F66` after `LDY #$0062`).

Other bits in the same word are house-construction state, not wives. `NightReset` (`$8282AC`) at the `$7F1F66` block after the pregnancy call: if bit `$0020` is set, it clears `$0020` and sets `$0040`; if bit `$0080` is set, it advances `development_rate`, and at 4 it sets `$0100`, stores 500 in `wood_need_for_upgrade`, clears `$0080`, and promotes `$7F1F64`. Two event writers set index 5 (`$0020`) at `$B3AD90` and `$B3CC50`, beside `JumpifFlag` on the house word. Those are the upgrade-order bit. A scan of `23 66 1F 7F` also hits `$B4B949` (Ann's bit, after 1-byte opcodes `$4F` and `$50`, then a place-the-player script). The five rows above are the shared proposal shape.

## Proposal flow

Each of the five sites is the same 19-byte fall-through. Ann at `$B3B124` is the pattern:

- `45` `JumpifBetweenValue` on that girl's hearts, range `0 .. $00F9`. Taken means hearts are 0..249 and the script jumps away. Fall-through is hearts ≥ 250.
- `1C` opens the text id in the table.
- `47` `SetValueSort` writes 0 to `tool_selected` (`$800921`).
- `23` `SetFlag` on `$7F1F66` with that girl's index.
- `23` `SetFlag` on `$7F1F5E` with a same-day index: Maria 7 (`$0080`), Ann 8 (`$0100`), Ellen 8 (`$0100`), Eve 9 (`$0200`), Nina 9 (`$0200`).
- `3D` `TransitiontoX` destination `$2D`.

`NightReset` later stores 0 to `$7F1F5E` (the `STA #$0000` series at `$7F1F5A`..`$7F1F62`). The same-day bits are gone by morning. They are not the score bits.

The instruction that jumps to each heart check is opcode `$4D` (`Event_UNKCCchanges17`, handler `$84BA72`). It takes its 16-bit destination only when all three are true:

- the running CC structure's byte at offset `$0C` is nonzero
- `player_action` (`$D4`) is not `$000A` (tool-use action, `ram.asm`)
- `$7F1F5E & $0040` is set

`ToolUsedBlueFeather` (`$82A3FD`, tool id `$18` in the `Tool_Used_Table` comment and in `ram.asm`) is the `ORA #$0040` onto `$7F1F5E`. `IterateCCStructures` (`$84816F`) clears that bit with `AND #$FFBF` at `.setflagsandquit`, after the scripts of that pass. `GameLoop` (`$808083`) calls the player update, then `IterateCCStructures`. The feather bit is a same-pass signal. The proposal script clears `tool_selected`; it does not compare the tool id. The fall-through contains no `JumpifFlag` on `$7F1F64`. House level is not a marriage gate.

Ann's prefix that falls into one `$4D` (`$B3B0EF`) jumps away when `hour` (`$7F1F1C`) is `$11` or `$12`, and jumps away when Ann's hearts are 150..200. Eve (`$B3B7DE`) and Nina (`$B3C2DF`) have the same hour test in front of their `$4D`. Spring/summer text earlier in Ann's event jumps to `$B096` on both arms, so that season test does not select the wedding.

## Pregnancy counter

`WifePregnanacyandChilds` is `$828790`. The only caller is `JSL` from `NightReset` after the calendar increment, the 6:00 store, farm/cow/weather work, and `FindMostLovedName`.

`NightReset` increments weekday, then day. Day `$1F` rolls the season; season 4 rolls the year and sets day to 1. Seasons are 30 days. The sleep that ends day D runs the pregnancy routine with the calendar already on D+1.

The routine:

1. Clears `$7F1F6C` bits `$0080|$0100|$0200` (`AND #$FC7F`).
2. Returns immediately when none of the five wife bits are set. Unmarried nights do not increment the counter.
3. Increments `wife_pregnancy` (`$7F1F3B`, u16) and divides by 120 (`DivisionUnsigned`, `$838082`). Remainder 0 ORs `$0080` onto `$7F1F6C`. That bit is a one-morning pulse on counter values 120, 240, and so on. It is cleared again at the next call. It is not gestation and not a birth.
4. Reads the wife's hearts. Child 1 needs ≥ 450 (`CMP #450` for Maria, `CMP #$01C2` / `BCS` for the others). Child 2 needs ≥ 650 (`#650` / `#$028A`). The routine does not write hearts.
5. `.wifepregnant`: counter `CMP #20` / `BCC` skips the flag. Then `$7F1F64 & $0080` (max house) must be set. If `$7F1F6E & $0004` is clear, it ORs `$0004` (first child). If `$0004` is already set and `kid1_age` ≥ 90, it ORs `$0008` (second child). The decomp comment on the `$0008` ORA says "Set first child"; the compare above it is the second-child arm.
6. If `$0004` is set, it increments `kid1_age` (`$7F1F37`). The night that sets `$0004` also moves age 0→1. Age landing on 60 ORs `$0010` onto `$7F1F6E`, adds 50 happiness, and returns, so kid 2 is not aged that night. Age past 60 with `(age-60) % 120 == 0` ORs `$0100` onto `$7F1F6C`.
7. If `$0008` is set, the same pattern runs for `kid2_age` (`$7F1F39`): age 60 ORs the same `$0010` and adds 100 happiness; the later pulse is `$7F1F6C` bit `$0200`.

`STA.L $7F1F3B` appears at the increment (`$8287D4` region), the new-game zero, and the SRAM load. The counter is not cleared on a season change.

There is no second gestation timer. The score flags are the `$0004` / `$0008` stores on the night the counter is already ≥ 20 and the house bit is set. Age 60 is the later happiness and naming pulse.

Call count from a fresh marriage (counter 0, ages 0, flags clear), assuming the house bit and the heart floors are already true on the night they are tested:

| Night after the wife bit | Morning | Counter | Flags | Ages at end of call |
|---|---|---|---|---|
| 1 | marriage date + 1 | 1 | none | 0, 0 |
| 20 | marriage date + 20 | 20 | `$0004` | kid1 = 1 |
| 110 | marriage date + 110 | 110 | `$0004` and `$0008` | kid1 = 91, kid2 = 1 |

Kid1's age is 1 on the child-1 morning and is tested before that night's increment, so child 2 waits until the start-of-call age is ≥ 90. That is 90 nights after the child-1 night, 110 nights after the marriage night. If the house bit or the heart floor is late, the flag waits for the first night all of the conditions hold, and child 2 waits 90 nights after that later night. The counter keeps incrementing on every married night either way.

`RanchMasteryCalculator` adds 16 for `$7F1F6E & $0008` and 16 for `& $0004`. Its comments label `$0008` "Child 1" and `$0004` "Child 2". The writer and `KID_EXISTS_FLAGS` in `harvest_state.py` use the opposite order: `$0004` first, `$0008` second. Both bits are +16, so the 32-point term needs both. The best-ending candidate at `$83F075` tests `$0008`, which is the second flag.

## Naming

`Unk_NamesInput` (`$82E80C`) dispatches on `$099F`: 5 → `SetKid1Name` (`$808338`), 6 → `SetKid2Name` (`$8083AE`). Each copies `temp_name_1..4` into the short name (`$7F1F3D` or `$7F1F41`, four bytes) and the long name, ORs `$4000` onto `$0196`, and jumps to `GameLoop`.

The morning chain at `CODE_83E074`: if `$7F1F6E & $0010` is set, then

- `$7F1F5E & $4000` loads event group `$1F` event 2 and returns
- `$7F1F5E & $2000` clear loads group `$1F` event 0 and returns
- otherwise loads group `$1F` event 1 and stores `$099F = 5` when `kid1_age == $3C`, else `$099F = 6`

Group `$1F` is directory entry `$B3805D` → `DATA8_B49A3F`. Event 0 at `$B49A5F` starts with opcode `$50` (one byte) and then `SetFlag` `$7F1F5E` index `$0D` (bit `$2000`). The age-60 night sets `$0010` and, later in the same `NightReset`, zeros `$7F1F5E`. The first morning of that age takes event 0, which sets `$2000`. The namer is the branch that already has `$2000` set. The score bits do not wait on the name.

## Latest marriage for two children on Y3 Summer D29

Day index, Y1 Spring D1 morning = 0: `year_ram * 120 + season * 30 + (day - 1)`. RAM year 0 is Y1. Seasons are spring 0, summer 1, fall 2, winter 3.

Y3 Summer D29 morning is index 298. The child-2 flag is on the morning of night 110, so the wife bit has to be set on index 188, which is Y2 Fall D9, before that night's sleep.

| Morning | Index | What is true, house bit and heart floors already met |
|---|---|---|
| Y2 Fall D9 | 188 | Wedding day. Wife bit set before sleep. |
| Y2 Fall D29 | 208 | Child 1 (`$0004`), `kid1_age` 1, counter 20. |
| Y3 Summer D29 | 298 | Child 2 (`$0008`), `kid1_age` 91, `kid2_age` 1, counter 110. |

Y2 Fall D10 (index 189) puts the child-2 flag on Y3 Summer D30 morning (index 299), which is the D29→D30 sleep. `CODE_83F4D8` scores that morning. The D29 evening audit in the master plan reads the flags before that sleep, so the date that already has both children on D29 is Y2 Fall D9.

## Live step

Both tries mint from `Y1_Inside_House` with Maria's bit, `$7F1F64` bits `$0040|$0080` (value `0x00C0`), Maria at 450, `wife_pregnancy` 19, child flags and ages 0. `house_size` was left 0. No `LiveRamEditor`, no `--ram-set`, no `time_running` write. `HEADLESS=1`, `SDL_VIDEODRIVER=dummy`, `QT_QPA_PLATFORM=offscreen`, `--trials 1`.

Try 1, pin `LN_mgr-f4_whatif_preg`, clock forced to 18:00 inside tilemap `0x15`. `GoToSleepTask` stood at (72, 88), then left that stand. Frame 24001 still had day 2, pregnancy 19, hearts 450. NightReset did not run.

Try 2, pin `LN_mgr-f4_whatif_preg2`, clock 17:00 so the task walks outside and lets the ROM reach 18:00. Re-entry loaded tilemap `0x17` (the level-2 house the upgrade bits select) and slept.

| Frame | Day | Hour | Pregnancy | Maria hearts | `$7F1F6E` | kid ages |
|---|---|---|---|---|---|---|
| 0 | Spring D2 | 17 | 19 | 450 | 0 | 0, 0 |
| 1868 | Spring D3 | 18 | 19 | 448 | 0 | 0, 0 |
| 1973 | Spring D3 | 6 | 20 | 448 | 0 | 0, 0 |

The counter stepped 19→20 on that one night. Hearts were already 448 on the first frame of D3, before the increment, and the child flag stayed clear. That is the Maria `CMP #450` / `BCC` skip: the house bit was still `0x00C0`, and 448 does not set `$0004`. The pregnancy routine never writes hearts; the −2 happened earlier the same evening. `kid1_age` stayed 0, which is the skip path (age increments only after the flag is set).

Commands, cwd `snes/harvest` for try 2 (`uv run --project` the repo root):

```bash
setsid nohup env HEADLESS=1 SDL_VIDEODRIVER=dummy QT_QPA_PLATFORM=offscreen \
  uv run --project /home/v/01_projects/11_games/retro_rl \
  python /tmp/mgr_f4_preg.py --trials 1
```

Try 1 was the same script from the repo root, before the hour-17 edit. Frames: try 1 ended at 24001; try 2 changed the day at 1868 and finished the morning at 1973.
