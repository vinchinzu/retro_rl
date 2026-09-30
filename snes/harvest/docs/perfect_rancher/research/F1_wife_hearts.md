# F1 — wife hearts back into 496–511

PR-010. Research only. The pin below is a what-if. It is not route evidence.

Opcode `0x41` (`Event_UNKCCchanges14` at `$84B5A9`) adds a signed word through a 24-bit pointer. A negative sum stores 0. A sum at or above 999 stores 999. Girl hearts are Maria `$7F1F1F`, Ann `$7F1F21`, Nina `$7F1F23`, Ellen `$7F1F25`, Eve `$7F1F27`.

The USA ROM has 94 negative `0x41` sites on those five pointers, all in banks `$B3`–`$B5`. Sixty-eight are gifts, twenty are Winter D24 wife events, six are the married indoor hour-18 sink.

Pregnancy (`bank_82` `.abouttogivebirth`) reads hearts. Child 1 compares against 450, child 2 against 650 (`$01C2` / `$028A`). It does not write the heart word. NightReset stores 0 to `$7F1F5A`, `$7F1F5C`, `$7F1F5E`, `$7F1F60`, `$7F1F62`, `$7F1F74`, `$7F1F76`, `$7F1F78`. It leaves the heart words alone.

## Probed recipe

Married Maria, group `$44` script 0, poison mushroom (item `0x02`).

The farm map loader `$83CB69` runs for tilemaps 0–3. It loads group `$44` script 0 into CC index `$13` when wife bit `$0001` is set and `$7F1F6E` bit `$0002` is clear. The script at `$B597F1` skips itself when weather `$0196` bit 1 is set, when the hour is 6–10 or 15–18, or when the season is winter. Open hours are 11, 12, 13 and 14. A child age in `$3C`–`$59` only changes the `0x1A` bytes; both age arms fall into the same gift split at `$B59836`.

That split gives item `0x02` to `$B59880`: text `$03B0`, Maria −6 at `$B59884`, jump back to `$B59836`. Items `0x09`–`0x0C` are text `$01D9` and −4 at `$B598B9`, same jump. Neither arm sets a flag. The item word is not cleared by the arm. Try 1 left `$091D` at 2 after the gift.

Opcode `0x2D` takes the gift pointer when `player_action` is 5, `$091D` is nonzero, and A is a new press. It also requires CC+`$0C` nonzero. The scan at `$83AE26` clears that byte each pass and sets it when the player anchor (`$DF`/`$E1`) is within `$19` of the CC anchor at +`$1A`/+`$1C`.

Measured on `LN_mgr-f1_whatif_wife660`, loaded from `LN_mgr-hearts_whatif_maria_flower` and then saved under the new name before the steps:

- Scalars on the pin: Maria 660, `marriage_flags` bit 0, hour 12, season 0, weather bit 1 clear, item `0x02`, action 5. CC index 1 pointed at `$B597F1`.
- Try 1 planted those once and pressed A. Text `$03B0` at frame 9. Hearts 660→654 at frame 88. The script then sat on the `0x2D` through frame 888. Action had returned to 0 and the proximity byte was not refreshed, so the split did not take the gift again.
- Try 2 reloaded the pin and, each frame, copied the player anchor onto that CC and held action 5, item 2, and CC+`$0C`. The header had already fallen through to `$B5982B` (the jump into `$B59836`) by the first snap. Hearts then fell by 6 every 78 frames, text `$03B0` each time, pointer writes 0. Frame 1950: Maria 510. Twenty-five drops. Ann, Nina, Ellen, Eve and happiness stayed 0. `(510 & 0x1FF) >> 4` is 31.

510 is inside 496–511. From 660, offers 25, 26 and 27 land on 510, 504 and 498. Offer 24 lands on 516 and offer 28 lands on 492. The measured stop was 25 offers.

The per-frame anchor copy stands in for standing next to the wife. The church CC on the source pin is not her sprite. A route offers the mushroom while the real group-44 wife script is loaded.

## Other wives, same outdoor script

Group `$44` still applies after marriage, because the loaders install it from the wife bits.

| Wife | Loader | Poison / disliked | Open hours |
| --- | --- | --- | --- |
| Maria | tilemaps 0–3, script 0 | `0x02` −6 text `$03B0`; `0x09`–`0x0C` −4 text `$01D9` | 11–14, not winter, weather bit 1 clear |
| Ann | tilemap `$26` `$83E6E3`, script 1 | `0x02` and `0x09`–`0x0C` both −6 text `$03B4` at `$B59A7B` | same 11–14 window |
| Nina | tilemaps 0–3, script 2 | item `0x19` −8 text `$03B1`; `0x09`–`0x0C` −4 text `$03BA` | same 11–14 window |
| Ellen | tilemaps 0–3, script 3 | `0x02` −8 text `$01D9`; `0x09`–`0x0C` −4 text `$03BA` | weather bit 1 clear, not winter, hour outside 17–18 |
| Eve | tilemap `$26` `$83E6F4`, script 4 | `0x02` −8 text `$03B4`; `0x09`–`0x0C` −4 text `$03BA` | weather bit 1 clear, not winter, hour outside 6–10 and outside 16–18 |

Every one of those adds is followed by a jump back to its `0x2D`. Nina's wife sinks are item `0x19` and the `0x09`–`0x0C` band. Her outdoor script has no poison-mushroom subtract.

## Indoor script and the automatic −2

Group `$43` is loaded from `$83E144` when that wife bit is set, her hearts are at least 200, and `$7F1F6E` bit `$0010` is clear. Hearts of 660 take this path. `$83E210` clears `$7F1F5E` bit 12, then loads the wife's group-43 script.

The first branch of each script is hour == 18: Maria −2 at `$B5805A`, Ann `$B58514`, Nina `$B589CA`, Ellen `$B58E88`, Eve `$B5934D`. The add runs, then the script sets `$7F1F5E` bit 12 and parks on an A-wait. One −2 per entry of the script at hour 18. It does not loop back to the hour test. Eve has a second −2 at `$B597BB` later in the same script; the next opcode is a jump to `$B59389`.

The gift split sits behind `$7F1F5E` bit 11. That bit is set at `$81D8FF` when the routine at `$81D7C5` sees `$096F == 3` and any wife bit set. NightReset clears the word, so the bit lasts until the next sleep once that routine has run. Group `$44` does not test bit 11. The probed farm gift is the one that stayed available.

Indoor gift deltas, same items as the outdoor table except Ann's poison is −8 (text `$03B4`) and Eve's `0x09`–`0x0C` band is −5 (text `$03BA`). Each is followed by a jump back to its own `0x2D`.

## Winter D24 −30

Group `$0F` loads on season 3, day `$18`:

- Script 0 at `$83E3FB` with no `$7F1F74` test.
- Script 1 at `$83E62D` if `$7F1F74` bit 0 is set. Wife bits 0, 1, 2 and 4 take −30 (Maria `$B48165`, Ann `$B4816E`, Nina `$B48177`, Eve `$B48180`). Ellen's bit 3 jumps to `$B481A5`.
- Script 2 at `$83D669` if bit 1 is set. Ann `$B4831A`, Nina `$B48323`, Ellen `$B4832C`, Eve `$B48335`. Text at the head of the chain is `$02C0`.
- Script 3 at `$83DBE1` if bit 2 is set. Maria `$B484FF`, Ann `$B48508`, Nina `$B48511`, Ellen `$B4851A`. Text `$03D9`. Eve's bit 4 jumps to `$B4853F`.
- Script 4 at `$83E65C` if bit 3 is set. Maria `$B48653`, Ann `$B4865C`, Ellen `$B48665`, Eve `$B4866E`. Nina's bit 2 jumps to `$B486AD`.
- Script 5 at `$83E685` if bit 4 is set. Maria `$B48737`, Nina `$B48740`, Ellen `$B48749`, Eve `$B48752`. Ann's bit 1 jumps to `$B487AF`.

Each −30 is `41` then a jump. The arm sets no flag. The wife-bit test means the subtract hits the married girl. Four −30s for one girl are −120. From 660 that lands on 540. `540 & 0x1FF` is 28, and `28 >> 4` is 1, above the 496–511 window. `$7F1F74` is one of the words NightReset clears, so a bit-gated script runs on a day that bit is set.

## Town gift copies

Banks `$B3` hold the same gift shape on the town and festival copies. The arm does not test `$7F1F66`. Deltas are per copy. Nina's town poison mushroom is −4, and her −6 there is item `0x18`. Several Maria and Ellen copies use text `$01DA` on the `0x09`–`0x0C` arm where another copy uses `$01D9`. The wife path above is groups `$43` and `$44`.

## Every negative site

`0x09`–`0x0C` is the band that shows text `$01D9` on the Maria ladders. Item `0x02` is the poison mushroom. Item `0x18` and `0x19` are the ids the compares use; the decomp names them only as those bytes.

| PC | Girl | Delta | Kind | Trigger | Text |
| --- | --- | ---: | --- | --- | --- |
| $B39614 | maria | -8 | gift | 2D $B395AC; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B39648 | maria | -2 | gift | 2D $B395AC; between $80091E 0x9..0xc | $01D9 |
| $B39821 | nina | -6 | gift | 2D $B3978F; eq $80091E==0x18 0x18 | $03B1 |
| $B3982E | nina | -4 | gift | 2D $B3978F; eq $80091E==0x2 poison_mushroom | $01D9 |
| $B39A10 | ellen | -8 | gift | 2D $B39997; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B39A44 | ellen | -4 | gift | 2D $B39997; between $80091E 0x9..0xc | $01D9 |
| $B39C4D | eve | -8 | gift | 2D $B39BB4; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B39C74 | eve | -4 | gift | 2D $B39BB4; between $80091E 0x9..0xc | $03BA |
| $B3A2E6 | ann | -8 | gift | 2D $B3A278; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B3A30D | ann | -2 | gift | 2D $B3A278; between $80091E 0x9..0xc | $03B4 |
| $B3B18A | ann | -8 | gift | 2D $B3B113; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B3B1B1 | ann | -2 | gift | 2D $B3B113; between $80091E 0x9..0xc | $03B4 |
| $B3B254 | ann | -2 | gift | 2D $B3B1C6; between $80091E 0x9..0xc | $03B4 |
| $B3B34F | ellen | -8 | gift | 2D $B3B2CC; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B3B383 | ellen | -2 | gift | 2D $B3B2CC; between $80091E 0x9..0xc | $01DA |
| $B3B652 | maria | -8 | gift | 2D $B3B5D9; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B3B686 | maria | -2 | gift | 2D $B3B5D9; between $80091E 0x9..0xc | $01DA |
| $B3B871 | eve | -8 | gift | 2D $B3B7F1; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B3B898 | eve | -2 | gift | 2D $B3B7F1; between $80091E 0x9..0xc | $03BA |
| $B3B99D | nina | -6 | gift | 2D $B3B90B; eq $80091E==0x18 0x18 | $03B1 |
| $B3B9AA | nina | -4 | gift | 2D $B3B90B; eq $80091E==0x2 poison_mushroom | $01D9 |
| $B3BFCD | maria | -8 | gift | 2D $B3BF65; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B3C001 | maria | -2 | gift | 2D $B3BF65; between $80091E 0x9..0xc | $01DA |
| $B3C0AD | ann | -8 | gift | 2D $B3C045; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B3C0D4 | ann | -2 | gift | 2D $B3C045; between $80091E 0x9..0xc | $03B4 |
| $B3C192 | ellen | -8 | gift | 2D $B3C132; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B3C1C6 | ellen | -2 | gift | 2D $B3C132; between $80091E 0x9..0xc | $01DA |
| $B3C395 | nina | -6 | gift | 2D $B3C2F2; eq $80091E==0x18 0x18 | $03B1 |
| $B3C3A2 | nina | -4 | gift | 2D $B3C2F2; eq $80091E==0x2 poison_mushroom | $01D9 |
| $B3C66A | maria | -8 | gift | 2D $B3C60B; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B3C69E | maria | -2 | gift | 2D $B3C60B; between $80091E 0x9..0xc | $01D9 |
| $B3C7E6 | nina | -6 | gift | 2D $B3C766; eq $80091E==0x18 0x18 | $03B1 |
| $B3C7F3 | nina | -4 | gift | 2D $B3C766; eq $80091E==0x2 poison_mushroom | $01D9 |
| $B3C8E5 | ann | -8 | gift | 2D $B3C886; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B3C90C | ann | -2 | gift | 2D $B3C886; between $80091E 0x9..0xc | $03B4 |
| $B3CA46 | ellen | -8 | gift | 2D $B3C9DF; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B3CA7A | ellen | -2 | gift | 2D $B3C9DF; between $80091E 0x9..0xc | $01DA |
| $B3D2A4 | eve | -8 | gift | 2D $B3D22C; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B3D2CB | eve | -2 | gift | 2D $B3D22C; between $80091E 0x9..0xc | $03BA |
| $B3D57A | maria | -8 | gift | 2D $B3D516; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B3D697 | nina | -5 | gift | 2D $B3D64F; between $80091E 0x1..0x4 | $03B2 |
| $B3D6B1 | nina | -8 | gift | 2D $B3D64F; eq $80091E==0x19 0x19 | $03B1 |
| $B3D791 | ann | -8 | gift | 2D $B3D73B; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B3D8F0 | ellen | -2 | gift | 2D $B3D869; between $80091E 0x9..0xc | $01DA |
| $B3E1C0 | maria | -2 | gift | 2D $B3E13A; between $80091E 0x9..0xc | $01DA |
| $B3E302 | nina | -8 | gift | 2D $B3E290; eq $80091E==0x19 0x19 | $03B1 |
| $B3E30F | nina | -2 | gift | 2D $B3E290; between $80091E 0x9..0xc | $03BA |
| $B3E3DC | ann | -8 | gift | 2D $B3E37D; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B3E403 | ann | -2 | gift | 2D $B3E37D; between $80091E 0x9..0xc | $03B4 |
| $B48165 | maria | -30 | festival | Winter D24 group 0F, script 1, wife bit 0 | $02B9 |
| $B4816E | ann | -30 | festival | Winter D24 group 0F, script 1, wife bit 1 | $02B9 |
| $B48177 | nina | -30 | festival | Winter D24 group 0F, script 1, wife bit 2 | $02B9 |
| $B48180 | eve | -30 | festival | Winter D24 group 0F, script 1, wife bit 4 | $02B9 |
| $B4831A | ann | -30 | festival | Winter D24 group 0F, script 2, wife bit 1 | $02C0 |
| $B48323 | nina | -30 | festival | Winter D24 group 0F, script 2, wife bit 2 | $02C0 |
| $B4832C | ellen | -30 | festival | Winter D24 group 0F, script 2, wife bit 3 | $02C0 |
| $B48335 | eve | -30 | festival | Winter D24 group 0F, script 2, wife bit 4 | $02C0 |
| $B484FF | maria | -30 | festival | Winter D24 group 0F, script 3, wife bit 0 | $03D9 |
| $B48508 | ann | -30 | festival | Winter D24 group 0F, script 3, wife bit 1 | $03D9 |
| $B48511 | nina | -30 | festival | Winter D24 group 0F, script 3, wife bit 2 | $03D9 |
| $B4851A | ellen | -30 | festival | Winter D24 group 0F, script 3, wife bit 3 | $03D9 |
| $B48653 | maria | -30 | festival | Winter D24 group 0F, script 4, wife bit 0 |  |
| $B4865C | ann | -30 | festival | Winter D24 group 0F, script 4, wife bit 1 |  |
| $B48665 | ellen | -30 | festival | Winter D24 group 0F, script 4, wife bit 3 |  |
| $B4866E | eve | -30 | festival | Winter D24 group 0F, script 4, wife bit 4 |  |
| $B48737 | maria | -30 | festival | Winter D24 group 0F, script 5, wife bit 0 |  |
| $B48740 | nina | -30 | festival | Winter D24 group 0F, script 5, wife bit 2 |  |
| $B48749 | ellen | -30 | festival | Winter D24 group 0F, script 5, wife bit 3 |  |
| $B48752 | eve | -30 | festival | Winter D24 group 0F, script 5, wife bit 4 |  |
| $B5805A | maria | -2 | calendar | group 43 script 0, hour == 18 |  |
| $B580E2 | maria | -6 | gift | 2D $B5808E; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B58117 | maria | -4 | gift | 2D $B5808E; between $80091E 0x9..0xc | $01D9 |
| $B58514 | ann | -2 | calendar | group 43 script 1, hour == 18 |  |
| $B58595 | ann | -8 | gift | 2D $B58548; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B585CA | ann | -4 | gift | 2D $B58548; between $80091E 0x9..0xc | $03B4 |
| $B589CA | nina | -2 | calendar | group 43 script 2, hour == 18 |  |
| $B58A4B | nina | -8 | gift | 2D $B589FE; eq $80091E==0x19 0x19 | $03B1 |
| $B58AA2 | nina | -4 | gift | 2D $B589FE; between $80091E 0x9..0xc | $03BA |
| $B58E88 | ellen | -2 | calendar | group 43 script 3, hour == 18 |  |
| $B58F11 | ellen | -8 | gift | 2D $B58EBC; eq $80091E==0x2 poison_mushroom | $01D9 |
| $B58F53 | ellen | -4 | gift | 2D $B58EBC; between $80091E 0x9..0xc | $03BA |
| $B5934D | eve | -2 | calendar | group 43 script 4, hour == 18 |  |
| $B593D5 | eve | -8 | gift | 2D $B59381; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B5942C | eve | -5 | gift | 2D $B59381; between $80091E 0x9..0xc | $03BA |
| $B597BB | eve | -2 | calendar | group 43 script 4, later −2, jump $B59389 |  |
| $B59884 | maria | -6 | gift | 2D $B59836; eq $80091E==0x2 poison_mushroom | $03B0 |
| $B598B9 | maria | -4 | gift | 2D $B59836; between $80091E 0x9..0xc | $01D9 |
| $B59A7B | ann | -6 | gift | 2D $B59A34; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B59C6F | nina | -8 | gift | 2D $B59C28; eq $80091E==0x19 0x19 | $03B1 |
| $B59CA4 | nina | -4 | gift | 2D $B59C28; between $80091E 0x9..0xc | $03BA |
| $B59E5F | ellen | -8 | gift | 2D $B59E10; eq $80091E==0x2 poison_mushroom | $01D9 |
| $B59EA1 | ellen | -4 | gift | 2D $B59E10; between $80091E 0x9..0xc | $03BA |
| $B5A065 | eve | -8 | gift | 2D $B5A017; eq $80091E==0x2 poison_mushroom | $03B4 |
| $B5A0BC | eve | -4 | gift | 2D $B5A017; between $80091E 0x9..0xc | $03BA |

## Probe command

Cwd `snes/harvest`. `HEADLESS=1`, `SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy`, `QT_QPA_PLATFORM=offscreen`, `--trials 1`, one emulator, `setsid nohup`. Interpreter `/home/v/01_projects/11_games/retro_rl/.venv/bin/python` so the repo path file is on `sys.path`. No `LiveRamEditor`. No `--ram-set`. RAM writes were `memory.assign` on this what-if pin only. The flower pin was loaded and never saved.

Try 1: 888 frames observed, one −6 at frame 88. Try 2: 1950 frames, 660→510. Detail is in `F1_wife_hearts_probe.json`.
