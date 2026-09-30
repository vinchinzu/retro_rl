# F9 hearts

Date: 2026-09-29. Bead PR-018. Status: measured.

The D1 intro talks that ran match their scripts: Maria text `0x00F9` hearts +0, Ann text `0x0103` hearts +0. That actor is opcode `0x1F` (one A destination, no item arm), so planting item `0x05` and `player_action` 5 on the church sprite left the ladder unentered. The event-1 flower ladders were then run on the same what-if pin by writing CC index 1's script pointer (parked at `$B3EF44`) to the ladder and holding old item `$091E` at `0x05`. Maria hearts went 0→5 with text `0x03B2` in 109 frames. Ann hearts went 0→6 with text `0x03B2` in 109 frames. Both match the decoded `$B3` ladders. Happiness stayed 0. The other three girls stayed 0 on each run.

## Writer

Opcode `0x41` `Event_UNKCCchanges14` at `$84B5A9` adds a signed word through a 24-bit pointer and clamps the result to 0..999. Hearts:

| Girl | Pointer | Live catalog |
| --- | --- | --- |
| Maria | `$7F1F1F` | `maria_hearts` |
| Ann | `$7F1F21` | `ann_hearts` |
| Nina | `$7F1F23` | `nina_hearts` |
| Ellen | `$7F1F25` | `ellen_hearts` |
| Eve | `$7F1F27` | `eve_hearts` |

A pattern scan of `41 <girl> 1F 7F` across banks `$B3`–`$B5` finds 361 sites. Aligned counts sit in the plan's 60–80 band: Maria 74, Ann 76, Nina 77, Ellen 70, Eve 64. Each girl has one `+100`, two `+50`, five `+30`, and four `−30`.

## D1 first meeting (measured)

Town load is `CODE_83D46D`. It returns without an NPC script when `$0196` has bit `0x0008`, `0x0200`, or `0x0002`, when `weather_tomorrow` is 3, or when `$7F1F64` bit `0x0020` is set. Otherwise weekday 0 loads event 3, weekday 6 loads event 2, and any other weekday loads event 1, script 0. The power-on diary landed weekday 1, season 0, day 1, 7:00, so event 1 was loaded. The first-meeting lines live at the end of event 11 and use opcode `0x1F` (A press, one destination, no item arm) then `SetFlag` on `$7F1F74`.

| Flag index | Mask bit | Text | Hearts in that block | Live |
| --- | --- | --- | --- | --- |
| 0 | `0x01` Ann | `0x0103` "I'm Ann. Nice to meet you." | none, then jump `$B3EAB0` | [MEAS] text `0x0103`, hearts 0→0, happiness 0, 240 frames, mask `0x20`→`0x21`, player (378, 906) tilemap `0x04` |
| 1 | `0x02` | `0x0116` "The bar is closed today." | none | not measured |
| 2 | `0x04` Nina | `0x0107` "I'm Nina." | none | not measured |
| 3 | `0x08` | `0x0105` crop-seed offer | none | not measured |
| 4 | `0x10` | `0x010C` meadow / livestock | none | not measured |
| 5 | `0x20` Maria | `0x00F9` "I'm Maria. Nice meeting you." then branch `0x00FA` "don't give up" | none | [MEAS] `0x00F9` at frame 6547, hearts 0→0, happiness 0, 282 frames, church (108, 409) tilemap `0x1B`. Next mash showed `0x00FA`, hearts still 0 |

There is no heart opcode on these intros, and the live deltas are 0. The Spring daytime scripts below are a different actor.

## Spring daytime talk (decoded, not the D1 line that ran)

Opcode `0x2D` at these sites splits A: word0 is talk, word1 is gift. Gift is taken only when `player_action` word is 5 and item `$091D` is nonzero. The handler then copies `$091D` into `$091E` and jumps. `game_state` bit `0x0004`, and actions `$0A`, `$0C`, `$0D`, `$1B`, skip the press.

| Girl | `0x2D` | Talk dest |
| --- | --- | --- |
| Maria | `$B395AC` | `$B39651` |
| Nina | `$B3978F` | `$B39844` |
| Ellen | `$B39997` | `$B39A4E` |
| Eve | `$B39BB4` | `$B39BDD` |
| Ann | `$B3A278` | `$B3A316` |

Maria's talk dest, hand-disassembled: season `$7F1F19` selects the body. Spring (`== 0`) and day `== 22` shows `0x00BF` and adds nothing. Otherwise RNG `0..2`:

- 0: text `0x0082`, no heart opcode
- 1: text `0x0086` ("Do you believe in God?"). Choice 0 → `0x0087` and Maria `+8`. Choice 1 → `0x0088` and Maria `+2`
- else: exit text `0x043D`, no heart opcode

Ann spring RNG texts `0x0096` and `0x009E` have no heart opcode on that path; exit text is `0x043E`. Nina texts `0x00AE` / `0x00B5`, exit `0x043F`. Ellen text `0x00CB` opens a choice (the scanner's `+8` / `+2` rows); `0x00C5` has no add; exit `0x0440`. Eve texts `0x00E1` / `0x00DF`, exit `0x0441`. Hour 17–18 on these dests shows `0x0391` and adds nothing. No `SetFlag` sits in front of the Maria `+8` / `+2`, so that choice is not a once-per-day flag in this script.

## Gift ladders (decoded)

Compare is `$80091E` (old hand). Default text on all five primary ladders is `0x0200` ("Thank you.") with no heart opcode. Forage names are `FORAGE_HELD` in `interact.py`. Ids `0x10`–`0x17` are the eight shipped goods; `SPRING_ECONOMY.md` infers corn, tomato, potato, turnip, egg, milk S, milk M, milk L and does not have a ROM name table. `0x06`, `0x09`–`0x0C`, `0x18`, `0x19`, and `0x49` stay numeric.

Maria `$B395D5`:

| Item | Delta | Text |
| --- | --- | --- |
| `0x02` poison mushroom | −8 | `0x03B0` |
| `0x01`, `0x03`–`0x05`, `0x10`–`0x17` | +5 | `0x03B2` "It looks delicious." |
| `0x06` | +3 | `0x03B5` |
| `0x19`, `0x49` | +4 | `0x0212` |
| `0x09`–`0x0C` | −2 | `0x01D9` "How rude!" |

Ann `$B3A2A7`:

| Item | Delta | Text |
| --- | --- | --- |
| `0x02` poison mushroom | −8 | `0x03B4` "nasty trick" |
| `0x01`, `0x03`–`0x07`, `0x10`–`0x17` | +6 | `0x03B2` |
| `0x19`, `0x49` | +2 | `0x0212` |
| `0x09`–`0x0C` | −2 | `0x03B4` |

Flower `0x05` is inside Ann's `0x01`–`0x07` range, so the decoded Ann flower delta is +6. Live on item `0x05`: Maria `$B39637` text `0x03B2` and `41 1F 1F 7F 05 00` applied +5; Ann `$B3A2EF` text `0x03B2` and `41 21 1F 7F 06 00` applied +6. See the live log.

Nina `$B397C1`: poison mushroom and `0x09`–`0x0C` are −4 text `0x01D9`. Flower `0x05` and `0x06` are +8 text `0x0212`. `0x01`, `0x03`–`0x04`, `0x07`, `0x10`–`0x17` are +4 text `0x03B2`. `0x18` is −6 text `0x03B1`. `0x19` is +6 text `0x0143`.

Ellen `$B399C9`: poison mushroom −8 text `0x03B0`. `0x01`, `0x03`–`0x05`, `0x07`, `0x10`–`0x13` are +6 text `0x03B2`. `0x14`–`0x17` are +8 text `0x03B3` ("bake a cake"). `0x19` and `0x49` are +5 text `0x0212`. `0x09`–`0x0C` are −4 text `0x01D9`.

Eve `$B39C07`: poison mushroom and fish `0x07` are −8 text `0x03B4`. `0x01`, `0x03`–`0x05`, `0x10`–`0x17` are +5 text `0x03B2`. `0x06`, `0x19`, `0x49` are +8 text `0x0212`. `0x09`–`0x0C` are −4 text `0x03BA`.

The same girls have more ladders (wife and festival copies). They are not the same numbers. Ann's copy at `$B3C06E` pays +5 where the primary pays +6. Do not merge them. Other event-1 `0x2D` sites still unclassified include `$B393E8`, `$B394B2`, `$B39D26`, `$B3A464`, `$B3A73D`, `$B3AA11`.

## Other triggers

Choice, day, and hour rows are the pattern scanner's heart-block hits, so the address is the `0x41`, not the compare.

- Hour `== 18` writes −2 for every girl, then a flag-bit jump on `$7F1F64` index 7: Maria `$B5805A`, Ann `$B58514`, Nina `$B589CA`, Ellen `$B58E88`, Eve `$B5934D`. Bytes at Maria are `41 1F 1F 7F FE FF 14 64 1F 7F 07`. Whether the loader runs this once a night is not traced. Marriage bit `$7F1F66` was not found on this window.
- Day `== 30` is +5 twice each for Maria, Ann, Nina, and Ellen. Eve has no day-30 row in the scan.
- Repeatable choice adds of +8 / +2 / +1 / +4 / +10 show up on later copies of the same question. Nina has one choice-3 site at +30. Those are separate from the D1 intro.
- One-shots, one address each: `+100` Maria `$B4A106`, Ann `$B4A19B`, Nina `$B4A269`, Ellen `$B4A2E7`, Eve `$B4A362`. `+50` pairs at `$B490AA` / `$B49B47` (Maria) and the same stride for the other four. `+30` and `−30` clusters sit at `$B3F167`, `$B3F1C0`, `$B48165`, `$B484FF`, `$B48653`, `$B48737`, `$B4A6D4`, `$B4A72D`.

Wife and indoor copies in bank `$B5` are a second set of sinks (Maria `$B5809C` −6 and `$B580CF` −4, and the matching Ann / Nina / Ellen / Eve pairs, plus event-68 copies). F1 owns the 650→496 recipe. Pregnancy checks hearts and does not write them: child 1 wants `≥ 450`, child 2 wants `≥ 650`, wife bits in `$7F1F66`.

## Schedule

PR-021 owns the timetable. What this pass can say: Spring D1 7:00, clear weather, weekday 1.

- Maria is inside the church, tilemap `0x1B`, stand about (103, 405). The live talk was at (108, 409), face up. Route `d1_town_to_maria` plus `d1_church_to_maria`.
- Ann is on town tilemap `0x04`. Route `d1_town_to_ann` ends near (392, 914). She was not on that pixel; `TrackNpcUntilBitTask` caught her and finished at (378, 906).
- Ellen is not one of the six `$7F1F74` D1 bits (`0x3F` is Ann, the `0x02` bar line, Nina, the flower shop, the livestock dealer, and Maria).
- Indoor clocks do not advance. Outdoor hour caps at 18. The bed is what reaches the night reset.

## Live log

Power-on, `HEADLESS=1`, `SDL_VIDEODRIVER=dummy`, `QT_QPA_PLATFORM=offscreen`, `--trials 1`, cwd `snes/harvest`. Entry was power-on (5755 frames to the town gate, (712, 424), tilemap `0x04`). No `LiveRamEditor`. No `--ram-set` on the walk. RAM writes on the talk walk are 0. Item plants and the ladder pointer writes were in memory on `LN_mgr-hearts_whatif_maria_flower` (and the earlier carry pin). The pin file was left as saved. Those sessions are research, not route evidence. CC structs live at `$7EB586`, stride `0x40`, script pointer at `+0x30`. Index 1 was the used church slot.

| Try | What ran | Hearts | Frames |
| --- | --- | --- | --- |
| Maria intro | text `0x00F9`, then `0x00FA` | 0 | power-on 5755, talk 282 ending frame 6823, second mash 700 |
| Maria gift, action word 5 | text stayed on the intro (`0x00FA` once). Item stayed 5. No `0x03B2` | 0 | 900 on the pin |
| Maria gift, action word 4 | text `0x036A` "Eat / Don't eat", then `0x00FA`. Held fell to 0 | 0 | 900 |
| Maria gift, action 5 poked every frame, `game_state` bit `0x0004` already clear (`0xC001`) | text stayed `0x00F9` | 0 | 500, then 180 |
| Ann intro | text `0x0103` | 0 | 1268 to reach her from the Maria pin, talk 240 ending frame 1508 |
| Maria flower ladder | CC index 1 set to `$B395D5`, item `0x05`. Frame 1: text `0x03B2`, lock 2, PC on `41 1F 1F 7F 05 00`. Frame 109: lock 1, Maria 5, PC `$B395A9` | Maria 0→5, Ann 0 | 109 |
| Ann flower ladder | same pin, CC index 1 set to `$B3A2A7`, item `0x05`. Frame 1: text `0x03B2`, PC on `41 21 1F 7F 06 00`. Frame 109: Ann 6, Maria still 0, PC `$B3A27D` (exit `0x4D` then the idle `0x2D`) | Ann 0→6 | 109 |

Decoded flower deltas and the live deltas are the same: Maria +5, Ann +6, text `0x03B2`. Intro talks are +0, and the live intros were +0. Probe JSON: `F9_hearts_probe.json`, `F9_hearts_ann_try.json`, `F9_hearts_gift_try.json`, `F9_hearts_gift_try2.json`, `F9_hearts_ladder_maria.json`, `F9_hearts_ladder_ann.json`.
