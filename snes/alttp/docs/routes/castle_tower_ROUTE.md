# Dungeon route — Agahnim's Tower

Copy of `docs/routes/DUNGEON_TEMPLATE.md`. A save-state pin is not continuous. Facts below are from `refs/z3-json-data/regions/dungeons/castletower/main.json` (also `dungeon/route_data/castle_tower.yaml`) or a live scan on `CastleTowerEntry`. Not natural. Not a STATUS claim.

## Entry

| Field | Value |
|-------|--------|
| Pin | CastleTowerEntry |
| Room | 0xE0 |
| First hop | `south_to_0x1b` (map door, DOWN, to overworld screen 0x1B). verification=isolated. Not natural. |

`CastleTowerEntry` is kind poke (`source_run` `poke:castle_tower`, frame 120). Entrance table index `0x24` (decimal 36 in `pin_report.json`). The pin starts indoors. `audit_pins.py --pin CastleTowerEntry` and `pin_probe` agree: room `0xE0`, module `0x07`, submodule `0x00`, dungeon `0x08`, floor `0x01`, hp `0x18`/`0x18`, sword 2, shield 0, armor 0, lamp 1, keys 0, spawn `(120,7640)`. Audit issues: none. `pin_report.json` dark flag was false. Follower `$F3CC` was 5 and was not written. Sword was not written to open a barrier.

`room_engine room_e0 south_to_0x1b --no-clear` from this pin was green twice. Both runs were 220 frames. The exit phase landed screen `0x1B` at `(2040,1588)`, module `0x08`, indoors false, hp 24. Settle then reported `(2040,1627)`, module `0x09`, submodule `0x00`, hp 22, iframes 57, knockback 12, follower 5, sword 2, lamp 1, dark false. A separate two-load DOWN hold (`probe_south.py`) settled at `(2040,1628)` screen `0x1B` module `0x09` hp 24. Logs: `recordings/probe_AP-123/south_run1.json`, `south_run2.json`.

The north slot is an intra-room scroll that stays in `0xE0`. The east face is a type `0x18` shutter. Neither is this hop. `overworld.py` labels screen `0x1B` `hyrule_castle`. The z3 exit node is named Hyrule Castle Ledge; this landing was not matched to that node.

## Requirements

Items named by z3 for this region. The live pin kit is sword 2 and lamp 1, poked before the entrance trigger. That kit is not the DefeatAgahnim1 strat.

| Item | Why | z3 source |
|------|-----|-----------|
| hasLightCone | Agahnim 1 door, Tower Exit door, both KeyA1 chests | `castletower/main.json` nodes 1–4. `helpers.json`: hasLightCone requires canDarkNav and Lamp. `tech.json`: canDarkNav requires nothing. |
| Lamp | The lamp half of hasLightCone | `helpers.json` hasLightCone. Live pin lamp=1. |
| KeyA1:2 | DefeatAgahnim1 | strat id 1 |
| canSlashCurtains | DefeatAgahnim1 | strat id 1. `helpers.json`: swordless or hasSword. |
| swords.swordless and (Hammer or BugCatchingNet) | The only DefeatAgahnim1 strat | strat id 1 yields DefeatAgahnim1 |

## Room table

`CastleTowerEntry` entered only `0xE0` by walking. The other palace `0x08` rooms below were submodule `0x15` warps from that pin (`recordings/probe_AP-126/`). Isolated poke. Not walked doors, not an entry pin, not natural. Arena `0x0D` was not loaded. Floor byte was `0x01` on every one of these landings, the same as the lobby. Geometry for `0xE0` stays in `alttp/maps/room_e0.json`. The other rooms have no map. Do not copy approach xy into Python.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| 0xE0 | south slot 3 type `0x0A` entrance to screen `0x1B`; west slot 0 type `0x00` pos `$07C4`; north slot 1 type `0x00` pos `$121C`; stairs slot 2 type `0x20` pos `$025C`; layer slot 8 east type `0x18` shutter pos `$07B4`; layer slot 9 south type `0x18` shutter pos `$0D1C` | chest 0 KeyA1 z3=`0xEAB5` tile `(54,22)` layer 0 opened=0 | two ball-and-chain `0x6A` hp 16 at `(64,7264)` and `(176,7264)`; two assault soldiers `0x44` hp 6 at `(416,7264)` and `(416,7312)` | Pin `CastleTowerEntry`. North UP stays in `0xE0`. RIGHT from the west half stopped at x=200 on the door row. Stairs and the chest are on the unreached right half. |
| 0x30 | slot 0 north type `0x32` slashable pos `$021C`; slot 1 north type `0x00` pos `$121C`; layer slot 9 south type `0x18` pos `$0D1C` | none | two cutscene Agahnim `0xC1` hp 255 | Warp landing scrolled to `(120,1996)`. North `0x00` UP ended mode `0x0E` sub 2, room stayed `0x30`, xy `(120,1688)`. Not `0x0D`. |
| 0x40 | slot 0 west type `0x40` pos `$37CA`; layer slot 8 east type `0x40` pos `$37AE` | none | blue soldiers `0x41`, statue `0x1C`, red spear `0x43`, two blue archers `0x46` | Landing `(120,2520)` tile `(15,59)` byte `0x01` wall. Cardinal holds did not move. |
| 0xB0 | slot 0 north type `0x18` pos `$121C`; slot 1 west type `0x18` pos `$07C4`; slot 2 west type `0x1C` pos `$17C4`; layer pairs slots 8–10 | none | red spear `0x43`, keese `0x6F`, red javelin `0x48`, ball-and-chain `0x6A`, assault `0x44` | Landing `(120,6104)` tile `(15,59)` byte `0x04` wall. Key door no path. Shutter fight did not move. |
| 0xC0 | slot 0 west type `0x18` pos `$07C4`; slot 1 west type `0x00` pos `$17C4`; slot 2 north stairs type `0x20` pos `$025C`; layer slot 8 east type `0x00` pos `$07B4`; layer slot 9 east type `0x18` pos `$17B4` | none | blue soldiers `0x41` and blue archers `0x46` | Landing `(120,6616)` tile `(15,59)` byte `0x04` wall. Stairs and the regular door no path. |
| 0xD0 | slot 0 west type `0x1C` pos `$17C4`; slot 1 west type `0x00` pos `$07C4`; layer slot 8 east type `0x1C` pos `$17B4`; layer slot 9 east type `0x18` pos `$07B4` | chest 0 KeyA1 z3=`0xEAB2` tile `(6,26)` layer 0 loc `$0D0C` opened=0 byte `0x58` | keese `0x6F`, blue soldiers `0x41`, assault `0x44` | Landing `(120,7128)` tile `(15,59)` byte `0x04` wall. Row 58 is floor. Cardinal holds stayed. Chest and both west slots no path. |

Stopping locks, keys held 0. Lobby east shutter slot 8 type `0x18` (layer of slot 0): not passed. Stairs slot 2 and chest 0 in `0xE0` sit behind that shutter and were not reached. North slot 1 does not change room. The `0xD0` small-key slot and the `0xB0` small-key slot were not walked. The `0xC0` stair was not walked.

## Keys and big key

Two KeyA1 nodes in z3: `0xEAB5` Castle Tower - Room 03, and `0xEAB2` Castle Tower - Dark Maze. Both require hasLightCone. ROM chest records: `0xEAB5` is room `0xE0`, `0xEAB2` is room `0xD0`. The live chest in `0xE0` was opened=0. The live chest in the `0xD0` warp was opened=0 (`open_free_chests`: no path). Keys stayed 0. Lamp was 1 and magic was 0; the chest was not pressed. No big-key node is in `castletower/main.json`. No big-key bit was read rising in play. Big-key word stayed 0.

## Big chest

`castletower/main.json` has no big-chest node. No big chest was opened.

## Mechanics

The lobby east face is a shutter (type `0x18`, slot 8). `window.json` delay 0..248 step 8: every row ends at x=200, both `0x6A` stay hp 16, and some rows die (mode `0x12`). `east.json` door-row nudge dies at `(200,7299)` mode `0x12` with slot types unchanged. `chain.json` `fight_nearby` on type `0x6A` returns combat cycle limit, frames 672, defeated empty, both hp 16, Link hp 16, still room `0xE0`. The north mouth in `east.json` scrolls inside `0xE0` (submodule `0x01`, then `0x05`) and ends at `(120,7344)` submodule `0x00`. `pin_probe` on the entry pin listed only `0x6A` and `0x44`.

Warp submodule `0x15` into `0xD0`, `0xC0`, `0xB0`, and `0x40` settles on tile `(15,59)` with doorway `$006C` clearing in place. Eight-frame and forty-frame cardinal holds do not change xy (`step_in.json`). Reachable poses from that tile: 0. `0xB0` `fight_nearby` ran 480 frames, defeated none, xy stayed `(120,6104)`, shutter types unchanged. `$F3CC` stayed 5. These warps are not edges to room `0x0D`.

## Boss

`Scan_castle_tower_boss` is kind poke. Source `CastleTowerEntry` (room `0xE0`). Warp submodule `0x15` kept `$A2` and set `$A0` to `0x0D`. `$F3CC` stayed 5. Isolated poke. The arena door was not walked. Not continuous, and not a STATUS claim. `castle_tower.yaml` has no room word. z3 enemy id 122 is sprite `0x7A`. `$09:D62E` lists that type in palace `0x08` only in room `0x0D` (room `0x20` also lists it and is outside this palace). The room's sprite list is `0x7A`. z3 `dmgFromLink` for id 122 is sixteen zeros. `dmgToLink` is 8, 8, 8.

Landing `(6776, 472)`, doorway `$006C=1`, floor `0x01`, hp `0x18`/`0x18`, sword 2, lamp 1, shield 0, keys 0. South door slot 0 type `0x18` shutter. Slot 0 is `0x7A` state `0x09` hp 96, `$0D80`=4, at `(6776, 344)`. Coherence issues on that landing were empty. `audit_pins.py --pin Scan_castle_tower_boss`: hp `0x18`/`0x18`, kind poke, room `0x000D`, dungeon `0x08`, module `0x07`, 0 incoherent.

Idle 1800 frames on the saved pin (`recordings/probe_AP-127/load_agahnim.json`): Link stays `(6776, 472)` hp 24. `$0E50` stays 96. `$0D80` goes 4→5 at frame 36, 5→2 at 48, 2→3 at 88, 3→4 at 344, and that loop repeats (samples at frames 380, 404, 444, 700, 736, 760, 800, 1056, 1096, 1116, 1156, 1412, 1452, 1472, 1512, 1772). Ball `0x7B` (z3 id 123, hp 255) is born at frame 229, lead ai 0, vy byte 32. The lead reaches y 453 at frame 296 and Link's HP stays 24. Trail clones are ai 15. Four lead volleys in the idle (frames 232–296, 588–652, 944–996, 1300–1364), each dying near y 450–453. Type `0x55` appears at frame 653 and is not the boss.

The saved pose is the shutter tile. Reflect trials stand on floor rows of the live grid (row 52 at `(6780, 420)`, row 48 at `(6780, 388)`). Those coordinates were not written back to the pin. Each trial also writes `$F3C5` to 1 (`progress_poke` in `tap_sweep.json`) and does not save that byte. `$F3CC` was not written. Input is one `set_button_mask` B.

| Field | Value |
|-------|--------|
| Sprite ids | `0x7A` Agahnim slot 0, room `0x0D`. Ball `0x7B`. |
| HP byte | `$0E50` starts at 96 and stays 96 across the 1800-frame idle. |
| Phase byte | `$0D80`. Idle cycle is 4, then 5, then 2, then 3. |
| What hurts it | On row 52, one B while the ai-0 ball is north (`dy<0`) and vy is still positive flips that vy from 32 to −48 and drops `$0E50` 96→80. The body table is all zeros. |
| When | Press when chebyshev first reaches 12, 16, 20, 24, or 28: reflect at swing age 3, ball chebyshev 4, 8, 12, 16, 20, `$3C`=3. Press at 32: age 4, chebyshev 22, `$3C`=4. Press at 40: age 5, chebyshev 28, `$3C`=5. The 48 threshold used the same frame as 40 (the ball was already at chebyshev 40). Press at 8 peaks `$3C` at 2 and leaves 96. Row 48 never flips vy; `$0E50` stays 96. |

By the press frame, row-52 HP is already 16. The trial samples HP only after B, so that sample does not split a pre-press hit from the B frame. End HP is 16 after presses 20, 24, 28, and 32, and 8 after 8, 12, 16, 40, and 48. Row 48 ends at 8 or 12, including presses 8, 12, and 16 that never started a swing. The shutter idle takes 0. Log: `recordings/probe_AP-127/tap_sweep.json`.

z3 strat id 1 yields DefeatAgahnim1 and is the swordless line in Requirements. That strat was not run.

## Prize and exit

No prize byte was read. The isolated leave from `CastleTowerEntry` is overworld screen `0x1B`, module `0x09`. z3 names the region exit Hyrule Castle Ledge (`castle_tower.yaml` connection to that overworld node). Dark World was false on the pin and on the screen `0x1B` settle. The Pyramid warp was not seen.

## Save points

Power-on save points only. `CastleTowerEntry` is development only.

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| CastleTowerEntry | poke entrance index `0x24` from HyruleCastleGrounds | Indoors room `0xE0`, module `0x07`, dungeon `0x08`. Isolated south door to screen `0x1B`. Not natural. |
| Scan_castle_tower_boss | CastleTowerEntry via submodule `0x15` (kind poke) | Room `0x0D` warp landing. Not a walked arena door. Not an entry pin. |

## Open reds

What is still red. Do not weaken a check to close a row.

| Red | Evidence | Blocker |
|-----|----------|---------|
| East shutter and right half of `0xE0` | `window.json` ends at x=200. Entry slot 8 is type `0x18`. Chest opened=0. `chain.json` leaves both `0x6A` at hp 16. | No passage from the west stand. Keys stayed 0. |
| Dark Maze key `0xEAB2` | Warp room `0xD0`, chest opened=0, no path. Landing stuck on the south wall. | Keys stayed 0. Not a walked door. |
| Other palace rooms to the arena door | `0x30` north push stayed in `0x30` in text. `0x40`/`0xB0`/`0xC0`/`0xD0` landings did not move. | No edge isolated. Arena `0x0D` was not entered this step. |
| Agahnim fight and heart | Row 52 reflect drops `$0E50` 96→80 and also drops Link HP. Heart not taken. | Step 6. The arena door was not walked. |
| Natural approach | Pin kind is poke. | Needs AP-122 and AP-092. |
| Dark World and Pyramid | dark flag stayed false. Screen `0x1B` is not the Pyramid. | Chain step. `hops.py` is outside this edit list. |
