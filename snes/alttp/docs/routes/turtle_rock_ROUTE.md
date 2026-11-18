# Dungeon route — Turtle Rock

Copy of `docs/routes/DUNGEON_TEMPLATE.md`. A save-state pin is not continuous. Facts below are from `dungeon/route_data/turtle_rock.yaml` (refs pin `1eb7a785`) or a live scan. Hint lines are not authority.

## Entry

| Field | Value |
|-------|--------|
| Pin | `TurtleRockEntry` (kind poke). Room `0x00D6`, module `0x07`, dungeon `0x18`, indoors, entrance `$010E=0x35`, HP `0x18/0x18`. Audit: 1 pin, 0 incoherent. Quake `$F349=0`. Small keys `$F36F=0`. Follower `$F3CC=5` was already on the load; the kit does not write it. |
| Room | `0xD6`. US dungeon entrance table ROM `$02C813` (file `0x14813`). Index `0x09` reads room `0x0084`. Index `0x35` reads room `0x00D6`, the same room as compass chest file `0xEA22`. |
| First hop | Unisolated. The forward door is slot 2 north type `0x00` (pos `$025C`). Two tries stayed in `0xD6`. Isolated poke only. The hop is not natural. |

Try 1, `recordings/probe_AP-179/pushes.json`: hold UP 240 frames from `(3448,7128)` to `(3448,7016)`, module `0x07`, indoors, HP `0x18`. DOWN left indoors (module `0x09`, screen 71, `(3848,312)`). That south exit is the backtrack. LEFT and RIGHT stayed in `0xD6`.

Try 2, `recordings/probe_AP-179/ride.json`: ability `$F379=0xF8` on the pin. `primitives.lift` lifted `(3448,7021)` and `throw` reported `(3448,7021)->(3448,7017)`. A second lift returned nothing lifted. Link stayed `(3448,7016)`. `select_y_item("somaria")` then Y spent magic `$F36E` `128→120`. No somaria platform sprite (`0xED`/`0xEF`/`0xF0`/`0xF1`). Final room `0xD6`.

`find_path` from the snapped spawn to all three door anchors returned no steps, including with pits allowed (`scan.json`). Floor+door flood from the south door tile `(47,59)` is 206 cells, columns 37–58, rows 36–63, and does not reach row ≤ 6.

`room_engine.py run` was not green. The north mouth is still unisolated. `ride_platform` is in `primitives.py`. The live calls spawn a platform and leave `$02F5` at 0. See Mechanics.

## Requirements

Items the entrance gap names, from `turtle_rock.yaml` gate `Turtle Rock Entrance Gap`.

| Item | Why | z3 source |
|------|-----|-----------|
| RescueZelda | Entrance gap | gate Turtle Rock Entrance Gap |
| canLiftDarkRocks | Entrance gap | same gate |
| hasProperMedallion | Entrance gap (Quake is AP-185, in play) | same gate |
| isDarkLink | Entrance gap | same gate |
| CaneOfSomaria | Entrance gap and compass chest | gate; location Compass Chest |
| Hammer | Entrance gap | gate Turtle Rock Entrance Gap |

The pin kit set sword, shield, progress, gloves, pearl, somaria, hammer, lamp, and magic. It did not set quake, keys, or `$F3CC`. Those kit bytes are a poke, not a rise in play.

## Room table

Rooms reached with the keys on the scan pin (small keys 0). Geometry stays in `alttp/maps/room_d6.json`. Dumps: `Scan_turtle_rock_0xd6`, `recordings/probe_AP-179/pin_probe_0xd6.txt`.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| 0xD6 | slot 0 north invisible `0x1A` pos `$021C` tile (14,4); slot 1 south entrance `0x0A` pos `$1D5C` tile (46,58); slot 2 north regular `0x00` pos `$025C` tile (46,4) | chest 0 CompassD7 tile (15,36) layer 0 loc `$121E` opened=0 z3=`0xEA22` | laser eye down `0x97` HP 255 at (3192,6680); two medusa `0xC5` HP 0 at (3120,7008) and (3264,7008) | Pin `Scan_turtle_rock_0xd6` kind scan, module `0x07`, dungeon `0x18`, link (3448,7128), layer 0, magic 128, ability 248, keys 0. |

Stopping locks, with small keys 0. The far side of each was not entered.

- North regular slot 2. Layer 0 on `Scan_turtle_rock_0xd6` has `0xB6` only at `(47,16)` and `(47,33)`. Columns 46–49 rows 44–45 are pots `0x70`/`0x71`. Rows 36–39 of those columns are floor, and row 35 is pit `0x20`. `$03F4=2`. UP from the spawn still stops on the south side of the pots. The closed hold-UP was not repeated.
- North invisible slot 0 and the compass chest. The entrance floor+door flood stays in columns 37–58. The chest tile is column 15. The laser eye sits on that wing.

Side mouths on the same ROM table, full room word in palace `0x18`. None of these were loaded.

| Index | Room word | Tie |
|-------|-----------|-----|
| `0x35` | `0x00D6` | Front pin. Compass chest. z3 node Turtle Rock Exit (Front). |
| `0x19` | `0x0024` | Big-chest record file `0xEA19` item MirrorShield, big bit set. z3 region Turtle Rock (Big Chest) also names Turtle Rock Ledge Exit (East). |
| `0x18` | `0x00D5` | Eye-bridge chest files `0xEA31` KeyD7, `0xEA2E` TwentyRupees, `0xEA2B` FiveRupees, `0xEA28` OneRupee. z3 region Turtle Rock (Eye Bridge) also names Turtle Rock Isolated Ledge Exit. |
| `0x15` | `0x0023` | No chest record in this room. z3 names Turtle Rock Ledge Exit (West) in the Second Section. This row does not assign that name to the index. |

Other chest rooms from the ROM table, not entered: `0x0014` BigKeyD7 file `0xEA25`; `0x0004` KeyD7 file `0xEA34`; `0x00B6` KeyD7 file `0xEA16`; `0x00B7` MapD7 `0xEA1C` and KeyD7 `0xEA1F`.

## Keys and big key

No small key and no big key was taken in play. The scan pin shows `$F36F=0`. The ride probe also left the big key at 0. Chest rooms above are the z3 locations. The big key chest is room `0x0014`.

## Big chest

Room `0x0024`, item MirrorShield, file `0xEA19` (big bit set). z3 location Turtle Rock - Big Chest requires canGrapple or CaneOfSomaria, BigKeyD7, and KeyD7 count 2. The chest was not opened. The shield byte did not rise in play.

## Mechanics

Column 47 of room `0xD6` is a somaria rail. The live layer-0 pipe starts are `0xB6` at `(47,16)` and `(47,33)`. Sprite names for a platform are `0xED`, `0xEF`, `0xF0`, `0xF1`.

`ride_platform` selects the cane and presses Y. It does not write `$F350`. A live success is a new platform sprite on `0xB6` or `0xBC`, magic `$F36E` lower by 8, `$02F5` set, and the reverse of `$0DE0` moving Link or the platform while `$02F5` stays set.

On `Scan_turtle_rock_0xd6` the cane byte is already 1. That is the kit poke, not a chest rise. `$03F4=2`. Follower stayed 5. Two calls, after a lift from tile `(46,46)` rather than the closed column-47 hold-UP, are in `recordings/probe_AP-174/ride_platform.json`:

- `(46,36)` left Link at `(3440,6945)`. Sprite `0xED` at `(3452,6924)`, direction 1, ai 1, on tile `(47,33)`. Magic `128→120`. `$02F5` stayed 0.
- `(47,36)` left Link at `(3449,6944)` with the same platform pose. `$02F5` stayed 0.

The floor edge does not overlap the platform. The north door was not isolated. The big key stayed 0.

## Boss

| Field | Value |
|-------|--------|
| Sprite ids | `0xCB` trinexx, `0xCC` trinexx fire head, `0xCD` trinexx ice head. z3 ids 203, 204, 205. |
| HP byte | `$0E50` is 40 on each slot at the landing. |
| Phase byte | `$0D80` AI. Load is 0. Aux `$0DB0` is 96 on the body and 64 on both heads. |
| What hurts it | A disclosed `$F345` fire rod drops the ice head `$0E50` by 4 at AI 3. A disclosed `$F346` ice rod flashes the fire head and leaves `$0E50` at 40. The owned hammer, 24 px north and facing down during ice-head AI 4, drops that head by 8. |
| When | Ice head `$0D80` is 3 and the sprite is on screen. The saved landing never leaves AI 0. |

`Scan_turtle_rock_boss` is kind poke, indoors room `0xA4`, module `0x07`, submodule 0, dungeon `0x18`. Audit: hp `0x18`/`0x18`, 0 incoherent. Source `Scan_turtle_rock_0xd6` was room `0xD6`. Warp submodule `0x15` kept `$A2` at `0xD6` and set `$A0` to `0xA4`. `$F3CC` stayed 5 and was not written. Isolated poke. Not a walked door, not an entry pin, not continuous, and not a STATUS claim.

`turtle_rock.yaml` location Turtle Rock - Trinexx has no room word. It requires hasLightCone, BigKeyD7, KeyD7 count 4, and CaneOfSomaria. Address `0x180159` is BossHeartContainer. Prize item on Turtle Rock - Prize is Crystal7. `$09:D62E` lists `0xCB`, `0xCC`, and `0xCD` together. The only palace `0x18` room with `0xCB` is `0xA4`.

Landing `(2424, 5592)` is wall tile `(47, 59)` attr `0x04`, doorway `$006C=0`, layer 0, camera `(2304, 5392)`. South slot 0 is shutter `0x18` pos `$1D1C`, dest unknown. No chests. Sword 1, shield 1, hammer 1, somaria 1, magic 128, fire rod 0, ice rod 0, keys 0, big key 0. That kit is the scan pin's poke, not a chest. There is no `maps/room_a4.json`.

Slots at the landing, state `0x09`: body slot 0 `(2168, 5484)`; fire head slot 1 `(2154, 5440)`; ice head slot 2 `(2181, 5440)`. `boss_probe.py` for 1800 frames from the saved pin: HP, AI, aux, and xy unchanged, no new sprite type, no ancilla, Link stayed `(2424, 5592)` HP 24. The body x is left of camera x 2304.

An unsaved `$E2`/`$E8` write to the live body minus 128 and 96 landed camera `(2040, 5388)`. It was not saved. Over 900 frames with Link still at the wall, HP 24: fire head AI 0→2 at frame 103, then 1 and 2, and 3 at frame 278. Ice head AI 0→2 at frame 230, and 3 at frame 473. Body stayed AI 0 aux 96 until frame 604 (AI 1), frame 611 (AI 0, aux 96→99, y 5484→5487), frame 659 (AI 2), frame 739 (AI 3), and frame 803 (AI 0). Every `$0E50` stayed 40. No ancilla. Slot 15 was a second `0xCC` at frame 341. Slots 14 and 15 were `0xCD` at frame 536. Logs: `recordings/probe_AP-183/phase.json`.

The nearest south floor `0x00` at chebyshev 2 from the body, with that camera write, sent Link HP 24→0 in 4 frames. That stand was not saved (`wake.json`).

Damage stands are offsets from the live sprite onto floor `0x00`. They were not saved. A rod trial writes `$F345` or `$F346` to 1 and does not save it. A connecting rod spends magic `128→112`. `$F3CC` stayed 5.

- Fire rod ancilla `0x02`, ice head AI 3, 48 px east, floor `(27, 41)`: ice `$0E50` 40→36, `$0EA0` 15, Link HP stayed 24 (`hurt.json`). The same drop at 24 px east, floor `(24, 42)`, then left Link at 0 (`hurt2.json`). Fire head and body stayed 40 both times.
- Ice rod ancilla `0x0B` on the fire head, floor, AI 2 and again at AI 3: `$0EA0` 15 and `$0E50` stayed 40. Link HP stayed 24.
- Fire rod on the fire head did not reach AI 3 (700-frame wait missed it) and did not flash. Fire rod on the body at AI 2 did not flash. Ice rod on the ice head at AI 3 did not flash. Those `$0E50` values stayed 40.
- Fighter sword, a 48-frame spin, and the hammer on floor at the `hurt2.json` offsets did not flash and left `$0E50` at 40. That is not a hit inside the head.

z3 `dmgFromLink` sword class 1 is 0 on all three. Fire class 11 is 4 on the ice head only. Ice class 12 is 4 on the fire head only. Hammer class 3 is 8 on all three. The live fire-rod drop matches class 11. The live ice-rod flash did not drop `$0E50`. The live hammer drop of 8 matches class 3 on the ice head during AI 4 only.

`bosses/trinexx.py` ran from `Scan_turtle_rock_boss`, idle offset 0. `$F345` and `$F346` were written to 1 and not saved. Those are disclosed writes. The pin's bytes are 0. Camera `$E2`/`$E8` was written from the live body and not saved. `$F3CC` stayed 5. `$0D80` was not written. Eight fire-rod hits spend magic `128→0` and leave the ice head at 8 and the fire head at 40. `$F36C` stays put (`recordings/probe_AP-184/try1c_stage.txt`). The hammer follow-up, 24 px north facing down while ice AI is 4, drops 8. The ice head then goes `40→0` by frame 1738. The fire head stays 40. Slot 0 stays `0xCB` and its `$0E50` does not change. Magic ends at 0. Link stays HP 24 at `(2424, 5592)`. `$F36C` has not risen at frame 9000 (`try2_stage.txt`). Message `0x0020` reaches `$1CD4` 3 while `$1CE9` is 28, and A closes it. Two tries. The heart line needs 10 of 12 offsets. Isolated poke, not continuous, and not a STATUS claim.

## Prize and exit

Prize in the yaml is Crystal7. The only leave measured from `TurtleRockEntry` is the south entrance backtrack: indoors cleared, module `0x09`, screen 71, xy `(3848,312)`, HP `0x18`. That glance is the overworld side of the entry door, not a prize exit.

## Save points

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| `TurtleRockEntry` | HyruleCastleGrounds plus entrance index `0x35` (kind poke) | Indoors, module `0x07`, room `0xD6`, dungeon `0x18`. Development pin. Isolated. |
| `Scan_turtle_rock_0xd6` | `TurtleRockEntry` (kind scan, frame 0) | Same room, doors, compass chest, and column-47 attributes. Keys 0. |
| `Scan_turtle_rock_boss` | `Scan_turtle_rock_0xd6` via warp submodule `0x15` (kind poke) | Room `0xA4`. Audit 0 incoherent. Not a walked arena door. Not an entry pin. |

## Open reds

| Red | Evidence | Blocker |
|-----|----------|---------|
| First forward edge | `ride_platform` spawned `0xED` on `(47,33)` and spent 8 magic. `$02F5` stayed 0 at `(3440,6945)` and `(3449,6944)`. Room stayed `0xD6`. | The floor edge does not overlap the platform. North door still unisolated. |
| Big key in play | Chest room `0x0014` not entered. Big key stayed 0 on the ride probe. `$F36F` stayed 0. | Rooms past the rail. |
| Mirror Shield in play | Big chest room `0x0024` not entered. | AP-182. Shield byte was not poked. |
| Trinexx fight | Disclosed rods plus the owned hammer kill the ice head. Fire head and body stay at 40. `$F36C` does not rise. | AP-184 blocked. |
| Natural approach | AP-178 and AP-095 are still open headings. Quake on this pin is 0. | AP-185. Quake in play belongs on that step. |
