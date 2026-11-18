# Dungeon route — Palace of Darkness

Isolated development route. A save-state pin is not continuous and not natural. Facts below are from `refs/z3-json-data` (pin `1eb7a785`) or a live scan. Hint lines are not authority.

## Entry

| Field | Value |
|-------|--------|
| Pin | `Whatif_PalaceOfDarknessEntry_parallel` (kind poke, room `0x4A`, module `0x07`). Predecessor `PalaceOfDarknessEntry`. Not natural. |
| Room | `0x4A` dungeon `0x0C`, dark-world flag 1. Spawn `(5368, 2520)`. |
| First hop | `east_stairs_to_0x09` isolated 2/2. Not natural. |

`Whatif_PalaceOfDarknessEntry_parallel` writes, all pokes: big key `$F366=0x0200` (read 512), keys `$F36F=1`, magic `$F36E=128`. `PalaceOfDarknessEntry` is the unpoked indoor pin (keys 0, big key not this write). Neither pin writes `$F3CC` (follower stays 5). Hammer `$F34B` stays 0.

`room_engine room_4a east_stairs_to_0x09` from the parallel pin returns frames 0, `in_origin_room` false, blocker `not in room 0x4A`, while Link is indoors at `(5368,2520)` room `0x4A` with `dark_world` true. `in_room` still rejects Dark World (AP-097). That run is not the hop.

`scratch/probe_ap131.py` walks `maps/room_4a.json` and does not use `in_room`. Two trials from the parallel pin both land room `0x09` `(4985,55)`, module `0x07`, submodule 0, control ready, push 140 frames. Follower stayed 5. Hammer stayed 0. Keys stayed 1 and big key stayed 512 (still the pokes). HP `56→40` from the east mini-helmasaur. Log `recordings/probe_AP-131/isolate_0x09.json`.

Tolerance 16 on the map path stops at `(5496,2232)` on the green-door jamb and never reaches the cage. Tolerance 8 reaches the stair. The map path tolerances are 8 after that measurement.

Audit of `Whatif_PalaceOfDarknessEntry_parallel` and `PalaceOfDarknessEntry`: 0 incoherent. Parallel pin hp `0x38/0x38`, sword 2, kind poke, room `0x004A`, dungeon `0x0C`, module `0x07`.

`pin_probe --doors --chests` on the parallel pin: no chests. Slot 2 north type `0x1C` small key at `$023C` was not walked. Slot 4 south type `0x0A` entrance to screen `0x5E`. Slots 0 and 1 north type `0x18` shutters. Slot 3 north type `0x00`. The stair to `0x09` is not one of those slots.

`spine/hops.py` has no Palace of Darkness row. This lane cannot edit that file. Needed row: `east_stairs_to_0x09`, verification isolated, not natural.

## Requirements

From `dungeon/route_data/pod.yaml` / `regions/dungeons/pod/main.json`.

| Item | Why | z3 source |
|------|-----|-----------|
| RescueZelda, isDarkLink | Entrance doors and the shooter-room key | Bridge Room, Bonk Wall, Exit, Shooter Room |
| KeyD1 | Arena bridge and stalfos basement, or hammer plus arrows | locations `0xEA3D`, `0xEA49` |
| canShootArrows | Arena ledge key, map, and later chests | Arena Ledge `0xEA3A`, Map `0xEA52` |
| hasLightCone, Hammer | Big key chest and the north / maze chests | Big Key Chest `0xEA37` |
| BigKeyD1 | Big chest and Helmasaur | Big Chest `0xEA40`, Helmasaur |
| KeyD1:6 | Helmasaur | Helmasaur location |

Small-key item nodes are Shooter Room `0xEA5B` (ROM room `0x09`), Arena Ledge `0xEA3A` and Arena Bridge `0xEA3D` (both ROM room `0x2A`), Stalfos Basement `0xEA49` (ROM room `0x0A`), Dark Basement Right `0xEA4F` (ROM room `0x6A`), and Dark Maze Bottom `0xEA58` (ROM room `0x19`). That is six `KeyD1` nodes. The parallel pin's key and big key are pokes. They did not rise in play.

## Room table

Geometry stays in `maps/`. Do not copy these xy into Python.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| 0x4A | south slot 4 type 0x0A to screen 0x5E; north slot 2 type 0x1C `$023C` to 0x3A; north slots 0 and 1 type 0x18 no path; north slot 3 type 0x00 stays in 0x4A | none | statue hp 0; two mini helmasaur 0x13 hp 4 | East 0x5E stairs to 0x09 are not a door slot. |
| 0x09 | no door slots; UP stair returns to 0x4A | chest 0 KeyD1 `0xEA5B` tile (24,22) unopened, no path | two medusa 0xC5 hp 0; antifairy 0x15 hp 255 | Landing (4985,54), floor 0xFF. Rays do not leave except UP. |
| 0x3A | south slot 0 type 0x1C back to 0x4A; north slots 1 and 2 type 0x00 no path; west-north passage is not a slot | chest 0 BigKeyD1 `0xEA37` tile (31,18) unopened, separate floor island, not a big-chest bit | two terrorpin 0x8E hp 8; two blue bari 0x24 hp 2; two medusa hp 0 | Landing (5368,2008). Center UP stops (5368,1800) on wall 0x04, north byte 0x01. |
| 0x2A | UP slot 0 type 0x18 and slot 1 type 0x1C; RIGHT slot 2 type 0x2E (not in the rom_scan type table) and slot 3 type 0x02 regular; DOWN slots 4 and 5 type 0x00 | chest 0 KeyD1 `0xEA3A` tile (56,45); chest 1 KeyD1 `0xEA3D` tile (46,44); both opened=0, no path | crystal switch; bumper; hardhat beetles 0x26 | Entered from 0x3A tile (14,24) at (5240,1496). |
| 0x5A | south slot 0 type 0x18 shutter `$1D5C` dest unknown | none | helmasaur king 0x92 slot 0 hp 48 ai 1 at (5463, 2920) | `Scan_pod_boss` poke warp, not a walked door. Map `maps/room_5a.json`. |

Rooms not walked from the lobby pin: `0x1A`, `0x0A`, `0x6A`, `0x4B`, `0x3B`, `0x2B`, `0x1B`, `0x0B`, `0x19`. `0x5A` was loaded by submodule `0x15` and saved as `Scan_pod_boss`. This lane cannot add files under `custom_integrations/`.

Stopping locks, from `scratch/probe_ap132_scan.json`, `probe_ap133_chest.json`, `probe_ap133_around.json`, and `probe_ap133_2a.json`.

| Lock | Live evidence | Why it stops |
|------|---------------|--------------|
| 0x4A north shutters | slots 0 and 1 type 0x18 | No layer-0 path from the spawn. |
| 0x4A center north | slot 3 type 0x00 | UP stays in 0x4A at (5368,2320). |
| 0x4A south exit | slot 4 type 0x0A | DOWN leaves indoors to (3920,1595), module 0x09, keys 255. Not continued. |
| 0x09 KeyD1 chest | tile (24,22) opened=0 | No path from (4985,54). DOWN stops (4985,192). |
| 0x3A north | slots 1 and 2 type 0x00 | No path from the south landing. |
| 0x3A BigKeyD1 chest | tile (31,18) opened=0, floor around it is 0x00 | Wall 0x01 on rows 25-32 cols 28-35 separates it from the south platform. Center UP stops at (5368,1800) on 0x04. Not in the legend-walkable flood. |
| 0x3A chest stairs | 0x5E at (31,3) (31,5) (31,6) | On the chest island. Not in the south-platform flood. |
| 0x3A to 0x2A | not a door slot | UP from body cell (14,24) enters 0x2A (5240,1496). Big key stayed 512. |
| 0x2A KeyD1 chests | `0xEA3A` and `0xEA3D` opened=0 | No path from the landing. `open_free_chests` skipped both. |
| 0x2A doors | six slots, all no path from (5240,1496) | Includes small-key slot 1 and shutter slot 0. |
| 0x2A crystal | sprite at (5376,1392), `$C172` stayed 0 | `hit_switch` did not toggle the peg. Keys stayed 0. |

## Keys and big key

Six `KeyD1` chests in the ROM chest table. One live small-key passage was crossed: room `0x4A` slot 2 type `0x1C` into `0x3A`, and `0x3A` slot 0 type `0x1C` is the return. The parallel pin's poked `$F36F` went 1→0 on that door. It did not rise in play. Room `0x2A` was entered and both of its `KeyD1` chests stayed closed (`0xEA3A`, `0xEA3D`). The shooter chest in `0x09` stayed closed. Rooms `0x0A`, `0x6A`, and `0x19` were not entered. Helmasaur's `KeyD1:6` is the logic total, not six doors seen live.

Big key `$F366` stayed `0x0200` (512), the poke, after the chest-island sample and after the `0x2A` visit. The BigKeyD1 chest in `0x3A` stayed closed. That poke is not a rise in play.

## Big chest

z3 item Hammer, address `0xEA40`, ROM room `0x1A`, big-chest bit set, requires `BigKeyD1`. Room `0x1A` was not entered. Hammer `$F34B` stayed 0. The `0x3A` chest is the big key, not the hammer.

`hammer` selects the hammer and swings once. On `Scan_swamp_0x37` (hammer `$F34B` already 1, a prior poke, not this chest) a disclosed stand off wall tile (31,59) pounded object slot 0, tile (16,8), attribute `0x70` to `0x00`. Stun `$0B58` stayed 0. Hammer on the PoD pin is still 0. Room `0x1A` was not entered.

## Mechanics

Lobby east stairs `0x5E` are the first room change and do not spend a key. The north `0x1C` spends the held key and loads `0x3A`. Bombs `$F343` are 0. No reached door slot was type `0x28`, so `place_bomb` was not used. Terrorpins in `0x3A` did not lose HP to 50 sword cycles. The west-north walk into `0x2A` did not open a chest.

## Boss

`Scan_pod_boss` is a poke copy of the room `0x5A` warp landing, saved from `Whatif_PalaceOfDarknessEntry_parallel` by submodule `0x15`. It is not a walked arena door and not an entry pin. Landing `(5368, 3032)` is wall tile `(31, 59)` attr `0x01`, `$006C=1`. The king is sprite type `0x92` (146) slot 0 at `(5463, 2920)`, tile `(42, 45)`. z3 names Helmasaur (`canShootArrows`, `hasLightCone`, `BigKeyD1`, `KeyD1:6`, Hammer) and prize Crystal1. Give-damage row for type `0x92`: sword class 1 is 0, hammer class 3 is 8, arrow class 6 is 4, bomb class 8 is 4.

| Field | Value |
|-------|--------|
| Sprite ids | one `0x92` Helmasaur King. No other sprite at the warp settle. |
| HP byte | `$0E50` on slot 0 reads 48. |
| Phase byte | `$0D80`. Warp settle and a 600-frame idle on tile `(42, 56)` stay 1 while `$0E80` counts (42, then 139). Walking north: 1 until about frame 25, then 2 with boss xy locked near `(5440, 2928)`, then 3 after contact (`$0DE0` 1→2), then 0 when `$0DF0` reloads to 64. |
| What hurts it | A disclosed bomb (`$F343` written to 5; the pin has 0) planted after 28 UP frames from tile `(42, 56)` drops `$0E50` 48→44. Same drop on a second load. `$0DB0` and `$0E90` read 1 after that blast, and `$0EF0` peaked at 52. A hammer Y (disclosed `$F34B=1`) at chebyshev ≤12 during phase 3 drops `$0E50` 48→47 and then sets `$0DB0` and `$0E90` to 1. That 1-point drop repeated on a second load. It is not the table's 8. |
| When | The bomb plant that hit was Link `(5456, 2966)` hp 56, phase 2, boss `(5440, 2928)`, doorway 0. The hammer that hit was phase 3, Link about 10px from the sprite, `$5D=0`. Sword B on contact left `$0E50` at 48. An arrow ancilla `0x09` from `(5460, 2967)` (count 30→29) left `$0E50` at 48 across 90 frames. Eight hammer Y presses from y 2969, 48px south of the sprite, also left 48. |

Door-only map: `maps/room_5a.json` (`arena_floor` is the disclosed stand, not a walked approach). Audit of `Scan_pod_boss`: hp `0x38/0x38`, kind poke, room `0x005A`, dungeon `0x0C`, module `0x07`, 0 incoherent. `boss_probe.py` 1800 idle frames from that landing: `$0E50` stayed 48, `$0DB0` stayed 0, `$0D80` cycled 1→2 at frame 30, 2→3 at 62, 3→0 at 126, and kept cycling. Fireball sprite `0x70` spawned at frame 349. Evidence: `recordings/probe_AP-135/scan_rooms.json`, `fire.json`, `chip.json`, `flight.json`, `boss_probe.json`.

## Prize and exit

z3 prize is Crystal1 (Palace of Darkness - Prize). Leave is the step-8 chain. Not risen in play.

## Save points

Power-on save points only. A `*Entry` pin is development only.

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| `Scan_pod_boss` | `Whatif_PalaceOfDarknessEntry_parallel` via submodule 0x15 (kind poke) | Room 0x5A warp landing. Not a walked arena door. Not an entry pin. |

`PalaceOfDarknessEntry` and `Whatif_PalaceOfDarknessEntry_parallel` are development pins for the isolated lobby hop. They are not power-on save points.

## Open reds

| Red | Evidence | Blocker |
|-----|----------|---------|
| room_engine dark world | frames 0, `in_origin_room` false, `dark_world` true, still room 0x4A | AP-097 `in_room` rejects Dark World. Probe hop does not use it. |
| 0x4A north type 0x1C | slot 2 opened into 0x3A, keys 1→0 | The spent key was the poke. It did not rise in play. |
| 0x09 KeyD1 and 0x3A BigKeyD1 | both opened=0 | Chest island in 0x3A is walled off. 0x09 chest has no path. |
| 0x2A keys | both KeyD1 opened=0, peg `$C172` stayed 0 | No path from the landing. Crystal hit did not open them. |
| 0x3A terrorpins | hp stayed 8 at the center wall | Sword cycles did not clear a path onto the chest island. |
| hammer on the PoD pin | `$F34B` stayed 0; room 0x1A not entered | The live peg is `Scan_swamp_0x37` tile (16,8) `0x70`→`0x00`. This chest did not open. |
| KeyD1 rooms still out | 0x0A, 0x6A, 0x19 not entered | No live path from the rooms entered. |
| hop table | `spine/hops.py` has no PoD edge | This lane cannot edit `hops.py`. |
| Scan pins | `Scan_pod_boss` is the room 0x5A poke only | Lobby rooms still have no `Scan_pod_<room>` walk. |
| Helmasaur heart | bombs drop `$0E50` by 4; try 1 reached 28 then Link hp 0 (bomb 2, fireball 2, king 1). Try 2 flee died at boss 44 (bomb 5, fireball 1). Arrows after `$0DB0=1` stayed 44. `strike_if_safe` sword hits 0 | Blast and fireballs empty 56 HP before 12 bombs. Heart container not taken. |
