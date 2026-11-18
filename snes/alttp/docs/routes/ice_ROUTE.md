# Dungeon route — Ice Palace

Isolated development route. A save-state pin is not continuous and not natural. Facts below are from `refs/z3-json-data` (pin `1eb7a785`) or a live scan. Hint lines are not authority.

## Entry

| Field | Value |
|-------|--------|
| Pin | `Whatif_IcePalaceEntry_parallel` (kind poke, room `0x0E`, module `0x07`). Predecessor `IcePalaceEntry` (kind poke). Not natural. |
| Room | `0x0E` dungeon `0x12`, dark-world flag 0 (Light World entrance warp). Spawn `(7544,472)`, doorway `$6C=1`. |
| First hop | `north_to_0x1e` isolated 2/2. Not natural. |

`Whatif_IcePalaceEntry_parallel` writes, all pokes: big key `$F366=0x0040`, keys `$F36F=2`, hammer `$F34B=1`. `IcePalaceEntry` already has fire rod, Titan's Mitt (`gloves=2`), boots, flippers, pearl, master sword, magic `0x80`, keys `0`. Neither pin writes `$F3CC` (follower stays 5).

Freezor slot 0 type `0xA1` at `(7520,296)` hp 16. Four sword swings left hp 16. Fire rod (`select_y_item`, use `$0303=0x05`) from `(7531,344)` facing UP: three shots, magic `128→112→96→80`, sprite gone, submodule `0x05` (shutters) for 7 frames. West door slot 1 stays type `0x18`. Before the melt, a walk toward `(7232,367)` stuck at `(7464,377)`. After the melt, LEFT from `(7488,365)` reached `(7232,396)` still room `0x0E`.

`Play_ice_0x0e_melted` (kind resumed, source `play:Whatif_IcePalaceEntry_parallel:fire_rod`) is that post-melt pose, hp `0x40`, keys still 2. `room_engine room_0e north_to_0x1e` from that pin was green twice, both 582 frames, both land room `0x1E` `(7289,823)`, floor `0xFF`. Audit of `Play_ice_0x0e_melted`, `Scan_ice_0x1e`, `Whatif_IcePalaceEntry_parallel`, and `IcePalaceEntry`: 0 incoherent. `room_engine` does not press Y, so it cannot melt the Freezor; the edge does not start on the untouched entry pin.

`$F36F` went `2→1` during the stair (already 1 at submodule `0x14`, still 1 after settle). No slot was type `0x1C`. `IcePalaceEntry` keys are 0, so this is not a restore of the unpoked count. Cause not identified. Not a key that rose in play. Big key stayed `0x0040` (still the poke).

## Requirements

From `dungeon/route_data/ice.yaml` / `regions/dungeons/ice/main.json`. `mods.ohko` is off, so the `not mods.ohko` term holds and hookshot / `canInvul` are spare.

| Item | Why | z3 source |
|------|-----|-----------|
| `canLiftDarkRocks` | Entrance and exit | Ice Palace Entrance Room, Ice Palace Exit |
| `canMeltThings` | Entrance, exit, Freezor chest, prize | same nodes |
| `canSwim` | Entrance and exit | same nodes |
| `RescueZelda` | Entrance and exit | same nodes |
| Hammer | Big key chest, map chest, prize | Ice Palace - Big Key Chest `0xE9A4` |
| `canLiftRocks` | Big key chest, map chest, prize | same |
| `BigKeyD5` | Big chest, prize | Ice Palace - Big Chest `0xE9AA` |
| `KeyD5:2` or (`CaneOfSomaria` and `KeyD5`) | Prize only | Ice Palace - Prize |

The parallel pin's hammer, big key, and two keys are pokes. They do not satisfy "rose in play".

## Room table

Geometry stays in `maps/`. Do not copy these xy into Python.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| `0x0E` | north slot 0 `0x22` stairs to `0x1E`; west slot 1 `0x18` shutter (stopping lock); south slot 2 `0x0A` entrance, screen `0x36`; slot 9 is the other layer of the west shutter | none | Freezor `0xA1` until the fire rod; three blue Bari `0x24` | Floor `0x00`. Ice on the west. West door tile `(7365,377)` stays in `0x0E` on all four pushes. South DOWN from `(7538,407)` leaves indoors at `(3256,3531)`. |
| `0x1E` | east slot 0 `0x18` shutter (stopping lock); west slot 1 `0x18` shutter (stopping lock); north slot 2 `0x18` passable UP back to `0x0E`; south slot 3 `0x00` regular (stopping lock, not reached) | none | crystal switch; four red Bari; two Zol | `Scan_ice_0x1e` landing `(7289,822)`, floor `0xFF`, keys already 1. ROM tag byte `0x17` (north move-block). Landing flood is cols 5-26, rows 37-58. Crystal, east shutter, south door, and block `(47,48)` are outside it. |

## Keys and big key

Not taken on the walked rooms `0x0E` and `0x1E`. `IcePalaceEntry` keys `0`, big key `0`. The parallel pin pokes keys `2` and big key bit `0x0040`. Holding UP into the north stair mouth dropped keys `2→1` while the room was still `0x0E` at `(7291,304)`, and the completed stair also left keys at 1. No slot was type `0x1C`. Big key stayed `0x0040` (still the poke). Big key chest is yaml `Ice Palace - Big Key Chest` ROM `0xE9A4` (Hammer, `canLiftRocks`). The chest table joins that file offset to room `0x1F` (not a big-chest bit). ROM room `0x1F` has one chest object at tile `(45,54)`. No chest slots in `0x0E` or `0x1E`. `Whatif_Ice_0x1e_bk0` sets the poked big key back to 0; hammer was already 1. The bit did not rise. Ledger writes were empty.

`Whatif_Ice_0x1f_chest` (kind poke, source `poke:Whatif_Ice_0x1e_bk0`, frame 108) is an isolated room `0x1F` load. It is not natural and not continuous. Link stands at `(7528,944)`, origin `(7168,512)`, tile `(45,54)`, indoors, module `0x07`. Live chest 0 is `BigKeyD5` loc `$1B5A` opened=0, file `0xE9A4`. Layer 0 attribute `0x58` covers `(45,54)`, `(46,54)`, `(45,55)`, and `(46,55)`. Facing `$2F=0`, one-frame A on each of those cells left action 0 and `big_key` 0. Ledger empty. Hammer stayed 1. Follower stayed 5. Ability `$F379=0x1E` is the Ice entry lift-2/swim/dash poke; the chest A-press mask is `0xE0`, and `0x1E` does not include it. Not a rise.

`Whatif_Ice_0x1f_open` (kind whatif, source `whatif:Whatif_Ice_0x1f_chest`) is that same isolated load with one ability bit added. It is not natural and not continuous. `$07:9C2F` does `LDA $9BA2,X` / `AND.l $7EF379` / `BEQ`. Open-chest is action 5 (`$07:D442` when `$02E5` bit 0 is set) and the table byte is `0xE0`. The AND passes if any of `0x20`, `0x40`, or `0x80` is set. The copy ORs only `0x20`, so `$F379` is `0x3E` (`0x1E|0x20`). `pin_probe` reads `ability=62`, room `0x1F`, `(7528,944)`, tile `(45,54)`, facing `0`, chest 0 opened=0, follower 5. One idle frame left that pose and `big_key` 0. One A then left `big_key` `0x0000→0x0040`, chest opened=1 (live `$0100`, save `0x0010`), `$02E5=1`, `$036C=5`, `$5D` 0, follower 5. Ledger row: `big_key` `0→64` source `play` frame 2 room `in:001F` (`recordings/probe_AP-165/ap165_ledger.json`). Edges from the entry are not isolated 2/2. Not a Clean claim.

## Big chest

Yaml `Ice Palace - Big Chest` ROM `0xE9AA`, item `BlueMail`, requires `BigKeyD5`. The chest table joins that file offset to room `0x9E` with the big-chest bit set. Not entered. On `Play_ice_0x0e_melted` and after the west ice crossing, armor `$F35B` stayed 0. The parallel pin's big key is still the poke.

## Mechanics

Freezor `0xA1` is sword-immune. Fire rod (three shots, 16 magic each on this pin) removes it. North stairs change the room to `0x1E` (`room_engine` `north_to_0x1e` 2/2).

Ice is attribute `0x0F`. Measured on `Play_ice_0x0e_melted` (`recordings/probe_AP-163/ap164_ice.json`). Floor, body on floor at `(7488,365)`, hold UP 24 frames: step pair `(0,-1),(0,-2)` (1.5 px/frame), stop distance 0, end `(7488,329)`. Parked on ice at `(7335,376)`, all four body tiles ice. LEFT 24 frames: about 5 still frames, then mostly `-1`, coast 7 px, end `(7316,376)` still ice. LEFT 72 frames: same stall, then a ramp, then the steady pair `(-2,0),(-1,0)` (1.5 px/frame, opposite phase of the floor pair). Release coasts 18 px and stops at `(7237,376)`, still ice. RIGHT 24 frames leaves the ice; once the body is floor the pair is `(1,0),(2,0)` and the coast is 0, end `(7361,376)`. UP 24 frames ends `(7334,368)` still ice, coast 0. DOWN 24 frames ends `(7334,384)` still ice, coast 0. The band at this pose is wider east-west than north-south, so the vertical samples stall before a coast.

`walk_physics.py` records that coast as `ICE_STRAIGHT_STEP_PX` `(2, 1)` and `ICE_COAST_PX` 18 on `Play_ice_0x0e_melted`. The 18 px release is only after that horizontal cap. Floor UP still coasts 0. Room `0x1E` has no ice on the landing flood, so the same walker holds until 2 px on floor.

`push_block` watches object state 0. Room `0x0E` has no `$0500` slot and no attribute `0x70`–`0x7F`. Room `0x1E` slot 5 is state 0, position tile `(47,48)`, attribute `0x75`. The four `0x74` cells at `(42,18)` are slot 4 state `0x1111` (pots, not a resting block). The landing flood is 136 poses and does not include `(47,48)`, so `push_block` was not stood. No boss-door edge from that landing.

LEFT from `(7488,365)` enters the `0x83` mouth: doorway `$6C` 2, submodule `0x01` then `0x05`, then control returns on ice still in room `0x0E`. The west door type stays `0x18`. The north stair tiles `(15,35)` `0x5E` and `(15,36)` `0x34`, with `0xF0` on rows 37–38, are outside that ice component. A hold toward the `0x34` cell stopped at `(7249,336)` with the body still `0x0F`. Three blue Bari `0x24` share that ice; 30 sword cycles left defeated slots empty and at least two at hp 2. `$F3CC` stayed 5. Log `scratch/probe_ap166_mouth.json`.

## Boss

`ice.yaml` location `Ice Palace - Kholdstare` is `BossHeartContainer` at `0x180157`. That file has no room word. Enemy id 162 is sprite type `0xA2`. The `$09:D62E` sprite list has that type in one room, `0xDE`, with the shell `0xA3` and falling ice `0xA4`. Palace `0x12` includes `0xDE`.

`Whatif_Ice_kholdstare` is kind poke, indoors room `0xDE`, module `0x07`, dungeon `0x12`. Audit: 0 incoherent, hp `0x40`/`0x40`. Source `Play_ice_0x0e_melted` was room `0x0E` with `fire_rod=1`. Warp submodule `0x15` kept `$A2` at `0x0E` and set `$A0` to `0xDE`. `$F3CC` stayed 5. `pin_probe` shows the shell slot 0 state `0x09` hp 64 ai 0 and Kholdstare slot 2 state `0x09` hp 64 ai 3, both `(7544,6744)`, plus falling ice slot 1 hp 8 at `(7536,6736)`. Link's warp landing is `(7488,7021)`, tile `(40,45)` floor. Isolated poke. Not natural, not continuous, and not a STATUS claim. There is no `Scan_ice_boss` pin.

Rows 27–38 are wall on every column. No door slots. `move_to` toward the shell stuck at `(7510,6961)` still south of that wall. A fire shot from x=7488 died on the wall; shell hp stayed 64. That walk was not saved.

Idle 1800 frames on the saved pin (`recordings/probe_AP-167/boss_idle.json`): shell and Kholdstare hp and `$0D80` did not change. Falling ice slot 15 appears at frame 125 on Link, shards slots 12–14 at frame 191. Link hp `64→48` at frame 192, `→32` at 320, `→16` at 448, `→0` at 576. Green-mail contact from that ice is 16.

An unsaved position poke put Link on north floor. It was not written to the pin. `$F3CC` stayed 5. Master sword at `(7544,6760)`, 16 px south, facing UP, left the shell at 64 and the eyeball at 64 ai 3 (`hurt_sword.json`). Fire rod from `(7544,6777)` spent magic `80→64` and dropped the shell `$0E50` `64→56` at frame 11. The eyeball stayed 64 ai 3 (`hurt_north.json`). z3 give-damage for the shell is 0 on the sword classes and 8 on fire class 11. Index 13 is 64 and this pin's Bombos byte is 0. Kit on the pin: sword 2, fire rod 1, magic 80, armor 0, hammer 1, ice rod 0, bow 0, arrows 0, bombs 0, lamp 0, bottles 0.

The same unsaved poke leaves BG2 at `$E2=1D00` `$E8=1B01` (`recordings/probe_AP-168/probe_cam.py`, 250 frames). Attack ice still spawns on Link, at frame 190 in that stand, with `$0F70=224` and `$0D80=0`. It stays there. Link HP stays 64. On the saved south pose the same ice lands and hits for 16. `bosses/kholdstare.py` still leaves an attack-ice xy. Wall rows 27–38 are not walked.

Five aligned shots are the whole meter (`probe_volley.py`, then `stage_replay` offset 0). South of the shell the first four drop `$0E50` `64→56→48→40→32` while magic goes `80→64→48→32→16`. The fifth is from the west flank after that ice exists: shell `32→24` at fight frame 178, magic `16→0`, Link `(7507,6748)`. One drop of 8 per shot. `strike_if_safe` on the shell and the in-shell eye commits nothing (`sword_hits=0`). Eye stays hp 64 ai 3. `$F36C` stays 64. `hits_by_cause` is empty. `RESULT pass=0 frames=201`. The pieces and the heart do not spawn. Isolated poke. Not a STATUS claim.

`Whatif_Ice_kholdstare_magic` is that pin with `$F36E=128` only. Kind poke. Shell HP was not written. `$F3CC` stays 5. Eight fire shots drop the shell `64→56→48→40→32→24→16→8→0` (16 magic and 8 shell HP each). The shell sprite then leaves, and three eyes (`0xA2`, ai 0, hp 64) spawn. A master-sword swing from the south window (cross-axis within 8, gap 24–34, facing UP) drops one eye `64→60`. Falling ice is 16. Eye contact is 32. There are no bottles. Screen Y 305 latches `$0F00` if BG2 scrolls off the shell; `fight` rewrites `$E2`/`$E8` only, not `$0CAA` and not `$F3CC`. Wall rows 27–38 are not walked. Isolated poke. Not continuous. Not a STATUS claim.

Offset 0 still dies before `$F36C` rises. Two leaves were tried after the shell hit 0 (`recordings/probe_AP-168/offset0_exit_down2.log`, `offset0_exit_left.log`). DOWN from `(7482,6809)`: HP `64→32` at frame 1401 `(7525,6848)` with an eye at chebyshev 32, then `32→16` at frame 1756 with ice on the tile, dead frame 2001 (`kholdstare:1`, `falling ice:2`). One sword whiff, no committed hit. LEFT from the same stand: HP `64→48` at frame 1120 `(7464,6829)` with ice chebyshev 14, then `48→16` at frame 1369 `(7464,6828)` with an eye at chebyshev 4, dead frame 1410. The ledger named that 32-point drop falling ice; the eye was the close sprite. Sword hits 0. Same two hits either way. Heart not taken.

| Field | Value |
|-------|--------|
| Sprite ids | Shell `0xA3` slot 0, Kholdstare `0xA2` slot 2, falling ice `0xA4` slot 1. Room `0xDE`. |
| HP byte | `$0E50`. Shell and Kholdstare start at 64. Falling ice starts at 8. Five fire shots leave the shell at 24. |
| Phase byte | `$0D80`. Shell stays 0. Kholdstare stays 3 (the in-shell prep) after the shell reaches 24. |
| What hurts it | Fire rod drops the shell by 8. A master-sword swing does not, including after magic is 0. Falling ice hits Link for 16 on the saved south pose. |
| When | One north-floor shot: frame 11, shell `64→56`, magic `80→64`. Five-shot fight: frames 10, 46, 82, 118, 178, shell `64→24`, magic `80→0`. Sword whiff at frame 201. South-pin ice: frames 192, 320, 448, 576. |

## Prize and exit

Prize node grants `Crystal5` (`0xC705` and the other yaml addresses) and requires Hammer, `canMeltThings`, `canLiftRocks`, `BigKeyD5`, and (`KeyD5:2` or Somaria plus one key). Not reached. Leave is not measured.

## Save points

Power-on save points only. The pins below are development loads, not continuous.

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| `IcePalaceEntry` | entrance poke `0x2D` | Room `0x0E`, module `0x07`, audit clean, kind poke |
| `Whatif_IcePalaceEntry_parallel` | `IcePalaceEntry` | Same room, plus poked big key, keys, hammer |
| `Play_ice_0x0e_melted` | `Whatif_IcePalaceEntry_parallel` | Freezor gone in play; `north_to_0x1e` 2/2 |
| `Scan_ice_0x1e` | that edge | Settled room `0x1E` `(7289,822)` at save, hp `0x3C` from a Bari hit on the probe path |
| `Scan_ice_0x0e` | UP from `Scan_ice_0x1e` | Return stair landing room `0x0E` `(7288,308)`, kind scan |
| `Whatif_Ice_0x1e_bk0` | `Scan_ice_0x1e` | Same landing, big key set back to 0. Hammer was already 1. Kind whatif. Not a rise. |
| `Whatif_Ice_0x1f_chest` | `Whatif_Ice_0x1e_bk0` | Room `0x1F` on chest tile `(45,54)`. Kind poke. Big key stayed 0. |
| `Whatif_Ice_0x1f_open` | `Whatif_Ice_0x1f_chest` | Same pose, ability `0x1E→0x3E`. Kind whatif. One A opened the chest; `big_key` `0x0040` in play. Not an edge. |
| `Whatif_Ice_kholdstare` | `Play_ice_0x0e_melted` | Room `0xDE` via warp submodule `0x15`. Kind poke. Shell and Kholdstare on screen. Not an edge. |
| `Whatif_Ice_kholdstare_magic` | `Whatif_Ice_kholdstare` | Same pose, magic `$F36E=128`. Kind poke. Shell HP not written. |

## Open reds

| Red | Evidence | Blocker |
|-----|----------|---------|
| West edge has no room id | LEFT stays `0x0E`; `at_door_destination` is false when `toRoom` is null | `room_engine.py` would need an intra-room landing check. Not edited. |
| `room_engine` cannot melt | Sword hp stayed 16; Y is not in the clear | Edge starts on `Play_ice_0x0e_melted`, not the untouched entry pin |
| Key count `2→1` | Stair submodule `0x14`, no `0x1C` slot | Cause not identified |
| Big key, hammer, keys | Parallel pin big key is still a poke. Isolated `Whatif_Ice_0x1f_open` ledger `big_key` `0→64` source `play` is not the walked route | Clean still needs the walk, not this pin |
| Hop table | `north_to_0x1e` is not in `spine/hops.py` | Step 8 needs that row. This step does not edit `hops.py`. |
| Ice coast constant | Floor pair is 1.5 px/frame; ice LEFT 72 coasts 18 px after the same 1.5 px/frame cap | Recorded as `ICE_COAST_PX` 18. Room `0x1E` landing flood is not ice. |
| `0x0E` west `0x18` | Door tile `(7365,377)` and LEFT from `(7488,365)` stay in `0x0E` | Shutter does not open. Named stopping lock. |
| `0x1E` east `0x18`, west `0x18`, south `0x00` | Flood cols 5-26 rows 37-58 does not touch those mouths. Switch `0x24` at `(6,56)` changed no attribute and no door type. RIGHT stopped on floor `(25-26,48)` short of `0xF9` `(27,48)`. Block `(47,48)` stayed state 0. | That walk does not enter `0x1F`. |
| Room `0x1F` chest | `Whatif_Ice_0x1f_open` ability `0x3E`, one A, opened=1, `big_key` `0x0040` source `play` | Room load is not an edge from `0x1E`. Step 3 still wants every edge isolated 2/2. |
| Kholdstare approach | Warp landing `(7488,7021)` is south of wall rows 27–38. No door slots. North-floor sword and fire were not saved. | Not walked. On `Whatif_Ice_kholdstare_magic` eight fire shots clear the shell. Offset 0 exits DOWN and LEFT both die to falling ice (16) and eye contact (32) before `$F36C` rises. Heart not taken. |
| Blue mail and the block | Chest table room `0x9E` not entered. Armor stayed 0. `0x0E` has no state-0 block. `0x1E` slot 5 tile `(47,48)` is outside the landing flood. | `push_block` had no stand. West `0x34` hold and 30 Bari cycles did not change room or doors. |
