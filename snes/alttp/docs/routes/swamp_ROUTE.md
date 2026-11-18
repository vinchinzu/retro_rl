# Dungeon route — Swamp Palace

Isolated development notes. Not a continuous route and not a STATUS claim. Facts are from `refs/z3-json-data/regions/dungeons/swamp/main.json` or a live scan. `Whatif_SwampPalaceEntry_parallel` is a disclosed poke copy of `SwampPalaceEntry`: big key `$F366=0x0400` (reads 1024) and keys `$F36F=1`. Flippers were not written. `$F3CC` was not written. The source pin was not replaced.

## Entry

| Field | Value |
|-------|--------|
| Pin | Scan pin `Whatif_SwampPalaceEntry_parallel` (kind poke). Live 2026-10-18: room 0x28, dungeon 0x0A, module 0x07, (4344, 1496), layer 0, hp 24, keys 1, big key 1024, flippers 0, hookshot 0, hammer 1, follower 0. `SwampPalaceEntry` remains the source (kind none, keys 0, big key 0). |
| Room | `0x28`, layer 0, spawn (4344, 1496). Room save word `$F000` for 0x28 is 271. Every other palace room word was 0 on that load. |
| First hop | `north_to_0x38`. Live slot 0 north type 0x22 stairs, pos `$021C`, dest 0x38. Attribute 0x5E at tiles (15,3) and (16,3). Not isolated. |

`room_engine.py run room_28 --edge north_to_0x38 --state SwampPalaceEntry` stopped at (4344, 1376), still room 0x28, frames 550, blocker `stuck before canal_lip`. A second walk held UP to the same crest, then Link returned to about y=1495. East from the pocket stops at (4392, 1416). A, B, and the hammer there changed 0 attribute cells.

`Whatif_SwampPalaceEntry_flippers` is a disclosed copy of that pin (`$F356` written to 1, provenance kind whatif). `pin_probe` reads room 0x28, link (4344, 1496), inventory `flippers=1`, and slot 0 stairs dest 0x38. The same `north_to_0x38` hop on that copy used 415 frames and stopped at (4344, 1336), layer 1, `$5D=4`, module 0x07, still room 0x28. Blocker `door push failed (north_to_0x38)`. That pose is north of the flippers-0 crest and of `canal_lip` (4344, 1352). Room 0x38 did not load. Evidence: `recordings/probe_AP-139/room_engine_flippers.json`. Not natural and not continuous. `$F3CC` was not written.

## Requirements

| Item | Why | z3 source |
|------|-----|-----------|
| canSwim | Entrance moat. Also the indoor 0x08 plug rejects a body with flippers 0. | `Swamp Palace Moat` requires `canSwim` |
| canMirror | Entrance moat and exit | `Swamp Palace Moat`, `Swamp Palace Exit` |
| isDarkLink | Dark World entry | same gates |
| RescueZelda | Moat gate | same gates |
| Hammer | Big chest, compass, big key, west chest, north chests, Arrghus | location requires |
| KeyD2 | Map chest and every chest after the entrance key | `KeyD2` count 1 |
| BigKeyD2 | Big chest only | `Swamp Palace - Big Chest` |
| canGrapple | Flooded rooms, waterfall room, Arrghus | north locations |

## Room table

Only room 0x28 is reachable with the keys on the parallel pin. Flippers are 0, so the north stairs never load. Other palace rooms below were submodule 0x15 warps (kind poke), not walks.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| 0x28 | slot 0 north 0x22 stairs dest 0x38; slot 1 south 0x0A entrance screen 0x7B | chest 0 tile (22,12) KeyD2 `0xEA9D` opened=0 | 0x9A kyameron; 0x81 water tektite x3; 0x8A spike block | Reachable. `Scan_swamp_0x28` kind scan. Tag RAM [50, 27]. No sprite 0x21. Stopping lock: deep water 0x08. |

Reachable floor from the spawn, layer 0, is cols 25-38 on rows 50-58, the neck cols 30-33 on row 49, and cols 31-32 on row 48. North of that, 14 cells of 0x08. Shallow water (`0x09`) fills the north pool, including the key chest tile, and is not connected.

## Keys and big key

The entrance key is chest KeyD2 at tile (22,12), address `0xEA9D`, requires nothing in z3. It did not rise in play: the chest is in the north pool and stayed opened=0. The parallel pin's keys=1 and big key 1024 are the disclosed poke, not a chest. z3 big key chest is room 0x35 tile (23,16) `0xEAA6` BigKeyD2, opened=0 on `Scan_swamp_0x35`. That room was a warp, not a walk.

## Big chest

z3: `Swamp Palace - Big Chest`, item Hookshot, address `0xE989`, requires Hammer, BigKeyD2, and KeyD2. Live on the 0x36 warp: big chest tile (29,27) Hookshot opened=0. Hookshot byte stayed 0. `hookshot` fires one Y. The swamp scan pins have no floor, then pit or deep water, then a grabbable tile within 16 tiles. The live pull is `Whatif_Library` room `0x3d`: disclosed `$F342` 0→1, ancilla `0x1F` at (6960,10076), `$037E` 1 and `$5D` `0x13`, xy (6960,10176) tile (38,56) floor `0x00` across pit rows 44–55 `0x20` onto tile (38,43) attr `0xC0`. `$F3CC` stayed 5. That write is not the chest.

## Mechanics

The forward gap is deep water, not a door. Layer 0 tiles (30,47)-(33,47) and (31,46)-(32,46) are 0x08. Holding UP overlaps that plug and the room returns Link to the south shore. Layer 1 is 0x08 across the channel, with 0x0A at (31,48), (32,48), (18,39), (19,39), (36,39), and (37,39). Those cells were not a climb: layer stayed 0.

A 0x54 pair at (38,25)-(39,26) sits north of a floor corridor. That corridor is itself past the 0x08 plug, so it was not stood on. No `$C172`-style sprite was in the room. The hammer did not rewrite the grid.

`nav.py` treats 0x08 as a wall unless `allow` contains `deep_water`. The room_engine fallback is straight `move_to`, and that still cannot stay north of (4344, 1376) while flippers are 0. A disclosed `flippers=1` write does not paint the hop green either: the 2026-09-30 what-if ends on layer 1 at (4344, 1336) with `$5D=4` and never loads 0x38. Swim that isolates the stairs still has to be real movement after flippers rise in play (AP-093). `$F3CC` stays untouched.

AP-141 on `SwampPalaceEntry` wrote no inventory. `find_path` with `deep_water` is 322 poses against 113 without it (209 swim-only). The northmost swim-only pose is local (280, 208). The KeyD2 south stand has no path either way; the nearest swim pose is chebyshev 104 away, and layer 0 has no 0x70 peg. Walking that path puts 0x08 under the body at `$5D=6`, reaches y=1377, then frame 672 is back on south-door attr 0x8E at y=1488, still room 0x28. `hammer()` on the first overlap at (4344, 1398) missed and changed 0 cells on both layers; Link was still `$5D=6` at y=1379. `hookshot()` returns doorway while `$006C` is set beside the spawn on both pins, and "hookshot is not owned" once `$006C` is 0 on `SwampPalaceEntry` (`$F342` stayed 0; the parallel pin's hookshot byte is also 0). `dash()` at the dry-north pose y=1400 returns "no dash": boots 1, ability `$F379=0xF8` (bit 0x04 clear). Ledger changes empty. Big key stayed 0 and the chest stayed opened=0. The parallel pin's big key stayed 1024 with that chest still opened=0. No edge isolated. Evidence: `recordings/probe_AP-141/try.json`, `recordings/probe_AP-141/try2.json`.

Sprite 0x21 is the only water switch. Live on the poke scans: room 0x37 at (3760, 1605), room 0x35 at (2880, 1621), room 0x76 at (3472, 3637). Floor 0, ai 0, direction byte 0. Holding UP from south of the sprite never set `$0642`. The draw hit test runs only while the sprite is on screen, on the d=0 box (sprite origin +4,+20, size 8x6), with Link to the right of sprite y+19 and `$2F=4`. A disclosed stand at (sprite x, sprite y+11) plus scroll copies `$E2`/`$E8` (and `$011E`, `$0122`, `$0610`, `$0614`, `$0618`, `$061C`) let that check run. It is not a walk.

Each lever changes the room it is in. `$0642` rises inside the sprite routine and the loaded tag consumes it. The save word at `$F000` did not gain bit 0x800 in the same room. Live `$0402` became 0x800 and the tag bytes cleared to [0, 0]. Other rooms were not reloaded, so this does not record a cross-room write.

| Room | Tag before | What changed |
|------|------------|----------------|
| 0x37 | [0, 25] WaterOn | Deep water L0 2→10, L1 380→400. Cells became 0x08. Submodule settled back to 0. `$0402=2048`. Save word stayed 3. |
| 0x35 | [0, 25] WaterOn | Deep water L0 0→8, L1 416→436. `$0402=2048`. Save word stayed 3. |
| 0x76 | [0, 24] WaterOff | Submodule 0x0B seen. L0 deep water 12→4 (8 cells 0x08→0x1D). L1 unchanged. Shallow L0 stayed 240. `$0402=2048`. Save word stayed 10. |

Rooms 0x38 and 0x36 loaded with tag [0, 0] and no sprite 0x21, so this push does not name them as switch targets. Door-only maps: `maps/room_35.json`, `maps/room_37.json`, `maps/room_76.json`. Evidence: `scratch/probe_ap140.json`, `scratch/probe_ap140_switch.json`.

## Boss

`Scan_swamp_boss` is a poke copy of the `Scan_swamp_0x06` warp landing in room 0x06, not a walked door. Audit: hp 0x18/0x18, kind poke, room 0x0006, dungeon 0x0A, module 0x07, submodule 0, 0 incoherent. Link (3320, 472), hookshot 0, follower 0. z3 still names Arrghus (`canGrapple`, Hammer, KeyD2) and prize Crystal2.

| Field | Value |
|-------|--------|
| Sprite ids | one `0x8C` and thirteen `0x8D` |
| HP byte | `$0E50` on the core reads 32. Each puff reads 8. |
| Phase byte | `$0D80` on the core. Puffs stayed 0. |
| What hurts it | Phase 0 puff stays at 8 until `hookshot()` sets phase 1 (the call reports a retract; the pull is the phase byte). Two `strike_if_safe` swings then drop that puff 8→4→0. Core hp stayed 32 while puffs were alive. Core and puff contact each take Link from 24 to 8. |
| When | Sword from the wall pose (3320, 472) does not drop core hp. On water, offset 0 of `bosses/arrghus.py`: `lip_fight12.log` killed 4 puffs at Link hp 24, then the relocate hit. `lip_fight3.log` killed 7 of 13 puffs, Link hp 24→0, core still 32, no heart. `$F3CC` stayed 0. |

Orbiting puffs are sword targets only after the hookshot pull. The pin's hookshot byte is 0; the fight writes `$F342` once. That write is not the big chest. Evidence: `recordings/probe_AP-144/lip_fight3.log`, `recordings/probe_AP-144/lip_fight12.log`.

## Prize and exit

Prize in z3 is Crystal2. Not taken. South door slot 1 type 0x0A leaves to overworld screen 0x7B. That backtrack was already isolated; it is not the forward hop.

## Save points

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| `SwampPalaceEntry` | none on this lane (kind none) | Development load of room 0x28. Not natural, not continuous. |
| `Whatif_SwampPalaceEntry_flippers` | copy of `SwampPalaceEntry` (kind whatif) | `$F356=1` disclosed write. Probe reads `flippers=1` at (4344, 1496). The north hop still ends in 0x28. |
| `Whatif_SwampPalaceEntry_parallel` | copy of `SwampPalaceEntry` (kind poke) | Big key 1024 and keys 1. Flippers not written. Scan load of room 0x28. |
| `Scan_swamp_0x28` | `Whatif_SwampPalaceEntry_parallel` (kind scan) | Settled room 0x28 doors, chest, sprites. |
| `Scan_swamp_0x35` `Scan_swamp_0x37` `Scan_swamp_0x76` | same pin via submodule 0x15 (kind poke) | Water-switch rooms. Not reachable from the spawn. |
| `Scan_swamp_boss` | copy of `Scan_swamp_0x06` (kind poke) | Room 0x06 warp landing. Audit 0 incoherent. Not a walked arena door. |

## Open reds

| Red | Evidence | Blocker |
|-----|----------|---------|
| `north_to_0x38` not green | `room_engine_1.json` frames 550, flippers 0, end (4344, 1376). `room_engine_flippers.json` frames 415, disclosed flippers 1, end (4344, 1336) layer 1 `$5D=4`, still 0x28. | Door push from the north lip does not reach the stairs at tiles (15,3). The flippers byte write is not a room change. |
| Big key and hookshot not from play | 0x35 chest opened=0. 0x36 big chest opened=0. Hookshot byte 0 on the swamp pins. AP-141: KeyD2 opened=0, big_key 0, ledger empty (`probe_AP-141/try.json`). | Swim to the stairs is still AP-093. Off the door, `hookshot()` is not owned on `SwampPalaceEntry`. The pull check is the library pit, not this chest. |
| Arrghus heart not taken | `lip_fight3.log` 7/13 puffs, Link dead, core 32. `lip_fight12.log` 4 puffs at hp 24, then core contact. No heart. | After the east-edge kills the core sits on the south lip and the next relocate runs through a puff or the core. |
