# Dungeon route — Skull Woods

Copy of `docs/routes/DUNGEON_TEMPLATE.md`. A save-state pin is not continuous. Facts below are from `dungeon/route_data/skull.yaml` (refs `regions/dungeons/skwoods/main.json`) or a live scan on `SkullWoodsEntry`. Not natural. Not a STATUS claim.

## Entry

| Field | Value |
|-------|--------|
| Pin | SkullWoodsEntry |
| Room | 0x58 |
| First hop | `south_to_0x40` (map door, DOWN, to overworld screen 0x40). verification=isolated. Not natural. |

`SkullWoodsEntry` is kind poke. `pin_probe` and `audit_pins` agree: indoors, room `0x58`, module `0x07`, submodule `0x00`, dungeon `0x10`, hp `0x18`/`0x18`, spawn `(4216,3032)` tile `(15,59)` attr `0x8E`. Audit issues: none. `$0FFF` dark-world flag read 0. Follower `$F3CC` was 5 and was not written. Sword 1, pearl 1, keys 0, big key 0, fire rod 0, bombs 0.

`room_engine room_58 south_to_0x40 --no-clear` from this pin was green twice. Both runs were 198 frames and both settled on screen `0x40` at `(744,603)`, module `0x09`, indoors false, hp 24, follower 5. Two further trials left the same way, settled at that pose, then held UP and both returned to room `0x58` `(4216,3032)` module `0x07` dungeon `0x10` at frame 161. Logs: `recordings/probe_AP-147/south_run1.json`, `south_run2.json`, `reenter.json`.

The quoted entrance index `0x29` is not this measurement. The pin file was saved from entrance `0x2A` (kind poke). The live door on screen `0x40` is the one that loads room `0x58`.

## Requirements

Items named on `skull.yaml` locations. Not a claim that this pin holds them.

| Item | Why | z3 source |
|------|-----|-----------|
| RescueZelda, isDarkLink | First-section exit, bomb jump, south door, west door, map chest | locations Map Chest and those door nodes |
| BigKeyD3 | Big chest | location Big Chest, raw token `BigKeyD3 ` |
| FireRod | Big chest contents; bridge-room key; Mothula gate | locations Big Chest, Bridge Room, Mothula |
| KeyD3 ×3 | Mothula | location Mothula |
| canSlashCurtains, isDarkLink | Mothula | location Mothula |

## Room table

Keys held on the pin: 0. Big key: 0. The only indoors room entered is `0x58`. Geometry stays in `maps/room_58.json`.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| 0x58 | south slot 4 type `0x0A` entrance to screen `0x40`; west slot 5 type `0x00` mouth attr `0x81` cols 32-36 rows 48-49, walked east, stayed `0x58`; west slot 1 type `0x1C` pos `$1784`; south slot 2 type `0x1C` pos `$1D5C`; north slot 0 type `0x30` pos `$0D9E`; west slot 3 type `0x2E` pos `$07C4` | chest 0 FireRod big z3=`0xE998` tile `(14,43)` opened=0; chest 1 MapD3 z3=`0xE99B` tile `(56,40)` opened=0 | pin_probe: mini moldorm `0x18` hp 3, bumper, mini helmasaur hp 4, spark clockwise, pull switch down, hardhat beetle, red bari `0x23` hp 2 | Pin `SkullWoodsEntry`. Door-only skeletons for the unreached slots are in `maps/room_58.json` with no approach. Sword on the corridor bari dropped it from hp 2 to 0 and two hp-2 `0x23` remained. Lift on attr `0x27` lifted nothing. |

Stopping locks, keys 0 and big key 0. The far side of each was not entered.

| Lock | Live door | Why it stops |
|------|-----------|--------------|
| 0x58 west | slot 1 type `0x1C` pos `$1784` tile mouths `(3-4,48-49)` attr `0xF1` | Small-key door. keys=0. Not pushed. The entrance flood reaches the floor east of it. |
| 0x58 south | slot 2 type `0x1C` pos `$1D5C` | DOWN from the east floor `(4457,2936)` stalled at `(4484,3008)` tile `(48,56)` attr `0x00`, still `0x58`, doorway `$006C` 0, keys 0. |
| 0x58 north | slot 0 type `0x30` pos `$0D9E` | Explosion door. bombs=0. Not contacted. UP from the east floor walked attr `0x3B` at `(48,39)` then stalled on pit `0x20` at `(4480,2824)`, hp 24 to 8, still `0x58`. |
| 0x58 west upper | slot 3 type `0x2E` pos `$07C4` | Not reached. The north pit stopped the walk first. |

## Drop holes

z3 `connections/main.json` names the holes. That file has no room hex on these nodes. None of them was entered from `SkullWoodsEntry`. keys stayed 0. The indoor attr `0x20` walks in room `0x58` stayed in `0x58`.

| Origin | Connection | Destination |
|--------|------------|-------------|
| Skull Woods First Section Hole (East) | Door | Skull Woods First Section (Right) |
| Skull Woods First Section Hole (West) | Door | Skull Woods First Section (Left) |
| Skull Woods First Section Hole (North) | Door | Skull Woods First Section (Top) |
| Skull Woods Second Section Hole | Door | Skull Woods Second Section (Drop) |
| Skull Woods Second Section (Drop) | Drop | Skull Woods Second Section |

The East, West, and North holes and the First Section Door sit on overworld region Skull Woods Forest. The Second Section Hole sits on Skull Woods Forest (West). The only fall typed `Drop` is Second Section (Drop) into Second Section, where BigKeyD3 `0xE99E` lives. That fall was not taken.

Screen `0x40` after the south exit is a big overworld area, origin `(0,0)`, 128 by 128. Standing on attr `0x20` map16 `0x0817` at `(798,688)` for 90 DOWN frames stayed outdoors, module `0x09`, big key 0, no fall submodule. The same area has `0x20` pairs at `(488,360)` and `(632,712)`. No overworld path reached the stand `(808,680)` beside the near pair. Logs: `recordings/probe_AP-147/ap149_try.json`, `ap149_try2.json`, `ap149_pit.json`.

## Keys and big key

`skull.yaml` small keys are KeyD3 at Pinball Room `0xE9C8` and Pot Prison `0xE9A1`. Neither room was entered. keys stayed 0. Big key is BigKeyD3 at `0xE99E` in Skull Woods Second Section. big key stayed 0. No key was poked.

## Big chest

Big chest in room `0x58` is FireRod, address `0xE998`, tile `(14,43)`, opened=0. It requires BigKeyD3. `open_free_chests` from the spawn had no path to that chest or to the map chest at `(56,40)` (frames 0). Floor beside attr `0x81` also had no path: that byte is not a walkable legend kind. `$F345` stayed 0. Big key stayed 0. The chest was not opened.

`fire_rod()` on `IcePalaceEntry` (kind poke, rod already 1, not written) fired ancilla `(7544,424)` and magic `128->112`. That is not this chest and not a rise in Skull Woods. Screen `0x40` `(744,603)` still had `$F345` 0, so the shot refused. The AP-149 pit stands were not walked.

## Mechanics

The entrance corridor is separated from the east floor by attr `0x27` except the open mouth attr `0x81` on rows 48-49. Walking east through that mouth stays in `0x58`. North of the east floor, attr `0x20` at `(48,33)` stopped the walk and did not change rooms. `$F3CC` stayed 5.

## Boss

| Field | Value |
|-------|--------|
| Sprite ids | ROM room `0x29` list starts `0x88` (pointer table `$09:D62E`). Not seen live. |
| HP byte | not measured |
| Phase byte | not measured |
| What hurts it | not measured |
| When | Room `0x29` was not entered. Entrance `0x2B` loaded `0x59` (north slot 0 type `0x1C`, keys 0; UP died at hp 0 still `0x59`). Entrance `0x28` loaded `0x56`. `$A0` was not poked. No floor log. |

## Prize and exit

Prize in `skull.yaml` is Crystal3. Not taken. The leave measured here is the south entrance back to screen `0x40` `(744,603)`, module `0x09`. Not the crystal exit.

## Save points

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| SkullWoodsEntry | poke entrance load | Development only. Isolated south door to screen `0x40` and the walk back to `0x58`. Not natural. |

## Open reds

| Red | Evidence | Blocker |
|-----|----------|---------|
| Forward indoor room | East `0x81` stayed `0x58`. North stalled on pit `0x20`. South `0x1C` stalled in `0x58`. Outdoor attr `0x20` at `(798,688)` stayed module `0x09`. | keys=0, bombs=0. Big key did not rise. |
| Fire rod chest | Room `0x58` chest 0 opened=0. No path from the spawn or from a floor cell beside `0x81`. | BigKeyD3 stayed 0. A poke of `$F345` is not a rise. |
| Mothula room | ROM room `0x29` holds sprite `0x88`. Entrances `0x2B` and `0x28` loaded `0x59` and `0x56`. | North door of `0x59` is small-key and keys stayed 0. `$A0` was not poked. |
| Natural entry | Pin kind is poke | No predecessor walk |
