# Dungeon route — Thieves' Town

Isolated development pin. A save-state load is the ladder rung `isolated`. Facts below are from `refs/z3-json-data/regions/dungeons/thieves/main.json` or the 2026-09-30 live scan. Hint lines are omitted.

## Entry

| Field | Value |
|-------|--------|
| Pin | `ThievesTownEntry` (room `0xDB`, module `0x07`, dungeon `0x16`, entrance `$010E=0x34`). `Whatif_ThievesTownEntry_parallel` is a poke copy of that pin (kind poke): big key `$F366=0x0010` read back 16, keys `$F36F=1`, ability not written (`0x12`). Not natural. The big key did not rise in play. `Whatif_ThievesTownEntry_lift` is a what-if copy with ability `$F379` `0x12`→`0x52` (`0x12\|0x40`). Not natural. |
| Room | `0xDB`, southwest quadrant of the huge lobby. Link load `(5880,7128)`, layer 0, hp `0x38`. |
| First hop | `east_to_0xDC` (slot 0, type `0x18` shutter, dest `0xDC`). Red. `south_to_outcasts` is the overworld backtrack (slot 1, type `0x0A`, screen `0x58`) and was already isolated on 2026-09-21. |

`room_engine.py run room_db --edge east_to_0xDC --state ThievesTownEntry` on 2026-09-30:

- Try 1, map path `east_south` → `conveyor`: `ok=false`, stuck before `conveyor` at `(5928,7072)`, hp 56, still `0xDB`. Clear saw no nearby targets.
- Try 2, map path `north_band` → `east_band` → `walkway` → `pot_stand`: `ok=false`, stuck before `pot_stand` at `(5952,6731)` facing right, hp 56, still `0xDB`.

A 1-frame follower on the same floor reaches local `(432,64)` / world `(6064,6720)` against the northeast pot. The commit walker accepted the walkway 11px south of that phase and could not move east.

On `Whatif_ThievesTownEntry_lift` the same stand faces right at `(6064,6720)` dir 6. `primitives.lift` cleared the four `0x71` tiles to `0x00`. The thrown pot stayed on that tile. A later path to local `(464,248)` stopped at `(6088,6903)` local `(456,247)` dir 6 hp 56, and RIGHT did not enter `0xDC`.

## Requirements

From `thieves/main.json`. Entrance nodes also require `RescueZelda` and `isDarkLink`.

| Item | Why | z3 source |
|------|-----|-----------|
| BigKeyD4 | Attic, big chest, Blind's cell, Blind | nodes 8–11; chest node 3 address `0xEA04` |
| KeyD4 | Attic, big chest, Blind | nodes 8, 9, 11; cell node 10 address `0xEA13` |
| Hammer | Big chest | node 9 |
| TitansMitt | Big chest contents | node 9 address `0xEA10` |

## Room table

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| `0xDB` | east shutter `0x18` → `0xDC`; south entrance `0x0A` → screen `0x58` | layer 1: MapD4 tile `(16,20)` loc `$2A20`; BigKeyD4 tile `(49,44)` loc `$3662` | bunny `0xD1`, spark `0x5B`, red zazak `0xA6`, blue zazak `0xA5`, zol `0x8F`, firesnake `0x80` | Quadrant only. One same-floor `0xA5` kill left the shutter at `0x18`. |

Rooms past the east shutter are unscanned.

## Keys and big key

BigKeyD4 is the unopened chest at `0xDB` tile `(49,44)` layer 1, loc `$3662`, z3 `0xEA04`. On `ThievesTownEntry` the big-key word stays 0. On the parallel poke it is already 16 and the chest stays `opened=0` (`ap156_edge.json`). The small key in this file is Blind's cell, room `0x45`, address `0xEA13`, and it requires the big key. Neither rose in play.

## Big chest

Room `0x44`, address `0xEA10`, item TitansMitt. z3 requires Hammer, BigKeyD4, and KeyD4. The node note says the placement changes with key logic. Chest is unopened.

## Mechanics

Layer 0 of `0xDB` on this pin: no `0x68`–`0x6B` conveyor bytes. The red/orange stripes are attribute `0x1C` (1002 cells). The east chute is sealed by a 2×2 of `0x71` at tiles `(56,8)`, `(57,8)`, `(56,9)`, `(57,9)`. Object slot 1 at the matching position has state `0x1111`.

`$F379` ability is `0x12` on `ThievesTownEntry` via `read_inventory` (`wram_index`). `pin_probe` prints that as `ability=18`. Lift bit `0x40` is clear there, so `primitives.lift` returns nothing lifted. Gloves item is 1. A bomb ancilla born at `(6086,6732)` left the four tiles `0x71` after 200 frames. One hammer Y and sword taps from the adjacent stand also left them `0x71`.

`Whatif_ThievesTownEntry_lift` reads `ability=82` (`0x52`). Facing right at `(6064,6720)`, `primitives.lift` returned lifted `(6077,6725)` and tiles `(56,8)`–`(57,9)` became `0x00`. Object slot 1 stayed state `0x1111` pos `$0470`. Other layer-0 liftable 2×2s, same object-pos formula: `(6,8)` `0x70` slot 0 pos `$040C`; `(44,32)` `0x72` slot 2 pos `$1058`. Slot 3 pos `$333E` is tile `(31,38)` and is not in `0x70`–`0x73`. East shutter slot 0 type `0x18` pos `$0FF4` anchors tile `(58,31)`. RIGHT from `(6088,6903)` stayed in `0xDB`.

`opening_route/room_engine.py` lifts only through the Eastern `_open_lifted_shutter` role (hold UP). `nav.py` still treats `0x1C` as walkable stairs. `walk_physics.py` has no Thieves conveyor sample because those bytes are absent in `0xDB`.

AP-156 on the parallel pin (`ap156_scan.json`, `ap156_try2.json`, `ap156_edge.json`): both layers have zero `0x68`–`0x6B` cells. Layer 0 has 1002 `0x1C` cells. That byte is the stair overlay, not a sampled conveyor, and no px/frame per direction was measured. Needed shared change, not applied: a per-direction conveyor step next to the ice constants in `walk_physics.py`, and a conveyor cost in `nav.find_path`. Both files are outside this lane.

The layer-1 `0x3E` at tiles `(16,49)`–`(17,50)` drops to layer 1 still in `0xDB`, about `(5760,7013)` tile `(16,44)`. From that landing, `LINK_BODY` reaches columns 13–32 and rows 38–46 only (169 poses). BigKeyD4 at `(49,44)` is a separate layer-1 alcove, columns 46–62 and rows 38–48, touching the east edge. A body pose in that alcove has a 4-step path to the chest stand. The west stair shaft sends Link back to layer 0 inside `0xDB`. East shutter slot 0 stayed type `0x18` after the one zazak kill.

AP-158 on `Whatif_ThievesTownEntry_parallel` (writes 0, follower stayed 5, gloves stayed 1). The layer-0 flood is 938 cells. The northernmost legal cell is local `(64,48)`. Holding UP there rested at `(5704,6704)` tile `(9,6)` attr `0x04`, still `0xDB`, east slot still `0x18`. No cell has local y ≤ 16, and none has local x ≥ 480. The easternmost cell is local `(432,48)`, west of the `0x71` pots. `0xCB` did not load.

Floor-0 zol slot 4 at `(5808,6912)` took the sword from 4 to 0. The shutter stayed `0x18`. Bunny beam `0xD1` at `(5680,6720)` stayed hp 64 through swords from `(5696,6710)`. Firesnake `0x80` matched Link's speed along y=6704 with about a 100px lead, turned south at x=5668, and was not hit. A facing-up A at the `0x70` 2×2 tile `(6,8)` from `(5683,6731)` lifted nothing; those tiles stayed `0x70`. The `0x72` stand local `(368,256)` has no path that avoids the `0x3E` drop. Rooms `0x65` (attic) and `0x45` (cell) were not entered. Logs: `recordings/probe_AP-158/`.

## Boss

`thieves.yaml` has no room word. z3 enemy id 206 is sprite type `0xCE`. The `$09:D62E` sprite list has that type in one palace `0x16` room, `0xAC`.

`Scan_thieves_boss` is kind poke, indoors room `0xAC`, module `0x07`, dungeon `0x16`. Audit: 0 incoherent, hp `0x38`/`0x38`. Source `Whatif_ThievesTownEntry_parallel` was room `0xDB` with sword 2, hammer 1, bombs 10, follower 5. Warp submodule `0x15` kept the previous room in `$A2` and set `$A0` to `0xAC`. `$F3CC` stayed 5. Isolated poke. Not natural, not continuous, and not a STATUS claim.

Slot 0 stays type `0xCE` state 0 hp 90 when the room loads with `$0402` clear. A live `$0402` write before the warp is gone on arrival. ORing `0x0200` into the room `0xAC` save word `$F158` before the warp leaves live `$0402` at `0x2000`, slot 0 state `0x09`, hp 90, `$0D90` 0, `$0DB0` 1, at `(6288, 5456)`. ORing `0x0800` into that save word leaves live `$0402` at `0x8000` and state 0. The saved landing is `(6392, 5592)`, doorway 0. On the pre-light load the south door is slot 0 shutter `0x18` pos `$1D5C`. No chests. Logs: `recordings/probe_AP-159/`.

`$0E50` starts at 90. A connecting hit writes it to 128. The hit count is `$0F90`. `$0D80` stayed 0. The byte that advances while he is idle is `$0DB0`.

Master sword, 32 px south of the live sprite, facing UP (`$2F` 0): the B press is frame 0 (`$0E50` 90, `$0EA0` 0, `$0F90` 0, Link hp 56). `$0EA0` becomes 14 at frame 10. Frame 14 writes `$0E50` 90→128, `$0F90` 0→1, and `$0F10` 48. Link hp stays 56 (`trace.json`). The same sword 24 px south leaves `$0F90` at 0, and Link hp is 48 on that B press. Hammer 24 px north, facing left (`$2F` 4): `hammer()` took 92 frames and reported no peg. The next sample already has `$0E50` 128, `$0F90` 1, `$0EA0` 7. A bomb on the 32 px south stand placed (fuse 159) and left `$0F90` at 0 and `$0E50` at 90 through 200 frames. `$0DB0` went 1→2 at frame 62 of that wait.

Three south swords on one load: `$0F90` reads 1, then 2. The third hit leaves `$0F90` at 0, `$0DB0` at 6, and a second `0xCE` in slot 15 with `$0D90` 2, hp 90, at `(6326, 5465)`. Link hp goes 56→48 on that swing.

Idle 1800 on the pin (`boss_idle.json`): `$0E50` stays 90, `$0F90` stays 0, `$0EA0` stays 0. `$0DB0` goes 1→2 at frame 67, then 2→3 at 646, 3→4 at 730, 4→2 at 803, and that 3-4-2 loop repeats through frame 1795. Type `0x41` (sprite_ids name blue soldier) is born at frames 75, 83, and 547 with hp 6, beside the boss. Link stays `(6392, 5592)` hp 56.

| Field | Value |
|-------|--------|
| Sprite ids | Blind `0xCE` (z3 id 206) slot 0, room `0xAC`. The head is a second `0xCE` with `$0D90` 2. Idle spawns are type `0x41`. |
| HP byte | `$0E50` starts at 90. A hit writes 128. The count that advances is `$0F90`: 0, then 1, then 2, then 0 once the head exists. |
| Phase byte | `$0DB0`. Starts at 1. Idle: 2 at frame 67, 3 at 646, 4 at 730, 2 at 803. `$0D80` stayed 0. The third sword left `$0DB0` at 6. |
| What hurts it | Master sword 32 px south facing UP. Hammer 24 px north facing left. |
| When | Sword: `$0EA0` 14 at frame 10; `$0F90` 0→1 and `$0E50` 90→128 at frame 14. Hammer: inside the 92-frame swing; the next sample is `$0F90` 1 and `$0EA0` 7. |

`strike_verdict` on that 32 px south sword is whiff. `$0EF0` is `0x1F` at frame 10, and `$0E50` rises at frame 14, after the 13-frame swing. The landing `(6392, 5592)` is attribute `0x01` (wall) on every `LINK_BODY` tile. UP, DOWN, LEFT, and RIGHT for 40 frames, and a 2100-frame walk toward the boss, leave that xy. A 3600-frame walk with no soldier strikes, offsets 0–11, ended at the same xy: swings 0, hammers 0, hp_lost 0, hits none. Closest `0xCE` was slot 15 at dx 62, dy 33. Body hp stayed 90, `$0F90` stayed 0, no head, max HP stayed `0x38`. South door slot 0 stays shutter `0x18`. Eleven `strike_if_safe` kills on the `0x41` spawns did not open it. The head (`$0D90` 2) keeps `$0F70` at 23 and hp 90 under the same south sword and under the hammer pose. Further south swords on the body after the head exists leave both of those bytes. No heart. Logs: `recordings/probe_AP-160/`.

## Prize and exit

z3 node 12 is Crystal4. The south door returns outdoors on screen `0x58`. Prize and crystal-exit glances are unscanned.

## Save points

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| `ThievesTownEntry` | none on this sitting | Indoor dev load of room `0xDB`. Isolated development. |
| `Whatif_ThievesTownEntry_lift` | `ThievesTownEntry` | Ability write only (`0x52`). Still room `0xDB`. Not a hop. |
| `Whatif_ThievesTownEntry_parallel` | `ThievesTownEntry` | Disclosed poke of big key 16 and one small key. Chest stayed shut. Not a hop. |
| `Scan_thieves_boss` | `Whatif_ThievesTownEntry_parallel` | Warp submodule `0x15` into `0xAC` after save word `$F158` bit `0x0200`. Kind poke. Blind is state `0x09`. Not a walked door. |

## Open reds

| Red | Evidence | Blocker |
|-----|----------|---------|
| `east_to_0xDC` | room_engine 0/2 still `0xDB`: `(5928,7072)` and `(5952,6731)`. What-if lift cleared the `0x71`, then RIGHT stuck `(6088,6903)` dir 6 hp 56 ability `0x52`, still `0xDB`. | Shutter slot 0 type `0x18` pos `$0FF4` did not open. The `0x72` at `(44,32)` was not lifted. |
| BigKeyD4 | `ThievesTownEntry` big key stayed 0. The pathable layer-0 `0x1C` at tile `(19,48)` dropped to layer 1 `(5788,7024)` still `0xDB` (`ap157_overlay.json`). A bomb on layer-1 floor left those tiles `0x00`; both chests skipped, no path (`ap157_try.json`). | East shutter stayed `0x18`. The `(49,44)` alcove was not entered. |
| Attic / cell | AP-158 parallel pin. North hold `(5704,6704)` stayed `0xDB`. Zol died, shutter stayed `0x18`. `0x70` did not lift. `0x72` had no drop-avoiding path. | Rooms `0x65` and `0x45` were not entered. Gloves stayed 1. Follower stayed 5. |
