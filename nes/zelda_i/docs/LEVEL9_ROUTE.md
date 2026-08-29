# Level 9 — Death Mountain (route notes)

**Status:** backward endgame recon is live; the natural Level 9 route is still
unbuilt. Spectacle Rock is overworld `0x05`, the settled entrance is room
`0x76`, the final Patra room is `0x52`, Ganon is `0x42`, and Zelda is `0x32`.
The preserved endgame states are explicitly composed, route-ineligible
fixtures—not Clean or Survival route evidence.

**Beads:** `rr-sz8` (Level 9 epic), `rr-sz8.1` (pre-Ganon → credits),
`rr-sz8.2` (live final Patra → credits), `rr-sz8.3` (room `0x62` disproved;
play `0x03` stairs → cellar `0x77` → Patra **2/2**; `0x13` north wall, not a
clean predecessor; play `0x04` bomb-west → `0x03` → Patra **2/2** recon; play `0x30` stairs → cellar `0x67` right → `0x04` → Patra **2/2** recon; play `0x31` bomb-west → `0x30` → Patra **1/1** recon; play `0x21` south shutter sealed after Patra; play `0x41` north → `0x31` dest **YES** → Patra **1/1** recon; play `0x40` key-north → `0x30` dest **YES**, stays dirty; play `0x51` identified as south pred of `0x41`, north dest walk **NO**).

Planning sources:

- [Zelda Dungeon — Level 9: Death Mountain](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-9-death-mountain/)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)
- RAM: `ADDR_TRIFORCE`, `ADDR_RING`, `ADDR_ARROWS`, `ADDR_MAGIC_KEY`, bombs

All room IDs in the backward-recon section are **live**. Unvisited interior
route claims remain source planning until reached from their real predecessor.

## Backward endgame recon (live 2/2, 2026-08-14)

```text
live final Patra 0x52 (body 0x47 + 8 eyes 0x25)
  ─controller sword clear─► CurOpenedDoors north bit 0x08
  ─UP─► Ganon 0x42 (object 0x3E)
  ─four registered Magical Sword hits─► brown ObjState nonzero
  ─Silver Arrow─► LastBossDefeated $0672 = 1
  ─collect Power Triforce / north door─► Zelda 0x32 (object 0x37)
  ─clear two guard fires / center trigger─► ending → rolling credits → final page
```

Repeatable proof lives in `level9/` (`ganon.py`, `patra.py`, `stair_suffix.py`).
Isolated `run_level9_*.py` recon CLIs pruned. Composer binds the fixture dests.

Both builds begin from live `Level9EntranceReconFixture`, use the game room
loader for `0x52`, and explicitly write the full inventory. The older
pre-Ganon fixture additionally removes Patra and opens the north door. The new
`Level9FinalPatraReconFixture` stops before those forbidden writes: it preserves
body + eight eyes, `CurOpenedDoors=0`, and `OpenDoorwayMask=0`. Every state is
still fixture-only and route-ineligible because the inventory/loader setup is
composed.

### Candidate room `0x62` — RETARGET (live + ROM, 2026-08-14)

The `-0x10` south-neighbor hypothesis is **disproved**. The game loader will
scroll a fake `0x72` → `0x62` (and the older `0x62` → `0x52` Patra load), but
that is not a natural door.

Live uncleared settle (`Level9Room62ReconFixture`, loader `0x72` hold UP):

| Signal | Live value |
|--------|------------|
| Room | `0x62`, play mode, Level 9 |
| Objects | 8× Keese type `0x1B`, slots 1–8, HP 0 (type-alive) |
| Door bits | `CurOpenedDoors=0`, `OpenDoorwayMask=0` |
| Room item | `0x0F` (unknown; appears after clear as a center drop) |
| Kill-clear | 8 Keese die; door bits stay 0; north push sticks at `(120, 93)` |
| Bomb north | stands `(120,93/101/109)` consume a bomb; wall stays closed |
| Sides | west visual open / east keyhole; y=189 walks stay in `0x62` |

First-quest L7–9 ROM door bytes (iNES `0x18A10`/`0x18A90`):

| Room | N | S | W | E |
|------|---|---|---|---|
| `0x62` | wall (1) | wall (1) | open (0) | key (5) |
| `0x52` | shutter (7) | **wall (1)** | wall (1) | wall (1) |

`0x52` therefore has no south door. The walkthrough predecessor is a stairs /
underground-passage drop into the Patra room under Ganon.

Evidence: `recordings/l9_room62_patra_credits_recon_probe.json`,
`l9_room62_door_experiment.json`, `l9_room62_exit_probe.json`,
`l9_pred_retarget_probe.json`; start PNG
`l9_room62_patra_credits_recon_probe_start.png`.

### Stair cellar dest table — live `0x77` left → Patra `0x52` (2026-08-14)

First 6 bytes of ROM `0x19C10` (iNES `0x19C20`) are
`LevelInfo_CellarRoomIdArray`. CheckSubroom (mode 9): Y < `0x40` and UP;
X < `0x80` reads AttrsA (left mouth), else AttrsB (right). Dest is the
**current** RoomId's door-attr bytes, not a sequential pair.

Live dests (InitMode9 + mouth stand + controller UP; dest written by
CheckSubroom, not `NEXT_SCREEN`):

| Cellar RoomId | Left (AttrsA) | Right (AttrsB) | Patra live? |
|---------------|---------------|----------------|-------------|
| `0x60` | `0x14` LikeLikes | `0x55` | no |
| `0x70` | `0x63` Zols | `0x05` Wizzrobes | no |
| `0x72` | `0x71` empty | `0x74` | no |
| `0x75` | `0x20` Wizzrobes | `0x61` **other Patra** (body+8 eyes, not `0x52`) | no |
| `0x67` | `0x30` traps+Wizzrobes | `0x04` traps+Wizzrobes | no |
| **`0x77`** | **`0x52` body `0x47` + 8 eyes `0x25`, north door 0** | `0x03` Zol+LikeLike | **yes, left** |
| `0x00` | not in cellar array; InitMode9 stays mode 9 / 4 Keese | same | no |
| `0x4F` | not in cellar array; InitMode9 stays mode 9 / 4 Keese | same | no |

Play-room **0x03** is the CheckWarps source for cellar **0x77** (right mouth).
See the walk section below. Other cellar entries remain unfound.

### Play room 0x03 stairs → cellar 0x77 → live Patra (2026-08-14)

CheckWarps source is play room **0x03**, stair tile **0x72** at exact pixel
**(128, 141)** / `($80, $8D)`. ALIGN_TOL=3 is too loose (139 misses).

The center stairs sit in an 8-block diamond. West object `0x68` rests at
`(96, 144)`; after kill-clear, stand `(96, 170)` and hold UP until the block
slides to y=`$80`. Walk the vacated west slot `(96, 133)` then x-first onto
`(128, 141)`. Mode 16 / SCREEN `0x77` (no InitMode9, no `NEXT_SCREEN` poke).

Natural entry lands on the **right** cellar mouth. Traverse: DOWN the right
stairwell → pit → left column **x=`$30`** → UP. Stay on `$30` while climbing
(switching to `$50` at y=`$70` walks off the ladder). CheckSubroom left
(Y < `$40`, X < `$80`, UP) → live Patra `0x52` (8 eyes, north closed).

Materialize: neighbor-scroll `0x13` hold UP, Link `($78, $58)`, FULL_LOADOUT +
`Level9EntranceReconFixture`. That `0x13` scroll uses fixture door-staging
`0x0F/0x0F` — **not** a clean walk (see 0x13 dump below).
`route_eligible=false` (fixture inventory + room-loader settle). Survival
`--infinite-life` OK. Continuous **2/2** frame-exact (Patra live on entry →
credits 15428 → final page 16628; zero forbidden writes).

Stitch pin `Level9Room03StairsReconFixture` is the natural Patra landing
after this walk (not InitMode9). No `Level9Room13ReconFixture` — 0x13 is
not a clean predecessor.

```bash
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Evidence: `recordings/l9_play03_patra_credits_recon.json`,
`l9_play03_patra_credits_recon_t{0,1}_0x03_tiles.json`; PNGs
`t{0,1}_{settle,after_walk,patra_entry,credits,final_screen}.png`.

### Play room 0x04 bomb-west → 0x03 stairs → Patra (2026-08-14 19:28 CT)

**Yes: 0x04 bomb-west lands 0x03.** No door poke. Loader is `0x14` hold UP
(stages 0x14, not 0x03). ROM + live: 0x04 N/S/E wall, **W bomb**; 0x03 E bomb.

Live 0x04 settle: screen 4, Link (120, 189), doors 0, B=bombs. Objects: 4×
blade trap `0x49`, 2× blue wizzrobe `0x23`, 2× orange wizzrobe `0x24`,
pushable `0x68` at (96, 144).

Bomb: south-band approach **(48, 189)** then stand **(48, 141)** face LEFT
(`BombWallController`). Blast 16→15, ~357–373f, dest **SCREEN=0x03**,
Link (208, 141) in the east bomb hole, `CurOpenedDoors` east bit 0x01.
Stair tile 0x72 still at (128, 141).

Compose: kill-clear Zol+LikeLike (ignore invuln 0x2B) → west-block UP from
(96, 170) → stand (128, 141) → cellar 0x77 left x=`$30` → Patra 0x52 →
credits. East-open SE detour uses x=176 (not 208). Do not DOWN through the
unpushed 0x68 from the north. Pause RIGHT×4 reselects Silver Arrows after
the bomb. **2/2** recon (~13727f). `route_eligible=false`. No InitMode9,
no `NEXT_SCREEN` poke, no 0x03 door poke.

Stitch pin `Level9Room04BombWestReconFixture` is the live Patra landing
after this real bomb-west walk (0x04 start is still fixture-loaded).

```bash
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Evidence: `recordings/l9_room04_dump.json` +
`l9_room04_dump_{settle,after_bomb,dest}.png`;
`recordings/l9_play04_bombwest_patra_credits_recon.json`.

### Play room 0x30 stairs → cellar 0x67 right → bomb-west of stairs (2026-08-14 19:45 CT)

**Yes: 0x30 / cellar 0x67 right lands 0x04.** No InitMode9, no `NEXT_SCREEN`
poke, no 0x04 door poke. Loader is `0x40` hold UP. That scroll writes
`0x0F/0x0F` on the south key-room so the key-north can start — **not** a
clean `0x40 → 0x30` walk (same class as the 0x13 door-stage). 0x04 doors
are not poked. ROM: 0x30 N/W wall, S key, E bomb, secret **block_stairs**.

Live 0x30 settle: screen 0x30, Link (120, 205), south door only. Objects:
4× blade trap `0x49`, 2× blue wizzrobe `0x23`, 2× orange wizzrobe `0x24`,
pushable `0x68` at (96, 144). Kill-clear alone does not reveal stairs.

Push: south-band to (96, 170), hold UP (~49f). The 0x68 relocates to the
engine stand **(208, 96)** / tile `0x72`. CheckWarps needs that exact pixel
(ALIGN_TOL=3 at (206, 93) stays in play). Mode 16 / SCREEN `0x67`. Natural
spawn is the right stairwell `(208, 93)`; CheckSubroom right (Y < `$40`,
X ≥ `$80`, UP) → play **0x04**. 0x04 west bomb is still live.

Compose: 0x30 stairs → cellar 0x67 right → 0x04 → accepted bomb-west →
0x03 stairs → Patra → credits. **2/2** recon, frame-exact **18492f**
(credits 17202 / final 18402 both trials). `route_eligible=false`. Zero
forbidden runtime writes.

Stitch pin `Level9Room30StairsReconFixture` is the live Patra landing
after this real walk (0x30 start is still fixture-loaded).

Next clean 0x30 entry (ROM only until dumped):

| Room | N | S | W | E |
|------|---|---|---|---|
| `0x30` | wall (1) | key (5) | wall (1) | **bomb (4)** |
| `0x31` | open (0) | shutter (7) | **bomb (4)** | wall (1) |
| `0x40` | key (5) | key (5) | wall (1) | wall (1) |

**0x31 bomb-west is live** (see below). **0x40 key-north is also live**
(controller Magical Key walk; see 0x40 section). The 0x30 *loader*
`0x40` hold UP still door-stages 0x40 — that fake scroll is not the
clean walk.

```bash
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Evidence: `recordings/l9_room30_dump.json` +
`l9_room30_dump_{settle,stairs_enter,cellar,dest}.png`;
`recordings/l9_play30_cellar67_patra_credits_recon.json`.

### Play room 0x31 bomb-west → 0x30 stairs → cellar 0x67 right (2026-08-14 19:55 CT)

**Yes: 0x31 bomb-west lands 0x30.** No InitMode9, no `NEXT_SCREEN` poke, no
0x30 door poke. Loader is `0x41` hold UP. That scroll may write `0x0F/0x0F`
on 0x41 (north is already open; no-door-poke settle also loaded 0x31).
0x30 doors are not poked. ROM: 0x31 N open, S shutter, W **bomb**, E wall;
pairs 0x30 E bomb.

Live 0x31 settle: screen 0x31, Link (120, 189), south doorway. Objects:
3× Like-Like `0x17`, 2× blue wizzrobe `0x23`, 2× orange wizzrobe `0x24`,
invuln residual `0x2B`. Doors raw 0; mask north open. B=bombs, 16 bombs.
No stair tiles. Kill-clear the Like-Likes before the west stand — one on
the west corridor causes stand_timeout.

Bomb-west: stand **(48, 141)** LEFT (`BombWallController`, same as 0x04).
Blast 16→15, ~328–330f, dest **SCREEN=0x30**. 0x30 still has pushable
`0x68` @(96,144); block-stairs still work → cellar `0x67` right → `0x04`.

Compose: 0x31 bomb-west → 0x30 stairs → cellar 0x67 right → 0x04 →
accepted bomb-west → 0x03 stairs → Patra → credits. **1/1** recon,
**27676f** (credits 26386 / final 27586). `route_eligible=false`. Zero
forbidden runtime writes.

Stitch pin `Level9Room31BombWestReconFixture` is the live Patra landing
after this real walk (0x31 start is still fixture-loaded).

Next clean 0x31 entry (ROM only until dumped):

| Room | N | S | W | E |
|------|---|---|---|---|
| `0x31` | **open (0)** | shutter (7) | bomb (4) | wall (1) |
| `0x21` | open (0) | shutter (7) | wall (1) | bomb (4) |
| `0x41` | open (0) | shutter (7) | wall (1) | wall (1) |

**0x21 south is not a clean predecessor** (see dump below). Next candidate:
play **0x41 north** (ROM open; current 0x31 loader). `0x40` key-north is a
separate live predecessor of 0x30 and stays dirty — do not treat it as the
next pred.

```bash
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Evidence: `recordings/l9_room31_dump.json` +
`l9_room31_dump_{settle,dest,stairs_dest,no_door_poke_settle}.png`;
`recordings/l9_play31_bombwest_patra_credits_recon.json`.

### Play room 0x21 south → 0x31 — RETARGET (2026-08-14 20:10 CT)

**No: 0x21 south does not land 0x31.** No InitMode9, no `NEXT_SCREEN` poke,
no 0x31 door poke. Loader is `0x11` hold DOWN (stages 0x11, never 0x31).
ROM: 0x21 N open, S **shutter**, W wall, E bomb; 0x31 N open pairs.

Live 0x21 settle: screen 0x21, Link (120, 77) north doorway. Objects:
Patra body `0x47` HP `0xB0` + 8 eyes `0x25`. Plus geometry D=`0xA5`.
Doors raw 0 / mask 0. B=bombs, 16 bombs. No stair tiles.

Uncleared south: stand **(120, 189)** DOWN sticks in 0x21. After
kill-clear (and a separate `patra_action` south-stand kill **1467f**,
body dead, eyes 0, `RoomAllDead=18`) the south shutter **stays sealed**
(doors raw 0). Cleared south probe still SCREEN **0x21**, stuck y=189.
Compose `--compose-21` not run.

**0x11 south → 0x21 is live.** Load 0x11 from 0x01 hold UP (8× type
`0x3B`), kill-clear opens 0x11 south shutter (doors raw 4), walk south
lands play 0x21 at (120, 77). That hop does not open 0x21's south
shutter.

`route_eligible=false`. 0x40 stays dirty — not the next pred.

Next clean 0x31 entry:

| Room | N | S | W | E |
|------|---|---|---|---|
| `0x31` | open (0) | shutter (7) | bomb (4) | wall (1) |
| `0x41` | **open (0)** | shutter (7) | wall (1) | wall (1) |
| `0x21` | open (0) | **shutter sealed after Patra** | wall (1) | bomb (4) |

**0x41 north is live** (see below). 0x40 stays dirty.

```bash
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Evidence: `recordings/l9_room21_dump.json` +
`l9_room21_dump_{settle,south_probe,after_clear,cleared_dest,no_door_poke_settle}.png`;
`recordings/l9_probe11_south_21.json` (0x11 south → 0x21 dest YES).

### Play room 0x41 north → 0x31 bomb-west suffix (2026-08-14 21:10 CT)

**Yes: 0x41 north lands play 0x31.** No InitMode9, no `NEXT_SCREEN` poke,
no 0x31 door poke. Loader is `0x51` hold UP (stages 0x51, never 0x31).
ROM: 0x41 N **open**, S shutter, W/E wall; 0x31 S shutter pairs.

Live 0x41 settle: screen 0x41, Link (120, 189) south doorway. Objects:
4× blade trap `0x49` + 4× Like-Like `0x17`. Doors raw 0; mask north
open. No stair tiles (north mouth tile `0x24` @(120,77) is the door).

Uncleared north: door-column UP sticks at y=103 (Like-Likes). After
chase-clear of types `< 0x40` (Like-Likes; traps skipped), north walk
lands play **0x31** mode 5, Link (120, 189) in the south shutter,
`CurOpenedDoors` south bit. Dest objects: Like-Likes + wizzrobes +
invuln `0x2B`. 0x31 west bomb still live.

Compose: 0x41 north → 0x31 bomb-west @(48,141) LEFT → 0x30 stairs
(south-band chase so the plus does not wedge the 0x68) → cellar
`0x67` right → 0x04 suffix. **1/1** recon, **27148f** (credits 25858 /
final 27058). `route_eligible=false`. Zero forbidden runtime writes.

Stitch pin `Level9Room41NorthReconFixture` is the live Patra landing
after this real walk (0x41 start is still fixture-loaded).

```bash
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Evidence: `recordings/l9_room41_dump.json` +
`l9_room41_dump_{settle,north_probe,after_clear,cleared_dest,dest}.png`;
`recordings/l9_play41_north_patra_credits_recon.json`.

### Play room 0x51 north → 0x41 — dest NO (2026-08-15)

**No: 0x51 north walk did not land 0x41.** No InitMode9, no `NEXT_SCREEN`
poke, no 0x41 door poke. Loader is `0x61` hold UP (stages 0x61, never
0x41). ROM: 0x51 N **open** / S **open** / W **shutter** / E wall,
secret **all_dead**; 0x41 S shutter pairs. `route_eligible=false`.

Live 0x51 settle: screen 0x51, Link (120, 205) south doorway. Objects:
6× Like-Like `0x17` HP `0x90`. Doors raw 0; mask 0. No stair tiles.
Mouth tiles `0x24` @(120,77) north and @(120,213) south. no-door-poke
`0x61` hold UP also settles 0x51 (both doors are ROM-open).

Uncleared north: Like-Like sits on the south door; stand (120, 181)
stays in 0x51. After chase-clear (~1314f) west shutter **opens**
(doors raw 2, west bit; `RoomAllDead` nonzero). North mouth stays
visually black / mask north+south. Center-aisle UP sticks at
**(120, 117)** on the north vertex of the statue diamond. Thread
columns **x=104** and **x=144** at y=133 also stick. Compose
`--compose-51` not attached.

`0x51` is still the identified south predecessor (ROM + visual north
open). The live dest walk is not earned. 0x40 stays dirty — not this
chain.

Next: thread the statue diamond from the south-door spawn after
clear, or materialize play **0x61** (ROM N/S open, E open; current
0x51 loader).

| Room | N | S | W | E |
|------|---|---|---|---|
| `0x51` | **open (0)** | open (0) | shutter (7) after all_dead | wall (1) |
| `0x41` | open (0) | **shutter (7)** | wall (1) | wall (1) |
| `0x61` | **open (0)** | open (0) | wall (1) | open (0) |
| `0x50` | key (5) into dirty 0x40 | wall (1) | wall (1) | shutter (7) |

```bash
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Evidence: `recordings/l9_room51_dump.json` +
`l9_room51_dump_{settle,north_probe,after_clear,cleared_dest,no_door_poke_settle}.png`.

### Play room 0x40 key-north → 0x30 (2026-08-14 19:56 CT)

**Yes: 0x40 key-north lands play 0x30.** No InitMode9, no `NEXT_SCREEN`
poke, no 0x30 door poke. Opening the key door is walking into it with
Magical Key (FULL_LOADOUT). Loader is `0x50` hold UP (stages 0x50, never
0x30). ROM: 0x40 N/S key, W/E wall, secret foes_item; 0x30 S key pairs.

Live 0x40 settle: screen 0x40, Link (120, 205), south door only, north
keyhole closed. Objects: 3× blue wizzrobe `0x23` + 2× orange wizzrobe
`0x24`. Plus + C-block geometry. No-door-poke `0x50` hold UP also settles
0x40 (Magical Key opens 0x50 north).

Walk: from the south alcove hold UP on the door column
(`room40_to_30_step`). Hold UP through mode 4/6/7 scroll. Do **not**
kill-clear first — chase leaves Link in the plus. Uncleared controller
UP → play **0x30** mode 5, Link (120, 205) in the south key doorway.
Dest objects: blade trap + wizzrobes + pushable `0x68` at (96, 144).
Stair tile at (208,96) is `0x76` until the block push (secret
block_stairs still works; compose takes it).

Compose: 0x40 key-north → 0x30 stairs → cellar 0x67 right → 0x04 →
accepted bomb-west → 0x03 stairs (`room03_chase_mode=blocking`) → Patra
→ credits. **1/1** recon, **17305f** (credits 16015 / final 17215).
`route_eligible=false`. Zero forbidden runtime writes.

Stitch pin `Level9Room40KeyNorthReconFixture` is the live Patra landing
after this real walk (0x40 start is still fixture-loaded).

Next clean 0x40 entry (ROM only until dumped):

| Room | N | S | W | E |
|------|---|---|---|---|
| `0x40` | **key (5)** | key (5) | wall (1) | wall (1) |
| `0x50` | **key (5)** | wall (1) | wall (1) | shutter (7) |

Primary next: play **0x50 key-north** (same Magical Key walk). 0x40 W/E
are walls.

```bash
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Evidence: `recordings/l9_room40_dump.json` +
`l9_room40_dump_{settle,north_probe,dest,after_clear,no_door_poke_settle}.png`;
`recordings/l9_play40_keynorth_patra_credits_recon.json`.


### Play room 0x13 — RETARGET (live + ROM, 2026-08-14)

`0x03` was entered via **0x13 hold UP** only because the loader staged
`CurOpenedDoors`/`OpenDoorwayMask` `0x0F/0x0F` on the from-room. That is
fixture-only. Live 0x13 after the game room loader settles (no door poke
on 0x13 itself):

| Signal | Live value |
|--------|------------|
| Room | `0x13`, play mode, Level 9 |
| Loader | `0x23` hold UP, Link `($78, $58)` |
| Link start | `(120, 205)` south doorway |
| Objects | 2× invuln `0x2B` + 2× Zol `0x13` + 2× LikeLike `0x17` |
| Door bits | `CurOpenedDoors=0x04` south only; `OpenDoorwayMask=0x04`; north 0 |
| RoomAllDead / RoomObjCount | 0 / 6 |
| Kill-clear | north bit stays 0; UP sticks at `(120, 93)` |
| No-door-poke settle | still lands 0x13 (0x23 north is key; Magical Key in fixture) |

First-quest L7–9 ROM door bytes (iNES `0x18A10`/`0x18A90`):

| Room | N | S | W | E |
|------|---|---|---|---|
| `0x13` | **wall (1)** | key (5) | key (5) | wall (1) |
| `0x03` | wall (1) | **wall (1)** | wall (1) | bomb (4) |

`0x13` therefore has no north door, and `0x03` has no south door. Controller
UP after a no-0x13-door-poke settle stays in `0x13`. Do not compose
`0x13 → 0x03` as a clean walk. The 0x03 loader's door-staging scroll is a
fake transition, same class as the disproved `0x72 → 0x62` load.

`0x03` east is ROM bomb; `0x04` west is ROM bomb — **live** (see 0x04
section). Clean 0x04 entry is play `0x30` / cellar `0x67` right (see above).

```bash
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Evidence: `recordings/l9_room13_dump.json`; PNGs
`l9_room13_dump_{settle,after_clear,north_probe,north_after_clear,no_door_poke_settle}.png`.

`Level9Stair77PatraEnteredReconFixture` is the earlier InitMode9 CheckSubroom landing
(fixture-only: FULL_LOADOUT + `0x67` hold DOWN into `0x77` + InitMode9 +
left mouth `(0x50, 0x3D)` + UP). Suffix from that entry is **2/2**
(Patra 1652f → credits 9762 → final page 10962).

```bash
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Evidence: `recordings/l9_stair77_dest_table.json`,
`l9_stair77_patra_credits_recon.json`; dest PNG
`l9_stair77_dest_table_0x77_left_dest.png`; stitch pin
`Level9Stair77PatraEnteredReconFixture`.

Patra evidence: `recordings/l9_patra_credits_recon.json`; screenshots
`l9_patra_credits_recon_t{0,1}_{patra_start,patra_cleared,ganon_start,ganon_arrow_kill,ganon_defeated,zelda_room,ending_start,credits,final_screen}.png`.
Older Ganon-only evidence: `recordings/l9_ganon_credits_recon.json`; screenshots
`l9_ganon_credits_recon_t0_{before_ganon,ganon_start,ganon_arrow_kill,ganon_defeated,zelda_room,ending_start,credits,final_screen}.png`.

---

## Gates / required capabilities

| Cap | RAM | Source role |
|-----|-----|-------------|
| **All 8 TF shards** | `ADDR_TRIFORCE == 0xFF` | Old Man allows passage; L9 content locked without |
| Bombs | `ADDR_BOMBS` | OW rock entrance + interior walls |
| Sword | preferably Magical | Combat density |
| Bow + arrows | `ADDR_BOW`, `ADDR_ARROWS` | Silver Arrow is arrow-type upgrade |
| **Red Ring** (dungeon) | `ADDR_RING` value 2 (source) | Damage quartered vs base |
| **Silver Arrows** (dungeon) | `ADDR_ARROWS` value 2 (source) | Only way to kill Ganon after stun |
| Magical Key (optional) | `ADDR_MAGIC_KEY` | Route splits: Magical Key path vs key-farm path |
| Red Potion | `ADDR_POTION` | Source strongly recommends full red before entry |

**Predecessor:** all of L1–L8 Triforce bits. OW bomb rock can be mapped
earlier; interior Old Man blocks without full TF.

**Do not** poke TF bits / Silver Arrows for Clean STATUS.

---

## Overworld

### Spectacle Rock / bomb entrance (source)

From start (ZD): **right, up×5, left, up×2, left×2**. Two large rocks; bomb
**just below the left rock** → cave / Level 9.

| Landmark | Source hops from start `0x77` | Hypothesized id | Live? |
|----------|-------------------------------|-----------------|-------|
| Bomb-rock screen (Spectacle Rock) | R U×5 L U×2 L×2 | **`0x05`** | **yes** |
| Nearby potion shop | one screen left of rock | **`0x04`** | no |

Hop arithmetic:

```text
0x77 →R→ 0x78 →U×5→ 0x28 →L→ 0x27 →U×2→ 0x07 →L×2→ 0x05
```

Live recon reached `0x05` through the authentic overworld scroll loader and
bombed the left rock to settle in Level 9 room `0x76`. The full natural walk
from the earned L8 predecessor remains unverified.

**Scaffold:** `level9/overworld.py` — `LEVEL9_ROCK_HOPS`, `has_full_triforce()`,
bomb-entry notes (controller TBD).

### Remaining natural-entry goals

1. Walk to rock screen from the real post-L8 predecessor.
2. Bomb the left rock with naturally held bombs and full Triforce.
3. Settle `level==9`, room `0x76` without inventory/progression writes.
4. Continue through the Old Man gate from that natural entry.

---

## Interior (source summary)

Two routes: **with Magical Key** (ZD §10.2) vs **without** (§10.3). Prefer
Magical Key path for automation (fewer key bottlenecks). Room IDs **unknown**.

### Magical Key path (condensed source)

| Phase | Action | Notes |
|-------|--------|-------|
| Entry | UP | 12 Keese optional |
| Old Man | full TF check | pass only if `triforce == 0xFF` |
| LEFT / bomb N | Lanmola | head hits; push left block → stairs |
| Underground | tunnel | |
| Like-Likes | protect Magical Shield | key RIGHT |
| Patra #1 | **skippable** | orbiting eyes; leave DOWN |
| Patra #2 | kill for **Map** | bomb walls continue |
| Wizzrobe / blocks | clear, push left block | stairs → **Red Ring** |
| Backtrack | Magical Key doors | Old Man bomb hint LEFT |
| Stairs chain | more Wizzrobes / Patra | |
| Item | stairs → **Silver Arrows** | required for Ganon |
| Final Patra | clear → door UP | live as room `0x52`; see below |
| **Ganon** | stun then Silver Arrow | see below |
| Zelda | princess room | ending sequence |

### Final Patra (live 2/2)

| Signal | Live value |
|--------|------------|
| Room / body | `0x52` / type `0x47`, slot 1 |
| Body initial HP | `0xB0`; after eyes: `B0 → 70 → 30 → dead` |
| Eyes | 8× type `0x25`, slots 2–9, initial HP `0x60` |
| Eye damage | Magical Sword `60 → 20 → dead` |
| Natural clear | body + eyes absent; `CurOpenedDoors & 0x08` becomes true |
| Door micro | clear ends near x≈112; recenter x≈120, then hold UP to `0x42` |

`FinalPatraFightController` follows a point 30 px south of the moving body and
pulses UP+A every 12 release frames. The orbiting eyes cross that sword line,
so the policy does not chase them through the block geometry. Both trials were
frame-exact: eight eyes fell by controller frame 1,465; the body and north-door
bit completed at frame 1,883. The Patra segment preserved its full start
inventory and declared zero object/room/door/inventory/progression/capacity
controller writes.

After 45 door-settle frames, `final_patra_to_ganon_step` corrects the observed
x≈112 finish to the strict x≈120 north-door band. Holding UP without this
recenter was the only observed failed composition: Patra cleared, but Link
stuck at the north wall and never entered Ganon.

### Ganon (live)

| Signal | Live value |
|--------|------------|
| Room / object | `0x42` / type `0x3E` |
| Scene phase | `$0445 == 2` during the fight |
| Initial HP | `$0485 + slot == 0xF0` |
| Sword sequence | `F0 → B0 → 70 → 30`; the next registered hit resets `F0` and enters brown |
| Brown | `ObjState[$00AC + slot] != 0`; engine seeds `0xFF`, first external post-step value is commonly `0xFE` |
| B item | `$0656 == 2` selects arrows (`1` is bombs) |
| Dying | `$042C + slot != 0` after Silver Arrow collision |
| Persistent kill | `$0672 != 0` (`LastBossDefeated`) |

Pulse A; holding it does not start the next sword swing. The controller chases
Ganon's live coordinates, waits 12 frames between sword pulses, then axis-aligns
and pulses the Silver Arrow on B. Collect the Power Triforce after the kill;
the north-door bit (`0x08`) then opens the path to Zelda.

### Red Ring / Silver Arrows RAM (source + Data Crystal style)

| Item | Address | Planned nonzero value |
|------|---------|------------------------|
| Ring | `0x0662` (`ADDR_RING`) | 1 = blue, **2 = red** |
| Arrows | `0x0659` (`ADDR_ARROWS`) | 1 = wooden, **2 = silver** |

Confirm values live before stop predicates rely on them.

---

## Zelda / ending stops (live)

Zelda room `0x32` contains Zelda object `0x37` and two guard-fire objects
`0x3F`. Clear the flames while walking to Link x=`0x70..0x80`, y=`0x95`;
the rescue switches to ending mode `0x13`.

Mode initialization reuses submode numbers, so mode + submode alone yields a
false early match. Require `$0011` (`IsUpdatingMode`) to be nonzero:

| Stop | Predicate |
|------|-----------|
| Rolling staff credits | `mode == 0x13 && is_updating_mode != 0 && submode == 3` |
| Final “Press Start” page | `mode == 0x13 && is_updating_mode != 0 && submode == 4` |

The older Ganon-only replay first entered rolling credits at frame 3,395 and
the final page at 4,595. The composed live-Patra replay reaches them at total
frames 5,342 and 6,542, respectively (**2/2 exact**), with no state load after
the start fixture. `level9_ending_stop` accepts either update-loop endpoint.

Both proofs preselect Silver Arrows in the fixture and report
`selected_item_writes=0` during combat. Each Patra→ending trial restored four
filled-heart units—two in `0x52`, two in `0x42`—with zero deaths and zero
progression/capacity writes. Those counters do not legalize the inherited
fixture composition.

---

## Boss / item stop predicates

```text
level9_red_ring      — ADDR_RING == 2 (planned)
level9_silver_arrows — ADDR_ARROWS == 2 (planned)
level9_ganon_dead    — ADDR_LAST_BOSS_DEFEATED ($0672) != 0
level9_ending        — update mode 0x13 submode 3 (credits) or 4 (final)
```

Full-clear program stop is **not** `triforce & 0x80` alone (that is L8);
Death Mountain end is Zelda/credits after Ganon.

---

## Checkpoints

| State | When |
|-------|------|
| `Level9EntranceReconFixture` | live `level==9`, room `0x76`; composed full inventory |
| `Level9Room03StairsReconFixture` | live Patra after play-0x03 stairs walk (fixture start) |
| `Level9Room04BombWestReconFixture` | live Patra after 0x04 bomb-west → 0x03 stairs (fixture start) |
| `Level9Room30StairsReconFixture` | live Patra after 0x30 stairs → cellar 0x67 right → 0x04 suffix (fixture start) |
| `Level9Room31BombWestReconFixture` | live Patra after 0x31 bomb-west → 0x30 stairs suffix (fixture start) |
| `Level9Room41NorthReconFixture` | live Patra after 0x41 north → 0x31 bomb-west suffix (fixture start) |
| `Level9Room62ReconFixture` | uncleared `0x62`; 8 Keese; doors 0; loader `0x72` UP; **not** Patra predecessor |
| `Level9FinalPatraReconFixture` | room `0x52`; live body `0x47` + eight eyes `0x25`; north closed |
| `Level9FinalPatraClearedReconFixture` | Patra naturally dead; `CurOpenedDoors & 0x08`; controller writes 0 |
| `Level9BeforeGanonReconFixture` | live final-Patra room `0x52`, Patra fixture-cleared, north open; requested start |
| `Level9GanonReconFixture` | room `0x42`, scene phase 2, Ganon type `0x3E` |
| `Level9GanonDefeatedReconFixture` | `$0672=1`, Power Triforce collected, north open |
| `Level9ZeldaRoomReconFixture` | room `0x32`, Zelda type `0x37` |
| `Level9EndingStartReconFixture` | ending mode `0x13` entered |
| `Level9CreditsReconFixture` | update-loop submode 3, visible staff credits |
| `Level9FinalScreenReconFixture` | update-loop submode 4, final Press Start page |
| `Level9PatraFinalScreenReconFixture` | same final page after continuous live-Patra suffix |
| `Level9RedRing` | after Red Ring |
| `Level9SilverArrows` | after Silver Arrows |

Every `*ReconFixture` state has a `.provenance.json` sidecar that warns it is
development-only and not a natural-entry checkpoint.

---

## Runners / probes

```bash
# Isolated segment CLI pruned. Durable runner:
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1
# Isolated segment CLI pruned. Durable: `run_survival_spine.py --no-video`.
```

Modules: `level9/overworld.py`, `level9/ganon.py`, `level9/patra.py`,
`level9/path.py`, `level9/room62.py`, `level9/stairs.py`,
`level9/room51.py`.

---

## Evidence boundary

- Live: Spectacle Rock `0x05`; entrance `0x76`; final Patra `0x52` body/eye
  types and HP; natural Patra clear + north-door bit; Ganon `0x42`; Zelda
  `0x32`; combat states; credits and final-screen stops.
- Fixture-only in both tracks: full inventory and room-loader composition.
  The older Ganon-only fixture also removes Patra and opens its north door;
  the accepted Patra runner does neither after its start.
- Not live from predecessor: natural Level 9 interior and Red Ring/Silver
  Arrow acquisition. Play-room **0x03** tile `0x72` @(128,141) → cellar
  `0x77` is live CheckWarps, but the start is still fixture-only
  (`route_eligible=false`).
  Cellar *exit* dest `0x77` left → `0x52` is live (CheckSubroom).
  Play-source 03 is now **2/2** (credits 15428 / final 16628, both trials).
  Play **0x04** bomb-west → 0x03 → Patra → credits is **2/2** recon
  (~13727f); bomb-west walk is real, 0x04 start is fixture-loaded.
  Play **0x30** tile `0x72` @(208,96) → cellar `0x67` right → `0x04`
  is **2/2** recon (18492f both trials); 0x30 start is fixture-loaded.
  Play **0x31** bomb-west @(48,141) LEFT → `0x30` is **1/1** recon
  (27676f; credits 26386 / final 27586); 0x31 start is fixture-loaded.
  Play **0x21** south shutter stays sealed after Patra kill (1467f,
  RoomAllDead=18, doors raw 0); dest still `0x21` at y=189. Not a
  clean predecessor of 0x31.
  Play **0x41** north (after Like-Like clear) → play `0x31` is live
  (controller UP; no 0x31 door poke). Compose **1/1** 27148f
  (credits 25858 / final 27058) via 0x31 bomb-west suffix.
  0x41 start is fixture-loaded (`route_eligible=false`).
  Play **0x51** is the identified south predecessor of 0x41 (ROM N
  open pairs 0x41 S shutter; 6× Like-Like; west shutter after
  all_dead). Live north dest walk is **not** earned: statue diamond
  blocks center (120,117) and thread columns 104/144. 0x51 start is
  fixture-loaded (`route_eligible=false`).
  Play **0x40** key-north → play `0x30` is live (controller UP from
  south alcove; Magical Key; no 0x30 door poke). 0x40 start is
  fixture-loaded (`route_eligible=false`). Compose suffix through
  0x03 stairs was not pinned (0x68 pushed south).
- Disproved: candidate room `0x62` as cardinal predecessor of `0x52`
  (north wall / south wall; eight Keese; no live north transition).
- Disproved: play room `0x13` as a clean cardinal predecessor of `0x03`
  (ROM north wall / 0x03 south wall; controller UP sticks at y=93;
  0x03 loader door-staging is a fake scroll).
- Dest-NO: play room `0x51` north walk into uncleared `0x41` (ROM +
  visual north open; statue diamond blocks the live walk). 0x51 is
  still the identified south predecessor.
- TF bit map: shards 1–8 = bits `0x01`…`0x80`; full = `0xFF`.
