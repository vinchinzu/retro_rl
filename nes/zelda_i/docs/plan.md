# Plan — Zelda I (NES)

## Goal

Advance from M5 (Clean power-on → Level 1 Triforce shard 1) toward a verified
continuous clear of The Legend of Zelda using the shared `retro_harness.adventure`
route graph.

Tracker: **`bd ready -l zelda_i`**. Process: `docs/tasks/PROCESS.md`.

## Next pass — Survival spine from power-on (2026-08-15)

Watchable main spine is **one continuous Survival session from power-on**.
Do not overwrite Clean M5. Seamed compose is gone (`rr-cont`). L9 backward
recon, `run_level4_rooms` slim (`rr-ekwl`), and isolated L4 (`rr-q3n`) are
**parked**.

Beads: **`rr-4d53`** epic. Parent **`rr-4d53.3`** (L2 exit → L3 TF `0x04`).
Entrance `0x7c` is **`rr-4d53.3.0` closed**. West key `0x7b` is
The continuous spine now clears L3 and continues through the Raft dock to L4
entry `0x71` with TF `0x07`.
The `.3.4.*` suffix is verified on `l3_tf_continuous_video_v1`; its documented
bomb-count top-up at natural Raft remains Survival-only until the farm pass.
Next is the live L3-exit → L4 suffix.

Full spine (do not claim ahead of the tip):

| Bead | Segment | Status |
|------|---------|--------|
| `rr-4d53.1` | power-on → L1 TF → L2 `0x7d` | **closed** |
| `rr-4d53.2.1` | live `0x7d` → Boom `0x4f` | **closed** — 1/1 Survival, boom owned |
| `rr-4d53.2.2` | natural bombs (no `--poke-bombs`) | **closed**; L2 entry bombs=4 |
| `rr-4d53.2.3` | Boom → Dodongo → TF `0x02` | **closed** — 1/1 Survival; documented bomb/key top-up |
| `rr-4d53.3.0` | L2 TF → Manji entry `0x7c` | **closed** — 1/1 Survival 53918f |
| `rr-4d53.3.1.1` | live `0x7c` west key `0x7b` | **closed** — 1/1 Survival 54589f keys=5 |
| `rr-4d53.3.1.2` / `.3.1` | occupancy dest `0x5b` | **closed** — 1/1 Survival 57256f |
| `rr-4d53.3.3.1` | `0x5b` LEFT → Compass `0x5a` | **closed** — 1/1 Survival 57648f |
| `rr-4d53.3.3.2` | `0x5a` KEY-LEFT → `0x59` | **closed** — 1/1 Survival, keys 5→4 |
| `rr-4d53.3.3.3` | clear `0x59`, DOWN → `0x69` | **closed** — 1/1 Survival |
| `rr-4d53.3.3.4` / `.3.3` | clear `0x69`, passage → Raft | **closed** — 1/1 Survival, bombs=8 |
| `rr-4d53.3.2` | L3 bomb budget | **verified Survival** — documented count top-up 8→16; farm deferred |
| `rr-4d53.3.4.*` | Raft → Manhandla → TF `0x04` | **verified** — one-way controller, `state_restores=0` |
| `rr-4d53.3` | parent: L2 exit → L3 TF `0x04` | **verified** — 1/1 continuous power-on, 92948f |
| `rr-doua` | Natural bomb farm (power-on L2 entry is 0) | **parked** — Survival count poke until then |
| `rr-4d53.6` | L3 exit → L4 TF `0x08` | **closed** — `l4_tf_continuous_v1` 2/2 TF `0x0F` mode 18 room `0x03`; HC not mid-room |
| `rr-4d53.7` | L4 exit → L5 TF `0x10` (attach `.5` pin) | **closed** — `l5_tf_continuous_v1` 1/1 TF `0x1F` mode 18 room `0x14` |
| `rr-4d53.4` | one session power-on → L5 TF | **closed** — same tape; `validate_l5_endpoint` passes |
| `rr-g3c1` | L5 fanfare settle → L6 entry | **closed** — `l6_entry_continuous_v2` 1/1 play `0x79` `(120,205)` 179,355f TF=`0x1F`; east key + west 0x78 + compass enter 0x68 on tape (`l6_compass_continuous_v1` 182,636f). 0x1B west is y=141 LEFT after south-around x≈72, not `↓ ←×7` free / north-edge LEFT. |

Spine-only close contract + room DAG: `docs/LEVEL3_ROUTE.md` § Spine attach.
Isolated `Level3*` checkpoints cannot close these beads.

Exact continuous command (L2 TF), then Manji dest `0x5b` (after closed west key):

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level2 --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level3 --trials 1
```

Expected: `recordings/survival_spine.json` + `.mp4`; `continuous_emulator_session=true`;
`boot_frames` near 200–565; `boot_policy.file_slot=1`; `progression_writes=0`;
`capacity_writes=0`; **`--through level2`**: `triforce & 0x02` in room `0x0d`;
**`--through level3`**: post-L3 OW `0x74`, TF includes `0x04`
(`stop=level3_triforce_0x04`).
`inventory_assist` lists bomb/key count pokes (power-on L2 entry is bombs=0).
Default Clean paths stay untouched. `--no-video` skips the encode.
`--through level1` stops after shard 1.

Last watchable L1+L2 tape (`ok=true`, 50529f, 11 HUD hearts) is **not**
the current encoding. Two bugs from that video:

1. **Hearts 3→7 after TF1, 11 at L2 TF.** Assist wrote
   `(health & 0xF0) | 0x0F`. Zelda 1 `HeartValues` (`$066F`) low nibble
   is whole hearts; full is `lo==hi` (`0x22`=3/3). `World_FillHearts`
   (`INC HeartValues` until `CompareHeartsToContainers`) then grants a
   container each fill. Source: aldonunez `zelda1-disassembly` `Z_05.asm`
   `World_FillHearts` / `CompareHeartsToContainers`, plus `$0670`
   `HeartPartial=$FF`. Assist now writes `0x22`/`0x44`/`0x66` +
   `heart_partial=$FF`, and only accepts +1 on HC `0x1A` or leaving
   mode 18.
2. **Thousands of LEFT/RIGHT/DOWN frames in place.** `unstick_wiggle`
   reset and fought forever. Now one 16f cycle then idle. Dungeon
   combat idles when no live enemies; stuck+live skips the next patrol
   point. Collect stands after one waypoint lap.

**Last live power-on → L3 entrance (Survival, 2026-08-21):** `ok=true`
53918f, room `0x7c` (Manji entry), `tf=0x03`, bombs=8 keys=4, deaths 0,
`poke_bombs=16` `poke_keys=2`, `progression_writes=0`,
`capacity_writes=0`, `accepted_containers=5` (HUD 5 hearts; L2 TF
container increment not observed this tape). L2 entry bombs=0 keys=0;
Survival count top-up at L2 entry + `SPINE_BOMB_RETOPUP`.
`bomb_north_6f` 340f (was 1f `no_bombs`). Boom, Dodongo, L2 TF `0x02`,
OW hop `enter_level3` 12864f all live. That tape stopped at `0x7c`
(`.3.0` closed), not dest 0x5b. West key closed 2026-08-21:
`l3_west_key_spine.json` 54589f room `0x7b` keys=5. Dest `0x5b`
(`.3.1.2`) closed in `l3_dest_0x5b_v12`: 57256f, keys=5, bombs=8,
TF=0x03, deaths/progression/capacity writes 0. Farm is `rr-doua`. Do not
grant undiscovered items.

```bash
QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level3 --no-video --trials 1 --tag l3_raft_spine_v2
```

Dest 0x5b (`rr-4d53.3.1.2`) is closed. Occupancy 0x6b north is
`level3_dest_6b_stages`; combat occupancy_patrol remains 1435f / 5 Zol.
Live ladder that closed the boundary (do not regress):

- v5: LEFT+UP south-mouth clip **works** — inland `(96,133)`, then occupancy
  1px-miss boxed in (51 misses, stood).
- v6: no-path diamond thread reached door column `(120,117)`; UP never hits
  band y=109.
- v7: climb-UP at `(104,133)` still mid-diamond.
- v8/v9: `(112,117)` — UP and LEFT+UP both no-op (x≈112 north-wall stick).
- v10: cardinal LEFT at `(112,117)` no-ops 5500f (`l3_dest_0x5b_v10` samples
  f250 `(112,125)` then f500–6000 `(112,117)` `leave_column_x`).
- v11: DOWN oscillates at x=112, y=125–127 for 6000f.
- v12: RIGHT exits the diagonal pocket; room `0x5b` reached in 945 exit frames.

Natural Raft and the boss suffix are verified. The spine tops bombs 8→16 at
Raft under the documented Survival count shortcut; the isolated suffix remains
recon. The same power-on session now reaches L4 entry `0x71` in 95,281f:
`l4_entry_continuous_v1.json`, TF=`0x07`, keys=4, bombs=0, Raft=1, deaths 0,
state restores 0, progression/capacity writes 0. Do not close `.6` yet.

Exact continuation work:

```bash
UV_CACHE_DIR=/tmp/retro_rl_uv_cache QT_QPA_PLATFORM=offscreen \
  uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-entry --no-video --trials 1 --tag l4_entry_continuous_v1
```

The continuous spine now passes that boundary and clears `0x50`:
`l4_clear50_continuous_v1.json`, 1/1, TF=`0x07`, keys=5, bombs=15, deaths 0,
state loads 0, progression/capacity writes 0. At the verified `0x61` bomb gate,
the operator-authorized Survival exception tops bombs 0→16; the wall consumes
one. Keys remain 4 until the natural `0x51` key raises them to 5.

The continuous spine now reaches `0x40`, clears it, and collects its natural
key: `l4_room40_key_continuous_v7.json`, 1/1, 103,630f, TF=`0x07`, keys=6,
bombs=15, deaths/state loads/progression/capacity writes all 0. The verified
path uses coordinate gates through `(160,177)`, `(116,181)`, `(112,124)`,
`(128,103)`, then UP to y≈93, LEFT to x≈120, and long UP. This replaces the
two failed fixed paths and does not use emulator-state BFS.

The existing `0x40→0x30` north controller is now on the continuous tape from
that leftover `(136,125)`: `l4_room30_continuous_v1.json`, 1/1, 103,857f,
room `0x30` `(120,205)`, keys=6, bombs=15, TF=`0x07`, hop 227f, deaths/state
loads/progression/capacity writes all 0. `--through level4-room30` stops at
enter-`0x30` (Vires still live). Do not close `.6` until TF `0x08`.

The existing `0x30` Vire clear (ignore invuln `0x2b`) plus KEY-RIGHT @y141 is
now on the continuous tape from that south-mouth leftover `(120,205)`:
`l4_room31_continuous_v1.json`, 1/1, 104,524f, room `0x31` `(16,141)`,
keys 6→5, bombs=15, TF=`0x07`, hop 667f (clear 325f + KEY-RIGHT 342f),
deaths/state loads/progression/capacity writes all 0. `--through level4-room31`
stops at enter-`0x31` (maze Vires still live). Do not close `.6` until TF `0x08`.

The `0x31` maze Vire clear is now on the continuous tape from that west-door
leftover `(16,141)`: `l4_clear31_continuous_v7.json`, 1/1, 109,514f, room
`0x31` `(112,141)`, keys=5, bombs=15, TF=`0x07`, hop 4,990f (inland 172f +
clear 4,818f), deaths/state loads/progression/capacity writes all 0.
`--through level4-clear31` stops after the Vire clear (east door is open;
do not require `0x32`). West alcove cardinals stick at `(32,141)`; the
verified inland is RIGHT+UP then waypoints `(48,109)→(80,109)→(80,173)→(128,133)`.
Do not close `.6` until TF `0x08`.

Free RIGHT `0x31→0x32` is now on the continuous tape from leftover `(112,141)`:
`l4_room32_continuous_v11.json`, 1/1, 109,890f, room `0x32` `(16,141)`, keys=5,
bombs=15, TF=`0x07`, hop 376f, deaths/state loads/progression/capacity writes
all 0. `--through level4-room32` is an enter-stop (Zol/LikeLike still live).
North-strip cardinals and south-channel RIGHT at `(128,173)` are dead ends;
the verified exit is UP to y=113, RIGHT+DOWN clip into the east column
`(160,125)`, then `(160,173)→(192,173)→(192,141)→(200,141)`. Do not close
`.6` until TF `0x08`.

The `0x32` Zol+LikeLike clear is now on the continuous tape from leftover
`(16,141)`: `l4_clear32_continuous_v1.json`, 1/1, 113,702f, room `0x32`
`(80,109)`, keys=5, bombs=15, TF=`0x07`, hop 3,812f, deaths/state
loads/progression/capacity writes all 0. `--through level4-clear32` is an
empty-room stop (do not require push-block or stairs `0x60`). Existing
`ROOM_32_SPEC` controller; no inland/occupancy change. Invuln `0x2b` and
block `0x68` residual OK. Do not close `.6` until TF `0x08`.

`--through level4-stepladder` is **1/1** on `l4_stepladder_continuous_v34`
(`ADDR_LADDER` at `(136,141)`, 118,292f, deaths/progression/capacity 0, no
state load). Occupancy v26 over-blocked the east grey dock as south-water
`x=80–175` plus exit `x>=176`. Walkway: west aisle south → y=189 RIGHT to
x=175 → UP the dock → y-first to y=141 → LEFT onto the pedestal. Isolated
BFS is still banned. Do not close `.6` until TF `0x08`.

`--through level4-exit60` is **1/1** on `l4_exit60_continuous_v2` (0x32 play
`(192,189)`, ladder set, 118,806f, hop 514f, deaths/progression/capacity 0,
no state load). Reverse inbound dock after 150f item freeze: RIGHT y=141 to
x=175, DOWN the dock, LEFT y=189, UP west-aisle stairs. v1 leftover
`(176,173)` LEFT mid-dock solid; keep DOWN until y=189. Isolated BFS still
banned. Next: west `0x32→0x31→0x30` then KEY-UP `0x20` (waypoints, no
emulator-state BFS). Do not close `.6` until TF `0x08`.

`--through level4-west31` is **1/1** on `l4_west31_continuous_v1` (0x31 play
`(208,141)`, ladder set, 119,211f, hop 405f). South corridor LEFT around
the pushed 0x68, west-aisle UP, LEFT into 0x31. Isolated WEST_31_SAMPLE_PATH
is not this tape. Next: reverse 0x31 maze east/inland waypoints to 0x30,
then KEY-UP `0x20`. Do not close `.6` until TF `0x08`.

`--through level4-keyup20` is **1/1** on `l4_keyup20_continuous_v1` (0x20 play
`(120,205)`, ladder set, keys 5→4, 120,079f, maze-west 514f + KEY-UP 354f).
Reverse of the verified 0x31 east U then LEFT+UP clip onto the north strip
and inland west. Isolated maze BFS is not this tape. Do not close `.6`
until TF `0x08`.

`--through level4-room21` is **1/1** on `l4_room21_continuous_v22` (0x21
play `(16,141)`, ladder set, keys=4 bombs=15, 121,775f, clear 1249f +
path 447f). 0x20 Vire clear stays 1/1 (max_live=7, ignore 0x2b). Isolated
`map_21` used Vire-clear then **state-saving BFS** (banned). PNG H-water:
H-bar y=144–159, spines x=48–63 / 192–207, arms y=112–127 and 176–191,
water ends y=191, east door only at y=141 x=208. Gold walls classify as
gold in stills — live collision wins. v20 DOWN at x=200 is solid at
y=109 (16px spine). v21 RIGHT+DOWN from `(200,96)` clips into x=208.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| map v1 | `0x20` `(120,141)` map_solid | RIGHT along door y from the N-S gold |
| map v2 | `0x20` `(120,133)` map_solid | H-bar is 8px north of y=141 |
| map v3 | `0x20` `(120,205)` map_solid | south band RIGHT around the H |
| map v4 | `0x20` `(200,189)` map_solid | y=192 RIGHT stays on south gold |
| map v5 | `0x20` `(120,193)` timeout stall=0 | exact y=192 hold |
| map v6 | `0x20` `(120,199)` map_solid | window y=192–200 can RIGHT at x=120 |
| map v7 | `0x20` `(120,199)` after clear | same, without Vires |
| map v8/v9 | `0x20` `(120,193)` timeout stall=0 | door-column UP to y=192 then RIGHT |
| map v10 | `0x20` `(120,93)` timeout stall=0 | UP to y=96 then cardinal RIGHT (north door) |
| map v11 | `0x20` `(136,93)` timeout stall=0 | RIGHT+DOWN clip then y=94–98 |
| map v12 | `0x20` `(160,101)` map_solid | y=90–108 clears the top arm |
| map v13 | `0x20` `(200,96)` timeout stall=0 | y=96 north-around reaches x=208 (east **wall**, door is y=141) |
| map v14 | `0x20` `(160,93)` map_solid | y=88 at x=160 is walkable |
| map v15 | `0x20` `(192,93)` map_solid | RIGHT+UP clip from x=192 |
| map v16 | `0x20` `(200,93)` map_solid | RIGHT+UP clip at v13 leftover |
| map v17 | `0x20` `(136,189)` map_solid | south-door RIGHT+UP clip then DOWN to y=192 |
| map v18 | `0x20` `(120,95)` timeout stall=0 | cardinal RIGHT at y=96 from the north door (v10) |
| map v19 | `0x20` `(136,94)` timeout stall=0 | clip stops at x=136; DOWN/RIGHT yo-yo (v11) |
| map v20 | `0x20` `(200,109)` map_solid | DOWN the east column from v13 leftover |
| map v21 | `0x20` `(208,133)` push_solid | RIGHT+DOWN clip into x=208; PUSH y-slop 8 RIGHT into wall |
| map v22 | `0x21` `(16,141)` **play** | y=141 then RIGHT; hop 447f |

PNGs: `recordings/l4_room21_continuous_v{1..22}_final.png`.

`--through level4-map` is **2/2** on `l4_map_continuous_v15` (and `v15b`):
dark leftover `(16,141)` → spawn RIGHT+UP to `(48,93)` → RIGHT+DOWN clip
into the maze → `ADDR_MAP|0x08` at `(208,181)` in 297f; map=`0x0A`;
122,072f; deaths/progression/capacity 0; no state load. Isolated
`MAP_21_SAMPLE_PATH` is still state-BFS after gel thrash — not this tape.
Cardinal RIGHT at y=109 is still the vestibule wall; the exit is the
two-button clip from the north strip, not occupancy 4-connected. PNG
interior stays black (no candle). Do not call `level4_room_nav`. Do not
close `.6` until TF `0x08`.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| pick v1 | `0x21` `(48,141)` map_solid | UP at inland x=48 |
| pick v2 | `0x21` `(48,141)` map_solid | RIGHT along door y |
| pick v3 | `0x21` `(48,141)` map_solid | RIGHT+UP clip (0x31 alcove) |
| pick v4 | `0x21` `(48,141)` map_solid | DOWN at x=48 |
| pick v5 | `0x21` `(48,141)` map_solid | RIGHT+DOWN clip |
| pick v6 | `0x21` `(48,117)` map_solid | UP at x=32 then RIGHT at y=117 |
| pick v7 | `0x21` `(48,101)` map_solid | north-around y=96 then RIGHT |
| pick v8 | `0x21` `(32,93)` map_solid | UP to y=80 (north wall of west column) |
| pick v9 | `0x21` `(32,100)` timeout stall=0 | RIGHT+UP at north end clips east (yo-yo) |
| pick v10 | `0x21` `(48,173)` map_solid | DOWN at x=32 then RIGHT at y=173 |
| pick v11 | `0x21` `(48,189)` map_solid | south band y=189 then RIGHT |
| pick v12 | `0x21` `(32,189)` map_solid | RIGHT+DOWN clip at SE corner |
| pick v13 | `0x21` `(48,125)` map_solid | 0x31 x≥40 off-band exit, then cardinal RIGHT |
| pick v14 | `0x21` `(48,109)` map_solid | spawn RIGHT+UP lands `(48,93)`; DOWN then cardinal RIGHT |
| pick v15 | `0x21` `(208,181)` **map bit** | RIGHT+DOWN from `(48,93)` clips east; hop 297f 2/2 |

PNGs: `recordings/l4_map_continuous_v{1..15}_final.png`. Wired
`--through level4-map` stop is `ADDR_MAP & 0x08` on play `0x21` (not
gel-clear).

`--through level4-bomb11` is **2/2** on `l4_bomb11_continuous_v2` (and
`v2b`): leftover `(208,181)` → UP the east column to y=93 → LEFT to
bomb stand `(120,105)` → bomb-UP into play `0x11` `(120,189)` in 435f;
122,507f; bombs 16→15 (documented count top-up at this gate); map=`0x0A`;
keys=4; TF=`0x07`; deaths/progression/capacity 0; no state load. Isolated
BFS is still not a spine path. PNG interior stays black (no candle).

| tag | leftover | wrong belief |
|-----|----------|----------------|
| bomb v1 | `0x21` `(192,109)` timeout | cardinal LEFT at y=109 after UP the east column |
| bomb v2 | `0x11` `(120,189)` **play** | north-around y=93 then LEFT; hop 435f 2/2 |

PNGs: `recordings/l4_bomb11_continuous_v{1,2,2b}_final.png`.

`--through level4-key01` is **2/2** on `l4_key01_continuous_v3` (and
`v3b`): bomb-UP stand `(120,105)` 377f then Keese-clear + pickup
`(120,141)` 819f; leftover play `0x01` `(120,133)`; keys 4→5; bombs
15→14; 123,703f; map=`0x0A`; TF=`0x07`; deaths/progression/capacity 0;
no state load. Isolated BFS is still not a spine path. PNG interior
stays black (no candle).

| tag | leftover | wrong belief |
|-----|----------|----------------|
| key v1 | `0x11` `(120,93)` timeout | free UP of 0x11 (north is a bomb wall) |
| key v2 | `0x01` `(96,135)` timeout | floor-key hunt at `(96,125)` (key is east) |
| key v3 | `0x01` `(120,133)` **keys 4→5** | bomb-UP then pickup `(120,141)`; hop 1196f 2/2 |

PNGs: `recordings/l4_key01_continuous_v{1,2,3,3b}_final.png`.

`--through level4-clear12` is **2/2** on `l4_clear12_continuous_v1` (and
`v1b`): DOWN 0x01→0x11 244f, bomb-RIGHT `(192,141)` 392f, Vire clear
654f (ignore `0x68`); leftover play `0x12` `(128,117)`; bombs 14→13;
keys=5; 124,993f; map=`0x0A`; TF=`0x07`; deaths/progression/capacity 0;
no state load.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| clear12 v1 | `0x12` `(128,117)` **cleared** | DOWN + bomb-RIGHT + Vire clear; hop 1290f 2/2 |

PNGs: `recordings/l4_clear12_continuous_v{1,1b}_final.png`.

`--through level4-gleeok13` is **2/2** on `l4_gleeok13_continuous_v2`
(and `v2b`): v1 y-first leftover `(128,141)` DOWN solid on the door row;
x-first to `PUSH_12_STAND` `(112,144)`, hold LEFT, hold4
`PATH_12_TO_GLEEOK`; leftover play `0x13` `(32,141)` in 414f.

`--through level4` is **2/2** on `l4_tf_continuous_v1` (and `v1b`):
south-stand Gleeok 3564f, TF `0x07→0x0F`, mode 18 room `0x03` `(120,149)`,
128,971f; deaths/progression/capacity 0; no state load. HC was not
mid-room (`hc_collected=false`). Isolated BFS still not this tape.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| gleeok v1 | `0x12` `(128,141)` timeout | y-first DOWN to push stand (door-row solid) |
| gleeok v2 | `0x13` `(32,141)` **play** | x-first then token; hop 414f 2/2 |
| tf v1 | `0x03` `(120,149)` **TF 0x08** | south-stand 3564f 2/2; HC not mid-room |

PNGs: `recordings/l4_gleeok13_continuous_v{1,2,2b}_final.png`,
`recordings/l4_tf_continuous_v{1,1b}_final.png`. `.6` closed.

`--through level5-entry` is **1/1** on `l5_entry_continuous_v1`: L4 fanfare
settle 284f onto island `0x45`, then `POST_L4_TO_LEVEL5_HOPS` (not old
At4A) through Lost Hills into play `0x76` `(120,205)`; 134,393f; TF=`0x0F`
keys=5 bombs=13; deaths/progression/capacity 0; no state load.

`--through level5-clear66` v1 timeout 12,000f leftover `0x66` `(119,173)`
2/3 Gibdo north of the river (cardinal patrol never crossed). v2
occupancy miss-block **1/1** `l5_clear66_continuous_v2`: 4,241f leftover
`(32,101)` keys 5→6; 138,634f.

`--through level5-east77` is **1/1** on `l5_east77_continuous_v1`:
north-bank leftover RIGHT to ladder x=56 then DOWN (v1 DOWN at x=32
never crosses); Pols Voice clear leftover `(136,165)` keys 7; 142,958f.

`--through level5-whistle` is **1/1** on `l5_whistle_continuous_v1`:
return 0x66 bomb-west → 0x65 bomb-west → 0x64 stairs → cellar → key-west
0x05 → Recorder `0x04` mode 9 `(135,141)`; 160,648f hop 17,690f; keys
7→6 bombs 13→8; deaths/progression/capacity 0; no state load. Whistle
earned (not granted).

`--through level5` is **1/1** on `l5_tf_continuous_v1`: extract
`run_level5_whistle_tf` into `level5_boss_path`; leftover cellar
`(135,141)` mode 9 → `exit_whistle_04` play `0x05` `(144,141)`; 0x06
v1 center-idle `(112,141)` tile 119 never warps; v2
`take_block_stairs_06` RIGHT onto `(128,141)` → cellar `0x07`; 0x65
north shutter sealed, bomb-east `0x66`; skip-fight to Digdogger `0x24`;
whistle shrink `0x38→0x18`; TF `0x0F→0x1F` mode 18 room `0x14`
`(120,149)`; 173,961f hop 13,311f; keys 6→5 bombs=8; deaths/progression/
capacity 0; no state load. `validate_l5_endpoint` passes. `.7` and `.4`
closed.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| entry v1 | `0x76` `(120,205)` **play** | L4 settle + POST_L4 hops; hop 5,138f 1/1 |
| 66 v1 | `0x66` `(119,173)` timeout | cardinal patrol reaches north-bank Gibdos |
| 66 v2 | `0x66` `(32,101)` **keys 5→6** | occupancy miss-block; hop 4,241f 1/1 |
| east77 v1 | `0x77` `(136,165)` **play** | north-bank to ladder x=56 then DOWN; 1/1 |
| whistle v1 | `0x04` `(135,141)` mode 9 **Recorder** | EastKey→0x04 suffix; hop 17,690f 1/1 |
| tf v1 stairs | `0x06` `(112,141)` tile 119 | center idle `(120,141)` warps after 0x68 push |
| tf v1 | `0x14` `(120,149)` mode 18 **TF 0x10** | block-stairs RIGHT `(128,141)`; hop 13,311f 1/1 |

PNGs: `recordings/l5_entry_continuous_v1_final.png`,
`recordings/l5_clear66_continuous_v{1,2}_final.png`,
`recordings/l5_east77_continuous_v1_final.png`,
`recordings/l5_whistle_continuous_v1.json` (v1 PNG was a stale 0x77 obs;
RAM is cellar 0x04),
`recordings/l5_tf_continuous_v1_final.png`.

`--through level6-entry` is **1/1** on `l6_entry_continuous_v2` (play `0x79`
`(120,205)`, 179,355f hop 4,884f). `--through level6-east-key` **1/1**
`l6_east_key_continuous_v1` keys 5→6. `--through level6-west` **1/1**
`l6_west_continuous_v1` play `0x78` `(144,141)` keys 6→5, 182,415f.
`--through level6-compass` **1/1** `l6_compass_continuous_v1` play `0x68`
`(120,205)` 182,636f hop 221f. `--through level6-clear68` **1/1**
`l6_clear68_continuous_v1` play `0x68` `(120,149)` compass bit `0x20`,
187,575f hop 4,939f. `--through level6-keese` **1/1** `l6_keese_continuous_v1`
play `0x58` `(120,205)` hop 209f. `--through level6-clear58` **1/1**
`l6_clear58_continuous_v1` play `0x58` `(112,167)` hop 882f. `--through
level6-room48` **1/1** `l6_room48_continuous_v1` play `0x48` `(120,205)` hop
341f, keys=5 (free UP). `--through level6-room38` **1/1**
`l6_room38_continuous_v1` play `0x38` `(120,189)` hop 261f. `--through
level6-clear38` **1/1** `l6_clear38_continuous_v1` play `0x38` `(32,149)`
hop 5,487f, 194,755f, max_live=7. `--through level6-room28` **1/1**
`l6_room28_continuous_v6` play `0x28` `(120,189)` hop 3,207f, 197,962f.
Left 0x68 slot11 `(96,144)→(96,136)` then west-aisle north.
`--through level6-clear28` **1/1** `l6_clear28_continuous_v1` play `0x28`
`(120,181)` hop 2,587f, 200,549f, max_live=2 orange `0x24`. Combat
occupancy-patrol 0 misses. Isolated BFS banned.
`--through level6-room18` **1/1** `l6_room18_continuous_v7` play `0x18`
`(120,189)` hop 280f, 200,829f. LEFT+UP at y=181, hold UP, RIGHT+UP at
y=109. Keys=5 (no spend). `--through level6-settle18` **1/1**
`l6_settle18_continuous_v1` IDLE 512f, type **`0x44`** (never `0x43`/`0x46`
during idle) + `0x56`, leftover `(120,189)`, 201,341f. `--through
level6-gleeok18` **1/1** `l6_gleeok18_continuous_v1` south-stand body-gone
hop 2,848f, 204,189f leftover `(121,133)`. `0x46` mid-fight. East shutter
still closed. `--through level6-postgleeok18` **1/1**
`l6_postgleeok18_continuous_v2` hop 192f, 204,381f leftover `(156,133)`.
No `0x46`; `0x56` then gone; `cur_opened_doors` 0→5; `open_doorway_mask`
0. `--through level6-stairs18` **red** (v1–v5; north hole decorative).
`--through level6-room19` **1/1** `l6_room19_continuous_v1` play `0x19`
`(16,141)` hop 251f, 204,632f. Occupancy y=141 RIGHT; map still `0x0A`.
`--through level6-clear19` **1/1** `l6_clear19_continuous_v1` play `0x19`
`(176,158)` hop 4,213f, 208,845f. Census 2× Zol `0x13` + 2× Like-Like
`0x17`; RoomItemId `0x17`. `--through level6-map19` **red** (v1 boxed;
v2 on sprite `(120,181)` bit still `0x0A`). Do not grant Map/Rod. Do not
poke doors/keys.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| v25 | `0x1B` `(112,61)` LEFT solid | walkthrough `↓ ←×7` is a free north-edge LEFT |
| v26 | `0x1B` `(96,165)` LEFT solid | y=165 is south of the x≈72 rock (it is the bottom row) |
| v27 | `0x1B` `(71,189)` timeout | UP under the south face of that rock |
| v28 | `0x1B` `(48,189)` LEFT solid | south sand reaches the west edge |
| v31 | `0x1B` `(24,149)` LEFT solid | screenshot tan x<32 y=136–151 is a free walk (mountain dither) |
| v32 | `0x1B` `(32,140)` timeout | LEFT+UP clips west (UP-priority yo-yo) |
| v35 | `0x1A` `(224,133)` **play** | y=141 LEFT after south-around; hop_1_1a |
| v37 | `0x15` `(104,141)` timeout | door-Y LEFT through Lynels |
| v1 | `0x15` `(232,109)` timeout | continuous east-edge leftover; inland then south band |
| v38 | `0x14` `(112,189)` timeout | 0x14 south mouth is center x=112 |
| v40 | `0x23` `(160,141)` timeout | 0x23 DOWN is center x=112 (mountain splitter) |
| entry v2 | `0x79` `(120,205)` **play** | SE blue 0x14 x=160 / 0x23 x=208; hop 4,884f 1/1 |
| east v1 | `0x7a` `(120,141)` **keys 5→6** | wall-first RIGHT; hop 1,844f 1/1 |
| west v1 | `0x78` `(144,141)` **cleared** | key-LEFT; hop 1,216f 1/1 |
| compass v1 | `0x68` `(120,205)` **play** | occupancy UP; 8 miss-blocks x=144; hop 221f 1/1 |
| clear68 v1 | `0x68` `(120,149)` **compass** | occupancy-patrol Zols; hop 4,939f 1/1 |
| keese v1 | `0x58` `(120,205)` **play** | occupancy UP from 0x68; hop 209f 1/1 |
| clear58 v1 | `0x58` `(112,167)` **cleared** | occupancy-patrol 8× Keese; hop 882f 1/1 |
| room48 v1 | `0x48` `(120,205)` **play** | occupancy long-UP is free; hop 341f 1/1 |
| room38 v1 | `0x38` `(120,189)` **play** | occupancy run-UP through traps; hop 261f 1/1 |
| clear38 v1 | `0x38` `(32,149)` **cleared** | occupancy-patrol 7 live; Bubble residual; hop 5,487f 1/1 |
| room28 v1 | `0x38` `(32,149)` stand 6000f | cardinal occupancy from west door can north |
| room28 v2 | `0x38` `(120,93)` 6000f | occupancy UP at north shutter; looks open, is sealed |
| room28 v3 | `0x38` `(96,133)` 8000f | hardcoded `(96,157)` + 200f UP is a push |
| room28 v4 | `0x38` `(96,133)` 8000f | occupancy from the push plane can path around |
| room28 v5 | `0x38` `(120,164)` 8000f | UP @ x=120 from south band is a clear lane |
| room28 v6 | `0x28` `(120,189)` **play** | live 0x68 y-move then west aisle x=64; hop 3,207f 1/1 |
| clear28 v1 | `0x28` `(120,181)` **cleared** | occupancy-patrol 2× orange `0x24`; hop 2,587f 1/1 |
| room18 v1 | `0x28` `(120,181)` stand 6000f | occupancy UP from leftover can path (freeze-miss boxed 4 cardinals) |
| room18 v2 | `0x28` `(120,181)` 6000f | hold UP at leftover walks north (y never moved) |
| room18 v3 | `0x28` `(80,181)` 6000f | LEFT to x=80 then UP; LEFT works, UP at y=181 still solid |
| room18 v4 | `0x28` `(80,181)` 6000f | peel DOWN to y=189 then aisle UP crosses y=181 (walks to 181, then solid) |
| room18 v5 | `0x28` `(96,173)` 6000f | LEFT+UP clips the y=181 face then cardinal RIGHT to x=120 (RIGHT solid at clip cell) |
| room18 v6 | `0x28` `(96,109)` 6000f | hold UP from clip cell reaches north band; then cardinal RIGHT to x=120 (RIGHT solid at y=109) |
| room18 v7 | `0x18` `(120,189)` **play** | RIGHT+UP clip at y=109; hop 280f 1/1 |
| settle18 v1 | `0x18` `(120,189)` **play** | IDLE census; type **`0x44`** not `0x43`; hop 512f 1/1 |
| gleeok18 v1 | `0x18` `(121,133)` **body-gone** | LEFT+UP y=189 then south-stand `0x44`; hop 2,848f 1/1 |
| postgleeok18 v1 | `0x18` `(121,133)` 17f | `cur_opened_doors` RIGHT = walkable east (PNG shutter black; mask 0) |
| postgleeok18 v2 | `0x18` `(156,133)` **census** | mask stays 0; no `0x46`; hop 192f 1/1 |
| stairs18 v1 | `0x18` `(160,117)` 4000f | occupancy to (120,109) UP-first from x=156 |
| stairs18 v2 | `0x18` `(120,93)` 4000f | hold-UP on the hole is mode 9 |
| stairs18 v3 | `0x18` `(120,109)` 4000f | idle at y=109 (tile `0x76` diamond, south of hole) |
| stairs18 v4 | `0x18` `(120,101)` 4000f | idle at y=101 (tile `0x77`, still south of hole) |
| stairs18 v5 | `0x18` `(120,95)` 4000f | idle at y=96 (tile `0x77`, still south of hole; hole decorative) |
| room19 v1 | `0x19` `(16,141)` **play** | occupancy y=141 RIGHT despite mask 0 / PNG black; hop 251f 1/1 |
| clear19 v1 | `0x19` `(176,158)` **cleared** | 2× Zol + 2× Like-Like; hop 4,213f 1/1; Map on floor |
| map19 v1 | `0x19` `(176,93)` 6000f | occupancy to (120,173) from leftover (176,158) |
| map19 v2 | `0x19` `(120,181)` 6000f | idle on the Map sprite is `ADDR_MAP|0x20` |

PNGs: `recordings/l5_to_l6_v{25,26,31,35,42}_final.png`,
`recordings/l6_entry_continuous_v{1,2}_final.png`,
`recordings/l6_east_key_continuous_v1_final.png`,
`recordings/l6_west_continuous_v1_final.png`,
`recordings/l6_compass_continuous_v1_final.png`,
`recordings/l6_clear68_continuous_v1_final.png`,
`recordings/l6_keese_continuous_v1_final.png`,
`recordings/l6_clear58_continuous_v1_final.png`,
`recordings/l6_room48_continuous_v1_final.png`,
`recordings/l6_room38_continuous_v1_final.png`,
`recordings/l6_clear38_continuous_v1_final.png`,
`recordings/l6_room28_continuous_v{4,5,6}_final.png`,
`recordings/l6_clear28_continuous_v1_final.png`,
`recordings/l6_room18_continuous_v{1,2,3,4,5,6,7}_final.png`,
`recordings/l6_settle18_continuous_v1_final.png`,
`recordings/l6_gleeok18_continuous_v1_final.png`,
`recordings/l6_postgleeok18_continuous_v{1,2}_final.png`,
`recordings/l6_stairs18_continuous_v{1,2,3,4,5}_final.png`,
`recordings/l6_room19_continuous_v1_final.png`,
`recordings/l6_clear19_continuous_v1_final.png`,
`recordings/l6_map19_continuous_v{1,2}_final.png`. Dest `0x19` clear is
**on the tape** (`l6_clear19_continuous_v1` 1/1). Map sprite on floor;
`ADDR_MAP` still `0x0A`. v2 idle on `(120,181)` did not set bit 0x20.
Next: column x=120 then idle `(120,141)` (compass analog). Do not grant
Map/Rod. Do not close `rr-tne2`.

```bash
QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-map19 --no-video --trials 1 --tag l6_map19_continuous_v3
```

Wrong belief (clear58 leftover PNG): north shutter closed ⇒ sealed. Live
occupancy UP from `(112,167)` entered 0x48 with keys still 5.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| v1 | `0x60` `(48,133)` HUNT timeout | spawn band y±32; HUNT RIGHT into water |
| v3 | `0x32` `(192,189)` mode 10 | `MAZE_60_TO_LADDER` 18 DOWN from `(48,77)` hits **exit stairs** |
| v4 | `0x60` `(136,189)` | isolated token path from `(48,69)` dumps south corridor; UP is water |
| v5 | `0x32` `(192,189)` | cap DOWN at y=109 still RIGHT+UP to exit |
| v6/v7 | `0x60` `(48,133)` | west-aisle cardinal RIGHT solid at y=69..157 |
| v8 | `0x60` `(48,133)` | south corridor UP solid at x=80..144; x≥176 is exit |
| v9/v10 | `0x60` `(48,133)` | RIGHT+UP at y=133 oscillates, no east |
| v11 | `0x60` `(48,133)` clips_done | RIGHT+UP/DOWN at y=133/125/141/117 all miss x=48 |
| v12 | `0x60` `(48,68)` timeout | occupancy north-strip y=68 RIGHT is a causeway; live is north wall |
| v13 | `0x60` `(48,185)` timeout | x=152 UP from south corridor; UP+LEFT only slides west |
| v14 | `0x60` `(160,189)` stairs_up_solid | stairs-column x=160 UP (west of grey stairs) |
| v15 | `0x60` `(168,189)` stairs_up_solid | stand on grey stairs x=168 then UP |
| v16 | `0x60` `(48,157)` gap158_solid | 7px band y=158 RIGHT between west-brick and south-water |
| v17 | `0x60` `(48,161)` notch161_solid stall=0 | simultaneous RIGHT+UP at the occupancy gap; live is UP-priority, x stays 48 |
| v18 | `0x60` `(48,159)` notch161_solid stall=0 | 1-frame RIGHT/UP tap-hold walks the y=158 gap; live RIGHT at y=159 is solid (no gap) |
| v19 | `0x60` `(84,189)` corner80_solid stall=0 | south-corridor SW corner RIGHT+UP clips onto the island; live slides east along y=189, UP is water |
| v20 | `0x60` `(88,189)` rightdown84_solid stall=0 | RIGHT+DOWN at leftover `(84,189)` clips onto the island; live DOWN is south brick, RIGHT slides 4px east (same y=189 band) |
| v21 | `0x60` `(84,189)` leftup88_solid stall=0 | LEFT+UP at `(88,189)` clips NW through the water SW corner; live UP is water, LEFT slides 4px west |
| v22 | `0x60` `(48,165)` rightdown161_solid stall=0 | RIGHT+DOWN at SW notch `(48,161)` clips east onto the island; live RIGHT is west-brick, DOWN-priority slides 4px south |
| v23 | `0x60` `(48,157)` leftup161_solid stall=0 | LEFT+UP at `(48,161)` clips NW onto the island; live LEFT is west wall, UP-priority slides 4px north |
| v24 | `0x60` `(48,130)` leftup133_solid stall=0 | LEFT+UP at west-aisle `(48,133)` (v11 burned RIGHT+DOWN here); live same UP-priority, LEFT wall |
| v25 | `0x60` `(48,71)` rightdown68_solid stall=0 | RIGHT+DOWN at north-strip `(48,68)` clips east; live RIGHT is north-brick, DOWN 3px |
| v26 | `0x60` `(48,65)` leftup68_solid stall=0 | LEFT+UP two-wall corner at `(48,68)` clips onto the island; live LEFT wall, UP 3px into north brick |
| v27 | `0x60` `(171,189)` dock_solid | east-dock UP is x=175; abs(dx)>4 idled 4px short |
| v28 | `0x60` `(176,157)` dock_solid | reached dock; island x-first LEFT into water south of y=151 |
| v29 | `0x60` `(176,149)` dock_solid | UP past 151 then LEFT; 2px north of causeway |
| v30 | `0x60` `(176,151)` timeout stall=0 | LEFT at y=151 slides y; DOWN yo-yo never west |
| v31 | `0x60` `(176,149)` timeout stall=0 | y=150–152 LEFT still yo-yos north |
| v32 | `0x60` `(141,141)` controller done, RAM ladder=0 | y-first to 141 then LEFT reached island; abs<=6 idled 5px east of pickup |
| v33 | `0x60` `(138,141)` same | abs<=2 idled 2px east |
| v34 | `0x60` `(136,141)` **ADDR_LADDER** | exact pickup cell |
| exit v1 | `0x60` `(176,173)` exit_solid | LEFT from x=176 back to x=175 mid-dock |
| exit v2 | `0x32` `(192,189)` **play + ladder** | DOWN the dock column until y=189 |

PNGs: `recordings/l4_stepladder_continuous_v{1,2,4,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34}_final.png`
(v3/v5 exit `0x32`); `l4_exit60_continuous_v{1,2}_final.png`.

### Why isolated BFS found a path occupancy says does not exist

Checked against `l4_tib8_60_path.json`, `l4_tib8_stepladder_stepladder.json`
(`stepladder_bfs` start `[48,69]`, hold=4, q=4, n_cells=98), 
`_bfs_60_to_ladder` / `_follow_60_ladder_path` in `level4_room_nav.py`,
`room_60_grid()`, leftover PNGs v12–v19, and a hold-4 token replay on the
seeded grid.

1. **Goal-state restore, not a token walk.** `_bfs_60_to_ladder` saves
   `em.get_state()` per 4px cell and, when `ADDR_LADDER` increments, restores
   `goal_state`. `_follow_60_ladder_path` is a no-op if ladder is already set.
   Isolated 2/2 is checkpoint BFS + `set_state`. Spine token replay of
   `MAZE_60_TO_LADDER` (v3/v4) dumps the south corridor.
2. **Hold-4 / q=4 quantization.** Each BFS edge is 4 frames of one cardinal.
   A 1–3px wall-slide or knockback that changes the 4px cell is recorded as
   reachable. Occupancy is 1px 4-connected. Unblocked geometric replay of the
   70 tokens from `(48,69)` ends at `(84,25)` (out of bounds). Against the
   seeded grid the first RIGHT (token 22) is a no-move at `(48,125)→(49,125)`.
3. **Keese knockback.** The BFS docstring says the walkable pocket depends on
   Keese / entry-frame jitter. Search runs with live 4× `0x1b`. A hit that
   displaces Link ≥4px opens a cell static occupancy never has. Spine Keese
   after ~118k frames are not the isolated checkpoint.
4. **Spawn is the same.** Isolated start is `(48,69)`; continuous west-aisle
   leftover is the same column. Not a mode-9 spawn mismatch.
5. **Center water is unseeded (unknown=free).** v26 occupancy enclosed the
   island because south-water `x=80–175,y=158–180` and exit `x>=176` (all y)
   painted over the east grey dock. v34 carves that dock (`x>=175`, y<189)
   and the stairs lip (`x=80–174,y=158–188`). OccupancyWalker spawn→island
   is DOWN, RIGHT, UP, LEFT. Isolated BFS still jumps/restores goal state.
6. **Not a two-button clip.** Isolated dirs are `UP/DOWN/LEFT/RIGHT` one at
   a time. `RIGHT,UP,RIGHT,UP` tokens are sequential holds, not `nes_action("RIGHT","UP")`.

Live v34 collected `ADDR_LADDER` on the continuous tape. Inventory: TF=`0x07`
keys=5 bombs=15 ladder=1; deaths/state/progression/capacity 0.

Live v2 exited mode-9 `0x60→0x32` on waypoints (no BFS). Live v1 west
`0x32→0x31` leftover `(208,141)`. Live v1 KEY-UP `0x30→0x20` leftover
`(120,205)` keys 5→4. Live 0x20 Vire clear 1249f. Live v22
`0x20→0x21` leftover `(16,141)` play `0x21`. Map pickup is on the tape
(v15 2/2 leftover `(208,181)`, `ADDR_MAP|0x08`). Bomb-UP `0x11` is on
the tape (v2 2/2 leftover `(120,189)`). Do **not** call
`_bfs_60_to_ladder` or `level4_room_nav` / map_21 state-BFS. Do not
close `.6` until TF `0x08`.

Exact verified predecessor:

```bash
UV_CACHE_DIR=/tmp/retro_rl_uv_cache QT_QPA_PLATFORM=offscreen \
  uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-room21 --no-video --trials 1 \
  --tag l4_room21_continuous_v22
```

Map-pickup stop (verified v15 2/2):

```bash
UV_CACHE_DIR=/tmp/retro_rl_uv_cache QT_QPA_PLATFORM=offscreen \
  uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-map --no-video --trials 1 \
  --tag l4_map_continuous_v15
```

Bomb-UP 0x11 stop (verified v2 2/2):

```bash
UV_CACHE_DIR=/tmp/retro_rl_uv_cache QT_QPA_PLATFORM=offscreen \
  uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-bomb11 --no-video --trials 1 \
  --tag l4_bomb11_continuous_v2
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-key01 --no-video --trials 1 \
  --tag l4_key01_continuous_v3
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-clear12 --no-video --trials 1 \
  --tag l4_clear12_continuous_v1
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-gleeok13 --no-video --trials 1 \
  --tag l4_gleeok13_continuous_v2
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4 --no-video --trials 1 \
  --tag l4_tf_continuous_v1
```
Isolated 0x6b check:

```bash
uv run python nes/zelda_i/scripts/run_level3_north_chain.py --trials 2
```

L2-exit → L3 OW hops are 2/2 assisted from `Level2ExitOverworld`
(`run_l2_to_l3.py`). L3 dest 0x5b is on `--through level3` (`level3_dest_6b_stages`).
Isolated north-chain does not close `.3.1.2`.

Bomb/key **count** pokes are a documented Survival shortcut
(`docs/ASSIST_CONTRACT.md`). Do not grant undiscovered items or write
`max_bombs`. Not a Clean claim.

## Parked L7/L8 boundary (2026-08-14)

`rr-dnp` now has a deterministic Survival-assisted pond controller from
`PostSwordStart`. The live walk reaches `0x53` through
`77→78→68→58→57→56→55→65→64→54→53`. The `0x64` east-ledge escape and
`0x54→0x53` transition are encoded and unit-tested. The last trial stopped on
`0x53` at `(224,173)` while trying to align DOWN for the west hop to `0x52`;
it had zero deaths and zero progression/capacity writes. It did **not** reach
the pond, so `OW_L7Pond` was not saved.

Exact continuation command:

```bash
PYTHONPATH=nes uv run python nes/zelda_i/scripts/probe_level7_entry.py \
  --allow-missing-caps --infinite-life --save-state --max-frames 10000 \
  --tag l7_dnp_pond_assisted_v10
```

Before rerunning, add one `level7_overworld.py` micro for `0x53`: move LEFT
inland from the east edge before descending toward the lower west gap, then
push LEFT to `0x52`. Evidence to compare:
`recordings/l7_dnp_pond_assisted_v9.json` and its `_final.png`.

Level 8 is parked at the existing `0x6D` bush/candle boundary. Do not reopen
the old poke-burn result as a route claim. After L7 yields the natural Red
Candle, the smallest L8 boundary is Red Candle + `Level8BushOW` → burn `0x6D`
→ live entry room; otherwise the residual is natural 60R farm → Blue Candle
buy → burn. No new L8 checkpoint or claim was made in this pass.

## Parked — predecessor of blade-trap/Like-Like room 0x41 (2026-08-14)

Verified backward recon remains an explicit fixture, not Clean or Survival
route STATUS:

- Blade-trap/Like-Like room `0x41` settles with four traps and four
  Like-Likes. The north mask is visible, but live enemies block the walk.
  Controller-only clear followed by north lands east-bomb `0x31`; no door or
  next-room poke is used.
- Continuous fixture compose `0x41→0x31→0x30→0x67→0x04→0x03→0x52→credits`
  is **1/1**: credits 25,858, final page 27,058, total 27,148 frames. Runtime
  object/room/door/inventory/progression/capacity writes are all zero.
- Evidence: `recordings/l9_room41_dump.json`,
  `recordings/l9_play41_north_patra_credits_recon.json`, and
  `Level9Room41NorthReconFixture`. The start still inherits fixture inventory
  and loader setup, so `route_eligible=false`.
- `0x51` is the identified south predecessor of `0x41` (6× Like-Like `0x17`;
  loader `0x61` hold UP, no `0x41` door poke). North dest walk is **NO**:
  after clear, center-aisle UP sticks at `(120, 117)` on the statue diamond.
  `rr-sz8.4` closed dest-NO. Next leaf **`rr-yxy6`**: thread the diamond
  from south-door spawn `(120, 205)`, else materialize `0x61`. Keep `0x40`
  out. `route_eligible=false`.

```bash
uv run python nes/zelda_i/scripts/run_level9_stairs.py \
  --dump-51 --tag l9_room51_dump
```

The forward East Key `0x77` → natural Recorder → Whistle basement `0x04`
seam is closed (`rr-4d53.5`, `Level5WhistleFrom77`). Attach that pin to the
proven `0x04` → Digdogger → L5 TF suffix before claiming a continuous L5
reel.

## Strategy (finish easy → then tune)

**Order of work (agents):** pathfinding and puzzle solving first → full-game
route under Survival assist → Clean combat/heart harden using damage heatmaps.

1. **Infinite life + damage tracking** — `--infinite-life` Survival assist
   (`UnlimitedHealthAssist`) keeps agents alive. Telemetry records
   `total_damage`, `damage_by_location`, samples (see ASSIST_CONTRACT). Not
   Clean STATUS.
2. **Path + puzzles first** — overworld hops, door geometry, keys, bomb walls,
   push-blocks, item gates. Do **not** block route progress on sword kiting.
3. **Pure-first rooms** — isolated controller → natural-entry → graph promote
   (geometry + stop predicates; combat only as needed to open doors).
4. **Expand beads at the tip** — epics for L2–L9 + OW prep + Death Mountain;
   spawn room children when that dungeon is active (~80–120 total by credits).
5. **Clean pass later** — rank rooms by assist `damage_by_location`; heart farm /
   combat harden only after geometry is known. Never demote assisted greens
   into Clean STATUS rows.
6. **Adventure harness** — keep RAM/combat local; `RouteGraph`, `NamedRoute`,
   legs, waypoints stay on `retro_harness.adventure`.

## Next milestones

1. **Survival spine** — `rr-4d53.2.3` Boom→TF closed (documented bomb/key
   top-up). L3 entrance `0x7c` closed (`.3.0`). West key `0x7b` closed
   (`.3.1.1`). The full entrance→Raft corridor (`.3.3`) is closed; next is
   the verified Raft→TF `0x04` suffix, then `.6` L4 and `.7`
   L5, then `.4` one-session L5 TF. L6 entry through Gleeok enter `0x18` are
   on the continuous tape; Gleeok fight / TF `0x20` / Rod / Gohma residual.
   L7–L9 stay out.
2. **L9 backward** — parked P4 (`rr-yxy6` / `rr-sz8`). Fixture suffix stays
   `route_eligible=false`.
3. **M6 route graph** — L3–L5 NamedRoute / door_graph / composer now exist;
   use them to sequence the assisted checkpoint chain, then dry-run.
4. **M7–M8** — verified full-game capture (assisted first, Clean later).

## Bottleneck

**Power-on → L3 TF `0x04`** is verified on the continuous Survival spine.
The L3-exit → L4 suffix is the watchable tip.
West key `0x7b` (`.3.1.1`) and entrance `0x7c` (`.3.0`) are closed. Then
Raft (`.3.3.*`), bombs (`.3.2`), TF (`.3.4.*`). L9 dest walk is parked.

## Video / watchability (2026-08-06)

Hitbox-gated sword + faster boot landed (not a STATUS promote):

- `combat.py` sword rectangle + `should_swing_at`; dungeon + L1 early rooms
  slash only in blade range / contact (patrol walks clean).
- OW `walk_or_swing` — no air-swings on empty screens (`nav_common` /
  `overworld_nav` / `ow_path`).
- Boot `BOOT_PERIOD=50` (~565f ready vs ~1749); YouTube intro 90f; cave
  dialog idle 180f.

Residual room-by-room combat polish only if a clear regresses under hitbox gate.
See `docs/tasks/QUEUE.md`.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Graph package: `retro_harness.adventure` (first consumer; second consumer later for promotion of richer APIs).
- Sword cave geometry (probe-stable): approach ~(60,100) on 0x77, cave mode 11, align x=120, walk up to sword, exit down.
- Level 1 overworld path (probe-stable 2026-07-28): east-then-north via 0x78/68/58/48/38 → 0x37; door enter UP at x≈112 from y≈140.
- Dungeon prefix (probe-stable 2026-07-28):
  `0x73→E 0x74→first key→W 0x73→unlock N→0x63→clear→N 0x53→clear/key`.
- Cleared 0x53 branches `W→0x52` (six Keese, item `0x03`) and `E→0x54`
  (eight Keese, item `0x16`); north is closed.
- Room 0x54 clear is 2/2 isolated + 2/2 natural; west returns to 0x53 and east
  is blocked.
- Level 1 completion is 2/2 isolated + 2/2 Clean power-on natural. See
  `docs/LEVEL1_ROUTE.md`; the required suffix ends on `triforce & 0x01`.
- Level 2 approach: engine settle to 0x37, walk prefix to 0x4A (see
  `docs/LEVEL2_ROUTE.md`). Avoid 0x79 dead-end; do not rely on mid-fanfare
  `Level1Complete.state` reloads.
- External walkthroughs/maps are approved planning accelerators. Keep their
  claims source-linked and separate from live emulator verification.
- Use `scripts/dungeon_lab.py` and `docs/DUNGEON_LAB.md` for future rooms.

- Dungeon door-graph template: `door_graph.py` (`LEVEL_2_DOOR_GRAPH` seed). See `docs/DUNGEON_LAB.md` § Door graph (`rr-mhl`).

### Item gates (`rr-iri`)
- ### ZOW — early item gates (rr-iri pathing; rr-38p residual)
- Planned hop tables in `item_gate_hops.py` (geometry only, assisted OK):
- Probe: `scripts/probe_item_gate_hops.py --route all --infinite-life`.
