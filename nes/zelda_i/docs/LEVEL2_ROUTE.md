# Level 2 route — The Moon

Planning sources:

- [Zelda Dungeon — Level 2: The Moon](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-2-the-moon/)
- [IGN — Dungeon Two](https://www.ign.com/wikis/the-legend-of-zelda/Dungeon_Two)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)

Walkthrough claims below that are emulator-verified are marked; source-only
claims stay labeled.

## Post-Triforce return (verified)

Collecting shard 1 sets `ADDR_TRIFORCE & 0x01` and enters **mode 18** (fanfare).
After ~535 idle frames the engine transitions (modes 2→3→4) and places Link on
**overworld screen 0x37** at ~(112, 125) around frame **704**.

- Prefer `Level1ExitOverworld.state` or a live settle after collection.
- Reloading `Level1Complete.state` mid-fanfare can freeze mode 18.

## Verified walk prefix (0x37 → 0x4A)

```text
0x37 ─E@y≈140─► 0x38 ─S@x≈120─► 0x48 ─S@x≈112─► 0x58
  ─E@y148–162─► 0x59 ─N@x≈112─► 0x49 ─E@y≈141─► 0x4A
```

Stop: `level2_path_prefix_success` — overworld play, screen 0x4A, sword ≥ 1,
triforce & 0x01. Evidence: `recordings/level2_prefix_isolated.json` (3/3).

Per-screen hop timing (emulator frames, `RoomTimer`, 2026-07-29 1/1):
`recordings/room_timings/level2_prefix_isolated_timing.json` — six hops,
slowest `0x49→0x4A` (~539 location frames); transitions ~83–104f.

## Full door path (probe-verified geometry; health not Clean yet)

Walkthrough target door: overworld **0x3C**. Naive “right four from start”
hits rocky dead-end **0x79**. North-entry `0x4B→0x5B` **seals east** (BFS max
x≈144). Correct corridor enters **0x5B from the west via 0x5A**.

```text
0x37 E@y140 → 0x38 S → 0x48 S → 0x58 E → 0x59
  E@y120–145 → 0x5A E@y130–150 → 0x5B
  E@y80–95 (north bush corridor) → 0x5C
  [maze] E@y≈88 to x≈184, down to y≈128, E → 0x5D
  N@x≈48–56 → 0x4D W@y120–170 → 0x4C N@x112 → 0x3C (Moon door UP)
```

Hop tables: `overworld.LEVEL2_DOOR_HOPS`, maze pixels
`LEVEL2_5C_MAZE_WAYPOINTS`. Fixture states (dev): `Level2DoorOW.state`,
`Level2Entrance.state`, `Level2EntryFresh.state`.

### Graph / NamedRoute coverage (scaffold)

`build_early_route_graph()` seeds every door-path screen and tags forward hops:

| Segment | Screens / edges | `verification` |
|---------|-----------------|----------------|
| Shared with 0x4A prefix | 0x37→38→48→58→59 | `observed` (`to_level2_prefix`) |
| Door-only hops | 0x59→5A→5B→5C→5D→4D→4C→3C | `planned` / `probe_geometry` (`to_level2_door`) |
| Abstract walk | `ow_37_post_triforce` → `ow_3c` | `planned` |
| Enter Moon | `ow_3c` → `dungeon_level2` | `planned` |

Named routes: `zelda_level2_path_prefix` (STATUS-aligned 0x4A),
`zelda_level2_door_path` / aliases `to_level2`, `level2_door` (planned
milestones through 0x5A / 0x5C / 0x3C / dungeon; stop predicates are future
controller names, not Clean-backed). Do **not** promote door hops to
`observed` until 2/2 Clean isolated to 0x3C.

### 0x5C maze (required)

BFS path cells (cell size 8): east along row y≈88 to gx≈23, up/down channel at
gx≈24, then east at y≈128 into 0x5D. Plain `ScreenHop` RIGHT with a single
y-band is **not** enough.

### 0x5D north

North exit only near **x≈48–56** (not center). East also opens to 0x5E but is
not on the door route.

## Interior (walkthrough + live recon)

Assisted recon from `Level2Entrance` (Survival, 2026-08-06) is recorded in
`recordings/l2_recon_probe.json` + `l2_recon_*.png`. Durable runners below.
**Not Clean STATUS.**

### Verified rooms 0x7d / 0x6d / 0x6c / 0x7e (isolated pure Clean)

| Room | Role | Enemies (live) | Doors | RoomItemId | Notes |
|------|------|----------------|-------|------------|-------|
| **0x7d** | Entry (south mouth) | **none** combat types at ready (idle 120f empty); `room_obj_count=3` | `cur_opened_doors=0`; **north walkable** without open bit; **east open** via diamond-nav (not door bit) | **0x03** (no inventory) | mode 5, xy≈**(120, 205)**; level==2 |
| **0x6d** | North of entry | **5× Rope type `0x28`** | before clear `0x00`; after clear **`0x02` LEFT bit** | **0x03** | spawn **~100f** after `screen==0x6d`; TYPE_AND_HP; clear → `room_all_dead≥20` |
| **0x6c** | West of 0x6d (key) | **6× Rope `0x28`** | `0x00` at enter (no kill-door) | **0x19** fixed small key | key pickup near **(136, 141)** mid-fight; `RoomAllDead` lags after last kill |
| **0x7e** | East of entry (key) | **5× Rope `0x28`** | no kill-door required | **0x19** fixed small key | diamond-nav entry from 0x7d; LEFT→0x7d, UP→0x6e |

| Claim | Source | Live |
|-------|--------|------|
| OW door 0x3C | walkthrough + probe | verified entry (UP @ x≈112) |
| Entry room | live settle | **0x7d** (south mouth; mode 16→2→3→4→**5**) |
| Entry snapshot | assisted 2/2 | `level==2` mode **5** xy≈**(120, 205)**; doors `0x00`; empty combat spawn at ready |
| North of entry | live recon | **0x6d**, **5** Ropes type **0x28**, spawn **~100f** after screen change |
| Rope liveness | live recon | **TYPE_AND_HP** once play-settled (HP `0x10`); type present with HP 0 during mode-4 settle |
| Clear opens left door | live recon + pure | `cur_opened_doors` **bit1 (`0x02`)** at clear; physical LEFT @ **y≈141** |
| West key room | **isolated pure Clean** | **0x6c**, 6× Rope, fixed key `0x19`, keys 0→1 |
| Entry east | **isolated pure Clean** | **0x7e** via diamond-nav (not sealed); naive y≈141 RIGHT sticks @x≈128 |
| Entry west | live raster | **sealed** |
| East of 0x6d | live door probe | **0x6e**, **3× Rope `0x28`**, RoomItemId `0x03` (no key); RIGHT residual |
| Magical Boomerang room | **isolated pure Clean** | **0x4f** via 0x5f bomb-N; 3× type `0x05` + fireballs `0x55`; RoomItemId `0x1e` |
| Magical Boomerang RAM | Data Crystal + pure | `ADDR_BOOMERANG=0x0674`, `ADDR_MAGIC_BOOMERANG=0x0675` stop ≠0 |
| Dodongo needs bombs | walkthrough + live | **LIVE** bomb-mouth type `0x32` (assisted) |
| Triforce bit 0x02 | live assisted | **LIVE 2/2** Boom→TF (`rr-5dk`) |

Specs + controllers: `dungeon.ROOM_7D_SPEC`, `ROOM_6D_SPEC`, `ROOM_6C_SPEC`,
`ROOM_7E_SPEC`; `ROPE_OBJECT_TYPE=0x28`; `GenericDungeonRoomController`. Stops:
`level2_room_6d_cleared`, `level2_room_6c_key_success`,
`level2_room_7e_key_success`.

#### Isolated pure evidence (Clean, `Level2Entrance` / `Level2RopesCleared`)

| Segment | Start | Stop | Trials | Frames | Evidence |
|---------|-------|------|--------|--------|----------|
| 0x6d ropes clear | `Level2Entrance` | `level2_room_6d_cleared` (all_dead≥20, doors&0x02) | **2/2** | 674 | `recordings/level2_clear6d_isolated.json` |
| 0x6c west key | `Level2RopesCleared` | `level2_room_6c_key_success` (keys≥1, 0 live) | **2/2** | 708 | `recordings/level2_clear6c_isolated.json` |
| 0x7d→0x6d→0x6c chain | `Level2Entrance` | same key stop | **2/2** | ~710 (6c stage) | `recordings/level2_clear6c_from_entrance_isolated.json` |
| 0x7e east key | `Level2Entrance` | `level2_room_7e_key_success` (keys≥1, 0 live) | **2/2** | 1110 | `recordings/level2_clear7e_isolated.json` |
| 0x6f bomb N → 0x5f | `Level2Compass` | `level2_room_5f_ready` (sc 0x5f mode 5) | **2/2** | 5420 | `recordings/level2_bomb_north_isolated.json` |
| 0x5f key-LEFT → 0x5e Goriya clear | `Level2_5F` | `level2_room_5e_cleared` (5× Goriya 0x06 dead) | **2/2** | 12815 | `recordings/level2_clear5e_isolated.json` |
| 0x5f bomb N → 0x4f Magical Boomerang | `Level2_5F` | `level2_room_4f_magic_boomerang_success` (`ADDR_MAGIC_BOOMERANG≠0`) | **2/2 Clean** | bomb~613 + fight~581 | `recordings/level2_magic_boomerang_isolated.json` |

Policy (lab-promoted): 0x6d `attack_phase=4` `engage=64`; 0x6c `attack_phase=2`
`engage=64`; 0x7e `attack_phase=4` `engage=64` + diamond entry
`((120,157),(208,157),(208,141))` RIGHT. Checkpoints: `Level2RopesCleared.state`,
`Level2WestKey.state`, `Level2EastKey.state`. Lab sweeps: `recordings/lab_l2_6d/`,
`recordings/lab_l2_6c/`.

### Post-west-key live graph (assisted recon, 2026-08-06)

From `Level2WestKey` / `Level2Entrance`. Survival + optional bombs/keys poke.

```text
0x6c west key ──RIGHT──► 0x6d ropes
                 ◄─LEFT── (doors bit 0x02 after 0x6d clear)
0x6d ──DOWN──► 0x7d entry ──DOWN──► OW 0x3c
0x6d ──RIGHT─► 0x6e (3× Rope)   0x6d UP sealed
0x7d ──RIGHT─► 0x7e (5× Rope + key 0x19)   ★ entry-east LIVE
0x7e ──UP────► 0x6e                 0x7e LEFT → 0x7d
0x6e ──DOWN──► 0x7e
0x6e ──RIGHT─► 0x6f (6× Gel + compass 0x16)  ★ key door LIVE (rr-c6b)
0x6f ──bomb N @ (120,101)──► 0x5f   ★ walkthrough bomb-N LIVE (rr-ebe 2026-08-06)
0x5f ──LEFT (key)──► 0x5e (Goriya 0x06)   ★ LIVE
0x5f ──bomb N @ (120,101)──► 0x4f (boom candidate RoomItemId 0x1e)  ★ LIVE (rr-cjf)
0x5f RIGHT sealed (walk+bomb; not the boom path)
0x5e ──UP (free)──► 0x4e (5× Rope + key 0x19)  ★ LIVE (rr-cjf)
0x5e ──bomb R @ (176,141)──► 0x5f   ★ walk-RIGHT blocked max_x≈160
0x4e ──RIGHT──► 0x4f   0x4e ──UP──► 0x3e   0x4e ──DOWN──► 0x5e
0x6c LEFT/UP/DOWN sealed            0x7d LEFT sealed
```

**Trap — diamond solids (0x7d / 0x6e east):** naive `y≈141` RIGHT hits solid
near **x≈128–176**. Helper: `nav_common.diamond_east_phase` (also
`ROOM_7E_SPEC.entry` waypoints):

1. Free mid (east alcove: **LEFT** first; do not y-only spin)
2. Open **y-band** → RIGHT to wall (**x≥200**)
   - 0x7d: band **y≈157** (or 149)
   - 0x6e: band **y≈113** — prefer **WEST** entry via 0x6d (south from 0x7e can stick ~y=181)
3. At wall: **S2** LEFT×6 → vertical to **door y≥137** (poke-verified; y≤133
   never opens the key door) → RIGHT×10 (repeat)
4. **Pure push** RIGHT on y=141 — **no LEFT during push** (re-enters solid)

**Trap — 0x6f bomb N:** walkthrough bomb-north is **LIVE** only from stand
**(x≈120, y≈101)** facing UP with bombs selected (sel often already 0x01 when
bombs present). Dense stands at y=96–105 with free pathing often miss; poke
`(120,101)` then B+UP is the reliable recon open. Opens **0x5f** (doors bit
DOWN). R/D/L bomb walls on 0x6f: **no open** in full stand sweep.

| Room | Role | Enemies (live) | RoomItemId | Notes |
|------|------|----------------|------------|-------|
| **0x7e** | East of entry (east key) | **5× Rope `0x28`** | **`0x19` key** | **isolated pure 2/2** (`Level2EastKey`); LEFT→0x7d, UP→0x6e |
| **0x6e** | N of 0x7e / E of 0x6d | **3× Rope `0x28`** | `0x03` | **key-RIGHT → 0x6f** (consumes 1 key) |
| **0x6f** | Compass branch | **6× Gel `0x15`** TYPE-only (hp=0 alive) | **`0x16` compass** | **isolated pure 2/2** (`Level2Compass`); east-wall pickup ~(192–208,101); **bomb N → 0x5f** |
| **0x5f** | N of compass (bomb) | **5× Gel `0x15`** TYPE-only + map | `0x17` map | DOWN→0x6f; **LEFT key → 0x5e**; **bomb N → 0x4f**; RIGHT sealed |
| **0x5e** | W of 0x5f | **Goriya `0x06`** (5× peak, TYPE_AND_HP HP≈48) | `0x03` | **isolated pure 2/2** (`Level2_5E`); free **UP→0x4e**; bomb-R→0x5f |
| **0x4e** | N of Goriya | **5× Rope `0x28`** | **`0x19` key** | RIGHT→0x4f; UP→0x3e; DOWN→0x5e (rr-cjf); `ROOM_4E_SPEC` |
| **0x4f** | Magical Boomerang | **3× type `0x05`** TYPE_AND_HP (HP≈80) + fireballs `0x55` (ignore) | **`0x1e`** | **isolated pure 2/2 Clean** (`Level2Boom`); pickup ~(136,135); paths bomb-N 0x5f or 0x4e RIGHT |
| **0x3e** | N of 0x4e | residual | `0x19` key (seen) | free UP from 0x4e; exits not fully mapped |

Evidence: `recordings/l2_east_open.json`, `l2_6e_right_ok.json`,
`l2_east_map.json`, `level2_clear7e_isolated.json` (**Clean pure 0x7e**),
`level2_clear6f_isolated.json` (**Clean pure 0x6f**),
`l2_past6f_expand.json` / `l2_5f_explore.json` (**bomb N + 0x5e**),
`l2_6f_exits.json` (negative R/D/L bomb + door y probe).

**Walkthrough:** “right → 5 Ropes + key” = **0x7e**. “3 Ropes → key RIGHT →
compass” = **0x6e → 0x6f**. “optional bomb N” = **0x6f → 0x5f**. Carry **≥2
keys** into 0x6e (west + east) so one remains after the key door; another key
for **0x5f LEFT**.

#### Magical Boomerang (isolated pure Clean, rr-bsq / rr-ebe)

| Item | Value |
|------|--------|
| Inventory stop | `level2_room_4f_magic_boomerang_success` — `ADDR_MAGIC_BOOMERANG (0x0675) != 0` |
| RoomItemId | **`0x1E` on 0x4f** (L1 wooden boom was `0x1D`) |
| Enemies | **3× type `0x05`** TYPE_AND_HP HP≈80 (`BLUE_GORIYA_OBJECT_TYPE`); fireballs **`0x55`** not clear targets |
| Pickup | dense probe **~(136, 135)** after kill; also mid-combat near center |
| Primary path | `Level2_5F` → bomb N @(120,101) → 0x4f clear+collect (`Level2BoomBombNorthController` + `ROOM_4F_SPEC`) |
| Alt path | `Level2_5E` free UP → 0x4e (`ROOM_4E_SPEC`) key-RIGHT → 0x4f |
| Evidence | `recordings/level2_magic_boomerang_isolated.json` (**2/2 Clean**); via-4e `level2_magic_boomerang_via4e.json` (assisted 1/1) |
| Specs / runner | `ROOM_4E_SPEC`, `ROOM_4F_SPEC`; `scripts/run_level2_magic_boomerang.py`; checkpoint `Level2Boom` |

Walk-RIGHT on 0x5f is **not** required (bomb-UP is the boom open).

## Puzzle catalog (`rr-3pz`)

Path/puzzle geometry for lab use. **Data module:** `nes/zelda_i/level2_puzzles.py`
(pure constants + open predicates; no combat rewrite). Does **not** claim Clean
STATUS. Walkthrough-only future bomb walls (map room, Moldorm detour, Dodongo
corridor) stay residual until room IDs are live.

### Bomb walls

| Room | Stand (x,y) | Face | Opens to | Live / source | Evidence |
|------|-------------|------|----------|---------------|----------|
| **0x6f** (compass) | **(120, 101)** | UP | **0x5f** | **LIVE** (walkthrough bomb-N) | `l2_past6f_expand.json` (`bomb_tests` stand [120,101] ok); `l2_5f_explore.json` |
| **0x5f** (map gels) | **(120, 101)** | UP | **0x4f** | **LIVE** pure boom | `level2_magic_boomerang_isolated.json`, `l2_cjf_expand.json` |
| **0x5e** (Goriya) | **(176, 141)** | RIGHT | **0x5f** | **LIVE** (walk-R blocked) | `l2_cjf_expand.json` |
| 0x6f | various | R / D / L | — | **FAIL** (no open) | `l2_6f_exits.json`, `l2_6f_bombn.json` |
| 0x6f | dense y=96–105 (not 101) | UP | — | **FAIL** free-path miss | `l2_6f_exits.json` (stands e.g. (120,100),(120,105),(112,100)…) |
| 0x5f | various | RIGHT | — | **FAIL** walk+bomb | `l2_cjf_expand.json` |
| **0x4f** (boom) | **(120, 101)** | UP | **0x3f** | **LIVE** pure | `level2_bomb_north_4f_isolated.json` |
| **0x1e** (Goriya) | **(120, 101)** | UP | **0x0e** Dodongo | **LIVE** assisted | `l2_1e_up.json`, `level2_dodongo.json` |
| 0x1e | mid | walk UP | — | **FAIL** solid (doors=12) | `l2_1e_up.json` strict_x120 |
| 0x0e | various | bomb R / walk R | TF? | **FAIL** sealed after kill | `l2_boss_exits.json` |

**Open predicate (lab):** transition `0x6f → 0x5f` after B-place facing UP at
stand; destination often `cur_opened_doors` **DOWN=4**. Inventory: need
`bombs≥1`; B-item often already selected on fixtures (sel sometimes `0x01` live;
probe poke uses `0x02`). Constant: `BOMB_WALL_6F_NORTH` / `bomb_wall_open_predicate`.

**Place policy (recon):** goto (120,101) → face UP → B+UP → wait ~60–90f → push UP.
Generic probe `BOMB_STAND["UP"]=(120,109)` is **not** the verified open.

Walkthrough residual: Moldorm bomb-N; Keese/trap bomb-N. Map-gels bomb-N is
**LIVE → 0x4f**. Natural bomb farm for Dodongo is past **0x4f / 0x3e** residual.

### Push-blocks / diamond solids

No verified **push-block → new door** on live L2 graph yet. Mid-room **diamond
solids** block naive east corridors (not classic center-block stairs).

| Room | Band y | Sequence | Destination | Key? | Evidence |
|------|--------|----------|-------------|------|----------|
| **0x7d** | **157** (or 149) | `diamond_east_phase`: free → band → wall x≥200 → S2 (LEFT×6, vert door_y, RIGHT×10) → **pure RIGHT** on y=141 | **0x7e** | no | `level2_clear7e_isolated.json`, `l2_east_open.json` |
| **0x6e** | **113** | same; prefer **WEST** entry via 0x6d | **0x6f** | **yes** | `l2_6e_right_ok.json`, `l2_6e_band_scan.json` |
| 0x6f | 113 (probe try) | diamond-east RIGHT residual | — | — | `level2_puzzles.DIAMOND_EAST_ROOMS` |

**Door y poke:** east wall opens only for **y≥137** (y≤133 never). Constants:
`DIAMOND_BAND_7D=157`, `DIAMOND_BAND_6E=113`, `DOOR_Y_MIN_OPEN=137` (also
`nav_common`). **Trap:** no LEFT during final push; south entry into 0x6e from
0x7e can stick ~y=181.

**Push-block probe (negative):** centers
`(120,141),(136,141),(104,141),(120,125),(120,157)` cardinal pushes on 0x6f did
not open a new exit (`l2_6f_blocks.json`
push_log doors stayed closed until other nav).

### Key doors

| Room | Approach | Key cost | Destination | Live | Evidence |
|------|----------|----------|-------------|------|----------|
| **0x6e** RIGHT | WEST entry + diamond band≈113 + pure push door_y≥137 | **1** | **0x6f** | **LIVE** | `l2_6e_right_ok.json` (`key_consumed`), `l2_east_open.json` (k2→1) |
| **0x5f** LEFT | mid y≈141 after bomb-N | **1** | **0x5e** (Goriya) | **LIVE** | `l2_5f_explore.json` (keys 4→3), `l2_past6f_expand.json` |
| **0x5f** RIGHT | walkthrough guess | — | — | **SEALED** (boom = bomb-UP → 0x4f) | `l2_cjf_expand.json`, `l2_5f_policy.json` |

Carry **≥2 keys** into 0x6e (west + east) so one remains for 0x6f; another for
**0x5f LEFT**. Predicates: `KEY_DOOR_6E_RIGHT`, `KEY_DOOR_5F_LEFT`,
`key_door_open_predicate` (room pair + keys_before−after == cost).

Kill-clear doors (not key): **0x6d** clear sets LEFT bit `0x02` → 0x6c (combat;
see room specs, not this catalog).

### Negative probes / sealed exits

| Probe | Result | Evidence |
|-------|--------|----------|
| 0x6f bomb R / D / L (full stand sweep) | no room change; doors stay LEFT-only | `l2_6f_exits.json`, `l2_6f_bombn.json` |
| 0x6f bomb UP off-stand (y≠101 dense grid) | no open | `l2_6f_exits.json` |
| 0x6f door cycle R/U/D without bomb | no open (LEFT returns 0x6e) | `l2_6f_exits.json` door_results |
| 0x5f RIGHT after bomb entry / Goriya / gel clear | sealed (walk+bomb) | `l2_past6f_expand.json`, `l2_cjf_expand.json` |
| 0x5f walk-UP (no bomb) | sealed; **bomb-UP LIVE → 0x4f** | `l2_cjf_expand.json` |
| 0x5e walk-RIGHT | blocked max_x≈160; **bomb-RIGHT LIVE → 0x5f** | `l2_cjf_expand.json` |
| 0x7d LEFT | sealed | live raster / recon |
| 0x6c LEFT / UP / DOWN | sealed | live recon |
| 0x6d UP | sealed | live recon |
| OW 0x4B→0x5B north entry | east of 0x5B sealed | door-path section above |

Code lists: `BOMB_WALL_NEGATIVES_6F`, `SEALED_EXITS` in `level2_puzzles.py`.

## L2 assisted complete LIVE (`rr-5dk`, 2026-08-07)

**Assisted tip green:** continuous `Level2Boom` → Dodongo → TF south-band →
`ADDR_TRIFORCE & 0x02` **2/2**. Survival only (`--infinite-life`). **Not**
Clean STATUS / natural power-on (deferred).

| | |
|--|--|
| Start | `Level2Boom` (0x4f Magical Boomerang owned) |
| Stop | `tf & 0x02`, mode 18, room **0x0d** @(128,149) |
| Trials | **2/2** assisted |
| Fight | Dodongo type `0x32` bomb-mouth ~1632f |
| TF phase | south-band policy ~513f (`policy_live=True`) |
| Checkpoint | **`Level2Complete.state`** (+ provenance) |
| Runner | `scripts/run_level2_dodongo.py` / `scripts/run_level2_complete.py` |
| Evidence | `recordings/l2_complete_assisted.json` (+ `_t0`/`_t1`/`_summary`) |
| Prior geometry | `recordings/l2_tf02_encode.json` (`rr-n5i`) |

```bash
uv run python nes/zelda_i/scripts/run_level2_dodongo.py \
  --infinite-life --from-state Level2Boom --trials 2 \
  --tag l2_complete_assisted --save-state
# or:
uv run python nes/zelda_i/scripts/run_level2_complete.py \
  --infinite-life --trials 2 --save-state
```

**Compose scope:** post-boom tip only (one env session, no mid-run state load).
Earlier pure segments (entry ropes / keys / compass / boom) remain **isolated
Clean green** from their own checkpoints. Full **Entrance→TF** continuous
multi-controller compose and power-on L1+L2 are **PARTIAL / deferred** — do
not block L3 tip on them. Clean residual after damage heatmaps: `rr-4oz`.

## Dodongo / triforce 0x02 (`rr-n5i` assisted LIVE geometry, 2026-08-07)

Goal: bomb Dodongo → Heart Container → **LEFT** into TF room **0x0d** →
`ADDR_TRIFORCE & 0x02`.
**Boss + HC + TF collect path LIVE (assisted).** Not Clean STATUS / natural-entry.

**Geometry trap:** walkthrough “TF **east** of boss” is **wrong live**. After
kill, doors are **LEFT-only** → room **0x0d** is **WEST** of Dodongo 0x0e.
RIGHT is sealed (key/bomb/push fail). Catalog: `ROOM_L2_BOSS=0x0E`,
`ROOM_L2_TF=0x0D`, `L2_TF_COLLECT_WAYPOINTS` in `level2_puzzles.py`.

### Post-boom tip chain (assisted LIVE)

```text
0x4f bomb N @(120,101) → 0x3f Keese → LEFT 0x3e Moldorm → UP 0x2e ropes clear
  → UP 0x1e Goriya clear → bomb N @(120,101) → 0x0e Dodongo (type 0x32)
  → bomb-mouth → HC → LEFT 0x0d (WEST of boss) / RIGHT SEALED
  → south-band maze → tf & 0x02 (mode 18)
```

### 0x0d triforce collect (LIVE south-band maze)

Spawn after LEFT from boss: **~(224, 141)**. Do **not** sit in east alcove —
first free column is **x≈208**. Diamond floor is a maze open from the **south**
(not a solid seal). North-band green sprite is a **red herring** (full y=93 walk
never sets `tf&0x02`). Push/bomb not required.

```text
(208, 141) → (208, 189) → (128, 189) → (128, 149) → idle until mode 18 / tf&0x02
```

| | |
|--|--|
| Collect box | x∈[112,128], y∈[140,149] (hit ~(128,149)) |
| tol | ≈3 |
| Constants | `L2_TF_COLLECT_WAYPOINTS`, `POST_BOSS_TF_POLICY` (`live=True`) |
| Checkpoint | `Level2_0D_PostBoss` |
| Runner | `level2_boss_path` / `run_level2_dodongo.py` / `run_level2_complete.py` |
| Evidence | `recordings/l2_0d_tf_reach.json`, `l2_tf02_encode.json` |

| Claim | Live |
|-------|------|
| **0x4f** bomb N → **0x3f** | **isolated pure 2/2 Clean** (`run_level2_bomb_north_4f`) |
| **0x3f** LEFT → **0x3e** Moldorm | LIVE assisted; TYPE clear + key |
| **0x3e** UP → **0x2e** | LIVE; 8× Rope; clear opens UP bit |
| **0x2e** UP → **0x1e** | LIVE; south-band y≈189 lateral then x=120 UP (mid-y diamond trap) |
| **0x1e** walk-UP after clear | **SOLID** (doors=12 red herring; min_y≈117) |
| **0x1e** bomb N @(120,101) | **LIVE → 0x0e** Dodongo ★ |
| Dodongo type | **`0x32`** |
| Boss kill + HC | LIVE assisted (`heart_containers` rises; `Level2_0E`) |
| Post-kill doors | **LEFT=2 only** → **0x0d** (WEST); RIGHT sealed |
| **0x0d** TF collect | **LIVE assisted** south-band waypoints; `room_item` `0x1b` |
| `triforce & 0x02` | **LIVE 2/2 assisted** Boom→TF (`rr-5dk`; `l2_complete_assisted.json`) |

Bombs inventory notes (for future boss policy):

- `ADDR_BOMBS = 0x0658`; B places when bombs selected as B-item (sel often `0x01`
  with bombs already in inventory — no START menu required in recon).
- `Level2EastCleared` / entry-fresh fixtures often start with **bombs=4** and
  selected pos already bombs — useful for bomb-wall recon only.
- Natural bomb source for Dodongo (walkthrough: Red Goriya room drop) is
  **beyond** current residual (0x4f clear / 0x3e / boss approach).

Evidence: `recordings/l2_east_open.json`, `l2_6e_right_ok.json`,
`l2_dodongo_path_recon.json`, `l2_boomerang_partial.json`,
`l2_past6f_expand.json`, `l2_5f_explore.json`, `l2_cjf_expand.json`. Shared
residual: **0x4f Magical Boomerang pure** (`rr-bsq` / `rr-ebe` / `rr-n5i`).

### Assisted Moon entry (2026-08-06, Survival only)

`OverworldToLevel2Controller(door_path=True, require_dungeon=True)` +
`--infinite-life --enter-dungeon` from `Level1ExitOverworld` (prefix→0x4A
rejoin→door path→UP door→idle settle):

| Trial | ok | final | frames | evidence |
|-------|----|-------|--------|----------|
| t0 | True | lvl2 mode5 sc **0x7d** (120,205) | 5871 | `recordings/l2_entry_assisted_t0_probe.json` |
| t1 | True | lvl2 mode5 sc **0x7d** (120,205) | 5871 | `recordings/l2_entry_assisted_t1_probe.json` |

Stop predicate: `level2_entrance_success` — not mid mode-16 on OW `0x3C`.
Checkpoint: `Level2Entrance.state` rewritten as **room-ready** settle.
Contract: `docs/ASSIST_CONTRACT.md` (not Clean STATUS).

Speed-route sketch (source): N ropes → W key → return → E key → N/E with keys
→ optional Compass/Map/bomb shortcuts → Magical Boomerang → Moldorm key →
Ropes unlock → Goriya bombs → **Dodongo** (2 mouths) → Heart → Triforce shard 2.

## Traps

| Trap | Detail |
|------|--------|
| 0x79 rocky dead-end | No east exit from 0x78 east@y≈180. |
| 0x37 east lane | Only **y≈140** exits east; y≈125 re-enters Level 1. |
| 0x4B→0x5B north entry | East of 0x5B unreachable; use **0x5A→0x5B**. |
| 0x5C north pocket | Entering only at y≈93 without maze cannot reach 0x5D. |
| 0x5A damage corridor | Arrives low on hearts; Clean farm/heal still open. |
| 0x5D north x | Must align **x≈52**, not x112. |
| Room-ready | After dungeon room transition wait for enemy types (Ropes ~100f after `screen` change; HP may lag until mode 5). |
| 0x6d LEFT door | Need mid-height **y≈141**; wall-hug at y≈157 with x≈47 does not transition. |

## Controllers / runner

```bash
uv run python zelda_i/scripts/run_to_level2_prefix.py --trials 2
uv run python zelda_i/scripts/run_to_level2_prefix.py --room-timing --trials 1
uv run python zelda_i/scripts/probe_level2_suffix.py --help
# Assisted first-pass to Moon entry room (not Clean):
uv run python zelda_i/scripts/probe_level2_suffix.py --infinite-life --enter-dungeon --save-state --tag l2_entry_assisted
# Isolated pure key branch (Clean from Level2Entrance / Level2RopesCleared):
uv run python nes/zelda_i/scripts/run_level2_clear6d.py --trials 2
uv run python nes/zelda_i/scripts/run_level2_clear6c.py --trials 2
uv run python nes/zelda_i/scripts/run_level2_clear6c.py --from-entrance --trials 2
uv run python nes/zelda_i/scripts/run_level2_clear7e.py --trials 2
uv run python nes/zelda_i/scripts/run_level2_clear7e.py --save-state --trials 2
uv run python nes/zelda_i/scripts/run_level2_clear6f.py --trials 2
uv run python nes/zelda_i/scripts/run_level2_clear6f.py --save-state --trials 2
# Assisted Boom → Dodongo → TF (not Clean):
uv run python nes/zelda_i/scripts/run_level2_magic_boomerang.py --infinite-life
uv run python nes/zelda_i/scripts/run_level2_complete.py --infinite-life --trials 2
# Diamond-east: nav_common.diamond_east_phase / ROOM_7E_SPEC.entry.
# 0x6e RIGHT: WEST entry + key + band≈113 wall-vertical pure push → 0x6f (door y≥137).
# 0x6f bomb N: stand (120,101) UP+B → 0x5f; LEFT key → 0x5e Goriya.
# Puzzle constants (no emu): zelda_i.level2_puzzles — BOMB_WALL_6F_NORTH, KEY_DOORS, DIAMOND_*
```

- `level2_overworld.PostTriforceSettleController`
- `level2_puzzles` — bomb stands / key doors / diamond bands (lab import)
- `level2_overworld.OverworldToLevel2Controller` (default stop 0x4A;
  `door_path=True` + maze; `require_dungeon=True` → room-ready 0x7d)
- `dungeon.GenericDungeonRoomController` + `ROOM_6D_SPEC` / `ROOM_6C_SPEC` /
  `ROOM_7E_SPEC` / `ROOM_6E_SPEC` / `ROOM_6F_SPEC`
- `nav_common.diamond_east_phase` — reusable diamond-blocked east doors
- Door path + maze not yet promoted to a 2/2 Clean natural runner
- Opt-in hop timing: `chain.run_controller_stage(..., room_timer=)` /
  runner `--room-timing` → `recordings/room_timings/`

## Measured door-path fail (not route progress)

From `Level1ExitOverworld` with `LEVEL2_DOOR_HOPS` + `require_level2_screen`
(2/2, Clean): reaches **0x5C** then **dies** (mode 17) at ~(16,93) with
**0 filled hearts**. Health drain along the way: 3→2 on 0x48/58, 2→1 on
0x59/5A, 1→0 on 0x5B/5C. Slowest hops: `0x5B→0x5C` (~718f), `0x59→0x5A`
(~659f). Maze hop 0x5C→0x5D never starts. Timing artifact:
`recordings/room_timings/level2_door_path_probe_timing.json`.

**Next experiment:** farm hearts on 0x4A to ≥2 filled before entering the
0x5A corridor; only then wire `LEVEL2_5C_MAZE_WAYPOINTS` into the controller.

## Acceptance (not yet)

- [ ] 2/2 isolated walk 0x37→0x3C without health poke
- [x] 2/2 enter Level 2 assisted (`level==2`, mode 5, room **0x7d**) — Survival only
- [x] Assisted recon 0x7d→0x6d (5× Rope `0x28`, left door bit, enter 0x6c) — Survival only
- [ ] 2/2 enter Level 2 Clean (`level==2`, room-ready 0x7d)
- [x] Isolated pure clear 0x6d (2/2 Clean) + 0x6c west key (2/2 Clean) — checkpoint only (`rr-fcc`)
- [x] Isolated pure east key 0x7e (2/2 Clean) + `Level2EastKey.state` (`rr-il4`)
- [ ] Natural-entry L2 interior (OW → 0x6d / key) without mid-run state load
- [x] Post-west-key live graph + boomerang inventory map (`rr-ep8` **PARTIAL**)
- [x] Entry-east **0x7e** live (5× Rope + key `0x19`) via diamond-nav
- [x] Entry-east **0x7e** isolated pure 2/2 Clean (keys≥1, 0 live ropes)
- [x] Open **0x6e RIGHT → 0x6f** key door (`rr-c6b`; diamond band≈113; 6× Gel)
- [x] Isolated pure 0x6f gels + compass inventory 2/2 (`rr-bcd`; `Level2Compass`)
- [x] **0x6f bomb N → 0x5f** live (stand 120,101) + **0x5f LEFT key → 0x5e** Goriya (`rr-ebe` advance)
- [x] Isolated pure 0x6f bomb N → 0x5f 2/2 Clean (`rr-lzk`; `Level2BombNorthController` / `Level2_5F`)
- [x] Isolated pure 0x5e 5× Goriya clear 2/2 Clean (`rr-etl`; `ROOM_5E_SPEC` / `Level2_5E`)
- [x] Puzzle catalog bomb/key/diamond + `level2_puzzles.py` data (`rr-3pz`) — not Clean STATUS
- [x] Open past 0x5e/0x5f: **0x4e / 0x4f / 0x3e** live + graph edges (`rr-cjf`; `l2_cjf_expand.json`)
- [ ] 0x4f boom clear + Magical Boomerang pure 2/2 (`ADDR_MAGIC_BOOMERANG`; `rr-bsq` / `rr-ebe`)
- [x] Dodongo path recon (`rr-a1t` **PARTIAL**) — boss not reached; residual past 0x4f / 0x3e
- [x] Live path past **0x4f/0x3e** → Dodongo + TF (`rr-n5i` geometry; `rr-5dk` compose)
- [x] Assisted Boom→TF `0x02` **2/2** + `Level2Complete` (`l2_complete_assisted.json`)
- [ ] Clean residual TF / damage harden (`rr-4oz`); pure 2/2 optional later
- [ ] Natural-entry / Entrance→TF continuous compose (deferred; not tip-blocking)

## Room 0x5f further exits (`rr-cjf`, 2026-08-06)

Evidence: `recordings/l2_cjf_expand.json`
from checkpoint **`Level2_5E`** (post-Goriya, Survival + inventory poke).

| Claim | Live result | Evidence |
|-------|-------------|----------|
| 0x5f walk-RIGHT / key-RIGHT | **SEALED** (post Goriya clear, gel clear, diamond bands) | door_tests ok=False |
| 0x5f bomb RIGHT | **FAIL** | bomb_tests |
| **0x5f bomb UP @(120,101)** | **LIVE → 0x4f** boom candidate (item `0x1e`) | `l2_cjf_expand.json` |
| **0x5e free UP** | **LIVE → 0x4e** (5× Rope `0x28` + key `0x19`) | same |
| 0x5e walk-RIGHT | **blocked** max_x≈160 | right_probes |
| **0x5e bomb RIGHT @(176,141)** | **LIVE → 0x5f** | same |
| **0x4e RIGHT** | **LIVE → 0x4f** | same |
| **0x4e UP** | **LIVE → 0x3e** | same |
| Gel clear opens R/U? | **No** (rr-fvt stands) | `l2_5f_policy.json` |
| Kill-all Goriya opens 0x5f R/U doors? | **No** walk doors; bomb-UP works independently | post-clear expand |

**Boom path (two LIVE routes):**

1. Shortcut: `0x5f` bomb N @(120,101) → **0x4f**
2. Alt: `0x5e` free UP → `0x4e` RIGHT → **0x4f**

Graph/constants: `door_graph` (`L2_BOOM_CANDIDATE` / `L2_ROPES_NORTH` /
`L2_NORTH_OF_4E`), `level2_puzzles.BOMB_WALL_5F_NORTH` / `BOMB_WALL_5E_EAST`.

**Residual for `rr-bsq` / `rr-ebe`:** clear 0x4f (obj types + collect
`ADDR_MAGIC_BOOMERANG`); map 0x3e / Dodongo branch.

## Room 0x5f policy (`rr-fvt`, 2026-08-06)

Evidence from checkpoint **`Level2_5F`**
(idle 360–600f @ 60f ticks → optional gel clear → door push R/U/L/D).

| Claim | Live result | Evidence |
|-------|-------------|----------|
| Spawn delay | **5× Gel `0x15` already present at entry** (TYPE-only hp=0); stable through idle | `l2_5f_policy.json` timeline f=0…600 |
| Empty transit? | **No** — gels always live; earlier “empty on hop” missed TYPE-only | same; supersedes empty note |
| Walkthrough Red Goriya on 0x5f | **Wrong room** — Goriya `0x06` is **0x5e** (key-LEFT) | `l2_5f_explore.json` / 0x5e peak |
| `cur_opened_doors` on bomb entry | **DOWN only = 4** (stable idle) | entry `doors.raw=4` |
| Clear opens doors? | **No** — after 5-gel clear doors stay **4** | clear `doors 4→4` |
| Map `0x17` | RoomItemId **0x17** from entry; **`ADDR_MAP` 0→2** (L2 bit) after clear+wander | `map_gained=True` |
| Kill-gate for exits? | **No** — LEFT key works without clear; DOWN hole open | door_tests |
| LEFT key → 0x5e | **LIVE** (keys−1); after use door bits often **L\|D=6** on 0x5f | door_tests LEFT ok |
| walk-RIGHT / walk-UP after clear+map | **Still sealed** (doors stay 4 or 6) | door_tests RIGHT/UP ok=False |
| bomb-UP after clear / post-Goriya | **LIVE → 0x4f** (rr-cjf; not tested in rr-fvt) | `l2_cjf_expand.json` |
| `ROOM_5F_SPEC`? | **Not encoded** — clear is **not** a door-open gate; optional map pure later | docs-first / avoid clash with rr-etl 0x5e |

**Policy label:** `gels_present_key_left_no_kill_gate`.

- **Transit:** key-LEFT → 0x5e / DOWN hole → 0x6f without needing combat.
- **Map:** clear gels + mid-room wander for `ADDR_MAP` L2 bit (inventory only).
- **Boom (rr-cjf):** bomb-UP @(120,101) → **0x4f** (not key-RIGHT).

**Walkthrough:** “right → 5 Ropes + key” = **0x7e**. “3 Ropes → key RIGHT →
compass” = **0x6e → 0x6f**. “optional bomb N” = **0x6f → 0x5f**. Carry **≥2
keys** into 0x6e (west + east) so one remains after the key door; another key
for **0x5f LEFT**.

#### Magical Boomerang residual (not Clean, not collected)

| Item | Value |
|------|--------|
| Inventory stop | `ADDR_MAGIC_BOOMERANG (0x0675) != 0` (wooden `0x0674`) |
| RoomItemId correlate | **`0x1E` on live 0x4f** |
| Enemy correlate | 0x4f objs `0x05`/`0x55` (ID residual); Goriya `0x06` on 0x5e |
| Path live so far | … → **0x5f bomb N → 0x4f**; alt **0x5e UP → 0x4e RIGHT → 0x4f** |
| Pure controller | no boom pure yet — `rr-bsq` / `rr-ebe` |

Next (**rr-bsq** / **rr-ebe**): clear **0x4f** + `ADDR_MAGIC_BOOMERANG` pure 2/2.
