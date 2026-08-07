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

Assisted recon from `Level2Entrance` (Survival, 2026-08-06):
`scripts/probe_level2_rooms.py --infinite-life` →
`recordings/l2_recon_probe.json` + `l2_recon_*.png`. **Not Clean STATUS.**

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
| Magical Boomerang room | walkthrough | **not live** — path residual (see below) |
| Magical Boomerang RAM | Data Crystal | `ADDR_BOOMERANG=0x0674`, `ADDR_MAGIC_BOOMERANG=0x0675` |
| Dodongo needs bombs | walkthrough | not yet |
| Triforce bit 0x02 | walkthrough | not yet |

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
0x6c LEFT/UP/DOWN sealed            0x7d LEFT sealed
0x6f further R/U/D residual
```

**Trap — diamond solids (0x7d / 0x6e east):** naive `y≈141` RIGHT hits solid
near **x≈128–176**. Helper: `nav_common.diamond_east_phase` (also
`ROOM_7E_SPEC.entry` waypoints):

1. Free mid (east alcove: **LEFT** first; do not y-only spin)
2. Open **y-band** → RIGHT to wall (**x≥200**)
   - 0x7d: band **y≈157** (or 149)
   - 0x6e: band **y≈113** — prefer **WEST** entry via 0x6d (south from 0x7e can stick ~y=181)
3. At wall: **S2** LEFT×6 → vertical to y≈141 → RIGHT×10 (repeat)
4. **Pure push** RIGHT on y=141 — **no LEFT during push** (re-enters solid)

| Room | Role | Enemies (live) | RoomItemId | Notes |
|------|------|----------------|------------|-------|
| **0x7e** | East of entry (east key) | **5× Rope `0x28`** | **`0x19` key** | **isolated pure 2/2** (`Level2EastKey`); LEFT→0x7d, UP→0x6e |
| **0x6e** | N of 0x7e / E of 0x6d | **3× Rope `0x28`** | `0x03` | **key-RIGHT → 0x6f** (consumes 1 key) |
| **0x6f** | Compass branch | **6× Gel `0x15`** TYPE-only (hp=0 alive) | **`0x16` compass** | LEFT door on enter; further exits residual |

Evidence: `recordings/l2_east_open.json`, `l2_6e_right_ok.json`,
`l2_east_map.json`, `level2_clear7e_isolated.json` (**Clean pure 0x7e**).

**Walkthrough:** “right → 5 Ropes + key” = **0x7e**. “3 Ropes → key RIGHT →
compass” = **0x6e → 0x6f**. Carry **≥2 keys** into 0x6e (west + east) so one
remains after the key door.

#### Magical Boomerang residual (not Clean, not collected)

| Item | Value |
|------|--------|
| Inventory stop | `ADDR_MAGIC_BOOMERANG (0x0675) != 0` (wooden `0x0674`) |
| RoomItemId correlate | `0x1D` (L1 wooden boom / dungeon_ids) |
| Enemy correlate | Blue Goriya — type **unverified on L2** (L1 Goriya `0x06`) |
| Path live so far | … → **0x7e** → **0x6e** → **0x6f**; blocked past 0x6f |
| Compass on 0x6f | RoomItemId `0x16` live; inventory pickup not yet reliable pure |
| Pure controller | **none** — boom room not reached |

Next: open **0x6f** exits → Blue Goriya boom room; pure 2/2 on
`ADDR_MAGIC_BOOMERANG`.

### Dodongo / triforce 0x02 (`rr-a1t` / `rr-n5i` PARTIAL, 2026-08-06)

Goal: bomb Dodongo (2 mouths) → Heart Container → east TF room →
`ADDR_TRIFORCE & 0x02`. **Not reached live.** Boss room / HC / TF IDs unknown.

| Claim | Live |
|-------|------|
| Reachable graph | **0x6c↔0x6d↔0x6e↔0x6f↔0x7e↔0x7d** |
| Entry **0x7d** east | **LIVE → 0x7e** (diamond-nav / pure 7e) |
| **0x7e** east key | 5× Rope + `0x19`; UP→0x6e |
| **0x6e** RIGHT | **LIVE → 0x6f** key door (`rr-c6b`) |
| **0x6f** further | **residual** — shared boom/Dodongo blocker |
| Dodongo object type | **unverified** (not entered) |
| `triforce & 0x02` | **not set** |

Bombs inventory notes (for future boss policy):

- `ADDR_BOMBS = 0x0658`; B places when bombs selected as B-item.
- `Level2EastCleared` / entry-fresh fixtures often start with **bombs=4** and
  selected pos already bombs — useful for bomb-wall recon only.
- Natural bomb source for Dodongo (walkthrough: Red Goriya room drop) is
  **beyond** 0x6f residual.

Evidence: `recordings/l2_east_open.json`, `l2_6e_right_ok.json`,
`l2_dodongo_path_recon.json`, `l2_boomerang_partial.json`. Shared residual:
**past 0x6f** (`rr-ebe` / `rr-n5i`).

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
# Interior recon 0x7d → 0x6d Ropes + doors (Survival):
uv run python zelda_i/scripts/probe_level2_rooms.py --infinite-life --tag l2_recon
# Isolated pure key branch (Clean from Level2Entrance / Level2RopesCleared):
uv run python nes/zelda_i/scripts/run_level2_clear6d.py --trials 2
uv run python nes/zelda_i/scripts/run_level2_clear6c.py --trials 2
uv run python nes/zelda_i/scripts/run_level2_clear6c.py --from-entrance --trials 2
uv run python nes/zelda_i/scripts/run_level2_clear7e.py --trials 2
uv run python nes/zelda_i/scripts/run_level2_clear7e.py --save-state --trials 2
# Assisted path recon past west key (not Clean; boom residual past 0x6f):
uv run python nes/zelda_i/scripts/probe_level2_boomerang_path.py --infinite-life
# Diamond-east: nav_common.diamond_east_phase / ROOM_7E_SPEC.entry.
# 0x6e RIGHT: WEST entry + key + band≈113 → 0x6f.
```

- `level2_overworld.PostTriforceSettleController`
- `level2_overworld.OverworldToLevel2Controller` (default stop 0x4A;
  `door_path=True` + maze; `require_dungeon=True` → room-ready 0x7d)
- `dungeon.GenericDungeonRoomController` + `ROOM_6D_SPEC` / `ROOM_6C_SPEC` /
  `ROOM_7E_SPEC`
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
- [ ] 0x6f compass inventory pure + exits → boom branch
- [ ] Magical Boomerang isolated pure 2/2 (`ADDR_MAGIC_BOOMERANG`)
- [x] Dodongo path recon (`rr-a1t` **PARTIAL**) — boss not reached; residual past 0x6f
- [ ] Live path past **0x6f** → Dodongo (`rr-n5i` / `rr-ebe`)
- [ ] Dodongo bomb-mouth pure + Heart + `triforce & 0x02` isolated 2/2
- [ ] Natural-entry Moon complete (`rr-5dk`, blocked on pure TF)
