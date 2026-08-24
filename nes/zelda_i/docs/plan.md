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
| `rr-4d53.6` | L3 exit → L4 TF `0x08` | **in progress** — continuous clear `0x32`; `0x60` island causeway blocked v19 |
| `rr-4d53.7` | L4 exit → L5 TF `0x10` (attach `.5` pin) | blocked on `.6` |
| `rr-4d53.4` | one session power-on → L5 TF | blocked on `.2` `.3` `.6` `.7` |

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

`--through level4-stepladder` is wired (`make_stepladder_controller(clear_first=False)`,
stop `ADDR_LADDER` / `level4_stepladder_success`) but **live blocked** after
v1–v19. Push+stairs enter `0x60` mode-9; the island/pedestal is not reachable
from the continuous leftover without emulator-state BFS. Do not close `.6`.

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

PNGs: `recordings/l4_stepladder_continuous_v{1,2,4,8,11,12,13,14,15,16,17,18,19}_final.png`
(v3/v5 exit `0x32`).

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
5. **Center water is unseeded (unknown=free).** Island `(136,141)` is
   passable. Occupancy still has **no** cardinal path because west-brick
   `x=49–90,y=53–161` (v18 extended past the fake y=158 gap) and south-water
   `x=80–175,y=158–180` enclose the island from both aisles. Isolated BFS
   does not 1px-walk those free interior cells; it jumps, then restores
   goal state.
6. **Not a two-button clip.** Isolated dirs are `UP/DOWN/LEFT/RIGHT` one at
   a time. `RIGHT,UP,RIGHT,UP` tokens are sequential holds, not `nes_action("RIGHT","UP")`.

Live v17–v19 (one new geometry each) did not cross the moat. `ADDR_LADDER`
stays 0. Inventory on miss: TF=`0x07` keys=5 bombs=15 ladder=0;
deaths/state/progression/capacity 0.

Next worker: do **not** call `_bfs_60_to_ladder` on the spine. Remaining
one-frame policies that are none of v1–v19: RIGHT+DOWN or LEFT+UP at
`(84,189)`; a Keese-timed knock from the west aisle (RNG, not a geometry
clip). Push from `(80,109)` is still OK.

Exact verified predecessor:

```bash
UV_CACHE_DIR=/tmp/retro_rl_uv_cache QT_QPA_PLATFORM=offscreen \
  uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-clear32 --no-video --trials 1 \
  --tag l4_clear32_continuous_v1
# blocked hop (do not expect ok=true):
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-stepladder --no-video --trials 1 \
  --tag l4_stepladder_continuous_v19
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
   L5, then `.4` one-session L5 TF. L6–L9 stay out of this pass.
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
