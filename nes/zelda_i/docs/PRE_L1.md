# Pre-L1 loadout

## Current sitting leftover (2026-09-24)

The no-beam L9 Patras clear with ordinary sword input under guarded
last-heart refill. From `LastHeart30_level9_natural_silver_arrows`, the
resumed route reached the credits in 35,288 frames, with one state load,
TF `0xFF`, 14 containers, zero deaths, and six refills. Room 0x61 cost 3
hearts; final Patra 0x52 cost 2. This is development evidence, not a Clean or
power-on result.

`lastheart_poweron34` had zero state loads and reached L8 0x1F with TF
`0x7F`, two keys, five bombs, and 8.48/13 hearts. It timed out in that room
after 16,150 frames; the beam is absent below full hearts. Its real
predecessor pin is `LastHeart34_level8_magic_key_stairs`. Next: fix that
Darknut clear from this pin, replay from the preceding L8 stage, then repeat
power-on. The assisted result must keep its 7 target / 8 safety refills
visible; zero is the Clean goal.

## Hidden rupees pay the ring (2026-09-24, rr-t49c)

The 73→250 ring write is gone: the chain opens the hidden rupee caves on
and beside its walk (`SECRET_RUPEE_CAVES`, `overworld/locations.py`) with
`make_secret_rupee_controller(screen, hops)`. ROM payouts (cave table at
file `$18610`): `$21`=30R, `$22`=100R, `$23`=10R.

| stage | screen | opens | pays | note |
|---|---|---|---|---|
| `rupees_2d` | `0x2D` | bomb rock (80,80) from (80,85) UP | 30 | a bomb after `0x2C`'s heart; NE walk starts here |
| `ne_100` | `0x0F` | centre arch | 100 | unchanged |
| `rupees_28` | `0x28` | tree (208,160) from (208,133) DOWN | 30 | on the white walk; candle selected at `0x0C` |
| `rupees_48` | `0x48` | tree (208,96) from (188,93) RIGHT | 30 | unchanged |
| `rupees_5b` | `0x5B` | tree (32,160) from (32,133) DOWN | 10 | row 5 east of `0x58` |
| `rupees_6b` | `0x6B` | tree (128,160) from (128,133) DOWN | 100 | down the x=48 gap |
| `rupees_56` | `0x56` | tree (160,160) from (160,133) DOWN | 10 | on the ring road; wallet hits the 255 cap |
| `rupees_62` | `0x62` | tree (128,96) from (148,93) LEFT | 100 | after the ring; exit is west of the bush column, back north via `0x52/0x53/0x54` |

The wallet caps at 255: 0x62 taken before the ring counted 0. Segmented
from pins (2026-09-24) the chain bought the ring with 255R (5 left) and
reached the `0x37` mouth with 107R, 6/6 hearts, and no rupee write.
`--through pre-l1` no longer writes `$066D`; `bomb_topup` hunts the coast
when short.

## Blue Ring main-spine revision (2026-09-23, rr-scum)

The default gathering chain now visits the Armos shop on `0x34` after the
`0x47` heart and buys the Blue Ring before entering L1. The route goes via
`0x48/0x58` and the western forest to `0x54/0x44/0x34`, then reverses to
`0x58`, heals at the `0x39` pond, and reaches the `0x37` mouth. The shop's
left-column Armos exposes stairs at `(64,125)`. Link walks past them while
pushing UP, then turns DOWN onto the exposed stairs. The middle pedestal is
`(120,141)` after contact from y=165. Purchase proof is `$0662: 0→1` and
the 250R debit; the route does not poke the Ring byte.

The standalone `chain:exit_47` replay from its predecessor pin was green:
8,987 frames through `ring`, `exit_ring`, `ring_return`, pond, and `walk_37`;
`GatherChain_ring` has Ring 1, 0R, and 6 containers. The power-on
`blue_ring_gather1` run was 1/1 green to the L1 mouth in 41,670 frames,
with `set_state=0`, Ring 1, White Sword, 6 containers, and 1R. Its Survival
inventory report records the only new ring-budget write, rupees 73→250
before the shop; `progression_writes=0`, `capacity_writes=0`. This is an
assisted purchase and leaves a natural 250R farm open. The earlier ringless
L1/L5 and later checkpoints are invalid for the main spine; resume rejects
ringless saves after the `ring` stage. The 2026-09-22 chain and L1 timings
below are historical, not current route results.

`blue_ring_l1_1` then continued power-on through L1 in 53,801 frames,
`set_state=0`, with Ring 1 at `BlueRing_enter_level1` and Ring 1 at the
Triforce leave (`TF=0x01`, 7 containers). The run remained Survival: its
inventory writes were rupees 73→250 at `ring` and key 0→1 at
`backtrack44`; no progression or capacity write. Next work is a natural
250R farm and a new L5+ baseline from this ring-bearing power-on prefix.
`blue_ring_l1_verify_20260923` repeated the power-on route in 53,801 frames:
ring purchase at 36,303f, L1 entry at 41,778f, and Triforce at 53,801f.
The final glance passes on fanfare `0x36` `(128,149)`, `TF=0x01`, Ring 1,
7 full containers, 8 bombs, 0 keys, and 8R; deaths, state loads, progression
writes, and capacity writes are all zero. The same two inventory count writes
remain, so this is a second Survival verification, not a natural farm.

Zelda Dungeon, The Gathering, first quest only. Grid is `screen = (row << 4) | col`. Start is `0x77`.

This prefix is how Level 1 gets more than three heart containers and a wooden sword. The Clean gate stays the 18909f wooden clear. Do not STATUS from a pin. Heart assist stays off. Do not poke bombs, Food, keys, rupees, the candle, or `$066F`.

Composer row is `overworld/gathering.py`: sword, coast walk, top-up, then the buy. `--through pre-l1` runs that row and forces the health assist off. A short arrival hunts the coast in `bomb_topup` before the buy.

## Leftover (2026-09-22, second sitting)

Bead `rr-ecmp` (gather with no refill) is in progress. Nothing is committed yet. These are dev tapes, not a STATUS result.

Measured now, all power-on, one run each:

| Run | Result |
|-----|--------|
| M5 `run_level1_complete --natural-entry` | green, 18909f, TF `0x01`. The oracle did not move. |
| Default spine (script default: L1 health assist on) | green to the L1 Triforce, 57292f. Gather refill: 8 writes, 25 half-hearts restored. L1: 16 writes, 5 restored. Pokes: `$066D` 11→20, keys 0→1. Tape `s1_final_default10`. |
| Rung 2 (`--no-infinite-life`) | red at L1 `clear23_key`, 53521f. It was green before this sitting (46193f). Tape `s1_final_nolife7`. |
| Rung 3 (`--gather-engage-hearts 0`) | red at `heart_7b`, a death on 0x7B. Before this sitting it died one stage earlier, in `walk_7c`. Tape `s1_final_r3b`. |

What changed, and how it was measured:

- **Pond fairy.** The ROM room table (`$697E`/`$69FE`) puts object list `0x2F` on 0x39 and 0x43 only. `UpdatePondFairy` fills every heart once Link's Y is exactly `$AD` and X is `$70..$80`. `heart_farm.PondFairyController` does that with 0 writes: `OW_39`, 1→3 hearts in 184 frames. The chain now detours after `heart_7b`: stages `walk_pond` → `pond_39` → `walk_2c`. 0x4B is two rooms (the x=192..223 corridor is sealed from the west half), and 0x39 opens only south onto 0x49. Lanes are in `POND_WALK_HOPS` and `POND_RETURN_HOPS`.
- **Overworld lattice.** `dungeon.tilemap.ow_walkable_nodes` applies the ROM collision (`GetCollidingTileMoving`): feet row y+11, both feet tiles, threshold `$034A` (`$89` on the overworld, `$78` in a dungeon), and the overworld-only `WalkableTiles` rewrite. Nodes are x≡0 and y≡5 (mod 8). That is the measured turn grid: off the grid, a perpendicular press first slides Link to the nearest line. `walk.physics.lattice_route` is a fewest-turns Dijkstra.
- **Geo hop rung** (`path.HOP_RUNG_GEO`, 95). It routes around rock only when the straight align-then-push is blocked. Fixed the 0x1E letter jam (2528 frames of `unstick_wait`). It found that **0x79's y=165 dead-ends at x=192**; y=133 and 141 scroll east, measured by walking them. When a hop's align row has no reachable edge node, it falls back to the nearest one.
- **Shot escape** (`common.shot_escape`, used by `_spit_escape`). Each held input is simulated on the lattice against each shot's straight flight, using the ROM hit box (9 px on both axes). Candidates are ranked by: never hit, then latest first hit, then distance off the shot's line. The old `perpendicular` crossed the *bearing*; walled on that side, it walked with a diagonal shot (6 of 7 Zora hits in trace `z1`). Rocks are included.
- **Scoop gate.** A drop in a live body's pad, or with a body in Link's pad, is not a pickup. 10 of 33 0x7B hits were this.
- **Parry.** It uses `blade_lands` (near end) and gets 90 frames per slot. 0x7D held `evade_parry` for 3047 frames when the gather walkers had evade on.
- **Bomb wall.** No B press inside a swing. A press with no bomb leaving `$0658` re-arms after 12 frames. The last 8 px to the cell are a push, not a swing: a leever in the lane pinned Link 3 px off the cell for 3827 frames. Frames where the cave text halts Link (ObjState `$40`) are not charged to `interior_budget`.
- **Dungeon entry replan** (`route_entry._lattice_step`). It runs only after a measured stall, ahead of the one-pixel `measured_walker`. It fixed L1 0x43→0x33 (the centre block at x 112..143).
- **White stage.** It refills at 2 whole hearts in the spine (`GATHER_WHITE_ENGAGE_HEARTS`), as the standalone segment always did. The 0x0A ring is not closed: the bottom band is rock at x 80..103, so the top band past the Lynel is the only way to the cave.

Coast eval, pre-l1 walk only, sixteen RNG offsets (idle after boot): **12/16 reach the shop, against 1/12 before**. Hits per run went from 5.8 to 3.8. Zora hits went from 29 to 8, rocks from 17 to 10, and leevers are 15. A single tape cannot score a combat change, because one change reshuffles everything downstream. Use the offset eval (`scratch/eval_coast_offsets.py`, `scratch/eval_chain_offsets.py`), not a tape.

Tried and reverted, with no measured gain or an M5 break:
- Evade on the gather walkers: 1/6 against 2/6.
- A close-peel hold in the hunt: 11/16 against 12/16.
- A Lynel-avoiding route on 0x0A: the ring is not closed.
- A lattice detour in the dungeon engage: it fixes 0x23 from a pin, but every gate tried (sim, stand-still, 150 frames without closing) rerouted M5's own 0x23 and broke M5 at `backtrack44`.

Next, in order:
1. L1 0x23 under rung 2. A red Goriya walks the inner ring's bottom row while Link chases on the top row. The only vertical passages are x=64 and x=176. Any fix must leave M5's 0x23 frame-exact.
2. Rung 3 dies on 0x7B at the bomb cell among red leevers on the last heart. The re-pinned `PreL1BombLeave` leaves at 2/3 hearts. The walk back to 0x7B costs about 1.
3. The White Sword Lynel still needs its 2-heart floor.

## Geometry split (2026-09-22)

The stops after the bombs are one table in `overworld/gather_segments.py`, walked from `BFS_*` pins, not from the coast. `overworld/gather_run.py` clamps a `BFS_*` heart byte of `0x2F` (15 hearts in 3 containers) to `0x22` before the walk. That is not a new container. `LastHeartAssist` then refills to the owned maximum only once `whole_hearts <= 1`. It is not on `--through pre-l1`. A `$0670` refill there hides the 10-kill 5-rupee. `ok` is the stop predicate. A saved pose on a failed walk is not a leave.

| Stop | Result |
|------|--------|
| Letter `0x0E` | Taken. Old man `0x72` at `(120, 128)`, letter below. Line up on x=120 and walk UP. `$0666` 0→1, 684 frames, `GatherLetterLeave`. The stop is the letter byte. Cave mode alone read 0. |
| Candle `0x0C` | Bought. Candle byte 0→1 and the wallet 80→20, so the 60 was seen leaving. 1555 frames, `GatherCandleLeave`. The 80 is the entry stamp. |
| Heart `0x7B` | Taken. Take-any cave: old man `0x6B` at `(120, 128)`, potion left, heart right. Touch `(152, 149)`, containers 3→4, mode 11, bombs 7, 641 frames. `GatherHeartL8Leave`. |
| Potion `0x64` | Play mode `(128, 77)` from `BFS_65` LEFT at y=141, 534 frames. A push did not open the cave. The old `GatherPotionLeave` marked that closed cave `ok` and was deleted. A shut opening fails. |
| 100 rupees `0x0F` | Taken. `NE_HOPS` had `0x1D→0x1E→0x1F` as LEFT; those columns go east. `0x1D` RIGHT needs y=141: at y=149 (inside tolerance 5 of the old 146) `(136, 149)` is a wall, and Link idled 1713 frames. Centre-arch secret cave: moblin `0x7C` at `(120, 128)`, text freezes Link ~225 frames, UP at x=112 misses the rupee, x=120 takes it. Wallet 7→107 counted in from `$067D`, 2324 frames, `GatherNE100Leave`. The stop is the whole 100 from cave entry. A floor rupee (6→7) graded the old stop green. |
| White sword `0x0A` | Walk reaches the pedestal. `0x29` and `0x2A` are walled on top. Row 2 west to `0x27`, UP at x=144 (x 112..160 cross), row 1 east to `0x1A` on y=141, then `white_sword.py`'s measured climb. `0x28` is staggered bushes. Enter at y=117 and walk the `WAYPOINTS` corners. Cave at about 4800 frames. The Old Man gates on five containers, and every pin here has three, so the segment fails on cave entry with `white_cave_reached_containers_3_of_5` and writes no leave. The `0x0A` Lynel takes 0x21/0x7E to 0 in one hit, so this segment refills at 2 hearts. |
| Heart `0x2C` | Taken. The doorway is on the rock's bottom face: a bomb from `(144, 165)` facing UP opens it, for x 136 to 152. The right face opened nothing, and the old `(176, 122)` bomb was placed facing the wrong way. The walk goes down the west column to y=165, then east. Same take-any cave as `0x7B`: touch `(152, 149)`, containers 3→4, mode 11, bombs 7, 1536 frames. `GatherHeartM3Leave`. |
| Burn row `0x48/0x47/0x46` | Taken in the chain. The old walk stood on the bush coordinate. The secrets are tile objects `0x64` at `(208, 96)`, `(176, 176)`, `(144, 176)`. |
| Blue ring `0x34` | `0x37` does not scroll west. Every align on that edge died at `(32, 133)`. |

The table's hop fixes and waypoints are 2026-09-22 dev-pin walks under the gather assist, not `--through pre-l1`. None of these pins is a natural-entry route, and none of them moves the 18909 frame gate.

2026-09-22, `heart_m3` from `GatherHeartM3Enter` (`BFS_2C`, bombs stamped 8, heart byte clamped `0x2F` to `0x22`). A recon sweep poked Link's position and facing and bombed each cell. Facing UP from y=165 or 173, x 136 to 152 opened the rock; the right face at x 176 and 184, facing LEFT, did not. The segment is not poked: 70 frames of approach, 894 walking the bomb row with swings, 1 bomb, then 31 frames of `cave_settle` and 228 to the heart. `ok` true, containers 4, health `0x32`. Assist on (last heart), 1 damage, 0 writes. The 894-frame bomb row is the cost to cut. Not a natural-entry route, and not the 18909 frame gate.

2026-09-22, two `heart_l8` boots from `GatherHeartL8Enter` (bombs stamped 8, heart byte clamped `0x2F` to `0x22`; that clamp is not a container). Boot 1 graded the gate: the bomb-mouth lock at `(144, 93)` ran 31 frames of `cave_settle`, then `0x6B` appeared at `(120, 128)` beside two fires `0x40` at `(72, 128)` and `(168, 128)`. `0x6B` is the take-any old man, not the heart. With the goal still on `(116, 128)` Link stood at `(115, 141)` after 146 misses, 782 frames, containers 3. y=141 is the old man's row. Boot 2 moved only the goal to the right-hand item, `(152, 149)`: 641 frames, `ok` true, containers 4, health `0x32`, `GatherHeartL8Leave` written. Assist on (last heart), 1 damage, 0 writes. Not a natural-entry route and not the 18909 frame gate.

## Gathering chain (2026-09-22)

`python -m zelda_i.overworld.gather_segments pin` runs power-on `--through pre-l1` (assist off) and saves `PreL1BombLeave` in the `0x6F` cave: bombs 4, rupees 0, health `0x20`, 3 containers. Its one RAM write is the spine's disclosed `$066D` to 20. `... chain` then runs `chain_stages()` in one env from that pin, to the Level 1 mouth screen. `chain:<stage>` resumes after a green stage's `GatherChain_<stage>` pose.

One full pass is green: 21609 frames, 22 stages, `ok` true, 0 deaths. The chain refills at 2 hearts (9 health writes, 25 damage). There are no inventory, progression, or container writes. The candle goes on B through the pause menu, not a poke. Leave: `0x37` `(240, 141)`, 6 containers, White Sword, blue candle, letter, 72 rupees, 2 bombs.

| Stage | Leave | Frames |
|-------|-------|--------|
| `walk_7c` | `0x7C` `(240, 141)`, coast reversed | 1143 |
| `heart_7b` | 4 containers, bombs 3 | 874 |
| `walk_2c` | `0x2C` `(0, 85)` via `0x6B`, `0x5B`, `0x4B`, `0x3B`, `0x2B` | 2106 |
| `heart_2c` | 5 containers, bombs 2 | 620 |
| `ne_100` | rupees 0→100 | 2201 |
| `letter` | `$0666` 1, via `0x1F`, `0x1E` | 1265 |
| `candle` | candle 1, rupees 100→41 | 1685 |
| `white` | sword 2, `0x0A` cave | 4597 |
| `back_1a`, `walk_48` | detour return, then row 1 west, `0x27` down, `0x28` east, down x=120 | 653 + 2231 |
| `rupees_48` | burn `(208, 96)` from `(188, 93)` RIGHT, keeper `0x7B`, rupees 42→72 | 745 |
| `heart_47` | burn `(176, 176)` from `(176, 157)` DOWN, take-any, 6 containers | 746 |
| `walk_37` | `0x48`, `0x38`, `0x37` | 741 |

Cave exits run 103 to 322 frames each. What the chain found that the pins hid:

- `0x7D`'s north edge is solid. The coast climbs column B. `0x7B`, `0x6B`, `0x5B`, and `0x0C` have `WAYPOINTS`.
- The candle shop's middle pedestal is a 100-rupee key. A buy walking y=149 with 101 rupees bought it. The lateral row is now y=165.
- A door-cave exit leaves Link on the mouth tile, and a sideways align slid him back in (11790 frames on `0x0C`). Door exits step 16 px clear. Burn caves exit by stairs (mode 10). DOWN held there walks Link back down, so those exits idle and clear nothing.
- The bomb cell took 6 px. A bomb from y≈159 left the `0x2C` rock shut. It takes 2 px now, and the `0x2C` approach fixes y first.
- A timing change in one stage reshuffles every later stage. Re-run the whole chain before quoting it.

Burn secrets (from the disassembly, `Z_04.asm` `UpdateTree`): each is a hidden tile object, type `0x64` for a tree or `0x63` for a rock wall, at a ROM-fixed spot. It shows in `snap.objects` (slot 11) the moment the screen loads, so read it before guessing a bush. The flame must walk 16 px before it stands. The tree reveals when the standing flame's timer drops below 2. Flush with the tree (`(192, 93)` RIGHT) never reveals it. Room flag bit `$80` is "secret found". `0x46`'s tree is `(144, 176)`: `(144, 157)` DOWN opens the shield shop, and the chain does not burn it yet.

Not natural entry, and not the 18909 frame gate. The chain's health refill is not Clean. Still open from the order: arrows `0x4A` (needs 80, chain has 72), potion, blue ring (250), shield `0x46` (90).

## In the default spine (2026-09-22)

Gathering is now the spine's default prefix: `run_survival_spine.py` with no `--through` runs power-on, the pre-l1 bombs (assist off), `gather_stages()` (the same 22 stages, one env, no pin), then L1 from the `0x37` door (`OverworldToLevel1Controller(phase=APPROACH_DOOR)`), first_key through clear53, and the Survival TF suffix. `--through gather` stops on `0x37`. `--no-gather` is the old wooden-sword prefix, and the ROM tier's legacy tests pin it.

Assist ladder, each rung one live power-on run (deterministic, so one run is the measurement):

| Rung | Gather refill | L1 health | Result |
|------|---------------|-----------|--------|
| 1 | 2 hearts, 9 writes | unlimited, 12 writes | L1 TF, 47089f |
| 2 (default) | last-heart, 6 writes | **off** | L1 TF, 46193f, 7 containers, 0 deaths |
| 3 | off | — | red: `walk_7c` death on `0x7F`, 298f after the shop |

Other writes on rung 2: `$066D` 10→20 before `bomb_topup`, and keys 0→1 before `backtrack44`. Rung 3 is red because Link leaves the `0x6F` cave on his last heart: the rung-2 refill fires on the chain's first frame (10666). The next rung is a heal before `exit_6f`, not a new refill threshold.

What the White Sword run broke in L1, and the fixes:

- A latched assist. Booting under the L1 assist latched 3 containers at power-on and rewrote the 6 gathered hearts back down to 3. The gathered boot now runs with no assist.
- `0x33` key. It is dropped where the carrier Stalfos dies (the beam killed it at `(128, 148)`), and the engine leaves the drop's position in slot 1. The fixed `(96, 173)` nudge sat there for 6000f. The key walk now reads slot 1 and seeds its own walker from `$6530`, because the fight walker has no geometry. The heart-wait fail-closed now triggers at 2 or fewer whole hearts, not "not full". With 3 containers that is the same rule.
- `0x45` entry. 0x44 can end on the north band `(168, 109)`. The south-aisle `y_first` route hits the statue below, and the stall skip then pushed RIGHT at `(208, 109)` for 9000f. `Room45SurvivalController` takes `(208, 109)`→`(208, 141)` from the north band.

Not natural entry and not Clean: the gather refill and the two count pokes remain. The 18909f oracle is untouched.

## Where the bomb walk is

`overworld/shop_p7.py` `SHOP_P7_HOPS`. Screens:

`0x77 → 0x78 → 0x79 → 0x7A → 0x7B → 0x7C → 0x7D → 0x7E → 0x7F → 0x6F`

Map-1.png names that sequence. It does not pick the row. Measured lanes, from `scratch/probe_coast_lane.py` tag `l1`:

| Screen | East scroll |
|--------|-------------|
| `0x79` | y=165 only, the south beach. Hop target `0x7A` uses `SCREEN_79_BEACH_Y`. |
| `0x7A` | y=133 and y=141. Hop target `0x7B` uses `SCREEN_7A_EAST_BAND`. |
| `0x7B`, `0x7C` | Every swept row. Those hops use `SCREEN_ANY_ROW_BAND`. An align_y here is a shuffle through leevers. |
| `0x7D` | Every row for its own exit. The hop that leaves `0x7D`, target `0x7E`, uses `SCREEN_7E_EAST_BAND` 137 to 145, so the dead row is fixed before the scroll. |
| `0x7E` | 117, 125, and 141 and below. y=133 does not scroll. `align_y` with a tolerance of 5 accepts that dead row. |
| `0x7F` | No east exit. The hop to `0x6F` is UP. |

`0x4A`, `0x5C`, `0x5E`, `0x58`, `0x59`, `0x68`, `0x67`, `0x6A`, `0x6B`, `0x6C`, and Lost Hills `0x1B` are in `SHOP_P7_NOT_ON_WALK`. The inland join through `0x68` was measured and rejected. Do not merge it back.

Price is 20 rupees. While the wallet is under that, the coast hunt stays open out to the destination cap, so a pass can arrive with more than 20. Cave mouth on `0x6F` is about `(48, 77)`. The walk may stop on the screen. The errand stops at `ADDR_BOMBS >= 1`. A short wallet ends the walk. `overworld/topup.py` fights north `0x5F` at x=122, then west `0x6E` inside y 109 to 189, and comes back. Still short, it keeps hunting the nearest coast screen `RoomHistory` has dropped — after those two neighbours that is `0x7C` — and comes back again. `0x7B` and `0x7D` were crossed, not cleared, so they are not that hunt. Inland `0x68` is not that hunt. The buy does not run until the wallet can pay. Both neighbours were measured as six-body waves. The back hop arrives on the reverse edge, so the hunt holds until the wave is done or the screen budget hits. Do not occupancy-search `0x6E`. It is a bush maze.

`RoomHistory` at `$621` keeps six screens. Turning around into the last six respawns nothing. `laps` stays 0. A screen this stage already hunted is not hunted again. Entering it only evicts another slot.

## Latest tape

`rr-ttyu.3` is open. Default arm is not accepted.

2026-09-20, `--through pre-l1 --no-video --trials 1 --rollout`, tag `pre_l1_7e_band1`. One trial. `set_state` 0. Assist null.

| | `pre_l1_topup_live` | `pre_l1_7e_band1` |
|--|---------------------|-------------------|
| Leftover | Dead on `0x7E` at `(40, 131)`, mode 17, 19 rupees | Cave `0x6F` `(120, 149)`, mode 11, bombs 4, 20 rupees |
| `0x7D` | 1236f, 3 hits | 424f, 0 hits, then `band_down` |
| Buy | Did not run | Top-up 1 frame, buy 483f, bombs 4 |

Glance on the green trial: triforce `0x00`, keys 0, health `0x21` so 2/3 and the partial is not full. The buy frame still showed 20 rupees while bombs read 4. Rollout was on. Do not call the default reactive walk green off this tape. M5 was not re-measured. The shop hops are not the Clean mouth.

Next measure on this bead: flag-off against flag-on damage on `0x7B`, `0x7C`, and shot type `0x55`, with no new stall. Keep the `0x7E` band on the hop that leaves `0x7D`.

2026-09-21, default arm, `--through pre-l1 --no-video --trials 1`, tag `pre_l1_shortfall2`. No `--rollout`. Assist null. `set_state` 0. No inventory poke.

The walk reaches the shop. The buy does not run. Glance at the death: overworld, mode 17, room `0x6F`, `(57, 165)`, sword, bombs 0, 10 rupees, health `0x20`, triforce `0x00`, keys 0, 3 containers. `bomb_topup` failed at 149 frames. `bomb_buy` did not start.

`0x7D` is 699 frames and 0 hits, hearts 1.488 in and out. The previous default tape died on that screen. The hop that leaves `0x7D` still carries `SCREEN_7E_EAST_BAND`. A spit in the water south of that band used to be read as an east shot, and the duck walked UP off the band onto the octorok row. The dodge now stays in the band when the shot is outside it. A shot already on the band still leaves the row.

What kills the errand is the wallet and the hearts together. The walk banks 10 rupees (16 dropped, 6 left). `0x7B` takes 3 Zora hits and heals 1. `0x7C` takes 3 Zora hits plus a leever and heals 1. `0x7E` takes one rock. `0x7F` takes one blue octorok. Arrival on `0x6F` is 0.484 hearts. The top-up's first hop is UP to `0x5F` at x=122. Link gets onto that column, then `spit_duck` walks west at y=173 for 56 frames into a `0x55` that started on the south edge, and the one hit is fatal. Neighbours `0x5F` and `0x6E` never start. A `0x23` fairy on `0x7B` exists only while health is already `0x22` and is gone before that screen's 1.5 hearts of damage. It is not a heal for the shop.

Not STATUS. The 18909 frame Level 1 Triforce gate was not re-measured. `rr-ttyu.3` stays open. The flagged A/B is still the next measure on this bead, and it still wants a buy that did not write `$066D`.

2026-09-21, same default walk, tag `pre_l1_rupee_poke1`. `--through pre-l1 --no-video --trials 1`. No `--rollout`. Assist null. `set_state` 0. End frame 10624.

The walk matches `pre_l1_shortfall2`. 9212 frames. 10 rupees banked, 16 dropped, 6 left. Arrival on `0x6F` is 0.484 hearts. The six left are on `0x7A`: one 5-rupee, state `0x0F`, and one 1-rupee, state `0x18`. That screen banked 0. `0x7D` is 699 frames and 0 hits.

Before `bomb_topup` the spine writes `$066D` from 10 to 20. `inventory_assist` records that one write. `select_bomb` is false. No bomb, key, Food, candle, or `$066F` write. `bomb_topup` returns on frame 1, note `path_stop`. `bomb_buy` is 463 frames and 0 hits. From the south edge it walks UP, `door_up_first` for 93 frames and `door_up_first_slash` for 8, then aligns and enters the cave.

Glance at the stop: cave mode 11, room `0x6F`, `(120, 149)`, sword, bombs 4, rupees 20, health `0x20`, triforce `0x00`, keys 0, 3 containers. B slot reads 1, bombs. The poke did not write that byte. `$066D` is still 20 on the grant frame. The price leaving the wallet was not observed. The stage stops on `ADDR_BOMBS >= 1`.

Not STATUS. The 18909 frame gate was not re-measured. `rr-ttyu.3` stays open.

## Order after the bombs

Catalog names are `overworld/locations.py`.

| Stop | Screen | How it opens | Walk |
|------|--------|--------------|------|
| Bombs | `0x6F` `shop_p7` | Open, 20 rupees | Coast table above |
| Heart | `0x7B` `heart_l8` | Bomb the north wall | From `0x6F`, down 1, left 4. Come from the east. Do not enter `0x79` inland. |
| Heart | `0x2C` `heart_m3` | Bomb the lower-right of the center rock | After the northeast rupee stops. Five containers. |
| 100 rupees | `0x0F` | Open centre arch on `0x0F` (secret-to-everybody cave) | `0x2C` to `0x2D`, up to `0x1D`, east to `0x1E` and `0x1F`, up to `0x0F` |
| Letter | `0x0E` | Open | `0x1F` to `0x1E`, stairs north. No pickup controller yet. |
| Candle | `0x0C` `shop_m1` | Open, 60 rupees | `0x0E` to `0x1E` to `0x1D` to `0x0D` to `0x0C`. Not the bomb shop at `0x66`. |
| White Sword | `0x0A` | Open, needs 5 containers | Not through Lost Hills. From `0x0C` go `0x1C` to `0x2C`, west along row 2 to `0x27`, up to `0x17`, east along row 1 to `0x1A`, north at x=208 to `0x0A`. Blue Lynel on the top band. |
| 30 rupees | `0x48` | Burn the top-right bush | Already on the Level 1 overworld path |
| Heart | `0x47` `heart_h5` | Burn the fifth bush from the right | `0x48` LEFT at y=141. Six containers. |
| Shield | `0x46` `shop_g5` | Burn the corner bush | `0x47` LEFT. Magical Shield, 90 rupees. Bait is on the same counter. Buy it later. |
| Arrows | `0x4A` | Open, 80 rupees | Existing Level 2 prefix. Buy controller exists. The farm is still the gap. |
| Blue Ring | `0x34` | Armos, top-middle | From `0x51`, right 3, up 2. 250 rupees. Read the slot order on a pin before a buy. |

Then the Level 1 mouth at `0x37`.

Lost Hills `0x1B` wraps on all four sides. The fourth UP from inside it is Level 5. `0x5C` is a maze with its own waypoint list, not a free RIGHT. `0x67` is a dead end going north from the start. Entering `0x67` from the east is a different bomb wall and is fine.

Three catalog openings disagree with the walkthrough. Pin them before a hop: `0x3D` rupees, `0x56` gamble, `0x51` gamble.

## Rules the coast already paid for

- `ButtonsPressed` is an edge. Hunt presses A for one frame, then idles. A held A does not swing again. A turn and a swing do not share a frame. Hold the direction until `$0098` matches, then press A. Do not gate the sword shot the same way. A wrong-way beam still crosses the screen.
- The blade has a near end. Inside `HUNT_BLADE_MIN_FWD`, peel. `pad <= MIN_DODGE_BODY` means a step cannot clear the body. It is not a cue to swing.
- A leever with `ObjState` 0 is under the sand. `combat.dormant_body` keeps it out of the wave. States 1 and 2 are the rise and stay threats.
- Tektites are not a beam-stand kind. Leevers still are. On `0x79` and `0x7A`, chase the tektites until they are gone or the destination cap hits, about 2400 frames. Retiring them at 600 left the wave up.
- Align the short axis only when the cross-track gap is greater than `lane_tol` 6.
- Scoop a floor drop that is inside pickup radius and outside `MIN_DODGE_BODY` of a live body. A drop in a body's pad loses to the chase. `scoop_rupees` is on for this walk. `need_rupees` is 0, so there is no restock lap. While rupees are under 20 the coast screen cap is the destination budget, not 600. At 20 it drops back.
- Never fight a Zora. Dodge and leave. The rung is `OverworldPathController._spit_duck`, above the evader and the hunt. Facing byte `0x03` is still absent from `dungeon/threat.py` `_FACING_AXIS`, so a firing-line test does not see that shot. The shot type `0x55` is not blocked by the small shield.
- A dodge is a walk. A direction that does not move Link gets written off per direction and 16-pixel cell. Do not poison the whole screen.
- The sword shot in `zelda_i/beam.py` fires at full hearts only. One wooden chip drops the partial below the gate and the shot is gone until a heart comes back. Offer `take_beam` from the hop above stall-escape. A 600-frame escape commit used to hold the frame for the whole shot.
- Standing on a rupee is not instant. If a body is inside `MIN_DODGE_BODY`, give the frame back.
- `@dataclass` defaults are baked into `__init__`. Ablation flags have to wrap `__init__` or pass the value in. Assigning the class attribute after the class exists measures the unablated hunt.
- `$066F` low nibble is whole hearts minus one. `0x22` is 3/3. Read `ram.whole_hearts`. A wooden chip lands in `$0670` and does not move that nibble.
- Overworld waves are one-shot while the screen stays in the six-slot history. A kill streak is not the same as rupees in the wallet. Read dropped against left.

## Wiring

`gathering.pre_l1_stages` is sword, `bomb_walk`, `bomb_topup`, `bomb_buy`. `spine/survival.py` writes `$066D` to `SHOP_P7_PRICE` before `bomb_topup` when the count is lower, with `allow_pokes` still false. The top-up stage is always in the list and returns on the first frame when the wallet already has the price. Short of the price it does not finish. The walk's stop is the shop screen. The top-up's stop is the money, on that screen. The buy's stop is the bombs.

`take_beam`, then scoop, sit above `_stall_escape` on the hop ladder. The reactive evader still runs first.

Re-measure Level 1 after this prefix is actually green on the default arm. Until then the oracle stays 18909f.
