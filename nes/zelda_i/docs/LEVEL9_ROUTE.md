# Level 9 — Death Mountain (route notes)

Route notes for this dungeon. The live sitting is the gathering prefix in [PRE_L1.md](PRE_L1.md). Do not treat this file as the current plan, and do not write STATUS from it.

## 2026-09-24: natural bomb budget after Level 8

`natl8_3` leaves Level 8 on overworld `0x6D` with TF `0xFF`, 0 bombs,
38 rupees, and 14 heart containers. A resumed ROM run from that measured
predecessor (`post_l8_bomb_check2.json`) bought four bombs at `0x4A` for
20 rupees and reached Level 9 room `0x76` with three bombs and no inventory
write. The return from the shop enters `0x59` from the north at `(112,61)`;
it must descend to the west passage before walking left. The old policy
assumed an east arrival at `y=141` and timed out for 12,000 frames.

A longer resumed run (`l9_bomb_suffix_credits.json`) collected Silver Arrows
in room `0x10`, then failed the Patra join with zero bombs. The four-bomb
pack was exhausted by the Level 9 entrance and prefix. Room `0x16` lists
a bomb room item, but a room `0x16` recon fixture did not yield it while
Patra was alive. The route now schedules two 20-rupee packs at `0x4A`; this second
purchase still needs a wallet of at least 40 rupees at the Level 8 leave
and a continuous power-on verification. Neither resumed run is a Clean or
continuous credits claim.

> **Probe names below are historical provenance labels, not paths.** The
> one-shot probe CLIs under `scratch/` were deleted 2026-09-07 once their
> beads closed; the measurements they produced live on in the constants and
> route tables this document describes. Git history is the restore path.



**Status:** backward endgame recon is live; the natural Level 9 route is still
unbuilt. Spectacle Rock is overworld `0x05`, the settled entrance is room
`0x76`, the final Patra room is `0x52`, Ganon is `0x42`, and Zelda is `0x32`.
The preserved endgame states are explicitly composed, route-ineligible
fixtures—not Clean or Survival route evidence.

**2026-09-10 — Clean dodge (`rr-npv.5`, fixture-live).** Ganon/Patra no longer
`idle(n)` through attack cooldown: dodge the nearest fireball/eye (manhattan
≤14, horizontal, flip at x=56/200), face, then fire. Brown Ganon commits to
the silver-arrow axis (do not dodge off the column). Door holds in
`natural_path` / `path.py` bind the column from leftover+hold_dir
(`door_band_goal`), not a frozen spawn x. Dest is RAM.

Fixture-live, no assist, no pokes, deaths 0, `route_eligible=false`:

| lab | pin | dest RAM | frames |
|-----|-----|----------|--------|
| Patra north | `Level9FinalPatraReconFixture` | doors bit 0x08, body gone | 3716 |
| Ganon silver-arrow | `Level9BeforeGanonReconFixture` | `$0672 != 0` | 1136 |
| credits | same, through wait_credits | mode 0x13 | 3167 e2e |

`Level9EntranceReconFixture` leftover: play `0x76` `(120,205)` TF `0xFF`.
Entry→Patra from that pin is a later hop (prefix + join). Not spine-green.

**2026-09-05 — the L8 predecessor is now measured.** `--through level8` is
power-on spine-green 2/2 (`rr-6o7.3`); the shard fanfare settles Link on OW
`0x6D` `(96,93)` mode 5, TF `0xFF`, MK 1, bombs 14, hc 10, B = bombs. That is
`level9.dungeon.MEASURED_POST_L8_HANDOFF`. `continue_level9_spine` is wired
into `spine/survival.py` carrying it, and `L9_THROUGH` is in `SPINE_THROUGH`,
so `--through level9-entry` is reachable — but every natural L9 chapter is
still a fail-closed `NaturalRouteUnavailableController`. First real build:
`Level9PostL8OverworldController` for the `0x6D → 0x5D → … → 0x78 →
LEVEL9_ROCK_HOPS → 0x05` walk (the `0x6D → 0x78` connector is unmapped).

**2026-09-06 — first real power-on run of the L9 chain, past a spine-dispatch
bug (`rr-mzxn`).** A genuine `--through level9-credits` power-on attempt had
never reached past L6 before: `survival.py`'s L6/L7/L8 dispatch only remapped
`through` for the *immediate* next level, so an L9 target either stranded
Link inside the L6 dungeon or hit a predecessor's own `raise ValueError`
guard. Fixed (one remap helper, all three call sites; now the `handoff`
column of the `SPINE_LEVELS` row table, `SpineLevel.target`) —
see `rr-mzxn`. That also fixed a real L6 bug the fixture-only path never
exercised: `INLAND29_SPEC`'s generic occupancy grid was too narrow for its
own LEFT+UP clip, stranding the BFS walker at the west wall; restored the
dedicated `Level6Inland29Controller`.

With those fixed, power-on `--through level9-credits` reaches TF `0xFF` and
`Level9PostL8OverworldController` for the first time. Two more never-live-
tested bugs turned up and got fixed in that controller (screens `0x59` and
`0x58` — both a bare y-threshold re-checked every frame, ping-ponging
forever instead of converging; see commit `5587fa1b`). Power-on now crosses
`0x6D → 0x5D → 0x5C → 0x5B → 0x5A → 0x59 → 0x58 → 0x48 → 0x38` in one run.

**2026-09-06 (later) — `0x38` and `0x27` fixed; the full reverse OW walk is
now power-on-verified end to end (0x6D → 0x05).** Both hops shared a bug
deeper than a re-checked threshold: the "realign to N" branch pressed the
*opposite* direction of the hop's final UP commit whenever position dipped
under N, and that final UP commit routinely overshoots N by 30-80px in one
continuous motion — so the realign branch kept firing on every later frame
and walked Link back, undoing real progress instead of just ping-ponging
near a boundary (probe evidence: `0x38` reached y=105 from y=133 before
being walked back to y=134; `0x27` reached y=105 from y=131, same pattern).
Fixed with a one-time latch per hop (`_cleared_38_bridge`, `_cleared_27_gap`),
matching the `_cleared_58_south_wall` pattern: once the first sub-threshold
read lands, never re-test position again, just keep committing to the final
direction. See `probe_l9_38_bridge.py` and commit `e62f2154`. The full walk
(`0x6D` → ... → `0x05`, Spectacle Rock) now completes in 6225 frames.

**Important correction, same session:** the paragraph above (2026-09-05)
claiming "live recon reached 0x05 ... and bombed the left rock to settle in
room 0x76" does not hold up under a genuine natural walk. Driving
`Level9SpectacleRockBombController` for real from `Level8OWLeaveLive` gets
Link to the coded stand position, consumes exactly one bomb, but a
before/after `save_rgb_png` pixel diff of the left rock-pile region shows
**zero visual change** — same silhouette, same tiles. A second manual trial
bombing the right pile (still fully intact in both screenshots) also did
nothing. This isn't combat interference (health/damage telemetry never
changes) — Link is standing at the base of ordinary, non-bombable Death
Mountain terrain. So either `SCREEN_LEVEL9_ROCK_HYP` (OW `0x05`) is not
actually the Spectacle Rock screen, or the real bombable formation is a
visually distinct small boulder pair elsewhere on it that hasn't been
found yet — not a coordinate-tuning problem on the current hypothesis.
`Level9EntranceReconFixture` (the `0x76` fixture chapters build on) was
evidently composed some other way, not from a genuine bomb-and-walk.
See `probe_l9_05_entry_sweep.py` for the reproducible sweep/evidence and
`rr-sz8.5` notes. Next step needs ROM overworld-map data for the real warp
tile, or a visual survey of `0x05` and its neighbors for the actual
formation — not another guess at the current one's coordinates.

**2026-09-06 (later still) — Spectacle Rock bomb geometry fixed via ROM
archaeology, not a guess.** The aldonunez `zelda1-disassembly`
`UpdateObject_JumpTable` shows `ObjType` `0x63`/`0x67` dispatch to
`UpdateRockWall`, the overworld's real bombable-rock handler (checks a
bomb's midpoint against the tile object's own midpoint via
`CheckTileObjWeaponCollision`, replaces the tile on hit). Dumping live RAM
objects on OW `0x05` after the real post-L8 walk
(`probe_l9_05_objects.py`) finds exactly one such object: type `0x63` at
`(80, 160)` — the two large piles are just background tiles; this one
small object *is* the secret, anchored 8px off from where the controller
stood (`x=72`). `probe_l9_05_rockwall_bomb.py` confirms a real pixel-diff
hole opens bombing at `(88,178)`/facing UP; `probe_l9_05_rockwall_enter.py`
sweeps entry columns from a post-blast savestate and finds `x=80` is the
only one that crosses `level==9`, settling at the documented `(120,205)`
room `0x76`. Fixed `Level9SpectacleRockBombController`'s two `x=72`
targets → `x=80` (commit `38694ecc`). `probe_l9_05_rock_bomb.py` now runs
the real controller class end to end to `DONE`/`success=True`.

**Same session — power-on reaches genuinely into the L9 dungeon for the
first time, then dies at `level9_stairs_05`.** With the rock-bomb fix,
`--through level9-credits` clears `level9_post_l8_overworld` and
`level9_spectacle_rock_bomb` power-on, then `level9_natural_silver_arrows`
clears hops `0x76→0x66→0x65→0x55→cellar 0x60→0x14→0x15→0x16→0x06→0x05`
(hops 0–8) before dying at hop 9 (`level9_stairs_05`, the block-push room
after the `0x06` bomb-west). This is a genuine deterministic deadlock, not
RNG variance — it reproduces byte-identical at `(98,149)` whether given
the original 4000f budget or a bumped 12000f (commit `fe431b07`,
bumped anyway since the room being hard was a reasonable prior; it just
wasn't the actual cause). Root-caused in `repro_l9_05_stairs_from_pin.py`:
the block-approach logic's exact-equality x check plus a
`_push_attempts > 250` escape valve permanently loops between
`recenter_y` and `clear_wizzrobe` once `link_x` incidentally lands on
exactly 96 (an engine corner-slide side effect of pressing DOWN, not a
deliberate x-correction), never once reaching `push_block_up`. Prototyped
two fixes against a *real* power-on savestate
(`L9Room05EntryReal`, pinned via `pin_l9_room05_entry.py` — runs the
actual `run_survival_spine()` pipeline and aborts the instant Link
settles in room `0x05`, so further iteration is seconds, not ~5 minutes):
room `0x05` actually has **5** live Wizzrobes (`0x23`/`0x24`), not the 1–2
assumed by the original policy. An always-fight variant
(`experiment_l9_05_push_v3.py`, mirroring `Level9Stairs55Controller`'s
proven Lanmola-clear-first pattern) landed **zero kills** in 12000f — the
scripted chase-and-slash can't reliably hit these teleporting enemies as
written. An ignore-them variant (`experiment_l9_05_push_v4.py`, correct
x-alignment + one-time y-latch, relying on `UnlimitedHealthAssist`) shows
Link repeatedly knocked back between `y≈93` and `y≈125`, never reaching
the `y=165` stand-off row needed before the push — a genuinely difficult
live encounter, not just a math bug. **Not fixed this session** —
`Level9Stairs05Controller.policy()` is unchanged (still the buggy loop);
whoever picks this up next has the real pin and three prototype scripts
to iterate from directly. See `rr-sz8.6` notes.

**2026-09-06 (fork continuation) — stairs_05 fixed, chain now reaches
`0x61` and `0x10` power-on; new statue-diamond blocker in `0x10`.**
`Level9Stairs05Controller`'s Wizzrobe combat had the real bug: it mashed A
on a period without ever checking the sword hitbox (0 kills observed).
Replaced with `should_swing_at`-gated engage combat plus a
backstep-when-stuck fallback (ported from `level6.wizzrobe`), full-clear
before touching the block, and a one-time y-recenter latch instead of the
re-checked `y<165` threshold. Verified against `L9Room05EntryReal`: all 5
Wizzrobes dead, block pushed, settles cellar `0x70` (192,93) (commit
`f3fb5eaa`).

The same bug class (distance-gated mash-A, no hitbox check) was also in
`Level9Stairs61Controller`'s Patra fight — fixed by reusing
`patra.patra_action` (the policy already proven for the final Patra,
room `0x52`) instead of reinventing combat (commit `4cb50f4a`). Two
full power-on `--through level9-credits` runs then died byte-identically
at frame 332954 in room `0x61`, stuck push-looping at `(32,93)` with all
8 eyes still alive — yet the `L9Room61EntryReal` pin kept succeeding in
isolation. Root cause: that pin was captured 60 "stable" frames after the
room loaded, but the real hop handoff runs the new hop's `policy()` on
the *exact* frame the predecessor's `arrived()` check fires — before
Patra has spawned (body frame 1, eyes +2f, the same race this doc already
documents for room `0x52`'s `WAIT_PATRA` phase). Reading "no live
eyes/body" on that literal first frame looked like an already-cleared
room, so the controller jumped straight to the block push while Patra
was fully alive. A corrected pin (`stable==1`, not 60) reproduced the
exact live failure; fixed by requiring Patra observed at least once
before trusting a cleared reading (commit `56912285`). Also added a
90-frame no-progress stuck-escape (step toward room center) for genuine
`0x61` geometry stalls (commit `0a8574f0`).

With both fixes, power-on `--through level9-credits` now clears `0x61`
correctly (full 8-eye kill, push, stairs, cellar `0x75`) and lands in
room `0x10`.

### Room `0x10` holds no item — the Silver Arrows are in cellar `0x4F`

**SOLVED 2026-09-06.** Two earlier sittings assumed the Silver Arrows were
a floor item at the centre of `0x10`'s statue grid and burned themselves on
maze-threading it. There is no floor item in `0x10` at all. ROM decode
(`dump_l9_rom_rooms`, self-validating against 41 in-repo live
anchors) plus live probing agree:

| Fact | Source |
|---|---|
| `0x10` item byte = `0x03` (none) | ROM table4 `& 0x1F`; live `room_item_id` on every entry |
| `0x10` secret = `5` `block_reveals_stairs` | ROM table5 `& 7`; same gating as the proven `0x05` / `0x30` push-stairs hops |
| L9's cellar array is **8** entries `60 70 72 75 67 77 00 4F` | level info block PRG `0x19C10`; truncating it at 6 is what hid the arrows |
| cellar `0x4F` item byte = `0x09` **Silver Arrow** | ROM table4; the only `0x09` in either quest-1 block |
| cellar `0x4F` both stair mouths → `0x10` | ROM table0/table1 (for a cellar row these are destinations, not door bitfields) |

Quest-1 underworld room attrs live at PRG `0x18700` (L1–6) and `0x18A00`
(L7–9), 6 × 128 bytes: `item = t4 & 0x1F`, `secret = t5 & 7`,
`N = t0>>5 & 7`, `S = t0>>2 & 7`, `W = t1>>5 & 7`, `E = t1>>2 & 7`. The
same decode independently reproduces every dungeon's signature cellar item
(L1 Bow `0x7F`, L4 Ladder `0x60`, L5 Recorder `0x04`, L6 Rod `0x75`, L7 Red
Candle `0x4A`, L8 Magic Key `0x0F` + Book `0x6F`, L9 Red Ring `0x00`).

So `0x10` was always the right destination and the wrong entity. The 17th
prefix hop now: clear the 5 Wizzrobes (3 × `0x2B` are invulnerable traps and
never clear — gate on `room_all_dead`, not a visible count), push the `0x68`
at `(192,144)` **east** from `(176,141)`, which exposes a staircase at cell
`(208,96)`; reach it via the west lane `x=32` → north band `y=93` → east;
in cellar `0x4F` drop to the floor `y=189`, climb the `x=176` shaft into the
upper chamber, walk `LEFT` to the arrow at `(128,141)`; reverse out through
the west exit shaft `x=48`, landing back in `0x10`.

Live from the `L9Room10EntryReal` pin: success in 6,659 controller frames,
`ADDR_ARROWS` 1 → 2, settled `0x10` `(96,157)`, 0 memory writes. Live from
power-on: `level9_natural_silver_arrows` **passes** in 24,759 frames.

Two traps this room set, both worth remembering:

- The `0x68` is a genuine push block. An earlier blind 120-frame hold
  concluded it "wanders on its own"; it does not — with zero input for 240
  frames it does not move at all. That test simply never got Link to a push
  face, because the room was still full of live Wizzrobes shoving him.
- In cellar `0x4F` the exit shaft and the item chamber are **both** above
  the floor corridor, so height alone cannot tell them apart. Splitting on
  height first parks Link at the top of the exit shaft holding `RIGHT` into
  a wall forever. Split on `x` first.

### Patra join: three fixes, and the sword gap it exposed

**2026-09-07 (rr-sz8.7).** With the arrows collected, `--through
level9-credits` failed at `level9_natural_patra_join`, timing out at its
full 24,000 frames without leaving room `0x10`. Iterated from a real
power-on pin (`pin_l9_post_arrows` → `L9PostArrowsReal`, room
`0x10` `(96,157)`, arrows 2, TF `0xff`) so each attempt costs ~25s instead
of a ~6 minute run. Three distinct bugs, all live-confirmed:

1. **`SOUTH_10` pressed into the statue band.** The phase aligned `x` to the
   mouth column 120 and held `DOWN`. That only ever worked from the `0x10`
   *entry* leftover, where Link already stands in the doorway at
   `(120,189)`. Coming back out of cellar `0x4F` he lands at `(96,157)`,
   one band above — and `0x10`'s statue band at y~176 blocks every column
   except the west lane `x=32`, so `DOWN` moved him zero pixels for 24,000
   frames. Fixed by routing through the same west lane the prefix hop
   already proves; the routing is now one shared `prefix.room10_lane_step`
   instead of two copies.
2. **`CLEAR_20` chased a Wizzrobe back through the bomb hole.** `0x20`'s
   north wall is the hole this join just came through, so
   `chase_sword_step` could follow an enemy up into `0x10` with a `0x20`
   phase still latched, making every `0x20` waypoint meaningless. Added a
   re-entry guard that re-derives the phase from the room Link is *in*, a
   no-fight band along `0x20`'s north edge, and a skip-the-fight fallback
   after two bounces.
3. **`CLEAR_03` burned 10,529 of the 24,000-frame budget.** Two causes.
   Its exit gate counted live non-`0x2B` objects, but room `0x03`'s block is
   **clear-gated**: with any enemy alive, standing south of the `0x68` and
   holding `UP` for 240 frames moves it *zero* pixels; once `room_all_dead`
   the same 240 frames slide it `144 → 128`. (Capping the phase instead just
   stranded `STAIRS_03` pressing an immovable block forever — verified from
   `L9Stairs03StallReal`.) Gate on `room_all_dead`, as room `0x10`'s
   Wizzrobes already do. Second, the naive chase walks only along the
   dominant axis, so a wall between Link and a wandering flyer pinned him at
   `(144,165)` for 7,000 frames chasing one `0x13` that the same policy
   kills in ~1,100 frames from an unblocked start; reusing the `stairs_61`
   90-frame no-progress escape cut the phase to 6,060 frames.

Live from the `L9PostArrowsReal` pin, the join now walks all ten hops —
`0x10 → 0x20 → 75 → 0x61 → 0x51 → 0x41 → 0x31 → 0x30 → 67 → 0x04 → 0x03 →
77 → 0x52` — and reaches the Patra room in **19,401 frames** (~4,600 of
margin), against 24,000 spent going nowhere before.

**Remaining blocker — the sword.** At `0x52` every `level9_live_patra_stop`
term is met except one:

| term | live value |
|---|---|
| `sword >= MAGICAL_SWORD (3)` | **`1` (wooden)** ← only miss |
| `triforce == 0xff`, `bow`, `arrows == 2`, `screen == 0x52` | met |
| `final_patra_live`, 8 × `0x25` eyes, north door shut | met (eyes spawn by +40f) |

The eye spawn is *not* a race — `WAIT_PATRA`'s 120-frame window is ample.
The natural power-on run simply never acquires a sword upgrade: it reaches
Patra with the wooden sword and **10** heart containers. The Magical Sword
needs 12 containers, so it is out of reach without two more; the White Sword
needs 5 and is not.

### The White Sword detour — cave `0x0A`, reached from row 0

**SOLVED 2026-09-07.** `overworld/white_sword.py` now takes the White Sword
on the way into Level 9, and the ending contracts ask for `WHITE_SWORD` (2)
instead of `MAGICAL_SWORD` (3). The boss policies are hitbox-driven — they
swing until the boss dies — so the weaker sword costs frames, not outcomes.

`route/item_gate_hops.py` had the *screen* right and the *approach* wrong,
and its approach is what had kept the cave unvisited. Both of its candidate
routes are now falsified live, band-exhaustively:

| candidate | result |
|---|---|
| west off the L5 door `0x0B` | **sealed** — its north half is mountain, and the walkable block has no west transition at any band |
| Lost Hills `0x1B` | **wraps to itself** in all four directions, so a BFS from it sees no edges at all |
| `0x09` (west neighbour of `0x0A`) | a sealed pocket — no east exit, no south exit |

The way in is row 0, which the Level 9 approach already walks:

```text
0x05 →E y=141→ 0x06 →E y=141→ 0x07 →S x=64→ 0x17
     →E y=141→ 0x18 →E y=141→ 0x19 →E y=141→ 0x1A
     →N from (208,157)→ 0x0A
```

`0x1A` sits beside the Lost Hills, but its north is plain geometry, not a
maze count: **x=208 is the only column that crosses**, and it always does.
An earlier probe "got through on the 8th UP" only because Link had wandered
onto that column. (`0x1A` also has a second cave mouth at x=96.)

Inside `0x0A` a lake fills the middle: the only north-south lane is the
`x=208` sand corridor, and the cave mouth is far west on the top band
(`y≈85`, `x≈34`). The cave puts Link back out at the top-**left**, so the
return has to re-cross the top band eastward before descending — heading
straight DOWN walks into the lake's west shore and stalls at `(32,189)`.
The pedestal is at `x=120`; walking UP there takes `ADDR_SWORD` 1 → 2.

ROM corroborates the screen independently
(`dump_ow_rom_screens`, self-validated by reproducing all six
live dungeon-entrance anchors): `0x0A` carries a **unique** overworld cave
id 18, sitting between the wooden sword cave `0x77` (id 16) and the Magical
Sword grave `0x21` (id 19, an `anchors.py` anchor). Overworld cave ids live
at PRG `0x18480` as `(byte >> 2) & 0x3F`; the same decode groups both known
bomb shops (`0x4A`, `0x6F`) under one id and both known candle shops
(`0x5E`, `0x66`) under another.

Two traps worth keeping:

- **Don't tile-dump the overworld.** `dump_room_tiles` reads
  `colliding_tile`, a dungeon-only field, so on an overworld screen it
  returns a uniform value for every cell — `0x0A` dumped as one solid block
  of `0x24`. Screenshots and real movement are the tools here.
- **Don't re-check a travel alignment during the move it enables.** `y=157`
  on `0x1A` is the lane that carries Link *east* to the opening, not where
  the climb starts. Re-deriving it every frame made the align and the climb
  fight each other and Link oscillated at `(208,150..155)` for 20,000
  frames. Latch once the column is reached, then only hold UP. (Same class
  as the `0x58` note in `level9/overworld.py`.)

Live from the real power-on pin `OW_05_Row0Real`:
`WhiteSwordDetourController` runs `0x05` → cave → `0x05` in **5,287
frames**, `ADDR_SWORD` 1 → 2, TF still `0xff`, 0 memory writes. It is staged
in `level9_entry_chapter` between `level9_post_l8_overworld` (which already
ends on `0x05`) and `level9_spectacle_rock_bomb`, and refuses up front below
5 heart containers — the Old Man gates on **containers**, which
`UnlimitedHealthAssist` never raises.

### Power-on `--through level9-credits` passes

**2026-09-07.** `ok=True`, end frame 354,346, 0 deaths, 0 memory writes of
any kind. Zelda I completed from power-on on the natural route.

| stage | frames | budget |
|---|---|---|
| `level9_post_l8_overworld` | 4,184 | 12,000 |
| `level9_white_sword` | 5,287 | 20,000 |
| `level9_spectacle_rock_bomb` | 709 | 4,000 |
| `level9_natural_silver_arrows` | 20,059 | 44,000 |
| `level9_natural_patra_join` | 17,933 | 24,000 |
| `level9_final_patra` | 2,189 | 6,000 |
| `level9_ganon` | 4,200 | 7,000 |
| `level9_wait_credits` | 1,492 | 12,000 |

Both boss budgets keep good headroom on the White Sword, confirming the
`WHITE_SWORD` relaxation costs frames rather than outcomes.

Getting there took six more fixes, and **every one of them was the same
bug**: a policy that drives one axis (or holds one button) with no way out
when that direction is blocked. Worth stating as a rule for this codebase —
*any* unconditional directional hold needs either a progress check or a cap:

| where | symptom | fix |
|---|---|---|
| `east_14` waypoints | Like Like bump left Link off the `y=93` lane; LEFT into stone at `(176,101)` for 3,373f | no-progress escape on the other axis |
| `stairs_05` entry | `bomb_west_06` now lands Link *in* the east hole at `(208,141)` with Wizzrobes camped in that wall → chased back out, `unexpected_play_0x06` | step off the door row first; never fight the east wall on it |
| `stairs_05` `recenter_y` | bare DOWN hold from `(144,125)`, 10,597f | sideways escape, plus give up after 1,200f |
| join waypoints | `NAV_BLOCK_20` spent 22,623f of 24,000 on waypoint 0 | shared `_wp_step` escape |
| `CLEAR_20` | false clear — Wizzrobes read hp 0 mid-teleport, so `NAV_BLOCK_20` then ran with the room still live | gate on `room_all_dead` |
| `BOMB_04` | sub-controller failed and returned its idle `"failed"` action forever — 13,197f doing nothing | recover (re-align to the door row, push) then fail loudly |

Two of these are worth generalising beyond their site:

- **A failed sub-controller must not be handed straight back.** A
  `BombWallController` in `FAILED` returns `idle "failed"` every frame; a
  parent that forwards it stalls silently until its own cap. Check the
  child's terminal state, recover if you can, and surface its note if you
  cannot.
- **A no-progress escape must be keyed on progress, not stillness.** The
  first cut of `_wp_step` tested "position unchanged", which never fired:
  the surviving Wizzrobes kept nudging Link a pixel at a time while he was
  wedged. Track the best distance to the target instead.

## Natural-spine seam (Wave A, implementation only)

The new natural-route seam lives in `level9/{dungeon,natural_path,hops,spine}.py`
and exposes exactly four cumulative public targets:

```text
level9-entry → level9-silver-arrows → level9-patra → level9-credits
```

`level9-entry`, `level9-silver-arrows`, and `level9-patra` attach one-frame
fail-closed controllers.  Magical Key topology is now a labeled **hypothesis**
graph (ROM cellar dests + walkthrough + live suffix), not live natural-segment
evidence.  Controllers refuse without TF `0xFF` / bombs and never write TF,
bomb capacity, rooms, or doors.  Exact missing-evidence reasons:

- `post_l8_ow_leftover_unmeasured`
- `spectacle_rock_0x05_bomb_entrance_unverified_from_post_l8`
- `old_man_room_0x66_full_tf_gate_unobserved`
- `silver_arrow_room_0x10_unobserved`
- `0x51_north_dest_walk_unverified_statue_diamond`

`door_graph/level9_exits.py` keeps the observed fixture suffix separate from
`LEVEL_9_NATURAL_DOOR_GRAPH` (`level_9_natural_hypothesis`).

### Selected Magical Key route (ZD §10.2 cut; hypothesis; route_eligible=false)

This is Zelda Dungeon **§10.2**, not §10.3, and not the full PNG red line.
Survival refill instead of Red Potion / Red Ring. Skip Compass + Map-Patra.
Hex IDs are RAM `$EB`; the wiki never names them. Wiki locked-door-UP after
the Red Ring backtrack is taken here on the **first visit**.

Red Ring `0x07` excluded (Survival refill; not negligible). Join is the proven
fixture suffix at `0x41`. `requires_51_to_41=True` — do not spend a sitting on
the statue diamond until this prefix is live.

```text
0x76 → 0x66 Old Man TF gate → 0x65 bomb-N → 0x55 Lanmola
  → cellar 0x60 → 0x14 → 0x15 → 0x16 skip Patra → 0x06 bomb-W → 0x05
  → cellar 0x70 → 0x63 → 0x62 (8 Keese corridor, NOT Patra south)
  → 0x61 Patra stairs → cellar 0x75 → 0x20 bomb-N → 0x10 Silver Arrows
join: 0x10 → 0x20 → 0x61 → 0x51 → 0x41 → 0x31 bomb-W → 0x30
  → cellar 0x67 → 0x04 bomb-W → 0x03 → cellar 0x77 left → 0x52
```

Dead beliefs: `0x62` is not a south neighbor of Patra `0x52` (ROM walls; live
8 Keese W/E only). `0x13→0x03` remains a fake loader scroll.

Fixture-live dest hops (`rr-sz8.6`, `route_eligible=false`): play `0x76`
leftover `(120,205)` hold UP → `0x66` **2/2** P1/P2 251 controller frames;
play `0x66` leftover `(120,205)` after west-shutter census (`doors=10`)
hold LEFT → `0x65` **2/2** W1/W2 `(224,141)`. Pin
`Level9Interior65WestReconFixture`. Play `0x65` leftover `(224,141)`
north-band approach `(208,141) -> (208,93) -> (120,93)` bomb-N → `0x55`
Lanmola **2/2** BN1/BN2 424 controller frames / 484 total with census,
leftover `(120,189)` facing UP, doors 4, 10× Lanmola `0x3A` HP32 + 1× `0x68`
stairs trigger HP176 at `(96,144)`. Pin `Level9Interior55NorthReconFixture`.
Play `0x55` leftover `(120,189)` dispatch 10× Lanmola `0x3A` (~404f), align
x=96, push UP block `0x68` from `(96,144)` to `(96,128)`, walk vacated slot
`(96,133)` to center stairs `(128,141)` to trigger mode 16 → cellar `0x60`
**2/2** S1/S2 506 controller frames / 626 total with census, leftover
`(192,93)` facing DOWN on right ladder, doors 0, 4× Keese `0x1B` HP 0. Pin
`Level9Interior60CellarReconFixture`.
Cellar `0x60` walk to west ladder → play `0x14` **2/2** C1/C2 (412f / 532f),
leftover `(96,157)` facing DOWN. Pin `Level9Interior14LikeLikeReconFixture`.
Play `0x14` east key door → `0x15` **2/2** E1/E2 (560f / 680f), leftover
`(16,141)` facing RIGHT. Pin `Level9Interior15ReconFixture`.
Play `0x15` east open door → `0x16` **2/2** E15_1/E15_2 (197f / 317f), leftover
`(32,141)` facing RIGHT. Pin `Level9Interior16PatraReconFixture`.
Play `0x16` skip Patra, north key door → `0x06` **2/2** N16_1/N16_2 (479f / 599f),
leftover `(120,205)` facing UP. Pin `Level9Interior06OldManReconFixture`.
Play `0x06` bomb west → `0x05` **2/2** BW06_1/BW06_2 (574f / 694f), leftover
`(208,173)` facing LEFT. Pin `Level9Interior05StairsReconFixture`.
Play `0x05` block push UP + stairs → cellar `0x70` **2/2** S05_1/S05_2 (482f / 602f),
leftover `(192,93)` facing DOWN on right ladder. Pin `Level9Interior70CellarReconFixture`.
Cellar `0x70` west ladder → play `0x63` **2/2** C70_1/C70_2 (412f / 532f),
leftover `(160,157)` facing DOWN. Pin `Level9Interior63ZolsReconFixture`.
Play `0x63` west key door → `0x62` (8 Keese) **2/2** W63_1/W63_2 (384f / 504f),
leftover `(224,141)` facing LEFT. Pin `Level9Interior62KeeseReconFixture`.
Play `0x62` west open door → `0x61` (other Patra) **2/2** W62_1/W62_2 (197f / 317f),
leftover `(224,141)` facing LEFT. Pin `Level9Interior61PatraReconFixture`.
Play `0x61` defeat Patra + push block + stairs → cellar `0x75` **2/2** S61_1/S61_2
(748f / 868f), leftover `(192,93)` facing DOWN. Pin `Level9Interior75CellarReconFixture`.
Cellar `0x75` west ladder → play `0x20` (Wizzrobes) **2/2** C75_1/C75_2 (412f / 532f),
leftover `(96,157)` facing DOWN. Pin `Level9Interior20ReconFixture`.
Play `0x20` perimeter walk + bomb north → `0x10` (Silver Arrows room) **2/2**
BN20_1/BN20_2 (533f / 653f), leftover `(152,189)` facing UP. Pin
All prefix hops are now 100% fixture-live (16/16 hops).
Natural Silver Arrows prefix (`rr-sz8.6`, `route_eligible=false`): `NaturalSilverArrowsController`
in `natural_path.py` sequences all 16 prefix hops (`0x76 → 0x66 → 0x65 → 0x55 → 0x60 → 0x14 → 0x15 → 0x16 → 0x06 → 0x05 → 0x70 → 0x63 → 0x62 → 0x61 → 0x75 → 0x20 → 0x10`)
into `level9_silver_arrows_chapter(handoff=...)` with 0 memory writes, 0 capacity writes, 0 progression writes. Fails closed without TF 0xFF or complete predecessor handoff.

Natural Patra Join (`rr-sz8.7`, `route_eligible=false`): play `0x10` Silver Arrows leftover `(152,189)`
facing UP through `0x20` → cellar `0x75` → `0x61` → `0x51` (threaded statue diamond corridor)
→ `0x41` (Like-Likes) → `0x31` (bomb west) → `0x30` (block push) → cellar `0x67` → `0x04`
(Keese, y=93 clear aisle corridor) → `0x03` (Zols, block push) → cellar `0x77` → live Patra `0x52`
(body `0x47` + 8 eyes `0x25`). Join runs in 21,156 frames (max 24,000) with 0 deaths, 0 loads,
0 memory writes.

The `level9-credits` chapter executes continuously from the exact live-Patra endpoint: room `0x52`,
body `0x47`, eight eyes `0x25`, north closed, TF `0xFF`, naturally owned Silver Arrows and Bow,
and the Magical Sword. Its fresh controller stages adapt Patra, Ganon, Power Triforce, Zelda, and
credits input policies:
- `level9_select_silver_arrows` (101f): pause-menu cursor navigation only; never assigns `ADDR_SELECTED_ITEM`.
- `level9_final_patra` (1,252f): Patra defeat + north shutter opened.
- `level9_enter_ganon` (301f): north into room 0x42.
- `level9_ganon` (1,534f): 4 Magical Sword hits + Silver Arrow defeat ($0672 != 0).
- `level9_power_triforce` (9f): Power Triforce collected.
- `level9_enter_zelda` (190f): north into room 0x32.
- `level9_rescue_zelda` (71f): fire strikes + center trigger rescue.
- `level9_wait_credits` (1,496f): ending cutscene to credits rolling (mode 0x13, submode 3).
Total end-to-end continuous execution: 26,109 frames (~7.25 minutes) with 0 deaths, 0 loads,
0 memory writes.

This is structural evidence only. It does not promote the ending suffix or make any `*ReconFixture`
route-eligible. Stitch still needs L8 TF `0xFF` + Magic Key + a measured post-L8 OW leftover.

**Beads:** `rr-sz8` (Level 9 epic), `rr-sz8.1` (pre-Ganon → credits),
`rr-sz8.2` (live final Patra → credits), `rr-sz8.3` (room `0x62` disproved;
play `0x03` stairs → cellar `0x77` → Patra **2/2**; `0x13` north wall, not a
clean predecessor; play `0x04` bomb-west → `0x03` → Patra **2/2** recon; play `0x30` stairs → cellar `0x67` right → `0x04` → Patra **2/2** recon; play `0x31` bomb-west → `0x30` → Patra **1/1** recon; play `0x21` south shutter sealed after Patra; play `0x41` north → `0x31` dest **YES** → Patra **1/1** recon; play `0x40` key-north → `0x30` dest **YES**, stays dirty; play `0x51` identified as south pred of `0x41`, north dest walk **NO**),
`rr-yxy6` (statue diamond corridor threaded in 0x51 -> uncleared 0x41),
`rr-sz8.6` (16 prefix hops 0x76 to 0x10 Silver Arrows complete and 2/2 byte-identical),
`rr-sz8.7` (natural Patra join 0x10 -> 0x52, Ganon, Zelda, credits in 26,109 continuous frames, 0 writes, 0 loads, 0 deaths).

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

## Interior (ZD §10.2 Magical Key, CUT)

This is **Zelda Dungeon §10.2**, not §10.3, and not the full PNG red line.
The in-repo selected path is a **cut** of 10.2: Survival refill instead of
Red Potion / Red Ring, and skip Compass + Map-Patra. Hex IDs are **ours
from RAM `$EB`**; the wiki never names them.

Two wiki routes exist: **with Magical Key** (§10.2, selected) vs **without**
(§10.3, not this sitting). Prefer Magical Key for automation.

### Wiki 10.2 → repo `$EB`

| Wiki 10.2 | Repo `$EB` |
|-----------|------------|
| 10.1: bomb left Spectacle Rock; potion one screen left | OW `0x05` left rock; potion **skipped** (Survival refill) |
| Entrance UP → Old Man full-TF gate → LEFT → bomb north → Lanmola → left-block stairs | `0x76` → `0x66` → `0x65` bomb-N → `0x55` → cellar `0x60` |
| Like-Likes, key RIGHT, skip first Patra | `0x14` → `0x15` → `0x16` skip Patra |
| Locked door UP from first Patra ("GO TO THE NEXT ROOM"), bomb LEFT, stairs → Silver Arrows | `0x16` → `0x06` bomb-W → `0x05` → cellar `0x70` → `0x63` → `0x62` → `0x61` → cellar `0x75` → `0x20` bomb-N → `0x10` |
| Like-Likes UP, blade-traps UP, bomb LEFT, stairs, cellar, last Patra, UP Ganon | `0x51` → `0x41` → `0x31` bomb-W → `0x30` → cellar `0x67` → `0x04` → `0x03` → cellar `0x77` left → `0x52` → `0x42` → `0x32` |

Wiki does the locked-door-UP only after the Red Ring backtrack. **We take it
on the first visit.**

### Cut from the PNG red path (do not walk)

- Compass (south of `0x15`)
- First-Patra DOWN → gels → Map Patra ("PATRA - HAS THE MAP") → bomb north → Red Ring (potion icon, `0x07`)
- Red Potion before entry

### Wiki was wrong live (do not follow the map here)

- `0x62` is not south of Patra `0x52` (both walls; live 8 Keese)
- `0x51` north is the right predecessor of `0x41`, but the dest walk is **NO** (statue diamond) — bead `rr-yxy6`; do not spend this sitting on it
- `0x13` → `0x03` is a fake loader scroll

Selected rooms already in `level9/dungeon.py`:
`L9_SELECTED_PREFIX_ROOMS` = `0x76 0x66 0x65 0x55 0x60 0x14 0x15 0x16 0x06 0x05 0x70 0x63 0x62 0x61 0x75 0x20 0x10`;
`L9_SELECTED_JOIN_ROOMS` = `0x10 0x20 0x61 0x51 0x41 0x31 0x30 0x67 0x04 0x03 0x77 0x52`;
plus Ganon `0x42` / Zelda `0x32`. Red Ring `0x07` **out**.

ROM L7–9 door bytes (iNES `0x18A10` / `0x18A90`), not dest-hop proof:

| Room | N | S | W | E |
|------|---|---|---|---|
| `0x76` entry | **open (0)** | open (0) | wall (1) | wall (1) |
| `0x66` Old Man | shutter (7) | open (0) | **shutter (7)** | wall (1) |
| `0x65` | bomb (4) | wall (1) | open (0) | open (0) |

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
| `Level9Interior65WestReconFixture` | live `level==9`, room `0x65`; east mouth after 0x66 west hop |
| `Level9Interior55NorthReconFixture` | live `level==9`, room `0x55`; south mouth after 0x65 bomb-north hop |
| `Level9Interior60CellarReconFixture` | live `level==9`, cellar `0x60`; right ladder after 0x55 stairs hop |
| `Level9Interior14LikeLikeReconFixture` | live `level==9`, room `0x14`; emerged from cellar 0x60 west ladder |
| `Level9Interior15ReconFixture` | live `level==9`, room `0x15`; west mouth after 0x14 east key door |
| `Level9Interior16PatraReconFixture` | live `level==9`, room `0x16`; west mouth after 0x15 east open door |
| `Level9Interior06OldManReconFixture` | live `level==9`, room `0x06`; south mouth after 0x16 north key door |
| `Level9Interior05StairsReconFixture` | live `level==9`, room `0x05`; east mouth after 0x06 bomb-west hop |
| `Level9Interior70CellarReconFixture` | live `level==9`, cellar `0x70`; right ladder after 0x05 stairs hop |
| `Level9Interior63ZolsReconFixture` | live `level==9`, room `0x63`; staircase emergence after cellar 0x70 |
| `Level9Interior62KeeseReconFixture` | live `level==9`, room `0x62`; east mouth after 0x63 west key door |
| `Level9Interior61PatraReconFixture` | live `level==9`, room `0x61`; east mouth after 0x62 west open door |
| `Level9Interior75CellarReconFixture` | live `level==9`, cellar `0x75`; right ladder after 0x61 stairs hop |
| `Level9Interior20ReconFixture` | live `level==9`, room `0x20`; staircase emergence after cellar 0x75 |
| `Level9Interior10SilverArrowsReconFixture` | live `level==9`, room `0x10`; south mouth after 0x20 bomb-north hop |
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

# Fixture-live 0x76 north dest hop (rr-sz8.6). Glance leftover first.
QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/`probe_l9_76_north` \
    --from-state Level9EntranceReconFixture --tag 20260904_P1 \
    --infinite-life --no-video
```

Modules: `level9/overworld.py`, `level9/ganon.py`, `level9/patra.py`,
`level9/path.py`, `level9/room62.py`, `level9/stairs.py`,
`level9/room51.py`.

---

## Evidence boundary

- Live: Spectacle Rock `0x05`; entrance `0x76`; all 16 prefix fixture-live dest
  hops from `0x76` through Silver Arrows `0x10` **2/2** (`rr-sz8.6`: `0x76→0x66`,
  `0x66→0x65`, `0x65→0x55`, `0x55→0x60`, `0x60→0x14`, `0x14→0x15`, `0x15→0x16`,
  `0x16→0x06`, `0x06→0x05`, `0x05→0x70`, `0x70→0x63`, `0x63→0x62`, `0x62→0x61`,
  `0x61→0x75`, `0x75→0x20`, `0x20→0x10`); final Patra `0x52` body/eye
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
  all_dead). Live north dest walk is **earned** (`rr-yxy6`): statue diamond
  threaded via waypoint corridor `(120,205) -> y<=189 -> x<=96 -> y<=141 -> x>=128 -> y<=93 -> x<=120 -> UP`.
  Lands uncleared `0x41` (traps + Like-Likes, doors=4) with no door poke
  (`recordings/l9_room51_dump.json`, 1303 total frames from 0x61).
  Play **0x40** key-north → play `0x30` is live (controller UP from
  south alcove; Magical Key; no 0x30 door poke). 0x40 start is
  fixture-loaded (`route_eligible=false`). Compose suffix through
  0x03 stairs was not pinned (0x68 pushed south).
- Disproved: candidate room `0x62` as cardinal predecessor of `0x52`
  (north wall / south wall; eight Keese; no live north transition).
- Disproved: play room `0x13` as a clean cardinal predecessor of `0x03`
  (ROM north wall / 0x03 south wall; controller UP sticks at y=93;
  0x03 loader door-staging is a fake scroll).
- Solved: play room `0x51` north walk into uncleared `0x41` (ROM +
  visual north open; statue diamond threaded via waypoint navigation, dest YES).
- TF bit map: shards 1–8 = bits `0x01`…`0x80`; full = `0xFF`.
